"""Stage 3: Student answer extraction with alignment algorithm.

This is the critical stage that solves the alignment problem:
matching student work regions to assignment questions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from google import genai

from .. import config
from ..models import ExtractedAnswer, QuestionManifest, StudentExtraction
from ..utils.image import crop_region_to_part
from ..vision import (
    call_vision,
    detect_content_regions,
    extract_json_from_response,
    load_image_part,
    load_images_from_dir,
    image_paths_from_dir,
)


async def _detect_blank_pages_vision(
    client: genai.Client,
    student_paths: list[Path],
    *,
    flash_model: str | None = None,
) -> dict[int, bool]:
    """Use vision to detect which student pages have NO student-added content.

    Sends all student pages and asks: which pages have student work?
    A page with only the printed template (questions, pre-marked answer key,
    headers) and NO student additions (handwriting, colored bubbles, pasted
    images) is considered blank.
    """
    if not student_paths:
        return {}

    images = [load_image_part(p) for p in student_paths]
    image_labels = [f"Student page {i+1}" for i in range(len(student_paths))]

    prompt = f"""I'm showing you {len(student_paths)} pages of a student's assignment submission.

Images in order:
{chr(10).join(f"- Image {i+1}: {label}" for i, label in enumerate(image_labels))}

For EACH page, determine whether the student added ANY of their own work. Student work includes:
- Handwriting (equations, text, calculations)
- Pasted images of handwritten solutions
- Colored/filled MCQ bubbles that the student selected (distinctly colored — green, orange, blue, etc.)
- Drawn diagrams or graphs
- Checkmarks, circles, or any marks the student made

A page is BLANK if it contains ONLY the printed template — the pre-printed questions, headers,
answer key markings, page numbers, class codes, etc. — with NO student additions whatsoever.

IMPORTANT: Some templates have a pre-marked answer key (correct answers shown as filled/highlighted
bubbles). These are TEMPLATE markings, NOT student work. If a page only has these template markings
and no additional student selections, it is BLANK.

Return ONLY valid JSON:
```json
{{
  "pages": {{
    "1": {{"has_student_work": true, "evidence": "Student wrote equations in the answer area"}},
    "2": {{"has_student_work": false, "evidence": "Only printed template, no student additions"}},
    "3": {{"has_student_work": true, "evidence": "Pasted image of handwritten solution"}}
  }}
}}
```"""

    model = flash_model or config.FLASH_MODEL
    response = await call_vision(
        client, model, prompt, images, temperature=0.1
    )

    data = extract_json_from_response(response)

    result = {}
    pages_data = data.get("pages", {})
    for page_str, info in pages_data.items():
        page_num = int(page_str)
        is_blank = not info.get("has_student_work", True)
        result[page_num] = is_blank

    return result


async def extract_student_answers(
    client: genai.Client,
    manifest: QuestionManifest,
    student_dir: Path,
    student_name: str,
    blank_dir: Path,
    *,
    flash_model: str | None = None,
) -> StudentExtraction:
    """Stage 3: Extract and align student answers to questions.

    Pipeline:
    0. Vision-based blank page detection on student pages (no template comparison)
    1. For non-blank pages: send to vision model for answer extraction
    2. Questions on blank pages are marked as unanswered

    When config.ENABLE_ZOOM is True, adds crop-and-zoom steps:
    0. Blank page detection (unchanged)
    1. Region detection on non-blank pages (parallel)
    2. Crop detected regions for higher-resolution extraction
    3. Send crops + full pages (as reference) to vision model

    The blank template is used ONLY for structural context (understanding
    question layout), never for comparison against student work.
    """

    student_dir = Path(student_dir)
    blank_dir = Path(blank_dir)

    # Load paths
    blank_paths = image_paths_from_dir(blank_dir)
    student_paths = image_paths_from_dir(student_dir)

    if not student_paths:
        return StudentExtraction(
            student_name=student_name,
            answers={},
            unanswered=[q.id for q in manifest.questions],
            alignment_warnings=["No student images found"],
        )

    flash = flash_model or config.FLASH_MODEL

    # ── Identify MCQ-heavy pages (exempt from blank detection) ──
    # Pages where any question is MCQ should never be auto-skipped, because
    # a student who answered MCQs correctly produces pages visually identical
    # to the blank template (their selected bubble matches the answer key).
    mcq_pages: set[int] = set()
    for q in manifest.questions:
        if q.type == "mcq":
            mcq_pages.add(q.page)

    # ── Step 0: Vision-based blank page detection ──
    blank_pages = await _detect_blank_pages_vision(client, student_paths, flash_model=flash)

    # Override: never mark MCQ pages as blank
    for page_num in mcq_pages:
        if page_num in blank_pages and blank_pages[page_num]:
            logger.info(
                "Page %d has MCQ questions — overriding blank detection (was marked blank)",
                page_num,
            )
            blank_pages[page_num] = False

    blank_page_nums = sorted([p for p, is_blank in blank_pages.items() if is_blank])
    non_blank_page_nums = sorted([p for p, is_blank in blank_pages.items() if not is_blank])

    # Build question ID lists
    all_question_ids = []
    question_lookup: dict[str, dict] = {}
    questions_on_blank_pages: list[str] = []

    for q in manifest.questions:
        qids = []
        if q.sub_parts:
            qids = [f"{q.id}{sp}" for sp in q.sub_parts]
        else:
            qids = [q.id]

        for qid in qids:
            all_question_ids.append(qid)
            question_lookup[qid] = {
                "page": q.page,
                "type": q.type,
                "points": q.points,
                "text_snippet": q.text_snippet,
                "options": q.options,
            }
            if q.page in blank_page_nums:
                questions_on_blank_pages.append(qid)

    # If ALL pages are blank, skip the extraction call entirely
    if not non_blank_page_nums:
        return StudentExtraction(
            student_name=student_name,
            answers={},
            unanswered=all_question_ids,
            alignment_warnings=[
                f"All {len(blank_page_nums)} pages detected as blank (no student work found via vision analysis)"
            ],
        )

    # Questions to extract (only those on non-blank pages)
    extractable_qids = [qid for qid in all_question_ids if qid not in questions_on_blank_pages]
    manifest_json = manifest.model_dump_json(indent=2)

    blank_page_note = ""
    if blank_page_nums:
        blank_page_note = f"""

BLANK PAGES DETECTED:
Pages {blank_page_nums} have been confirmed as having NO student work (via separate vision analysis).
Questions on those pages are pre-marked as unanswered — do NOT extract answers for them.
Only extract answers for questions on pages: {non_blank_page_nums}.
"""

    if config.ENABLE_ZOOM:
        response = await _extract_with_zoom(
            client,
            student_paths,
            non_blank_page_nums,
            blank_dir,
            extractable_qids,
            manifest_json,
            blank_page_note,
            flash_model=flash,
        )
    else:
        response = await _extract_without_zoom(
            client,
            student_paths,
            non_blank_page_nums,
            blank_dir,
            extractable_qids,
            manifest_json,
            blank_page_note,
            flash_model=flash,
        )

    data = extract_json_from_response(response)

    # Parse answers
    answers = {}
    for qid, ans_data in data.get("answers", {}).items():
        answers[qid] = ExtractedAnswer(**ans_data)

    # Combine unanswered: blank-page questions + model-reported unanswered
    unanswered = list(set(questions_on_blank_pages + data.get("unanswered", [])))

    warnings = data.get("alignment_warnings", [])
    if blank_page_nums:
        warnings.insert(
            0,
            f"Pages {blank_page_nums} detected as blank (no student work) — questions on these pages marked unanswered",
        )

    return StudentExtraction(
        student_name=student_name,
        answers=answers,
        unanswered=unanswered,
        alignment_warnings=warnings,
    )


async def _extract_without_zoom(
    client: genai.Client,
    student_paths: list[Path],
    non_blank_page_nums: list[int],
    blank_dir: Path,
    extractable_qids: list[str],
    manifest_json: str,
    blank_page_note: str,
    *,
    flash_model: str | None = None,
) -> str:
    """Original extraction path: send full page images to the model."""
    all_images = []
    image_labels = []

    # Include blank template pages for structural context only
    for i, img_part in enumerate(load_images_from_dir(blank_dir)):
        all_images.append(img_part)
        image_labels.append(f"BLANK template page {i+1} (for question structure reference only)")

    # Only send non-blank student pages
    for page_num in non_blank_page_nums:
        idx = page_num - 1
        if idx < len(student_paths):
            all_images.append(load_image_part(student_paths[idx]))
            image_labels.append(f"STUDENT submission page {page_num}")

    prompt = _build_extraction_prompt(
        image_labels, extractable_qids, manifest_json, blank_page_note
    )

    model = flash_model or config.FLASH_MODEL
    return await call_vision(
        client, model, prompt, all_images, temperature=0.1
    )


async def _extract_with_zoom(
    client: genai.Client,
    student_paths: list[Path],
    non_blank_page_nums: list[int],
    blank_dir: Path,
    extractable_qids: list[str],
    manifest_json: str,
    blank_page_note: str,
    *,
    flash_model: str | None = None,
) -> str:
    """Zoom extraction path: detect regions, crop, and send crops + full pages."""

    # ── Step 1: Region detection (parallel) ──
    detection_tasks = []
    page_path_map: dict[int, Path] = {}
    for page_num in non_blank_page_nums:
        idx = page_num - 1
        if idx < len(student_paths):
            page_path_map[page_num] = student_paths[idx]
            detection_tasks.append(
                detect_content_regions(client, student_paths[idx], page_num)
            )

    detection_results = await asyncio.gather(*detection_tasks)

    # Map page_num -> regions
    page_regions: dict[int, list[dict]] = {}
    for page_num, regions in zip(page_path_map.keys(), detection_results):
        page_regions[page_num] = regions

    # ── Step 2: Crop regions and build image list ──
    all_images = []
    image_labels = []

    # Include blank template pages for structural context only
    for i, img_part in enumerate(load_images_from_dir(blank_dir)):
        all_images.append(img_part)
        image_labels.append(f"BLANK template page {i+1} (for question structure reference only)")

    # Add cropped regions for each non-blank page
    for page_num in sorted(page_path_map.keys()):
        regions = page_regions.get(page_num, [])
        if regions:
            for region in regions:
                crop_part = crop_region_to_part(
                    page_path_map[page_num],
                    region,
                    pad_pct=config.REGION_PAD_PCT,
                )
                all_images.append(crop_part)
                location = (
                    f"x={region.get('x_pct', 0):.0f}% y={region.get('y_pct', 0):.0f}% "
                    f"w={region.get('w_pct', 0):.0f}% h={region.get('h_pct', 0):.0f}%"
                )
                content_type = region.get("content_type", "unknown")
                label_text = region.get("label", "")
                image_labels.append(
                    f"CROPPED region from page {page_num} [{content_type}] "
                    f"at ({location}): {label_text}"
                )
        else:
            # No regions detected — fall back to full page as a primary image
            all_images.append(load_image_part(page_path_map[page_num]))
            image_labels.append(
                f"STUDENT submission page {page_num} (no regions detected — full page)"
            )

    # Add full page images as reference context (after crops)
    for page_num in sorted(page_path_map.keys()):
        regions = page_regions.get(page_num, [])
        if regions:
            # Only add full-page reference if we sent crops (avoid duplicating
            # pages that were already sent as full-page fallback above)
            all_images.append(load_image_part(page_path_map[page_num]))
            image_labels.append(f"FULL PAGE (reference) — student page {page_num}")

    prompt = _build_zoom_extraction_prompt(
        image_labels, extractable_qids, manifest_json, blank_page_note
    )

    model = flash_model or config.FLASH_MODEL
    return await call_vision(
        client, model, prompt, all_images, temperature=0.1
    )


def _build_extraction_prompt(
    image_labels: list[str],
    extractable_qids: list[str],
    manifest_json: str,
    blank_page_note: str,
) -> str:
    """Build the extraction prompt for the non-zoom path."""
    return f"""You are an expert at analyzing handwritten student submissions for physics/math assignments.

I'm showing you images in this order:
{chr(10).join(f"- Image {i+1}: {label}" for i, label in enumerate(image_labels))}

The BLANK template images are provided ONLY so you understand the question structure and layout.
Do NOT compare blank template content against student pages to determine answers.
The blank template may contain a pre-marked answer key — those markings also appear on student
pages and are NOT student answers.

Here is the question manifest:
{manifest_json}

Question IDs to extract (ONLY these):
{json.dumps(extractable_qids)}
{blank_page_note}

YOUR TASK: For each question ID listed above, extract the student's answer from their submission pages.

CRITICAL MCQ RULES:
1. Student MCQ answers are ADDITIONAL markings beyond the printed template. They appear as
   distinctly colored bubbles (green, orange, blue, etc.), or hand-drawn circles/checkmarks.
2. The template itself may have pre-highlighted bubbles showing correct answers — these appear
   identically on every student's page. Ignore them. Only report bubbles that show a DISTINCT
   student selection (different color, additional marking, hand-drawn indicator).
3. If you cannot identify a clear student-added selection for an MCQ question, mark it as unanswered.
4. DO NOT guess or hallucinate MCQ answers. When uncertain, mark as unanswered.

FREE-RESPONSE RULES:
1. Look for handwriting, pasted images of handwritten work, typed text added by the student.
2. Transcribe equations, steps, and final answers.
3. Handle pasted images that may cover multiple questions — identify which question each
   part of the image answers using question numbers, sub-part labels, or content matching.

Return ONLY valid JSON:
```json
{{
  "answers": {{
    "Q1": {{
      "response_type": "mcq",
      "selected": "B",
      "work_shown": "",
      "final_answer": "B",
      "confidence": 0.9,
      "evidence": "Green bubble at option B, clearly a student addition distinct from template",
      "source_pages": [1],
      "alignment_method": "spatial",
      "flags": []
    }},
    "Q14A": {{
      "response_type": "free_response",
      "selected": null,
      "work_shown": "Student writes ΔU = mgh...",
      "final_answer": "0.287 J",
      "confidence": 0.85,
      "evidence": "Handwritten work below Q14A",
      "source_pages": [4],
      "alignment_method": "spatial+content",
      "flags": []
    }}
  }},
  "unanswered": ["Q2", "Q3"],
  "alignment_warnings": []
}}
```"""


def _build_zoom_extraction_prompt(
    image_labels: list[str],
    extractable_qids: list[str],
    manifest_json: str,
    blank_page_note: str,
) -> str:
    """Build the extraction prompt for the zoom path with crops and full-page references."""
    return f"""You are an expert at analyzing handwritten student submissions for physics/math assignments.

I'm showing you images in this order:
{chr(10).join(f"- Image {i+1}: {label}" for i, label in enumerate(image_labels))}

IMAGE LAYOUT EXPLANATION:
- BLANK template images: Provided ONLY for question structure and layout reference.
- CROPPED region images: High-resolution crops of detected student work areas. These are your
  PRIMARY source for reading student handwriting, equations, and bubble selections. Each crop
  is labeled with its source page, content type, and approximate location on the page.
- FULL PAGE (reference) images: Complete student pages provided for spatial context. Use these
  to understand where crops come from and how content relates across the page. The crops above
  are zoomed-in versions of regions on these pages.

Do NOT compare blank template content against student pages to determine answers.
The blank template may contain a pre-marked answer key — those markings also appear on student
pages and are NOT student answers.

Here is the question manifest:
{manifest_json}

Question IDs to extract (ONLY these):
{json.dumps(extractable_qids)}
{blank_page_note}

YOUR TASK: For each question ID listed above, extract the student's answer from the provided images.
Use the CROPPED regions as your primary source for reading fine details (handwriting, equations,
bubble colors). Use the FULL PAGE references for spatial alignment and context.

CRITICAL MCQ RULES:
1. Student MCQ answers are ADDITIONAL markings beyond the printed template. They appear as
   distinctly colored bubbles (green, orange, blue, etc.), or hand-drawn circles/checkmarks.
2. The template itself may have pre-highlighted bubbles showing correct answers — these appear
   identically on every student's page. Ignore them. Only report bubbles that show a DISTINCT
   student selection (different color, additional marking, hand-drawn indicator).
3. If you cannot identify a clear student-added selection for an MCQ question, mark it as unanswered.
4. DO NOT guess or hallucinate MCQ answers. When uncertain, mark as unanswered.

FREE-RESPONSE RULES:
1. Look for handwriting, pasted images of handwritten work, typed text added by the student.
2. Transcribe equations, steps, and final answers.
3. Handle pasted images that may cover multiple questions — identify which question each
   part of the image answers using question numbers, sub-part labels, or content matching.

Return ONLY valid JSON:
```json
{{
  "answers": {{
    "Q1": {{
      "response_type": "mcq",
      "selected": "B",
      "work_shown": "",
      "final_answer": "B",
      "confidence": 0.9,
      "evidence": "Green bubble at option B, clearly a student addition distinct from template",
      "source_pages": [1],
      "alignment_method": "spatial",
      "flags": []
    }},
    "Q14A": {{
      "response_type": "free_response",
      "selected": null,
      "work_shown": "Student writes ΔU = mgh...",
      "final_answer": "0.287 J",
      "confidence": 0.85,
      "evidence": "Handwritten work below Q14A",
      "source_pages": [4],
      "alignment_method": "spatial+content",
      "flags": []
    }}
  }},
  "unanswered": ["Q2", "Q3"],
  "alignment_warnings": []
}}
```"""
