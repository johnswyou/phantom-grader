"""Stage 3: Student answer extraction with alignment algorithm.

This is the critical stage that solves the alignment problem:
matching student work regions to assignment questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from google import genai

from .. import config
from ..models import ExtractedAnswer, QuestionManifest, StudentExtraction
from ..vision import (
    call_vision,
    extract_json_from_response,
    load_image_part,
    load_images_from_dir,
    image_paths_from_dir,
)


async def _detect_blank_pages_vision(
    client: genai.Client,
    student_paths: list[Path],
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

    response = await call_vision(
        client, config.FLASH_MODEL, prompt, images, temperature=0.1
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
    ocr_text: str | None = None,
) -> StudentExtraction:
    """Stage 3: Extract and align student answers to questions.

    Pipeline:
    0. Vision-based blank page detection on student pages (no template comparison)
    1. For non-blank pages: send to vision model for answer extraction
    2. Questions on blank pages are marked as unanswered
    
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

    # ── Step 0: Vision-based blank page detection ──
    blank_pages = await _detect_blank_pages_vision(client, student_paths)
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

    # ── Step 1: Vision extraction on non-blank pages ──
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

    ocr_context = ""
    if ocr_text:
        max_ocr = 4000
        truncated = ocr_text[:max_ocr] + ("..." if len(ocr_text) > max_ocr else "")
        ocr_context = f"""

OCR text from the student's submission (for cross-reference only — images are primary source):
---
{truncated}
---"""

    prompt = f"""You are an expert at analyzing handwritten student submissions for physics/math assignments.

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
{blank_page_note}{ocr_context}

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

    response = await call_vision(
        client, config.FLASH_MODEL, prompt, all_images, temperature=0.1
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
