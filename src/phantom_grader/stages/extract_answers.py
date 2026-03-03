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


async def extract_student_answers(
    client: genai.Client,
    manifest: QuestionManifest,
    student_dir: Path,
    student_name: str,
    blank_dir: Path,
    ocr_text: str | None = None,
) -> StudentExtraction:
    """Stage 3: Extract and align student answers to questions.

    This implements the multi-pass alignment algorithm:
    1. Full Scan Pass - identify all student content regions
    2. Alignment Pass - map regions to questions using the manifest
    3. Extraction Pass - extract answers from aligned regions
    4. OCR cross-reference (if available)
    """

    student_dir = Path(student_dir)
    blank_dir = Path(blank_dir)

    # Load all blank template images
    blank_images = load_images_from_dir(blank_dir)
    blank_paths = image_paths_from_dir(blank_dir)

    # Load all student page images
    student_images = load_images_from_dir(student_dir)
    student_paths = image_paths_from_dir(student_dir)

    if not student_images:
        return StudentExtraction(
            student_name=student_name,
            answers={},
            unanswered=[q.id for q in manifest.questions],
            alignment_warnings=["No student images found"],
        )

    # Build the complete question ID list (including sub-parts)
    all_question_ids = []
    question_lookup: dict[str, dict] = {}
    for q in manifest.questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                qid = f"{q.id}{sp}"
                all_question_ids.append(qid)
                question_lookup[qid] = {
                    "parent_id": q.id,
                    "page": q.page,
                    "type": q.type,
                    "points": q.points,
                    "sub_part": sp,
                    "text_snippet": q.text_snippet,
                    "options": q.options,
                }
        else:
            all_question_ids.append(q.id)
            question_lookup[q.id] = {
                "page": q.page,
                "type": q.type,
                "points": q.points,
                "text_snippet": q.text_snippet,
                "options": q.options,
            }

    manifest_json = manifest.model_dump_json(indent=2)

    # ── Combined Scan + Alignment + Extraction Pass ──
    # Send ALL images (blank + student) to the model for holistic analysis.
    # The blank images provide structural context; student images contain the work.

    all_images = []
    image_labels = []

    for i, img in enumerate(blank_images):
        all_images.append(img)
        image_labels.append(f"BLANK template page {i+1}")

    for i, img in enumerate(student_images):
        all_images.append(img)
        image_labels.append(f"STUDENT submission page {i+1}")

    ocr_context = ""
    if ocr_text:
        # Truncate OCR if too long to avoid token limits
        max_ocr = 4000
        truncated = ocr_text[:max_ocr] + ("..." if len(ocr_text) > max_ocr else "")
        ocr_context = f"""

Additionally, here is OCR-extracted text from the student's submission for cross-reference:
---
{truncated}
---
Use this to verify your visual reading where helpful, but trust the images as primary source."""

    prompt = f"""You are an expert at analyzing handwritten student submissions for physics/math assignments.

I'm showing you images in this order:
{chr(10).join(f"- Image {i+1}: {label}" for i, label in enumerate(image_labels))}

Here is the question manifest for this assignment:
{manifest_json}

The complete list of question IDs to extract (including sub-parts) is:
{json.dumps(all_question_ids)}
{ocr_context}

YOUR TASK: For each question ID, extract the student's answer by comparing the BLANK template pages against the STUDENT submission pages.

ALIGNMENT STRATEGY:
1. First compare each student page against the corresponding blank template page to identify student-added content (handwriting, pasted images of handwritten work, filled-in bubbles).
2. Use multiple signals for alignment:
   - **Spatial position**: Student work appearing below/near a question on the same page number
   - **Question labels**: Student may write "Q14" or "14)" or "a)" in their handwriting
   - **Content matching**: Mathematical content that matches the question topic
   - **MCQ bubbles**: Colored/filled/shaded bubbles vs empty outlines on MCQ pages
3. Handle edge cases:
   - Student may paste ONE image of handwritten work covering multiple questions
   - Student may continue an answer on the next page
   - Student may leave questions blank (mark as unanswered)
   - On MCQ pages, look for bubbles that are filled/colored/shaded (student answer) vs empty outlines (not selected)

EXTRACTION RULES:
- For MCQ: identify which option letter the student selected (filled/shaded bubble). Report the letter.
- For free-response: transcribe the student's work (equations, steps, final answer). Be thorough.
- For diagrams: describe what was drawn.
- Assign a confidence score (0.0-1.0) based on how clear the alignment and reading is.
- Note the source page(s) and alignment method used.
- Flag any ambiguous cases.

Return ONLY valid JSON:
```json
{{
  "answers": {{
    "Q1": {{
      "response_type": "mcq",
      "selected": "A",
      "work_shown": "",
      "final_answer": "A",
      "confidence": 0.95,
      "evidence": "Green filled bubble at option A, other bubbles are empty outlines",
      "source_pages": [1],
      "alignment_method": "spatial",
      "flags": []
    }},
    "Q14A": {{
      "response_type": "free_response",
      "selected": null,
      "work_shown": "Student writes ΔU = mgh, identifies h = L(1-cos45°)...",
      "final_answer": "0.287 J",
      "confidence": 0.85,
      "evidence": "Handwritten work in answer area below Q14A prompt",
      "source_pages": [4],
      "alignment_method": "spatial+content",
      "flags": []
    }}
  }},
  "unanswered": ["Q13"],
  "alignment_warnings": ["Q16: Student pasted single image covering all sub-parts"]
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

    unanswered = data.get("unanswered", [])
    warnings = data.get("alignment_warnings", [])

    return StudentExtraction(
        student_name=student_name,
        answers=answers,
        unanswered=unanswered,
        alignment_warnings=warnings,
    )
