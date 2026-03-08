"""Stage 4: Grade student work against rubric and solution manual."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from google import genai

from .. import config
from ..models import (
    CriterionGrade,
    QuestionGrade,
    QuestionManifest,
    Rubric,
    SolutionManual,
    StudentExtraction,
    StudentGrades,
)
from ..vision import (
    call_vision,
    extract_json_from_response,
    load_image_part,
    image_paths_from_dir,
)

logger = logging.getLogger(__name__)


def validate_grades(
    grades: dict[str, QuestionGrade],
    rubric: Rubric,
    manifest: QuestionManifest,
) -> list[str]:
    """Validate grade consistency. Returns list of issue descriptions."""
    issues: list[str] = []

    # Build qid → page mapping
    qid_to_page: dict[str, int] = {}
    for q in manifest.questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                qid_to_page[f"{q.id}{sp}"] = q.page
        else:
            qid_to_page[q.id] = q.page

    rubric_qids = set(rubric.rubric.keys())
    grade_qids = set(grades.keys())

    # Every question in rubric should have a grade entry
    missing = rubric_qids - grade_qids
    if missing:
        issues.append(f"Missing grade entries for rubric questions: {sorted(missing)}")

    # No grade entries for IDs not in rubric
    extra = grade_qids - rubric_qids
    if extra:
        issues.append(f"Grade entries for unknown question IDs (not in rubric): {sorted(extra)}")

    for qid, grade in grades.items():
        # Criterion earned <= possible
        for cb in (grade.criteria_breakdown or []):
            if cb.earned > cb.possible + 0.1:
                issues.append(
                    f"{qid}: criterion '{cb.criterion}' earned {cb.earned} > possible {cb.possible}"
                )

        # Points earned <= points possible
        if grade.points_earned > grade.points_possible + 0.1:
            issues.append(
                f"{qid}: points_earned {grade.points_earned} > points_possible {grade.points_possible}"
            )

        # Sum of criteria breakdown earned == points_earned
        if grade.criteria_breakdown:
            criteria_sum = sum(cb.earned for cb in grade.criteria_breakdown)
            if abs(criteria_sum - grade.points_earned) > 0.1:
                issues.append(
                    f"{qid}: criteria sum {criteria_sum} != points_earned {grade.points_earned} (auto-fixable)"
                )

    # Page totals check
    page_earned: dict[str, float] = {}
    for qid, grade in grades.items():
        page = qid_to_page.get(qid, 0)
        page_str = str(page)
        page_earned[page_str] = page_earned.get(page_str, 0) + grade.points_earned
    for page_str, cap in manifest.points_per_page.items():
        earned = page_earned.get(page_str, 0)
        if earned > cap + 0.1:
            issues.append(
                f"Page {page_str}: total earned {earned} > page cap {cap}"
            )

    return issues


async def grade_student(
    client: genai.Client,
    manifest: QuestionManifest,
    extraction: StudentExtraction,
    rubric: Rubric,
    solutions: SolutionManual,
    student_dir: Path,
    blank_dir: Path,
    *,
    pro_model: str | None = None,
) -> StudentGrades:
    """Stage 4: Grade a student's extracted answers against rubric."""

    student_dir = Path(student_dir)
    blank_dir = Path(blank_dir)

    student_paths = image_paths_from_dir(student_dir)
    blank_paths = image_paths_from_dir(blank_dir)

    # Build question ID → page mapping
    qid_to_page: dict[str, int] = {}
    for q in manifest.questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                qid_to_page[f"{q.id}{sp}"] = q.page
        else:
            qid_to_page[q.id] = q.page

    # Collect all question IDs we need to grade
    all_qids = list(rubric.rubric.keys())

    # Build grading context
    rubric_json = rubric.model_dump_json(indent=2)
    solutions_json = solutions.model_dump_json(indent=2)
    extraction_json = extraction.model_dump_json(indent=2)

    # Identify which student pages are relevant
    relevant_student_pages: set[int] = set()
    for qid in all_qids:
        if qid in extraction.answers:
            for p in extraction.answers[qid].source_pages:
                relevant_student_pages.add(p)
        if qid in qid_to_page:
            relevant_student_pages.add(qid_to_page[qid])

    # Load relevant images
    images = []
    image_labels = []

    # Include blank pages for context
    for i, bp in enumerate(blank_paths):
        images.append(load_image_part(bp))
        image_labels.append(f"BLANK template page {i+1}")

    # Include relevant student pages
    for page_num in sorted(relevant_student_pages):
        if 1 <= page_num <= len(student_paths):
            images.append(load_image_part(student_paths[page_num - 1]))
            image_labels.append(f"STUDENT submission page {page_num}")

    prompt = f"""You are an expert AP Physics / Math grader. Grade the student's work precisely according to the rubric.

I'm showing you images:
{chr(10).join(f"- Image {i+1}: {label}" for i, label in enumerate(image_labels))}

STUDENT: {extraction.student_name}

EXTRACTION (what the student wrote/selected per question):
{extraction_json}

SOLUTION MANUAL (correct answers):
{solutions_json}

RUBRIC (grading criteria):
{rubric_json}

GRADING INSTRUCTIONS:
1. For each question ID in the rubric, grade the student's answer.
2. For MCQ questions: Compare selected answer to correct answer. Award full points if correct, 0 if incorrect.
3. For free-response: Apply each rubric criterion. Award partial credit based on work shown.
4. If a question is listed as unanswered, award 0 points.
5. If the extraction has low confidence (< 0.7), look at the student images carefully for that question before grading.
6. Be fair but strict. Only award points when the criterion is clearly met.
7. Provide brief, constructive feedback for each question.

CRITICAL: Points for each criterion must not exceed the criterion's possible points.
CRITICAL: Total earned per question must not exceed total possible.

Return ONLY valid JSON:
```json
{{
  "grades": {{
    "Q1": {{
      "points_earned": 2,
      "points_possible": 2,
      "correct": true,
      "criteria_breakdown": [
        {{"criterion": "Correct answer", "earned": 2, "possible": 2, "note": ""}}
      ],
      "feedback": "Correct."
    }},
    "Q14A": {{
      "points_earned": 4,
      "points_possible": 5,
      "criteria_breakdown": [
        {{"criterion": "Correct setup", "earned": 2, "possible": 2, "note": ""}},
        {{"criterion": "Correct calculation", "earned": 2, "possible": 2, "note": ""}},
        {{"criterion": "Final answer with units", "earned": 0, "possible": 1, "note": "Missing units"}}
      ],
      "feedback": "Good work. Remember to include units."
    }}
  }}
}}
```"""

    model = pro_model or config.PRO_MODEL
    response = await call_vision(
        client, model, prompt, images, temperature=0.1
    )

    data = extract_json_from_response(response)

    # Parse grades
    grades: dict[str, QuestionGrade] = {}
    for qid, grade_data in data.get("grades", {}).items():
        criteria = [
            CriterionGrade(**c) for c in grade_data.get("criteria_breakdown", [])
        ]
        grades[qid] = QuestionGrade(
            points_earned=grade_data["points_earned"],
            points_possible=grade_data["points_possible"],
            correct=grade_data.get("correct"),
            criteria_breakdown=criteria,
            feedback=grade_data.get("feedback", ""),
        )

    # ── Validate and auto-fix grades ──
    validation_issues = validate_grades(grades, rubric, manifest)
    if validation_issues:
        for issue in validation_issues:
            logger.warning("Grade validation [%s]: %s", extraction.student_name, issue)

    # Auto-fix: recalculate points_earned from criteria sum when they mismatch
    for qid, grade in grades.items():
        if grade.criteria_breakdown:
            criteria_sum = sum(cb.earned for cb in grade.criteria_breakdown)
            if abs(criteria_sum - grade.points_earned) > 0.1:
                logger.warning(
                    "Auto-fix [%s] %s: recalculated points_earned %s -> %s (from criteria sum)",
                    extraction.student_name, qid, grade.points_earned, criteria_sum,
                )
                grade.points_earned = criteria_sum

    # Compute page totals
    page_totals: dict[str, float] = {}
    for qid, grade in grades.items():
        page = qid_to_page.get(qid, 0)
        page_str = str(page)
        page_totals[page_str] = page_totals.get(page_str, 0) + grade.points_earned

    # Enforce page caps from manifest
    scaled_pages: list[str] = []
    for page_str, cap in manifest.points_per_page.items():
        if page_str in page_totals and page_totals[page_str] > cap:
            overage = page_totals[page_str] - cap
            if overage > 2:
                # Large overage: log a prominent warning
                # TODO: implement re-grade loop for pages that exceed cap by > 2 points
                logger.warning(
                    "Page %s for %s exceeds cap by %.1f points (%.1f > %d). "
                    "Consider re-grading this page.",
                    page_str, extraction.student_name, overage, page_totals[page_str], cap,
                )
            # Scale down proportionally (for both small and large overages)
            ratio = cap / page_totals[page_str]
            for qid, grade in grades.items():
                if str(qid_to_page.get(qid, 0)) == page_str:
                    grade.points_earned = round(grade.points_earned * ratio, 1)
            page_totals[page_str] = cap
            scaled_pages.append(page_str)

    if scaled_pages:
        logger.warning(
            "Proportionally scaled grades on page(s) %s for %s to enforce page caps",
            scaled_pages, extraction.student_name,
        )

    total_earned = sum(g.points_earned for g in grades.values())
    total_possible = sum(g.points_possible for g in grades.values())

    return StudentGrades(
        student_name=extraction.student_name,
        grades=grades,
        page_totals=page_totals,
        total={"earned": total_earned, "possible": total_possible},
    )
