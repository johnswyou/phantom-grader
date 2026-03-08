"""Stage 2: Generate solution manual and grading rubric."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from google import genai

from .. import config
from ..models import (
    Question,
    QuestionManifest,
    QuestionRubric,
    Rubric,
    RubricCriterion,
    Solution,
    SolutionManual,
)
from ..vision import (
    call_vision,
    image_paths_from_dir,
    load_image_part,
)

logger = logging.getLogger(__name__)


SOLVE_PAGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "solutions": {
            "type": "OBJECT",
            "additionalProperties": {
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "key_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["answer", "explanation", "key_steps"],
            },
        },
        "rubric": {
            "type": "OBJECT",
            "additionalProperties": {
                "type": "OBJECT",
                "properties": {
                    "total_points": {"type": "INTEGER"},
                    "criteria": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "points": {"type": "INTEGER"},
                                "description": {"type": "STRING"},
                                "type": {"type": "STRING"},
                            },
                            "required": ["points", "description", "type"],
                        },
                    },
                },
                "required": ["total_points", "criteria"],
            },
        },
    },
    "required": ["solutions", "rubric"],
}


VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "correct": {"type": "BOOLEAN"},
        "issues": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["correct", "issues"],
}


async def _solve_page(
    client: genai.Client,
    page_num: int,
    page_questions: list[Question],
    page_image_path: Path,
    page_points: int,
    model: str,
) -> dict:
    """Solve questions for a single page and return raw parsed JSON data."""
    image = load_image_part(page_image_path)

    # Build question list for this page, expanding sub-parts
    question_ids = []
    for q in page_questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                question_ids.append(f"{q.id}{sp}")
        else:
            question_ids.append(q.id)

    questions_json = json.dumps(
        [q.model_dump() for q in page_questions], indent=2
    )

    prompt = f"""You are an expert AP Physics / Math teacher. I'm showing you page {page_num} of a blank assignment template.

Here are the questions on this page:
{questions_json}

For EVERY question on this page (and sub-part if applicable), provide:
1. A complete, correct solution with all work shown
2. A grading rubric with partial credit criteria

For questions with sub-parts (e.g., Q14 with sub_parts ["A", "B"]), create separate solution and rubric entries for each sub-part using IDs like "Q14A", "Q14B".

For MCQ questions: the rubric should be all-or-nothing (full points for correct, 0 for incorrect).

For free-response: break down into meaningful rubric criteria that add up to the question's points.

CRITICAL: For questions with sub-parts, distribute the question's total points across the sub-parts reasonably. The sub-part points must sum to the question's total points.

CRITICAL: The sum of all question points on this page MUST equal {page_points}."""

    response = await call_vision(
        client, model, prompt, [image], temperature=0.1,
        response_schema=SOLVE_PAGE_SCHEMA,
    )
    return json.loads(response)


async def generate_solutions_and_rubric(
    client: genai.Client,
    manifest: QuestionManifest,
    blank_dir: Path,
    *,
    pro_model: str | None = None,
) -> tuple[SolutionManual, Rubric]:
    """Stage 2: Solve all questions and generate rubric.

    Batches by page: each page's questions are solved in a separate LLM call,
    all pages run in parallel, and results are merged.
    """
    blank_dir = Path(blank_dir)
    image_paths = image_paths_from_dir(blank_dir)
    model = pro_model or config.PRO_MODEL

    # Group questions by page
    pages_with_questions: dict[int, list[Question]] = {}
    for q in manifest.questions:
        pages_with_questions.setdefault(q.page, []).append(q)

    # Launch per-page solve calls in parallel
    tasks = []
    page_order = []
    for page_num in sorted(pages_with_questions.keys()):
        page_questions = pages_with_questions[page_num]
        # Get the image for this page (1-indexed)
        if page_num - 1 < len(image_paths):
            page_image = image_paths[page_num - 1]
        else:
            logger.warning("No image found for page %d, skipping", page_num)
            continue
        page_points = manifest.points_per_page.get(str(page_num), 0)
        tasks.append(_solve_page(client, page_num, page_questions, page_image, page_points, model))
        page_order.append(page_num)

    results = await asyncio.gather(*tasks)

    # Merge all page results
    all_solutions: dict[str, Solution] = {}
    all_rubric_items: dict[str, QuestionRubric] = {}

    for page_num, data in zip(page_order, results):
        for qid, sol_data in data.get("solutions", {}).items():
            all_solutions[qid] = Solution(**sol_data)
        for qid, rub_data in data.get("rubric", {}).items():
            criteria = [RubricCriterion(**c) for c in rub_data.get("criteria", [])]
            all_rubric_items[qid] = QuestionRubric(
                total_points=rub_data["total_points"],
                criteria=criteria,
            )

    return SolutionManual(solutions=all_solutions), Rubric(rubric=all_rubric_items)


async def verify_solutions(
    client: genai.Client,
    manifest: QuestionManifest,
    solutions: SolutionManual,
    blank_dir: Path,
    *,
    flash_model: str | None = None,
) -> dict:
    """Verify generated solutions for correctness.

    Returns a dict: {question_id: {"verified": bool, "note": str}}
    For MCQ: if manifest has embedded_answer, compare directly (no LLM).
    For free-response: send solution + page image to Flash and ask for verification.
    """
    blank_dir = Path(blank_dir)
    image_paths = image_paths_from_dir(blank_dir)
    model = flash_model or config.FLASH_MODEL

    # Build question lookup
    question_by_id: dict[str, Question] = {}
    for q in manifest.questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                question_by_id[f"{q.id}{sp}"] = q
        else:
            question_by_id[q.id] = q

    results: dict[str, dict] = {}
    llm_tasks: list[tuple[str, asyncio.Task]] = []

    for qid, solution in solutions.solutions.items():
        q = question_by_id.get(qid)
        if q is None:
            results[qid] = {"verified": False, "note": "Question ID not found in manifest"}
            continue

        # MCQ with embedded_answer: direct comparison
        if q.type == "mcq" and q.embedded_answer:
            match = solution.answer.strip().upper() == q.embedded_answer.strip().upper()
            results[qid] = {
                "verified": match,
                "note": "" if match else f"Solution '{solution.answer}' != embedded answer '{q.embedded_answer}'",
            }
            continue

        # Need LLM verification
        page_idx = q.page - 1
        page_image = load_image_part(image_paths[page_idx]) if page_idx < len(image_paths) else None

        prompt = f"""Here is a physics/math question and a proposed solution. Is the solution correct?
Check: arithmetic, units, signs, logical steps.

Question ID: {qid}
Question type: {q.type}
Question snippet: {q.text_snippet}
Points: {q.points}

Proposed solution:
Answer: {solution.answer}
Explanation: {solution.explanation}
Key steps: {json.dumps(solution.key_steps)}"""

        images = [page_image] if page_image else None

        async def _verify(qid_inner: str, p: str, imgs):
            resp = await call_vision(client, model, p, imgs, temperature=0.1,
                                     response_schema=VERIFY_SCHEMA)
            return qid_inner, json.loads(resp)

        llm_tasks.append(asyncio.ensure_future(_verify(qid, prompt, images)))

    # Run all LLM verification calls in parallel
    if llm_tasks:
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        for result in llm_results:
            if isinstance(result, Exception):
                logger.warning("Verification call failed: %s", result)
                continue
            qid, data = result
            correct = data.get("correct", True)
            issues = data.get("issues", [])
            results[qid] = {
                "verified": correct,
                "note": "; ".join(issues) if issues else "",
            }

    return results
