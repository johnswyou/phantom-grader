"""Stage 2: Generate solution manual and grading rubric."""

from __future__ import annotations

import json
from pathlib import Path

from google import genai

from .. import config
from ..models import (
    QuestionManifest,
    QuestionRubric,
    Rubric,
    RubricCriterion,
    Solution,
    SolutionManual,
)
from ..vision import (
    call_vision,
    extract_json_from_response,
    image_paths_from_dir,
    load_image_part,
)


async def generate_solutions_and_rubric(
    client: genai.Client,
    manifest: QuestionManifest,
    blank_dir: Path,
) -> tuple[SolutionManual, Rubric]:
    """Stage 2: Solve all questions and generate rubric."""

    blank_dir = Path(blank_dir)
    image_paths = image_paths_from_dir(blank_dir)

    # Group questions by page for efficient batching
    pages_with_questions: dict[int, list] = {}
    for q in manifest.questions:
        pages_with_questions.setdefault(q.page, []).append(q)

    # Build the full question list with sub-parts expanded
    question_ids = []
    for q in manifest.questions:
        if q.sub_parts:
            for sp in q.sub_parts:
                question_ids.append(f"{q.id}{sp}")
        else:
            question_ids.append(q.id)

    # Send all pages + manifest to the model and ask for solutions + rubric
    images = [load_image_part(p) for p in image_paths]

    manifest_json = manifest.model_dump_json(indent=2)

    prompt = f"""You are an expert AP Physics / Math teacher. I'm showing you all pages of a blank assignment template.

Here is the question manifest:
{manifest_json}

For EVERY question (and sub-part if applicable), provide:
1. A complete, correct solution with all work shown
2. A grading rubric with partial credit criteria

For questions with sub-parts (e.g., Q14 with sub_parts ["A", "B"]), create separate solution and rubric entries for each sub-part using IDs like "Q14A", "Q14B".

For MCQ questions: the rubric should be all-or-nothing (full points for correct, 0 for incorrect).

For free-response: break down into meaningful rubric criteria that add up to the question's points.

CRITICAL: For questions with sub-parts, distribute the question's total points across the sub-parts reasonably. The sub-part points must sum to the question's total points.

CRITICAL: The sum of all question points on each page MUST match these page totals:
{json.dumps(manifest.points_per_page, indent=2)}

Return ONLY valid JSON in this format:
```json
{{
  "solutions": {{
    "Q1": {{
      "answer": "A",
      "explanation": "By Coulomb's law...",
      "key_steps": ["step1", "step2"]
    }},
    "Q14A": {{
      "answer": "ΔU_g = mgh = ...",
      "explanation": "...",
      "key_steps": ["Identify h = L(1-cosθ)", "Substitute values"]
    }}
  }},
  "rubric": {{
    "Q1": {{
      "total_points": 2,
      "criteria": [
        {{"points": 2, "description": "Correct answer (A)", "type": "all_or_nothing"}}
      ]
    }},
    "Q14A": {{
      "total_points": 5,
      "criteria": [
        {{"points": 2, "description": "Correct setup", "type": "partial"}},
        {{"points": 2, "description": "Correct calculation", "type": "partial"}},
        {{"points": 1, "description": "Final answer with units", "type": "partial"}}
      ]
    }}
  }}
}}
```"""

    response = await call_vision(
        client, config.PRO_MODEL, prompt, images, temperature=0.1
    )

    data = extract_json_from_response(response)

    # Parse solutions
    solutions = {}
    for qid, sol_data in data.get("solutions", {}).items():
        solutions[qid] = Solution(**sol_data)
    solution_manual = SolutionManual(solutions=solutions)

    # Parse rubric
    rubric_items = {}
    for qid, rub_data in data.get("rubric", {}).items():
        criteria = [RubricCriterion(**c) for c in rub_data.get("criteria", [])]
        rubric_items[qid] = QuestionRubric(
            total_points=rub_data["total_points"],
            criteria=criteria,
        )
    rubric = Rubric(rubric=rubric_items)

    return solution_manual, rubric
