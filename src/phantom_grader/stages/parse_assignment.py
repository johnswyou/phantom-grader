"""Stage 1: Parse the blank assignment template to extract question structure."""

from __future__ import annotations

import json
from pathlib import Path

from google import genai

from .. import config
from ..models import Question, QuestionManifest
from ..vision import call_vision, load_images_from_dir


MANIFEST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "page": {"type": "INTEGER"},
                    "type": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "points": {"type": "INTEGER"},
                    "text_snippet": {"type": "STRING"},
                    "embedded_answer": {"type": "STRING", "nullable": True},
                    "sub_parts": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["id", "page", "type", "points", "text_snippet", "options", "sub_parts"],
            },
        },
    },
    "required": ["questions"],
}


def parse_points_file(points_path: Path) -> dict[str, int]:
    """Parse MAX_POINTS_PER_PAGE.txt → {page_num_str: points}."""
    result = {}
    for line in Path(points_path).read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "PAGE 1: 14" or "Page 1: 14"
        parts = line.split(":")
        page_str = parts[0].strip().upper().replace("PAGE", "").strip()
        points = int(parts[1].strip())
        result[page_str] = points
    return result


async def parse_assignment(
    client: genai.Client,
    blank_dir: Path,
    points_path: Path,
    assignment_name: str | None = None,
    *,
    flash_model: str | None = None,
) -> QuestionManifest:
    """Stage 1: Analyze blank template pages and extract question structure."""

    blank_dir = Path(blank_dir)
    points_per_page = parse_points_file(points_path)
    total_pages = len(list(blank_dir.glob("*.jpg"))) or len(list(blank_dir.glob("*.png")))

    if assignment_name is None:
        assignment_name = blank_dir.name.replace("BLANK-", "")

    # Load all blank template images
    images = load_images_from_dir(blank_dir)

    prompt = f"""You are analyzing a blank physics/math assignment template. I'm showing you {total_pages} pages of the blank template.

The point allocations per page are:
{json.dumps(points_per_page, indent=2)}

For EACH page, identify ALL questions. For each question, determine:
1. **Question ID**: e.g., "Q1", "Q2", ..., "Q14", etc. Number them sequentially across pages.
2. **Page number**: Which page (1-indexed) the question appears on.
3. **Type**: "mcq" if it has multiple-choice bubbles/options, "free_response" otherwise.
4. **Options**: For MCQ, list the option letters (e.g., ["A", "B", "C", "D", "E"]).
5. **Points**: Estimated points for this question. The total per page MUST match the points_per_page values.
6. **Text snippet**: A brief excerpt of the question text (first ~100 chars).
7. **Embedded answer**: If the blank template has a pre-filled/highlighted answer (e.g., a filled bubble), note it. Otherwise null.
8. **Sub-parts**: If the question has labeled sub-parts (a, b, c or A, B, C), list them.

IMPORTANT RULES:
- MCQ questions that just require selecting a letter are typically worth 2 points each.
- Free response questions vary in points based on complexity.
- The sum of question points on each page MUST equal the page's point allocation.
- If a question has sub-parts, the question ID is like "Q14" and sub_parts would be ["A", "B"].
- Number questions sequentially across the entire assignment."""

    model = flash_model or config.FLASH_MODEL
    response = await call_vision(
        client, model, prompt, images, temperature=0.1,
        response_schema=MANIFEST_SCHEMA,
    )

    data = json.loads(response)
    questions = []
    for q in data["questions"]:
        # Normalize nulls to empty lists for list fields
        if q.get("options") is None:
            q["options"] = []
        if q.get("sub_parts") is None:
            q["sub_parts"] = []
        questions.append(Question(**q))

    total_points = sum(int(v) for v in points_per_page.values())

    manifest = QuestionManifest(
        assignment_name=assignment_name,
        total_pages=total_pages,
        total_points=total_points,
        points_per_page=points_per_page,
        questions=questions,
    )

    return manifest
