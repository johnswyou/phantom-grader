"""CLI entry point for phantom-grader."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import config
from .models import (
    QuestionManifest,
    Rubric,
    SolutionManual,
    StudentExtraction,
)
from .vision import get_client

app = typer.Typer(
    name="phantom-grader",
    help="Agentic auto-grading system for handwritten physics/math assignments.",
)
console = Console()


def _run(coro):
    """Run an async coroutine."""
    return asyncio.run(coro)


@app.command()
def grade(
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    student_dir: Path = typer.Option(..., "--student-dir", help="Directory of student submission subdirs"),
    points_file: Path = typer.Option(..., "--points-file", help="MAX_POINTS_PER_PAGE.txt"),
    output_dir: Path = typer.Option("graded_output", "--output-dir", help="Output directory"),
    ocr_dir: Optional[Path] = typer.Option(None, "--ocr-dir", help="Directory of OCR markdown files"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
    student: Optional[str] = typer.Option(None, "--student", help="Grade only this student (name filter)"),
):
    """Run the full grading pipeline."""
    from .pipeline import run_pipeline

    key = config.get_api_key(api_key)
    _run(run_pipeline(blank_dir, student_dir, points_file, output_dir, key, ocr_dir, student))


@app.command("parse-assignment")
def parse_assignment_cmd(
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    points_file: Path = typer.Option(..., "--points-file", help="MAX_POINTS_PER_PAGE.txt"),
    output: Path = typer.Option("question_manifest.json", "--output", help="Output JSON path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
):
    """Stage 1: Parse blank assignment template."""
    from .stages import parse_assignment as _parse

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_parse():
        manifest = await _parse(client, blank_dir, points_file)
        Path(output).write_text(manifest.model_dump_json(indent=2))
        console.print(f"Manifest saved to [green]{output}[/]")
        console.print(f"Found {len(manifest.questions)} questions, {manifest.total_points} total points")

    _run(_run_parse())


@app.command("solve")
def solve_cmd(
    manifest: Path = typer.Option(..., "--manifest", help="question_manifest.json"),
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    output: Path = typer.Option("solutions.json", "--output", help="Output JSON path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
):
    """Stage 2: Generate solutions and rubric."""
    from .stages import generate_solutions_and_rubric

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_solve():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        solutions, rubric = await generate_solutions_and_rubric(client, m, blank_dir)

        out = Path(output)
        sol_path = out.parent / "solution_manual.json"
        rub_path = out.parent / "rubric.json"
        sol_path.write_text(solutions.model_dump_json(indent=2))
        rub_path.write_text(rubric.model_dump_json(indent=2))
        console.print(f"Solutions: [green]{sol_path}[/]")
        console.print(f"Rubric: [green]{rub_path}[/]")

    _run(_run_solve())


@app.command("extract")
def extract_cmd(
    manifest: Path = typer.Option(..., "--manifest", help="question_manifest.json"),
    student_dir: Path = typer.Option(..., "--student-dir", help="Student image directory"),
    student: str = typer.Option(..., "--student", help="Student name"),
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    output: Path = typer.Option("extraction.json", "--output", help="Output JSON path"),
    ocr_file: Optional[Path] = typer.Option(None, "--ocr-file", help="OCR markdown file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
):
    """Stage 3: Extract student answers."""
    from .stages import extract_student_answers

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_extract():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        ocr_text = Path(ocr_file).read_text() if ocr_file else None
        extraction = await extract_student_answers(
            client, m, student_dir, student, blank_dir, ocr_text
        )
        Path(output).write_text(extraction.model_dump_json(indent=2))
        console.print(f"Extraction saved to [green]{output}[/]")

    _run(_run_extract())


@app.command("grade-student")
def grade_student_cmd(
    extraction: Path = typer.Option(..., "--extraction", help="student_extraction.json"),
    rubric: Path = typer.Option(..., "--rubric", help="rubric.json"),
    solutions: Path = typer.Option(..., "--solutions", help="solution_manual.json"),
    manifest: Path = typer.Option(..., "--manifest", help="question_manifest.json"),
    student_dir: Path = typer.Option(..., "--student-dir", help="Student image directory"),
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    output: Path = typer.Option("grades.json", "--output", help="Output JSON path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
):
    """Stage 4: Grade a student."""
    from .stages import grade_student as _grade

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_grade():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        ext = StudentExtraction.model_validate_json(Path(extraction).read_text())
        rub = Rubric.model_validate_json(Path(rubric).read_text())
        sol = SolutionManual.model_validate_json(Path(solutions).read_text())

        grades = await _grade(client, m, ext, rub, sol, student_dir, blank_dir)
        Path(output).write_text(grades.model_dump_json(indent=2))
        console.print(f"Grades saved to [green]{output}[/]")
        console.print(f"{grades.student_name}: {grades.total['earned']}/{grades.total['possible']}")

    _run(_run_grade())


if __name__ == "__main__":
    app()
