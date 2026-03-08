"""CLI entry point for phantom-grader."""

from __future__ import annotations

import asyncio
import json
import logging
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
    blank_dir: Optional[Path] = typer.Option(None, "--blank-dir", help="Directory of blank template images"),
    student_dir: Optional[Path] = typer.Option(None, "--student-dir", help="Directory of student submission subdirs"),
    points_file: Path = typer.Option(..., "--points-file", help="MAX_POINTS_PER_PAGE.txt"),
    output_dir: Path = typer.Option("graded_output", "--output-dir", help="Output directory"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
    student: Optional[str] = typer.Option(None, "--student", help="Grade only this student (name filter)"),
    flash_model: Optional[str] = typer.Option(None, "--flash-model", help="Override Flash model name"),
    pro_model: Optional[str] = typer.Option(None, "--pro-model", help="Override Pro model name"),
    blank_pdf: Optional[Path] = typer.Option(None, "--blank-pdf", help="Blank template PDF (converted to images)"),
    student_pdf: Optional[Path] = typer.Option(None, "--student-pdf", help="Directory of student PDF files"),
):
    """Run the full grading pipeline."""
    from .pipeline import run_pipeline
    from .utils.pdf import pdf_to_images

    logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')

    # Validate blank source
    if blank_dir and blank_pdf:
        raise typer.BadParameter("Provide either --blank-dir or --blank-pdf, not both.")
    if not blank_dir and not blank_pdf:
        raise typer.BadParameter("Provide either --blank-dir or --blank-pdf.")

    # Validate student source
    if student_dir and student_pdf:
        raise typer.BadParameter("Provide either --student-dir or --student-pdf, not both.")
    if not student_dir and not student_pdf:
        raise typer.BadParameter("Provide either --student-dir or --student-pdf.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert blank PDF to images if needed
    if blank_pdf:
        blank_images_dir = out / "blank_images"
        pdf_to_images(blank_pdf, blank_images_dir)
        blank_dir = blank_images_dir

    # Convert student PDFs to images if needed
    if student_pdf:
        student_pdf_dir = Path(student_pdf)
        student_images_dir = out / "student_images"
        for pdf_file in sorted(student_pdf_dir.iterdir()):
            if pdf_file.suffix.lower() == ".pdf":
                stem_dir = student_images_dir / pdf_file.stem
                pdf_to_images(pdf_file, stem_dir)
        student_dir = student_images_dir

    key = config.get_api_key(api_key)
    _run(run_pipeline(
        blank_dir, student_dir, points_file, output_dir, key, student,
        flash_model=flash_model, pro_model=pro_model,
    ))


@app.command("parse-assignment")
def parse_assignment_cmd(
    blank_dir: Path = typer.Option(..., "--blank-dir", help="Directory of blank template images"),
    points_file: Path = typer.Option(..., "--points-file", help="MAX_POINTS_PER_PAGE.txt"),
    output: Path = typer.Option("question_manifest.json", "--output", help="Output JSON path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
    flash_model: Optional[str] = typer.Option(None, "--flash-model", help="Override Flash model name"),
):
    """Stage 1: Parse blank assignment template."""
    from .stages import parse_assignment as _parse

    logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_parse():
        manifest = await _parse(client, blank_dir, points_file, flash_model=flash_model)
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
    pro_model: Optional[str] = typer.Option(None, "--pro-model", help="Override Pro model name"),
):
    """Stage 2: Generate solutions and rubric."""
    from .stages import generate_solutions_and_rubric

    logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_solve():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        solutions, rubric = await generate_solutions_and_rubric(client, m, blank_dir, pro_model=pro_model)

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
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key"),
    flash_model: Optional[str] = typer.Option(None, "--flash-model", help="Override Flash model name"),
):
    """Stage 3: Extract student answers."""
    from .stages import extract_student_answers

    logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_extract():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        extraction = await extract_student_answers(
            client, m, student_dir, student, blank_dir,
            flash_model=flash_model,
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
    pro_model: Optional[str] = typer.Option(None, "--pro-model", help="Override Pro model name"),
):
    """Stage 4: Grade a student."""
    from .stages import grade_student as _grade

    logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')

    key = config.get_api_key(api_key)
    client = get_client(key)

    async def _run_grade():
        m = QuestionManifest.model_validate_json(Path(manifest).read_text())
        ext = StudentExtraction.model_validate_json(Path(extraction).read_text())
        rub = Rubric.model_validate_json(Path(rubric).read_text())
        sol = SolutionManual.model_validate_json(Path(solutions).read_text())

        grades = await _grade(client, m, ext, rub, sol, student_dir, blank_dir, pro_model=pro_model)
        Path(output).write_text(grades.model_dump_json(indent=2))
        console.print(f"Grades saved to [green]{output}[/]")
        console.print(f"{grades.student_name}: {grades.total['earned']}/{grades.total['possible']}")

    _run(_run_grade())


if __name__ == "__main__":
    app()
