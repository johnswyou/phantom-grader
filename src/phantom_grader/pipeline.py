"""Main pipeline orchestrator for phantom-grader."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from google import genai

from . import config
from .models import (
    QuestionManifest,
    Rubric,
    SolutionManual,
    StudentExtraction,
    StudentGrades,
)
from .stages import (
    extract_student_answers,
    generate_reports,
    generate_solutions_and_rubric,
    grade_student,
    parse_assignment,
)
from .vision import get_client

console = Console()


def _save_json(data, path: Path) -> None:
    """Save a Pydantic model or dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump_json"):
        path.write_text(data.model_dump_json(indent=2))
    else:
        path.write_text(json.dumps(data, indent=2, default=str))


def discover_students(student_dir: Path) -> list[tuple[str, Path]]:
    """Discover student subdirectories.

    Returns list of (student_name, student_subdir_path).
    """
    student_dir = Path(student_dir)
    results = []
    for subdir in sorted(student_dir.iterdir()):
        if subdir.is_dir():
            # Extract student name from directory name
            # Format: "Assignment Name - Student Name"
            name = subdir.name
            if " - " in name:
                name = name.rsplit(" - ", 1)[-1]
            results.append((name, subdir))
    return results


def find_ocr_text(ocr_dir: Path | None, student_name: str) -> str | None:
    """Find OCR markdown file for a student, if available."""
    if ocr_dir is None:
        return None
    ocr_dir = Path(ocr_dir)
    if not ocr_dir.exists():
        return None

    # Look for a file matching the student name
    for f in ocr_dir.iterdir():
        if f.suffix == ".md" and student_name in f.name:
            return f.read_text()
    return None


async def run_pipeline(
    blank_dir: Path,
    student_dir: Path,
    points_file: Path,
    output_dir: Path,
    api_key: str,
    ocr_dir: Path | None = None,
    student_filter: str | None = None,
) -> None:
    """Run the full grading pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(api_key)

    # ── Stage 1: Parse Assignment ────────────────────────────────────
    console.print("\n[bold blue]Stage 1:[/] Parsing assignment template...")

    manifest_path = output_dir / "question_manifest.json"
    if manifest_path.exists():
        console.print("  Loading cached manifest...")
        manifest = QuestionManifest.model_validate_json(manifest_path.read_text())
    else:
        manifest = await parse_assignment(client, blank_dir, points_file)
        _save_json(manifest, manifest_path)
    console.print(
        f"  Found [green]{len(manifest.questions)}[/] questions across "
        f"[green]{manifest.total_pages}[/] pages "
        f"([green]{manifest.total_points}[/] points total)"
    )

    # ── Stage 2: Solution + Rubric ───────────────────────────────────
    console.print("\n[bold blue]Stage 2:[/] Generating solutions and rubric...")

    solutions_path = output_dir / "solution_manual.json"
    rubric_path = output_dir / "rubric.json"
    if solutions_path.exists() and rubric_path.exists():
        console.print("  Loading cached solutions and rubric...")
        solutions = SolutionManual.model_validate_json(solutions_path.read_text())
        rubric = Rubric.model_validate_json(rubric_path.read_text())
    else:
        solutions, rubric = await generate_solutions_and_rubric(
            client, manifest, blank_dir
        )
        _save_json(solutions, solutions_path)
        _save_json(rubric, rubric_path)
    console.print(
        f"  Generated solutions for [green]{len(solutions.solutions)}[/] questions, "
        f"rubric for [green]{len(rubric.rubric)}[/] questions"
    )

    # ── Discover Students ────────────────────────────────────────────
    all_students = discover_students(student_dir)
    if student_filter:
        all_students = [
            (name, path)
            for name, path in all_students
            if student_filter.lower() in name.lower()
        ]
    console.print(
        f"\n[bold]Processing [green]{len(all_students)}[/] student(s):[/] "
        + ", ".join(name for name, _ in all_students)
    )

    if not all_students:
        console.print("[yellow]No students found matching filter.[/]")
        return

    # ── Stage 3 + 4: Extract + Grade (per student, parallel) ────────
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_STUDENTS)
    all_grades: list[StudentGrades] = []

    async def process_student(name: str, stu_dir: Path) -> StudentGrades | None:
        async with sem:
            console.print(f"\n[bold blue]Stage 3:[/] Extracting answers for [cyan]{name}[/]...")

            ocr_text = find_ocr_text(ocr_dir, name)

            extraction_path = output_dir / f"student_extraction_{name.replace(' ', '_')}.json"
            if extraction_path.exists():
                console.print(f"  Loading cached extraction for {name}...")
                extraction = StudentExtraction.model_validate_json(
                    extraction_path.read_text()
                )
            else:
                extraction = await extract_student_answers(
                    client, manifest, stu_dir, name, blank_dir, ocr_text
                )
                _save_json(extraction, extraction_path)

            n_answered = len(extraction.answers)
            n_unanswered = len(extraction.unanswered)
            console.print(
                f"  {name}: [green]{n_answered}[/] answered, "
                f"[yellow]{n_unanswered}[/] unanswered"
            )
            if extraction.alignment_warnings:
                for w in extraction.alignment_warnings:
                    console.print(f"  [yellow]Warning:[/] {w}")

            console.print(f"\n[bold blue]Stage 4:[/] Grading [cyan]{name}[/]...")

            grades_path = output_dir / f"student_grades_{name.replace(' ', '_')}.json"
            if grades_path.exists():
                console.print(f"  Loading cached grades for {name}...")
                grades = StudentGrades.model_validate_json(grades_path.read_text())
            else:
                grades = await grade_student(
                    client, manifest, extraction, rubric, solutions, stu_dir, blank_dir
                )
                _save_json(grades, grades_path)

            console.print(
                f"  {name}: [bold green]{grades.total['earned']}/{grades.total['possible']}[/]"
            )
            return grades

    tasks = [process_student(name, path) for name, path in all_students]
    results = await asyncio.gather(*tasks)
    all_grades = [r for r in results if r is not None]

    # ── Stage 5: Reports ─────────────────────────────────────────────
    console.print("\n[bold blue]Stage 5:[/] Generating reports...")
    await generate_reports(all_grades, manifest, output_dir)
    console.print(f"  Reports saved to [green]{output_dir / 'reports'}[/]")
    console.print(f"  Class summary: [green]{output_dir / 'class_summary.csv'}[/]")

    # Final summary
    console.print("\n[bold green]Pipeline complete![/]")
    for g in all_grades:
        console.print(f"  {g.student_name}: {g.total['earned']}/{g.total['possible']}")
