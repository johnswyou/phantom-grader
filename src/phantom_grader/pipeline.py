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
    verify_solutions,
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


async def run_pipeline(
    blank_dir: Path,
    student_dir: Path,
    points_file: Path,
    output_dir: Path,
    api_key: str,
    student_filter: str | None = None,
    *,
    flash_model: str | None = None,
    pro_model: str | None = None,
    force_stages: set[str] | None = None,
) -> None:
    """Run the full grading pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    force = force_stages or set()

    client = get_client(api_key)

    def _use_cache(path: Path, stage: str) -> bool:
        """Return True if we should use cached output."""
        return path.exists() and stage not in force and "all" not in force

    # ── Stage 1: Parse Assignment ────────────────────────────────────
    console.print("\n[bold blue]Stage 1:[/] Parsing assignment template...")

    manifest_path = output_dir / "question_manifest.json"
    if _use_cache(manifest_path, "parse"):
        console.print("  Loading cached manifest...")
        manifest = QuestionManifest.model_validate_json(manifest_path.read_text())
    else:
        manifest = await parse_assignment(client, blank_dir, points_file, flash_model=flash_model)
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
    if _use_cache(solutions_path, "solve") and _use_cache(rubric_path, "solve"):
        console.print("  Loading cached solutions and rubric...")
        solutions = SolutionManual.model_validate_json(solutions_path.read_text())
        rubric = Rubric.model_validate_json(rubric_path.read_text())
    else:
        solutions, rubric = await generate_solutions_and_rubric(
            client, manifest, blank_dir, pro_model=pro_model
        )
        _save_json(solutions, solutions_path)
        _save_json(rubric, rubric_path)
    console.print(
        f"  Generated solutions for [green]{len(solutions.solutions)}[/] questions, "
        f"rubric for [green]{len(rubric.rubric)}[/] questions"
    )

    # ── Solution Verification ────────────────────────────────────────
    verification_path = output_dir / "solution_verification.json"
    if not _use_cache(verification_path, "solve"):
        console.print("  Verifying solutions...")
        verification = await verify_solutions(
            client, manifest, solutions, blank_dir, flash_model=flash_model
        )
        _save_json(verification, verification_path)
        flagged = {qid: v for qid, v in verification.items() if not v.get("verified", True)}
        if flagged:
            console.print(f"  [yellow]Warning:[/] {len(flagged)} solution(s) flagged during verification:")
            for qid, v in flagged.items():
                console.print(f"    [yellow]{qid}:[/] {v.get('note', 'unverified')}")

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
    all_extractions: dict[str, StudentExtraction] = {}

    async def process_student(name: str, stu_dir: Path) -> StudentGrades | None:
        async with sem:
            console.print(f"\n[bold blue]Stage 3:[/] Extracting answers for [cyan]{name}[/]...")

            extraction_path = output_dir / f"student_extraction_{name.replace(' ', '_')}.json"
            if _use_cache(extraction_path, "extract"):
                console.print(f"  Loading cached extraction for {name}...")
                extraction = StudentExtraction.model_validate_json(
                    extraction_path.read_text()
                )
            else:
                extraction = await extract_student_answers(
                    client, manifest, stu_dir, name, blank_dir,
                    flash_model=flash_model,
                )
                _save_json(extraction, extraction_path)

            all_extractions[name] = extraction

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
            if _use_cache(grades_path, "grade"):
                console.print(f"  Loading cached grades for {name}...")
                grades = StudentGrades.model_validate_json(grades_path.read_text())
            else:
                grades = await grade_student(
                    client, manifest, extraction, rubric, solutions, stu_dir, blank_dir,
                    pro_model=pro_model,
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
    generate_reports(all_grades, manifest, output_dir, extractions=all_extractions)
    console.print(f"  Reports saved to [green]{output_dir / 'reports'}[/]")
    console.print(f"  Class summary: [green]{output_dir / 'class_summary.csv'}[/]")

    # Final summary
    console.print("\n[bold green]Pipeline complete![/]")
    for g in all_grades:
        console.print(f"  {g.student_name}: {g.total['earned']}/{g.total['possible']}")
