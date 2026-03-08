"""Stage 5: Generate per-student reports and class summary."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..models import QuestionManifest, StudentExtraction, StudentGrades


def generate_student_report(
    grades: StudentGrades,
    manifest: QuestionManifest,
    extraction: StudentExtraction | None = None,
) -> str:
    """Generate a detailed markdown report for a single student.

    If extraction is provided, includes what the system extracted for each
    question (useful for debugging wrong grades).
    """

    lines = [
        f"# Grading Report: {grades.student_name}",
        f"## {manifest.assignment_name}",
        "",
        f"**Total: {grades.total['earned']}/{grades.total['possible']}**",
        "",
        "---",
        "",
    ]

    # Page-by-page summary
    lines.append("## Page Totals")
    lines.append("")
    lines.append("| Page | Earned | Possible |")
    lines.append("|------|--------|----------|")
    for page_str in sorted(manifest.points_per_page.keys(), key=int):
        earned = grades.page_totals.get(page_str, 0)
        possible = manifest.points_per_page.get(page_str, 0)
        lines.append(f"| {page_str} | {earned} | {possible} |")
    lines.append("")

    # Question-by-question detail
    lines.append("## Question Details")
    lines.append("")

    for q in manifest.questions:
        qids = []
        if q.sub_parts:
            qids = [f"{q.id}{sp}" for sp in q.sub_parts]
        else:
            qids = [q.id]

        for qid in qids:
            grade = grades.grades.get(qid)
            if grade is None:
                lines.append(f"### {qid} — 0/{q.points if not q.sub_parts else '?'}")
                lines.append("*Not answered or not graded*")
                lines.append("")
                continue

            lines.append(
                f"### {qid} — {grade.points_earned}/{grade.points_possible}"
            )

            if grade.correct is not None:
                status = "✓ Correct" if grade.correct else "✗ Incorrect"
                lines.append(f"**{status}**")
                lines.append("")

            # Show extraction data if available
            if extraction:
                if qid in extraction.answers:
                    ans = extraction.answers[qid]
                    lines.append("**Extracted:**")
                    if ans.response_type == "mcq" and ans.selected:
                        lines.append(f"- Selected: **{ans.selected}**")
                    if ans.final_answer:
                        lines.append(f"- Final answer: {ans.final_answer}")
                    if ans.work_shown:
                        work = ans.work_shown[:300]
                        if len(ans.work_shown) > 300:
                            work += "..."
                        lines.append(f"- Work shown: {work}")
                    if ans.confidence < 0.7:
                        lines.append(f"- ⚠️ Low confidence: {ans.confidence:.2f}")
                    if ans.flags:
                        lines.append(f"- Flags: {', '.join(ans.flags)}")
                    lines.append("")
                elif qid in extraction.unanswered:
                    lines.append("**Extracted:** *Unanswered (no student work detected)*")
                    lines.append("")

            if grade.criteria_breakdown:
                lines.append("| Criterion | Earned | Possible | Notes |")
                lines.append("|-----------|--------|----------|-------|")
                for cb in grade.criteria_breakdown:
                    lines.append(
                        f"| {cb.criterion} | {cb.earned} | {cb.possible} | {cb.note} |"
                    )
                lines.append("")

            if grade.feedback:
                lines.append(f"**Feedback:** {grade.feedback}")
                lines.append("")

    return "\n".join(lines)


def generate_class_csv(
    all_grades: list[StudentGrades],
    manifest: QuestionManifest,
) -> str:
    """Generate a class summary CSV."""

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    page_cols = [f"Page {p}" for p in sorted(manifest.points_per_page.keys(), key=int)]
    header = ["Student", "Total Earned", "Total Possible"] + page_cols
    writer.writerow(header)

    for grades in sorted(all_grades, key=lambda g: g.student_name):
        row = [
            grades.student_name,
            grades.total["earned"],
            grades.total["possible"],
        ]
        for page_str in sorted(manifest.points_per_page.keys(), key=int):
            row.append(grades.page_totals.get(page_str, 0))
        writer.writerow(row)

    # Summary stats
    if all_grades:
        totals = [g.total["earned"] for g in all_grades]
        writer.writerow([])
        writer.writerow(["--- Statistics ---"])
        writer.writerow(["Mean", f"{sum(totals)/len(totals):.1f}"])
        if len(totals) > 1:
            sorted_t = sorted(totals)
            mid = len(sorted_t) // 2
            median = (
                sorted_t[mid]
                if len(sorted_t) % 2
                else (sorted_t[mid - 1] + sorted_t[mid]) / 2
            )
            writer.writerow(["Median", f"{median:.1f}"])
            mean = sum(totals) / len(totals)
            variance = sum((t - mean) ** 2 for t in totals) / len(totals)
            writer.writerow(["Std Dev", f"{variance**0.5:.1f}"])

    return output.getvalue()


def generate_reports(
    all_grades: list[StudentGrades],
    manifest: QuestionManifest,
    output_dir: Path,
    extractions: dict[str, StudentExtraction] | None = None,
) -> None:
    """Generate all reports and save to output directory."""

    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Per-student reports
    for grades in all_grades:
        extraction = extractions.get(grades.student_name) if extractions else None
        report = generate_student_report(grades, manifest, extraction)
        safe_name = grades.student_name.replace(" ", "_").replace("/", "_")
        report_path = reports_dir / f"{safe_name}.md"
        report_path.write_text(report)

    # Class summary CSV
    csv_content = generate_class_csv(all_grades, manifest)
    csv_path = output_dir / "class_summary.csv"
    csv_path.write_text(csv_content)
