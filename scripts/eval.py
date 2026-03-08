#!/usr/bin/env python3
"""Evaluation harness: compare predicted grades against golden grades.

Usage:
    python scripts/eval.py --predicted <graded_output_dir> --golden <golden_grades.json>
    python scripts/eval.py --predicted <graded_output_dir> --golden <golden_grades.json> --manifest <question_manifest.json>

golden_grades.json format:
{
  "students": {
    "Student Name": {
      "Q1": {"earned": 2, "possible": 2},
      "Q2": {"earned": 0, "possible": 2},
      ...
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phantom_grader.models import QuestionManifest, StudentGrades


def load_predicted(predicted_dir: Path) -> dict[str, StudentGrades]:
    """Load all student_grades_*.json files from the predicted directory."""
    results: dict[str, StudentGrades] = {}
    for path in sorted(predicted_dir.glob("student_grades_*.json")):
        grades = StudentGrades.model_validate_json(path.read_text())
        results[grades.student_name] = grades
    return results


def load_golden(golden_path: Path) -> dict[str, dict[str, dict]]:
    """Load golden grades JSON."""
    data = json.loads(golden_path.read_text())
    return data["students"]


def compute_metrics(
    predicted: dict[str, StudentGrades],
    golden: dict[str, dict[str, dict]],
    manifest: QuestionManifest | None = None,
) -> dict:
    """Compute evaluation metrics across all matching student+question pairs."""
    exact_matches = 0
    within_one = 0
    total_pairs = 0
    abs_errors: list[float] = []

    per_student: dict[str, dict] = {}
    per_question_type: dict[str, dict] = {}

    # Build question type lookup from manifest if available
    qid_type: dict[str, str] = {}
    if manifest:
        for q in manifest.questions:
            if q.sub_parts:
                for sp in q.sub_parts:
                    qid_type[f"{q.id}{sp}"] = q.type
            else:
                qid_type[q.id] = q.type

    for student_name, golden_questions in golden.items():
        if student_name not in predicted:
            continue

        pred_grades = predicted[student_name]
        student_exact = 0
        student_total = 0
        student_pred_total = 0.0
        student_gold_total = 0.0

        for qid, gold in golden_questions.items():
            gold_earned = gold["earned"]
            gold_possible = gold["possible"]

            pred_grade = pred_grades.grades.get(qid)
            pred_earned = pred_grade.points_earned if pred_grade else 0

            total_pairs += 1
            student_total += 1
            error = abs(pred_earned - gold_earned)
            abs_errors.append(error)

            if pred_earned == gold_earned:
                exact_matches += 1
                student_exact += 1
            if error <= 1:
                within_one += 1

            student_pred_total += pred_earned
            student_gold_total += gold_earned

            # Per question type
            qtype = qid_type.get(qid, "unknown")
            if qtype not in per_question_type:
                per_question_type[qtype] = {"exact": 0, "within_one": 0, "total": 0, "errors": []}
            per_question_type[qtype]["total"] += 1
            per_question_type[qtype]["errors"].append(error)
            if pred_earned == gold_earned:
                per_question_type[qtype]["exact"] += 1
            if error <= 1:
                per_question_type[qtype]["within_one"] += 1

        per_student[student_name] = {
            "exact": student_exact,
            "total": student_total,
            "pred_total": student_pred_total,
            "gold_total": student_gold_total,
        }

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    return {
        "total_pairs": total_pairs,
        "exact_matches": exact_matches,
        "within_one": within_one,
        "mae": mae,
        "per_student": per_student,
        "per_question_type": per_question_type,
    }


def print_report(metrics: dict) -> None:
    """Print a formatted evaluation report."""
    total = metrics["total_pairs"]
    if total == 0:
        print("No matching student+question pairs found.")
        return

    exact_pct = metrics["exact_matches"] / total * 100
    within_one_pct = metrics["within_one"] / total * 100

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total question comparisons: {total}")
    print(f"Exact match rate:           {metrics['exact_matches']}/{total} ({exact_pct:.1f}%)")
    print(f"Within-1-point rate:        {metrics['within_one']}/{total} ({within_one_pct:.1f}%)")
    print(f"Mean Absolute Error (MAE):  {metrics['mae']:.2f}")
    print()

    # Per-student breakdown
    print("-" * 60)
    print(f"{'Student':<30} {'Exact':>8} {'Total':>8} {'Pred':>8} {'Gold':>8}")
    print("-" * 60)
    for name, data in sorted(metrics["per_student"].items()):
        print(
            f"{name:<30} {data['exact']:>5}/{data['total']:<3}"
            f" {data['pred_total']:>7.1f} {data['gold_total']:>7.1f}"
        )
    print()

    # Per question type breakdown (if available)
    if metrics["per_question_type"]:
        has_types = any(t != "unknown" for t in metrics["per_question_type"])
        if has_types:
            print("-" * 60)
            print("BY QUESTION TYPE")
            print("-" * 60)
            print(f"{'Type':<20} {'Exact':>10} {'Within 1':>10} {'MAE':>8}")
            print("-" * 60)
            for qtype, data in sorted(metrics["per_question_type"].items()):
                t = data["total"]
                exact_pct_t = data["exact"] / t * 100 if t else 0
                w1_pct_t = data["within_one"] / t * 100 if t else 0
                mae_t = sum(data["errors"]) / t if t else 0
                print(
                    f"{qtype:<20} {data['exact']:>4}/{t} ({exact_pct_t:>4.0f}%)"
                    f" {data['within_one']:>4}/{t} ({w1_pct_t:>4.0f}%)"
                    f" {mae_t:>7.2f}"
                )
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted grades against golden grades."
    )
    parser.add_argument(
        "--predicted", type=Path, required=True,
        help="Directory containing student_grades_*.json files",
    )
    parser.add_argument(
        "--golden", type=Path, required=True,
        help="Path to golden_grades.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional question_manifest.json for per-type breakdown",
    )
    args = parser.parse_args()

    if not args.predicted.is_dir():
        print(f"Error: {args.predicted} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.golden.is_file():
        print(f"Error: {args.golden} does not exist", file=sys.stderr)
        sys.exit(1)

    predicted = load_predicted(args.predicted)
    golden = load_golden(args.golden)

    manifest = None
    if args.manifest:
        manifest = QuestionManifest.model_validate_json(args.manifest.read_text())

    metrics = compute_metrics(predicted, golden, manifest)
    print_report(metrics)


if __name__ == "__main__":
    main()
