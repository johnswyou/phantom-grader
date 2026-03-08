# phantom-grader 👻

An agentic auto-grading system for handwritten physics and math assignments.

## The Problem

Students submit handwritten solutions via platforms like Classkick, where they paste images of their work onto assignment pages. The submissions have **arbitrary structure** — solutions may span multiple pages, cover unrelated questions, or appear in unexpected locations. Traditional page-by-page grading breaks down.

## The Solution

phantom-grader uses a multi-stage vision AI pipeline (Gemini) to:

1. **Parse** the blank assignment template to understand its structure
2. **Solve** all problems and generate a rubric (per-page, parallel)
3. **Verify** solutions against embedded answer keys and via independent checks
4. **Extract** student answers with crop-and-zoom for fine handwriting
5. **Grade** each answer against the rubric with partial credit and consistency validation
6. **Report** detailed per-student markdown reports and class summaries

The key innovation is the **alignment algorithm** in Stage 3: vision-based blank page detection, content region detection with cropping, and multi-signal alignment that maps student work to questions regardless of layout.

## Quick Start

```bash
pip install -e .
export GEMINI_API_KEY="your-key"

# From image directories
phantom-grader grade \
  --blank-dir "BLANK-Assignment/" \
  --student-dir "IMG-Assignment/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --output-dir graded_output/

# From PDFs (auto-converted to images)
phantom-grader grade \
  --blank-pdf "BLANK-Assignment.pdf" \
  --student-pdf "StudentPDFs/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --output-dir graded_output/
```

## Model Overrides

```bash
phantom-grader grade \
  --blank-dir "BLANK-Assignment/" \
  --student-dir "IMG-Assignment/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --flash-model "gemini-2.5-flash" \
  --pro-model "gemini-2.5-pro"
```

## Individual Stages (Debugging)

```bash
phantom-grader parse-assignment --blank-dir ... --points-file ...
phantom-grader solve --manifest manifest.json --blank-dir ...
phantom-grader extract --manifest manifest.json --student-dir ... --student "Name" --blank-dir ...
phantom-grader grade-student --extraction ... --rubric ... --solutions ... --manifest ... --student-dir ... --blank-dir ...
```

## Output

```
graded_output/
  ├── question_manifest.json
  ├── solution_manual.json
  ├── rubric.json
  ├── solution_verification.json
  ├── student_extraction_*.json
  ├── student_grades_*.json
  ├── class_summary.csv
  └── reports/*.md
```

All stage outputs are cached. Re-running skips completed stages. Delete a file to re-run its stage.

## Evaluation

Compare pipeline output against manually-graded golden data:

```bash
python scripts/eval.py \
  --predicted graded_output/ \
  --golden golden_grades.json \
  --manifest graded_output/question_manifest.json
```

## Requirements

- Python 3.11+
- Gemini API key

## License

MIT
