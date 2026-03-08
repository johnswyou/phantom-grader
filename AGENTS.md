# AGENTS.md — phantom-grader

## Project Overview

**phantom-grader** is an agentic auto-grading system for handwritten physics and math assignments (AP Physics 1, AP Physics 2, and general math). It handles arbitrary assignment structures exported from Classkick (or similar platforms) where students paste images of handwritten solutions.

## The Core Problem

Students submit work via Classkick. The submissions are PDFs/images where each page corresponds to an assignment page. But students often:
- Paste a single photo of handwritten work that answers multiple questions
- Write solutions in unexpected locations on the page
- Have their handwritten image cover/overlap the printed assignment questions
- Split one question's answer across multiple pages
- Leave questions blank

This creates an **alignment problem**: you can't assume "student page N = assignment page N" or that answers appear in the expected spatial locations.

## Architecture

The system is a **staged pipeline** where each stage produces structured JSON artifacts:

```
Stage 1: ASSIGNMENT PARSING (Flash model)
    Input: Blank template page images + MAX_POINTS_PER_PAGE.txt
    Output: question_manifest.json

Stage 2: SOLUTION + RUBRIC GENERATION (Pro model, per-page parallel)
    Input: question_manifest.json + blank template images
    Output: solution_manual.json, rubric.json

Stage 2b: SOLUTION VERIFICATION (Flash model, parallel)
    Input: solutions + manifest + blank images
    Output: solution_verification.json

Stage 3: STUDENT ANSWER EXTRACTION (per student, parallel) (Flash model)
    Input: Student page images + question_manifest.json
    Substeps:
      3a. Vision-based blank page detection
      3b. Content region detection (when zoom enabled)
      3c. Crop and upscale regions
      3d. Extraction with crops + full pages as reference
    Output: student_extraction_*.json

Stage 4: GRADING (per student, parallel) (Pro model)
    Input: student_extraction + rubric + solutions + student images
    Post-processing: consistency validation, auto-fix, page cap enforcement
    Output: student_grades_*.json

Stage 5: REPORT GENERATION (pure Python, no LLM)
    Input: All student grades
    Output: Per-student markdown reports + class summary CSV
```

## Directory Structure

```
phantom-grader/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── scripts/
│   └── eval.py                    # Evaluation harness
├── src/
│   └── phantom_grader/
│       ├── __init__.py
│       ├── cli.py                 # typer CLI entry point
│       ├── config.py              # Configuration constants
│       ├── models.py              # Pydantic models for all JSON schemas
│       ├── vision.py              # Gemini API abstraction + region detection
│       ├── pipeline.py            # Main pipeline orchestrator
│       ├── stages/
│       │   ├── __init__.py
│       │   ├── parse_assignment.py    # Stage 1
│       │   ├── solve_and_rubric.py    # Stage 2 + 2b (verify_solutions)
│       │   ├── extract_answers.py     # Stage 3 (zoom + non-zoom paths)
│       │   ├── grade.py               # Stage 4 (validate + regrade)
│       │   └── report.py             # Stage 5
│       └── utils/
│           ├── __init__.py
│           ├── pdf.py             # PDF → image conversion
│           └── image.py           # Crop utilities
└── tests/
    └── __init__.py
```

## Key Design Decisions

1. **Structured JSON at every stage boundary.** Makes the pipeline debuggable and resumable. If grading is wrong, inspect the extraction JSON to see if alignment or grading was at fault.

2. **File-based caching.** Every stage output is saved. Re-running skips completed stages. Delete a file to re-run that stage.

3. **Two-model strategy.** Flash for fast/cheap work (parsing, extraction, verification). Pro for reasoning-heavy work (solving, grading).

4. **Crop-and-zoom.** Content region detection + cropping gives the vision model higher effective resolution on small handwriting. Controlled by `config.ENABLE_ZOOM`.

5. **Per-page batching (Stage 2).** Questions are solved page-by-page in parallel instead of one monolithic call. Reduces token pressure and enables granular retries.

6. **Solution verification.** After generating solutions, a separate verification pass cross-checks MCQ answers against embedded keys and validates free-response solutions via independent LLM calls.

7. **Grade consistency validation.** Programmatic checks after grading: criteria sums, earned ≤ possible, rubric coverage. Auto-fixes trivially wrong sums.

8. **Page cap enforcement with re-grade.** If a page's grades exceed the point cap by >2 points, the page is re-graded with an explicit constraint. Falls back to proportional scaling only if re-grade still fails.

9. **Parallel student processing.** Students are processed concurrently with separate semaphores for Flash (8 slots) and Pro (3 slots) models.

10. **Conservative defaults.** When alignment is uncertain, award 0 and flag for review rather than guessing.

## Vision API

- **Flash model** (`config.FLASH_MODEL`): Parsing, extraction, verification, region detection
- **Pro model** (`config.PRO_MODEL`): Solving, grading
- Both overridable via `--flash-model` and `--pro-model` CLI flags
- Native async via `client.aio.models.generate_content()`
- Retry with exponential backoff (3 attempts)
- Thinking enabled (`thinking_budget=8192`)

## CLI

```bash
# Full pipeline
phantom-grader grade \
  --blank-dir "BLANK-Assignment/" \
  --student-dir "IMG-Assignment/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --output-dir graded_output/ \
  --student "Name"              # optional: filter to one student
  --flash-model "model-name"    # optional
  --pro-model "model-name"      # optional

# From PDFs
phantom-grader grade \
  --blank-pdf "BLANK.pdf" \
  --student-pdf "StudentPDFs/" \
  --points-file MAX_POINTS_PER_PAGE.txt

# Individual stages
phantom-grader parse-assignment --blank-dir ... --points-file ...
phantom-grader solve --manifest ... --blank-dir ...
phantom-grader extract --manifest ... --student-dir ... --student ... --blank-dir ...
phantom-grader grade-student --extraction ... --rubric ... --solutions ... --manifest ... --student-dir ... --blank-dir ...
```

## Dependencies

- `google-genai` — Gemini API (native async)
- `typer` — CLI framework
- `pydantic` — Data validation and JSON schemas
- `pymupdf` — PDF to image conversion
- `pillow` — Image manipulation (cropping, resizing)
- `rich` — Terminal output (progress, colors)

## Sample Data

For testing, reference the data in `/home/john/Documents/Meritus/Winter 2026/Grading/`:
- Each assignment directory contains:
  - `BLANK-<name>/` (blank template as images)
  - `IMG-<name>/` (student submissions, one subdir per student)
  - `MAX_POINTS_PER_PAGE.txt`

## Evaluation

```bash
python scripts/eval.py \
  --predicted graded_output/ \
  --golden golden_grades.json \
  --manifest graded_output/question_manifest.json
```

Golden grades format:
```json
{
  "students": {
    "Student Name": {
      "Q1": {"earned": 2, "possible": 2}
    }
  }
}
```

## Error Handling

- Vision API errors: retry up to 3 times with exponential backoff
- Empty student directory: skip and note in report
- Page cap violations: re-grade the page, then scale as fallback
- Partial results always saved so the pipeline can be resumed
