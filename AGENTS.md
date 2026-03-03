# AGENTS.md — phantom-grader

## Project Overview

**phantom-grader** is an agentic auto-grading system for handwritten physics and math assignments (AP Physics 1, AP Physics 2, and general math). It handles arbitrary assignment structures exported from Classkick (or similar platforms) where students paste images of handwritten solutions.

## The Core Problem

Students submit work via Classkick. The submissions are PDFs where each page corresponds to an assignment page. But students often:
- Paste a single photo of handwritten work that answers multiple questions
- Write solutions in unexpected locations on the page
- Have their handwritten image cover/overlap the printed assignment questions
- Split one question's answer across multiple pages
- Leave questions blank

This creates an **alignment problem**: you can't assume "student page N = assignment page N" or that answers appear in the expected spatial locations.

## Architecture

The system is a **staged pipeline** where each stage produces structured JSON artifacts:

```
Stage 1: ASSIGNMENT PARSING
    Input: Blank template page images + MAX_POINTS_PER_PAGE.txt
    Output: question_manifest.json
    
Stage 2: SOLUTION + RUBRIC GENERATION
    Input: question_manifest.json + blank template images
    Output: solution_manual.json, rubric.json

Stage 3: STUDENT ANSWER EXTRACTION (per student, parallel)
    Input: Student page images + OCR markdown (if available) + question_manifest.json
    Output: student_extraction.json (per student)
    
Stage 4: GRADING (per student, parallel)
    Input: student_extraction.json + rubric.json + solution_manual.json + student images
    Output: student_grades.json (per student)

Stage 5: REPORT GENERATION
    Input: All student_grades.json files
    Output: Per-student markdown reports + class summary CSV
```

## Directory Structure

```
phantom-grader/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── src/
│   └── phantom_grader/
│       ├── __init__.py
│       ├── cli.py              # typer CLI entry point
│       ├── config.py           # Configuration (API keys, model selection)
│       ├── models.py           # Pydantic models for all JSON schemas
│       ├── vision.py           # Vision API abstraction (Gemini primary)
│       ├── pipeline.py         # Main pipeline orchestrator
│       ├── stages/
│       │   ├── __init__.py
│       │   ├── parse_assignment.py    # Stage 1: Parse blank template
│       │   ├── solve_and_rubric.py    # Stage 2: Solution manual + rubric
│       │   ├── extract_answers.py     # Stage 3: Student answer extraction + alignment
│       │   ├── grade.py               # Stage 4: Grading
│       │   └── report.py             # Stage 5: Report generation
│       └── utils/
│           ├── __init__.py
│           ├── pdf.py          # PDF → image conversion
│           └── image.py        # Image preprocessing utilities
└── tests/
    └── ...
```

## Detailed Stage Specifications

### Stage 1: Assignment Parsing (`parse_assignment.py`)

**Input:**
- Directory of blank template page images (e.g., `BLANK-<name>/page_0001.jpg`, ...)
- `MAX_POINTS_PER_PAGE.txt` file

**Process:**
1. Send each blank template page to the vision model
2. Extract: question numbers, question types (MCQ/free-response), sub-parts, spatial locations on page
3. Detect embedded answer keys (some templates have correct MCQ answers pre-marked — filled/highlighted bubbles on the BLANK template)
4. Parse MAX_POINTS_PER_PAGE.txt for point allocations

**Output: `question_manifest.json`**
```json
{
  "assignment_name": "AP Physics 2 Class 03 Homework - Electrostatics, Part 1 (W26)",
  "total_pages": 7,
  "total_points": 60,
  "points_per_page": {"1": 14, "2": 8, ...},
  "questions": [
    {
      "id": "Q1",
      "page": 1,
      "type": "mcq",
      "options": ["A", "B", "C", "D", "E"],
      "points": 2,
      "text_snippet": "Two electric objects experience a repulsive force...",
      "embedded_answer": "A",  // if detected from blank template, else null
      "sub_parts": []
    },
    {
      "id": "Q12",
      "page": 3,
      "type": "free_response",
      "points": 8,
      "text_snippet": "An electric dipole is a pair of charged particles...",
      "sub_parts": []
    },
    {
      "id": "Q14",
      "page": 4,
      "type": "free_response",
      "points": 10,
      "text_snippet": "A small ball of mass 200g and charge 60μC...",
      "sub_parts": ["A", "B"]
    }
  ]
}
```

### Stage 2: Solution + Rubric Generation (`solve_and_rubric.py`)

**Input:** question_manifest.json + blank template images

**Process:**
1. For each question, send the question image region + question text to the vision model
2. Ask it to solve the problem fully, showing all work
3. For MCQ: verify against embedded answer key if available
4. Generate rubric with partial credit criteria, constrained by MAX_POINTS_PER_PAGE

**Output: `solution_manual.json`**
```json
{
  "solutions": {
    "Q1": {
      "answer": "A",
      "explanation": "By Coulomb's law, F ∝ 1/r². Doubling r → F/4.",
      "key_steps": []
    },
    "Q14A": {
      "answer": "ΔU_g = mgh = mg·L(1 - cos45°) = 0.2 × 9.8 × 0.5 × (1 - √2/2) ≈ 0.287 J",
      "key_steps": ["Identify h = L(1-cosθ)", "Substitute values", "Compute"]
    }
  }
}
```

**Output: `rubric.json`**
```json
{
  "rubric": {
    "Q1": {
      "total_points": 2,
      "criteria": [
        {"points": 2, "description": "Correct answer (A)", "type": "all_or_nothing"}
      ]
    },
    "Q14A": {
      "total_points": 5,
      "criteria": [
        {"points": 2, "description": "Correct geometric setup: h = L(1-cosθ)"},
        {"points": 2, "description": "Correct substitution and arithmetic"},
        {"points": 1, "description": "Correct final answer with units"}
      ]
    }
  }
}
```

### Stage 3: Student Answer Extraction (`extract_answers.py`)

**THIS IS THE CRITICAL STAGE — the alignment problem is solved here.**

**Input:** Student page images + OCR markdown (optional) + question_manifest.json

**Process — The Alignment Algorithm:**

1. **Full Scan Pass:** Send ALL student pages to the vision model in one call (or batched). Ask it to:
   - Identify all regions of student-added content (handwriting, pasted images)
   - Distinguish printed template text from student work
   - Note any question number references visible in student handwriting
   - For MCQ pages: identify which bubbles are filled (colored/shaded vs empty outline)

2. **Alignment Pass:** With the question manifest as context, ask the vision model to:
   - Map each detected student content region to the question it's answering
   - Use multiple signals: spatial position relative to template, question numbers written by student, mathematical content matching (e.g., if student writes Coulomb's law, it's probably answering an electrostatics question)
   - Handle edge cases: student pasted one big image covering multiple questions, student wrote "Q14" in their handwriting, student continued answer on next page
   - Flag low-confidence alignments

3. **Extraction Pass:** For each aligned region, extract the student's answer:
   - MCQ: which option was selected
   - Free-response: transcription of the student's work (equations, steps, final answer)
   - Diagrams/graphs: description of what was drawn

4. **Cross-reference with OCR** (if available): Compare vision-extracted content with OCR markdown. Note discrepancies.

**Output: `student_extraction.json`**
```json
{
  "student_name": "Roy Choi",
  "answers": {
    "Q1": {
      "response_type": "mcq",
      "selected": "A",
      "confidence": 0.95,
      "evidence": "Green filled bubble at position 1, other bubbles empty outlines",
      "source_pages": [1],
      "alignment_method": "spatial"
    },
    "Q14A": {
      "response_type": "free_response",
      "work_shown": "Student writes ΔU = mgh, identifies h = L(1-cos45°), substitutes m=0.2kg, g=9.8, L=0.5m. Gets ΔU = 0.287 J",
      "final_answer": "0.287 J",
      "confidence": 0.85,
      "evidence": "Handwritten work in answer area below Q14A prompt",
      "source_pages": [4],
      "alignment_method": "spatial+content"
    },
    "Q16A": {
      "response_type": "free_response",
      "work_shown": "...",
      "final_answer": "K = ke²/(2r)",
      "confidence": 0.70,
      "evidence": "Pasted image of handwritten work. Student labeled 'a)' at top. Contains Coulomb force = centripetal force derivation.",
      "source_pages": [5],
      "alignment_method": "content_label",
      "flags": ["pasted_image", "low_confidence_alignment"]
    }
  },
  "unanswered": ["Q13"],
  "alignment_warnings": [
    "Q16: Student pasted single image covering all sub-parts (A, B, C). Extracted based on sub-labels in handwriting."
  ]
}
```

### Stage 4: Grading (`grade.py`)

**Input:** student_extraction.json + rubric.json + solution_manual.json + student page images

**Process:**
1. For each question, compare student's extracted answer against the rubric
2. For MCQ: direct comparison — correct or not
3. For free-response: send the student's work image + extracted text + solution manual + rubric to the vision model and ask for point-by-point grading
4. Apply partial credit rules from rubric
5. Enforce page point caps from MAX_POINTS_PER_PAGE
6. For low-confidence extractions: use the original student images directly (re-examine)

**Output: `student_grades.json`**
```json
{
  "student_name": "Roy Choi",
  "grades": {
    "Q1": {
      "points_earned": 2,
      "points_possible": 2,
      "correct": true,
      "feedback": "Correct. Coulomb's law: F ∝ 1/r², doubling distance → force decreases to 1/4."
    },
    "Q14A": {
      "points_earned": 4,
      "points_possible": 5,
      "criteria_breakdown": [
        {"criterion": "Correct geometric setup", "earned": 2, "possible": 2},
        {"criterion": "Correct substitution", "earned": 2, "possible": 2},
        {"criterion": "Final answer with units", "earned": 0, "possible": 1, "note": "Missing units"}
      ],
      "feedback": "Good setup and calculation. Remember to include units (Joules) in final answer."
    }
  },
  "page_totals": {"1": 12, "2": 7, "3": 6, ...},
  "total": {"earned": 48, "possible": 60}
}
```

### Stage 5: Report Generation (`report.py`)

Generates:
1. Per-student markdown report (detailed, similar to the Andy_Chen.md format John already has)
2. Class summary CSV: `student, total, page1, page2, ..., pageN`
3. Optional: class statistics (mean, median, std dev, per-question stats)

## CLI Interface

```bash
# Full pipeline
phantom-grader grade \
  --blank-dir "BLANK-Assignment Name/" \
  --student-dir "IMG-Assignment Name/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --ocr-dir "OCR-Assignment Name/"  \  # optional
  --output-dir graded_output/ \
  --api-key "$GEMINI_API_KEY"

# Run individual stages (for debugging)
phantom-grader parse-assignment --blank-dir ... --points-file ... --output manifest.json
phantom-grader solve --manifest manifest.json --blank-dir ... --output solutions.json
phantom-grader extract --manifest manifest.json --student-dir ... --student "Roy Choi" --output extraction.json
phantom-grader grade-student --extraction extraction.json --rubric rubric.json --solutions solutions.json --output grades.json

# Grade a single student (useful for testing)
phantom-grader grade --blank-dir ... --student-dir ... --points-file ... --student "Roy Choi"
```

## Vision API

Use **Google Gemini** as the primary vision backend.

- Model: `gemini-2.5-flash` for most tasks (fast, cheap, good at vision)
- Model: `gemini-2.5-pro` for the grading stage (needs stronger reasoning)
- API: Use the `google-genai` Python SDK (official Google GenAI SDK)
- API key is passed via CLI flag or `GEMINI_API_KEY` env var

### Key Vision Prompting Patterns

For the alignment stage, send images in this order:
1. ALL blank template pages (so the model understands the full assignment structure)
2. ALL student submission pages
3. The question manifest JSON

Ask the model to process holistically — don't go page by page for alignment.

For grading, send:
1. The specific question from the blank template
2. The student's answer region (specific page image)
3. The solution and rubric for that question

### Image Handling
- Accept both directories of JPGs and PDF files as input
- If PDFs are provided, convert to images using PyMuPDF (same as existing pdfs_to_jpg_dirs.py)
- Images should be resized if over 4MB (Gemini limit) — use reasonable quality reduction

## Important Design Decisions

1. **Structured JSON at every stage boundary.** This makes the pipeline debuggable. If grading is wrong, you can inspect the extraction JSON to see if the alignment was wrong vs. the grading logic.

2. **Confidence scores + flags.** The extraction stage must output confidence and flag ambiguous cases. The grading stage should treat low-confidence extractions with extra scrutiny (re-examine the image).

3. **Conservative defaults.** When alignment is uncertain, award 0 points and flag for human review rather than guessing.

4. **Idempotent stages.** Running a stage twice with the same inputs produces the same outputs. Stages can cache results.

5. **Parallel student processing.** Use asyncio to process multiple students concurrently. Respect API rate limits with semaphores.

## Dependencies

- `google-genai` — Google GenAI SDK for Gemini API
- `typer` — CLI framework
- `pydantic` — Data validation and JSON schemas
- `pymupdf` — PDF to image conversion
- `pillow` — Image manipulation
- `rich` — Pretty terminal output (progress bars, tables)
- `aiofiles` — Async file I/O

## Sample Data

For testing, reference the data in `/home/john/Documents/Meritus/Winter 2026/Grading/`:
- Assignment 1: AP Physics 2 Thermodynamics (4 pages, 11 students)
- Assignment 3: AP Physics 2 Electrostatics (7 pages, 2 students)

Each assignment directory contains:
- `BLANK-<name>.pdf` and `BLANK-<name>/` (blank template as images)
- `<name>/` (student PDFs)
- `IMG-<name>/` (student submissions as images, one subdir per student)
- `OCR-<name>/` (Mistral OCR markdown output, optional)
- `MAX_POINTS_PER_PAGE.txt`

## Error Handling

- If the vision API returns an error, retry up to 3 times with exponential backoff
- If a student has no images (empty directory), skip and note in report
- If MAX_POINTS_PER_PAGE constraint is violated after grading, re-grade that page
- Always save partial results so the pipeline can be resumed
