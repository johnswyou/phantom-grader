# phantom-grader 👻

An agentic auto-grading system for handwritten physics and math assignments.

## The Problem

Students submit handwritten solutions via platforms like Classkick, where they paste images of their work onto assignment pages. The submissions have **arbitrary structure** — solutions may span multiple pages, cover unrelated questions, or appear in unexpected locations. Traditional page-by-page grading breaks down.

## The Solution

phantom-grader uses a multi-stage vision AI pipeline to:

1. **Parse** the blank assignment template to understand its structure
2. **Solve** all problems to create a solution manual and rubric
3. **Extract** student answers with intelligent alignment (the hard part)
4. **Grade** each answer against the rubric with partial credit
5. **Report** detailed per-student markdown reports and class summaries

The key innovation is **Stage 3**: an alignment algorithm that uses spatial, textual, and semantic signals to map student work to the correct questions — regardless of how the student organized their submission.

## Quick Start

```bash
pip install phantom-grader

# Set your API key
export GEMINI_API_KEY="your-key"

# Grade an assignment
phantom-grader grade \
  --blank-dir "BLANK-Assignment/" \
  --student-dir "IMG-Assignment/" \
  --points-file MAX_POINTS_PER_PAGE.txt \
  --output-dir graded_output/
```

## Requirements

- Python 3.11+
- Gemini API key (for vision processing)

## License

MIT
