"""Pydantic models for all phantom-grader JSON schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Stage 1: Question Manifest ──────────────────────────────────────────────

class Question(BaseModel):
    id: str
    page: int
    type: str  # "mcq" or "free_response"
    points: int
    text_snippet: str = ""
    options: list[str] = Field(default_factory=list)
    embedded_answer: str | None = None
    sub_parts: list[str] = Field(default_factory=list)


class QuestionManifest(BaseModel):
    assignment_name: str
    total_pages: int
    total_points: int
    points_per_page: dict[str, int]
    questions: list[Question]


# ── Stage 2: Solution Manual + Rubric ────────────────────────────────────────

class Solution(BaseModel):
    answer: str
    explanation: str = ""
    key_steps: list[str] = Field(default_factory=list)


class SolutionManual(BaseModel):
    solutions: dict[str, Solution]


class RubricCriterion(BaseModel):
    points: int
    description: str
    type: str = "partial"  # "all_or_nothing" or "partial"


class QuestionRubric(BaseModel):
    total_points: int
    criteria: list[RubricCriterion]


class Rubric(BaseModel):
    rubric: dict[str, QuestionRubric]


# ── Stage 3: Student Extraction ──────────────────────────────────────────────

class ExtractedAnswer(BaseModel):
    response_type: str  # "mcq" or "free_response"
    selected: str | None = None  # for MCQ
    work_shown: str | None = ""
    final_answer: str | None = ""
    confidence: float = 0.0
    evidence: str | None = ""
    source_pages: list[int] = Field(default_factory=list)
    alignment_method: str | None = ""
    flags: list[str] | None = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        # Normalize None → defaults
        if self.work_shown is None:
            self.work_shown = ""
        if self.final_answer is None:
            self.final_answer = ""
        if self.evidence is None:
            self.evidence = ""
        if self.alignment_method is None:
            self.alignment_method = ""
        if self.flags is None:
            self.flags = []


class StudentExtraction(BaseModel):
    student_name: str
    answers: dict[str, ExtractedAnswer]
    unanswered: list[str] = Field(default_factory=list)
    alignment_warnings: list[str] = Field(default_factory=list)


# ── Stage 4: Grading ────────────────────────────────────────────────────────

class CriterionGrade(BaseModel):
    criterion: str = ""
    earned: int | float = 0
    possible: int | float = 0
    note: str | None = ""

    def model_post_init(self, __context) -> None:
        if self.note is None:
            self.note = ""


class QuestionGrade(BaseModel):
    points_earned: int | float = 0
    points_possible: int | float = 0
    correct: bool | None = None  # for MCQ
    criteria_breakdown: list[CriterionGrade] | None = Field(default_factory=list)
    feedback: str | None = ""

    def model_post_init(self, __context) -> None:
        if self.criteria_breakdown is None:
            self.criteria_breakdown = []
        if self.feedback is None:
            self.feedback = ""


class StudentGrades(BaseModel):
    student_name: str
    grades: dict[str, QuestionGrade]
    page_totals: dict[str, int | float]
    total: dict[str, int | float]  # {"earned": X, "possible": Y}
