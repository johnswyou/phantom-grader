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
    work_shown: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    evidence: str = ""
    source_pages: list[int] = Field(default_factory=list)
    alignment_method: str = ""
    flags: list[str] = Field(default_factory=list)


class StudentExtraction(BaseModel):
    student_name: str
    answers: dict[str, ExtractedAnswer]
    unanswered: list[str] = Field(default_factory=list)
    alignment_warnings: list[str] = Field(default_factory=list)


# ── Stage 4: Grading ────────────────────────────────────────────────────────

class CriterionGrade(BaseModel):
    criterion: str
    earned: int | float
    possible: int | float
    note: str = ""


class QuestionGrade(BaseModel):
    points_earned: int | float
    points_possible: int | float
    correct: bool | None = None  # for MCQ
    criteria_breakdown: list[CriterionGrade] = Field(default_factory=list)
    feedback: str = ""


class StudentGrades(BaseModel):
    student_name: str
    grades: dict[str, QuestionGrade]
    page_totals: dict[str, int | float]
    total: dict[str, int | float]  # {"earned": X, "possible": Y}
