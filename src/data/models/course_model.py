"""
Course model for the data analysis system.

This module defines the data models for course evaluations and
learning outcome measurements for the 15.390 course.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.data.models.enums import Semester, ToolVersion
from src.utils.safe_ops import safe_lower


class ScaleValue(BaseModel):
    """Rating scale values for course evaluation questions."""

    one: Optional[str] = Field(default=None, alias="1")
    two: Optional[str] = Field(default=None, alias="2")
    three: Optional[str] = Field(default=None, alias="3")
    four: Optional[str] = Field(default=None, alias="4")
    five: Optional[str] = Field(default=None, alias="5")
    seven: Optional[str] = Field(default=None, alias="7")
    not_applicable: Optional[str] = Field(default=None, alias="N/A")

    class Config:
        allow_population_by_field_name = True


class Scale(BaseModel):
    """Scale information for course evaluation metrics."""

    values: Optional[ScaleValue] = None
    unit: Optional[str] = None
    best_value: Optional[int] = Field(default=None, alias="best_value")
    type: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class EvaluationQuestion(BaseModel):
    """Individual question in a course evaluation."""

    question: Optional[str] = None
    median: Optional[float] = None
    avg: Optional[float] = None
    responses: Optional[int] = None
    stdev: Optional[float] = None

    @property
    def is_high_impact(self) -> bool:
        """
        Check if this is a high-impact learning outcome question.

        High-impact questions are those identified as particularly relevant
        for measuring the tool's impact on learning outcomes.

        Returns:
            bool: True if high impact, False otherwise
        """
        if not self.question:
            return False

        # Key phrases that indicate high-impact learning outcome questions
        high_impact_phrases = [
            "learning objectives",
            "assignments contributed",
            "overall learning experience",
            "intellectually challenging",
            "learned a great deal",
            "useful feedback",
            "effectively structured",
        ]

        return any(
            phrase in safe_lower(self.question) for phrase in high_impact_phrases
        )


class EvaluationMetric(BaseModel):
    """Group of related evaluation questions in a specific section."""

    questions: List[EvaluationQuestion] = []
    section: Optional[str] = None
    scale: Optional[Scale] = None

    def get_high_impact_questions(self) -> List[EvaluationQuestion]:
        """
        Get only the high-impact learning outcome questions.

        Returns:
            List[EvaluationQuestion]: High-impact questions
        """
        return [q for q in self.questions if q.is_high_impact]

    def get_average_score(self, high_impact_only: bool = False) -> float:
        """
        Calculate the average score across all questions.

        Args:
            high_impact_only: Whether to only include high-impact questions

        Returns:
            float: Average score
        """
        if high_impact_only:
            questions = self.get_high_impact_questions()
        else:
            questions = self.questions

        if not questions:
            return 0.0

        # Get valid averages (non-None)
        valid_avgs = [q.avg for q in questions if q.avg is not None]

        if not valid_avgs:
            return 0.0

        return sum(valid_avgs) / len(valid_avgs)


class SemesterInfo(BaseModel):
    """Semester information for a course evaluation."""

    term: Optional[str] = None  # "Fall", "Spring", etc.
    order: Optional[int] = None  # For sorting
    year: Optional[int] = None  # e.g., 2023, 2024

    def get_semester_enum(self) -> Optional[Semester]:
        """
        Get the semester as an enum.

        Returns:
            Optional[Semester]: Semester enum
        """
        if not self.term or not self.year:
            return None

        semester_str = f"{self.term} {self.year}"

        for semester in Semester:
            if semester.value == semester_str:
                return semester

        return None

    def get_tool_version(self) -> ToolVersion:
        """
        Get the tool version used in this semester.

        Returns:
            ToolVersion: Tool version enum
        """
        semester = self.get_semester_enum()
        if not semester:
            return ToolVersion.NONE

        return Semester.get_tool_version(semester)


class CourseEvaluation(BaseModel):
    """
    Course evaluation model for the 15.390 course.

    Contains evaluation metrics for a specific semester of the course,
    including learning outcome measurements.
    """

    course_id: Optional[str] = None  # Course identifier (e.g., "15.390")
    evaluation_metrics: List[EvaluationMetric] = []
    tool_version: Optional[str] = None  # Tool version used in this semester
    semester: Optional[SemesterInfo] = None

    def get_tool_version(self) -> ToolVersion:
        """
        Get the tool version used in this semester as an enum.

        Returns:
            ToolVersion: Tool version enum
        """
        if self.tool_version:
            for version in ToolVersion:
                if version.value == safe_lower(self.tool_version):
                    return version

        if self.semester:
            return self.semester.get_tool_version()

        return ToolVersion.NONE

    def get_overall_score(self, high_impact_only: bool = False) -> float:
        """
        Calculate the overall score across all metrics.

        Args:
            high_impact_only: Whether to only include high-impact questions

        Returns:
            float: Overall score
        """
        if not self.evaluation_metrics:
            return 0.0

        scores = [
            metric.get_average_score(high_impact_only)
            for metric in self.evaluation_metrics
        ]

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def get_section_score(
        self, section_name: str, high_impact_only: bool = False
    ) -> float:
        """
        Calculate the score for a specific section.

        Args:
            section_name: Section name to filter by
            high_impact_only: Whether to only include high-impact questions

        Returns:
            float: Section score
        """
        for metric in self.evaluation_metrics:
            if metric.section and safe_lower(metric.section) == safe_lower(
                section_name
            ):
                return metric.get_average_score(high_impact_only)

        return 0.0

    def get_high_impact_questions_count(self) -> int:
        """
        Count the total number of high-impact questions.

        Returns:
            int: Count of high-impact questions
        """
        count = 0
        for metric in self.evaluation_metrics:
            count += len(metric.get_high_impact_questions())
        return count
