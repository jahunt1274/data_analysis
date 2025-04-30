"""
Course repository for the data analysis system.

This module provides data access and query methods for CourseEvaluation entities,
including learning outcome analysis and semester comparison.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path

from src.data.models.course_model import (
    CourseEvaluation,
    EvaluationMetric,
    EvaluationQuestion,
)
from src.data.models.enums import Semester, ToolVersion
from src.data.repositories.base_repository import BaseRepository
from src.data.db import InMemoryDatabase

from src.utils.safe_ops import safe_lower


class CourseRepository(BaseRepository[CourseEvaluation]):
    """
    Repository for CourseEvaluation data access and analysis.

    This repository provides methods for querying and analyzing course evaluation data,
    including learning outcome measurements and semester comparisons.
    """

    def __init__(self, db=None, config=None):
        """
        Initialize the course repository.

        Args:
            db: Optional database connection
            config: Optional configuration object
        """
        super().__init__("course_evaluations", CourseEvaluation)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._config = config
        # In-memory indexes for faster lookups
        self._semester_index = {}  # Maps semester to evaluation

    def connect(self) -> None:
        """
        Connect to the data source (JSON files or database).

        For JSON files, this loads the data into an in-memory structure.
        """
        try:
            # If _db is provided, use it (MongoDB-like interface)
            if self._db is not None:
                return

            # Otherwise, load from JSON file
            if self._config and hasattr(self._config, "COURSE_EVAL_DATA_PATH"):
                file_path = self._config.COURSE_EVAL_DATA_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/course_evaluation.json"

            # Create in-memory DB-like structure if needed
            if not isinstance(self._db, InMemoryDatabase):
                self._db = InMemoryDatabase()

            self._load_data_from_json(file_path)

            # Build indexes for faster lookups
            self._build_indexes()
        except Exception as e:
            self._logger.error(f"Error connecting to course evaluation data: {e}")
            raise

    def _load_data_from_json(self, file_path: str) -> None:
        """
        Load course evaluation data from a JSON file.

        Args:
            file_path: Path to the JSON file containing course evaluation data
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Insert data into in-memory DB
            if isinstance(data, list):
                self._db.insert_many(self._collection_name, data)
            else:
                self._db.insert_one(self._collection_name, data)

            self._logger.info(
                f"Loaded {self._db.count(self._collection_name, {})} course evaluations from {file_path}"
            )
        except Exception as e:
            self._logger.error(
                f"Error loading course evaluation data from {file_path}: {e}"
            )
            raise

    def _build_indexes(self) -> None:
        """
        Build in-memory indexes for faster lookups.
        """
        try:
            evaluations = self.get_all()

            # Build semester index
            for eval in evaluations:
                if eval.semester and eval.semester.get_semester_enum():
                    semester = eval.semester.get_semester_enum().value
                    self._semester_index[semester] = eval
        except Exception as e:
            self._logger.error(f"Error building indexes: {e}")

    def find_by_semester(
        self, semester: Union[str, Semester]
    ) -> Optional[CourseEvaluation]:
        """
        Find course evaluation for a specific semester.

        Args:
            semester: Semester name or enum

        Returns:
            Optional[CourseEvaluation]: Evaluation or None if not found
        """
        # Convert to string semester name if an enum is provided
        if isinstance(semester, Semester):
            semester_name = semester.value
        else:
            semester_name = semester

        # Check in-memory index first
        if semester_name in self._semester_index:
            return self._semester_index[semester_name]

        # Fall back to search by semester term and year
        for eval in self.get_all():
            if eval.semester and eval.semester.term and eval.semester.year:
                semester_str = f"{eval.semester.term} {eval.semester.year}"
                if semester_str == semester_name:
                    return eval

        return None

    def find_by_tool_version(
        self, tool_version: Union[str, ToolVersion]
    ) -> List[CourseEvaluation]:
        """
        Find course evaluations for a specific tool version.

        Args:
            tool_version: Tool version name or enum

        Returns:
            List[CourseEvaluation]: List of matching evaluations
        """
        # Convert to string version name if an enum is provided
        if isinstance(tool_version, ToolVersion):
            version_name = tool_version.value
        else:
            version_name = tool_version

        return [
            eval
            for eval in self.get_all()
            if eval.get_tool_version().value == version_name
        ]

    def get_evaluation_comparison(
        self, semester1: Union[str, Semester], semester2: Union[str, Semester]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare course evaluations between two semesters.

        Args:
            semester1: First semester name or enum
            semester2: Second semester name or enum

        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        eval1 = self.find_by_semester(semester1)
        eval2 = self.find_by_semester(semester2)

        if not eval1 or not eval2:
            return {}

        # Compare overall scores and high-impact questions
        result = {
            "overall_score": {
                (
                    semester1.value if isinstance(semester1, Semester) else semester1
                ): eval1.get_overall_score(),
                (
                    semester2.value if isinstance(semester2, Semester) else semester2
                ): eval2.get_overall_score(),
            },
            "high_impact_score": {
                (
                    semester1.value if isinstance(semester1, Semester) else semester1
                ): eval1.get_overall_score(high_impact_only=True),
                (
                    semester2.value if isinstance(semester2, Semester) else semester2
                ): eval2.get_overall_score(high_impact_only=True),
            },
        }

        # Compare section scores
        sections = set()
        for metric in eval1.evaluation_metrics + eval2.evaluation_metrics:
            if metric.section:
                sections.add(safe_lower(metric.section))

        for section in sections:
            result[f"section_{section}"] = {
                (
                    semester1.value if isinstance(semester1, Semester) else semester1
                ): eval1.get_section_score(section),
                (
                    semester2.value if isinstance(semester2, Semester) else semester2
                ): eval2.get_section_score(section),
            }

        return result

    def get_all_high_impact_questions(self) -> Dict[str, List[EvaluationQuestion]]:
        """
        Get all high-impact learning outcome questions by semester.

        Returns:
            Dict[str, List[EvaluationQuestion]]: Mapping from semester to question list
        """
        result = {}

        for eval in self.get_all():
            if eval.semester and eval.semester.get_semester_enum():
                semester = eval.semester.get_semester_enum().value

                high_impact_questions = []
                for metric in eval.evaluation_metrics:
                    high_impact_questions.extend(metric.get_high_impact_questions())

                result[semester] = high_impact_questions

        return result

    def get_tool_impact_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of the tool on course evaluations.

        Compares semesters with and without the tool.

        Returns:
            Dict[str, Dict[str, float]]: Impact analysis results
        """
        # Get evaluations by tool version
        no_tool_evals = self.find_by_tool_version(ToolVersion.NONE)
        v1_evals = self.find_by_tool_version(ToolVersion.V1)
        v2_evals = self.find_by_tool_version(ToolVersion.V2)

        # Calculate average scores for each group
        result = {
            "overall_score": {
                "no_tool": (
                    sum(eval.get_overall_score() for eval in no_tool_evals)
                    / len(no_tool_evals)
                    if no_tool_evals
                    else 0
                ),
                "v1": (
                    sum(eval.get_overall_score() for eval in v1_evals) / len(v1_evals)
                    if v1_evals
                    else 0
                ),
                "v2": (
                    sum(eval.get_overall_score() for eval in v2_evals) / len(v2_evals)
                    if v2_evals
                    else 0
                ),
            },
            "high_impact_score": {
                "no_tool": (
                    sum(
                        eval.get_overall_score(high_impact_only=True)
                        for eval in no_tool_evals
                    )
                    / len(no_tool_evals)
                    if no_tool_evals
                    else 0
                ),
                "v1": (
                    sum(
                        eval.get_overall_score(high_impact_only=True)
                        for eval in v1_evals
                    )
                    / len(v1_evals)
                    if v1_evals
                    else 0
                ),
                "v2": (
                    sum(
                        eval.get_overall_score(high_impact_only=True)
                        for eval in v2_evals
                    )
                    / len(v2_evals)
                    if v2_evals
                    else 0
                ),
            },
        }

        return result

    def get_average_scores_by_section(self) -> Dict[str, Dict[str, float]]:
        """
        Get average scores for each section across all semesters.

        Returns:
            Dict[str, Dict[str, float]]: Mapping from section to semester scores
        """
        result = {}

        # Collect sections
        sections = set()
        for eval in self.get_all():
            for metric in eval.evaluation_metrics:
                if metric.section:
                    sections.add(safe_lower(metric.section))

        # Calculate scores by section and semester
        for section in sections:
            result[section] = {}

            for eval in self.get_all():
                if eval.semester and eval.semester.get_semester_enum():
                    semester = eval.semester.get_semester_enum().value
                    result[section][semester] = eval.get_section_score(section)

        return result

    def get_question_scores_over_time(self, question_text: str) -> Dict[str, float]:
        """
        Track scores for a specific question over time.

        Args:
            question_text: Question text to search for

        Returns:
            Dict[str, float]: Mapping from semester to question score
        """
        result = {}
        question_text_lower = safe_lower(question_text)

        for eval in self.get_all():
            if not eval.semester or not eval.semester.get_semester_enum():
                continue

            semester = eval.semester.get_semester_enum().value

            # Search for matching question
            for metric in eval.evaluation_metrics:
                for question in metric.questions:
                    if question.question and question_text_lower in safe_lower(
                        question.question
                    ):
                        result[semester] = (
                            question.avg if question.avg is not None else 0.0
                        )
                        break

        return result

    def get_high_impact_question_count(self) -> Dict[str, int]:
        """
        Get the number of high-impact questions by semester.

        Returns:
            Dict[str, int]: Mapping from semester to question count
        """
        result = {}

        for eval in self.get_all():
            if eval.semester and eval.semester.get_semester_enum():
                semester = eval.semester.get_semester_enum().value
                result[semester] = eval.get_high_impact_questions_count()

        return result

    def get_semester_order(self) -> List[str]:
        """
        Get semesters in chronological order.

        Returns:
            List[str]: List of semester names
        """
        evaluations = self.get_all()
        semesters = []

        for eval in evaluations:
            if eval.semester and eval.semester.get_semester_enum():
                semesters.append(eval.semester.get_semester_enum().value)

        # Sort by year and term
        return sorted(
            semesters,
            key=lambda s: (
                int(s.split()[1]),  # Year
                0 if s.split()[0] == "Spring" else 1,  # Term (Spring before Fall)
            ),
        )

    def get_evaluation_metrics_by_section(
        self, section_name: str
    ) -> Dict[str, EvaluationMetric]:
        """
        Get evaluation metrics for a specific section across all semesters.

        Args:
            section_name: Section name

        Returns:
            Dict[str, EvaluationMetric]: Mapping from semester to metrics
        """
        result = {}
        section_name_lower = safe_lower(section_name)

        for eval in self.get_all():
            if not eval.semester or not eval.semester.get_semester_enum():
                continue

            semester = eval.semester.get_semester_enum().value

            # Find matching section
            for metric in eval.evaluation_metrics:
                if metric.section and safe_lower(metric.section) == section_name_lower:
                    result[semester] = metric
                    break

        return result

    def get_tool_version_by_semester(self) -> Dict[str, ToolVersion]:
        """
        Get the tool version used in each semester.

        Returns:
            Dict[str, ToolVersion]: Mapping from semester to tool version
        """
        result = {}

        for eval in self.get_all():
            if eval.semester and eval.semester.get_semester_enum():
                semester = eval.semester.get_semester_enum().value
                result[semester] = eval.get_tool_version()

        return result

    def get_average_question_score(self, question_text: str) -> float:
        """
        Get the average score for a specific question across all semesters.

        Args:
            question_text: Question text to search for

        Returns:
            float: Average score
        """
        question_text_lower = safe_lower(question_text)
        scores = []

        for eval in self.get_all():
            # Search for matching question
            for metric in eval.evaluation_metrics:
                for question in metric.questions:
                    if (
                        question.question
                        and question_text_lower in safe_lower(question.question)
                        and question.avg is not None
                    ):
                        scores.append(question.avg)

        return sum(scores) / len(scores) if scores else 0.0

    def get_pre_post_tool_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare average scores before and after tool implementation.

        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        pre_tool_evals = self.find_by_tool_version(ToolVersion.NONE)
        post_tool_evals = []
        post_tool_evals.extend(self.find_by_tool_version(ToolVersion.V1))
        post_tool_evals.extend(self.find_by_tool_version(ToolVersion.V2))

        if not pre_tool_evals or not post_tool_evals:
            return {}

        # Calculate average scores
        pre_tool_overall = sum(
            eval.get_overall_score() for eval in pre_tool_evals
        ) / len(pre_tool_evals)
        post_tool_overall = sum(
            eval.get_overall_score() for eval in post_tool_evals
        ) / len(post_tool_evals)

        pre_tool_high_impact = sum(
            eval.get_overall_score(high_impact_only=True) for eval in pre_tool_evals
        ) / len(pre_tool_evals)
        post_tool_high_impact = sum(
            eval.get_overall_score(high_impact_only=True) for eval in post_tool_evals
        ) / len(post_tool_evals)

        return {
            "overall_score": {
                "pre_tool": pre_tool_overall,
                "post_tool": post_tool_overall,
                "difference": post_tool_overall - pre_tool_overall,
                "percent_change": (
                    (post_tool_overall - pre_tool_overall) / pre_tool_overall * 100
                    if pre_tool_overall > 0
                    else 0
                ),
            },
            "high_impact_score": {
                "pre_tool": pre_tool_high_impact,
                "post_tool": post_tool_high_impact,
                "difference": post_tool_high_impact - pre_tool_high_impact,
                "percent_change": (
                    (post_tool_high_impact - pre_tool_high_impact)
                    / pre_tool_high_impact
                    * 100
                    if pre_tool_high_impact > 0
                    else 0
                ),
            },
        }
