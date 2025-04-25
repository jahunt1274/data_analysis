"""
Learning outcome analyzer for the data analysis system.

This module provides analytical methods for evaluating the educational impact
of the JetPack/Orbit tool on learning outcomes, course evaluations, and
educational effectiveness across cohorts.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict

from ..data.data_repository import DataRepository
from ..data.models.enums import (
    Semester,
    ToolVersion,
    UserEngagementLevel,
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
)
from ..data.models.course_model import CourseEvaluation
from ..utils.common_utils import (
    calculate_correlation,
    calculate_summary_statistics,
    calculate_distribution_percentages,
    calculate_time_differences,
    group_values_into_ranges,
    group_by_time_period,
)
from ..utils.framework_analysis_utils import (
    analyze_framework_bottlenecks,
    identify_high_impact_steps,
    classify_progression_pattern,
    analyze_step_relationships,
)
from ..utils.data_processing_utils import (
    get_step_completion_data,
    group_steps_into_sessions,
    get_activity_metrics_by_time,
    filter_steps_by_framework,
    extract_timestamps_from_steps,
)


class LearningAnalyzer:
    """
    Analyzer for learning outcomes and educational impact.

    This class provides methods for analyzing the relationship between
    tool usage and educational outcomes, comparing cohorts, and
    evaluating the tool's impact on learning.
    """

    def __init__(self, data_repo: DataRepository):
        """
        Initialize the learning analyzer.

        Args:
            data_repo: Data repository with access to all entity repositories
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._data_repo = data_repo
        self._course_id = "15.390"  # Default course ID for analysis

    def set_course_id(self, course_id: str) -> None:
        """
        Set the course ID for analysis.

        Args:
            course_id: Course identifier (e.g., "15.390")
        """
        self._course_id = course_id

    def correlate_tool_usage_with_course_ratings(
        self, semester: Union[str, Semester] = None
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between tool usage and course ratings.

        Args:
            semester: Optional semester to filter by

        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            # Get course evaluation data
            if semester:
                evaluation = self._data_repo.courses.find_by_semester(semester)
                if not evaluation:
                    return {"error": f"No course evaluation found for {semester}"}
                evaluations = [evaluation]
            else:
                evaluations = self._data_repo.courses.get_all()

            # Filter out semesters with no tool (need tool usage to correlate)
            evaluations = [
                eval
                for eval in evaluations
                if eval.get_tool_version() != ToolVersion.NONE
            ]

            if not evaluations:
                return {"error": "No course evaluations found with tool usage"}

            results = {}

            for eval in evaluations:
                sem = (
                    eval.semester.get_semester_enum().value
                    if eval.semester
                    else "Unknown"
                )

                # Get users for this semester
                users = self._data_repo.users.get_users_by_semester(
                    eval.semester.get_semester_enum(), course_id=self._course_id
                )

                if not users:
                    results[sem] = {"error": "No users found for this semester"}
                    continue

                # Group users by engagement level
                engagement_groups = {
                    UserEngagementLevel.HIGH: [],
                    UserEngagementLevel.MEDIUM: [],
                    UserEngagementLevel.LOW: [],
                }

                for user in users:
                    level = user.get_engagement_level()
                    engagement_groups[level].append(user)

                # Get average scores for each engagement group
                high_users_count = len(engagement_groups[UserEngagementLevel.HIGH])
                medium_users_count = len(engagement_groups[UserEngagementLevel.MEDIUM])
                low_users_count = len(engagement_groups[UserEngagementLevel.LOW])
                total_users = high_users_count + medium_users_count + low_users_count

                # Get key metrics from course evaluation
                overall_score = eval.get_overall_score()
                high_impact_score = eval.get_overall_score(high_impact_only=True)

                # Calculate engagement distribution
                engagement_distribution = {
                    "high": high_users_count / total_users if total_users > 0 else 0,
                    "medium": (
                        medium_users_count / total_users if total_users > 0 else 0
                    ),
                    "low": low_users_count / total_users if total_users > 0 else 0,
                }

                # Get content and completion scores by engagement level
                content_scores = {
                    "high": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.HIGH], "content"
                    ),
                    "medium": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.MEDIUM], "content"
                    ),
                    "low": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.LOW], "content"
                    ),
                }

                completion_scores = {
                    "high": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.HIGH], "completion"
                    ),
                    "medium": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.MEDIUM], "completion"
                    ),
                    "low": self._get_avg_score(
                        engagement_groups[UserEngagementLevel.LOW], "completion"
                    ),
                }

                # Calculate weighted engagement score
                weighted_engagement = (
                    3 * engagement_distribution["high"]
                    + 2 * engagement_distribution["medium"]
                    + 1 * engagement_distribution["low"]
                ) / 3

                results[sem] = {
                    "course_rating": {
                        "overall_score": overall_score,
                        "high_impact_score": high_impact_score,
                    },
                    "engagement": {
                        "distribution": engagement_distribution,
                        "weighted_score": weighted_engagement,
                        "content_scores": content_scores,
                        "completion_scores": completion_scores,
                        "user_count": total_users,
                    },
                    "tool_version": eval.get_tool_version().value,
                }

            # Calculate correlation coefficients if we have multiple semesters
            if len(results) > 1:
                correlation = self._calculate_rating_engagement_correlation(results)
                return {"semester_data": results, "correlation": correlation}
            else:
                return {"semester_data": results}

        except Exception as e:
            self._logger.error(f"Error correlating tool usage with course ratings: {e}")
            return {"error": str(e)}

    def _get_avg_score(self, users: List[Any], score_type: str) -> float:
        """
        Get average score for a list of users.

        Args:
            users: List of user objects
            score_type: Score type ("content" or "completion")

        Returns:
            float: Average score
        """
        if not users:
            return 0.0

        scores = []
        for user in users:
            if user.scores:
                if score_type == "content" and user.scores.content is not None:
                    scores.append(user.scores.content)
                elif score_type == "completion" and user.scores.completion is not None:
                    scores.append(user.scores.completion)

        if not scores:
            return 0.0

        # Using utility function to get more comprehensive statistics
        stats = calculate_summary_statistics(scores)
        return stats["mean"]

    def _calculate_rating_engagement_correlation(
        self, semester_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate correlation between course ratings and engagement metrics.

        Args:
            semester_data: Data for each semester

        Returns:
            Dict[str, float]: Correlation coefficients
        """
        # Extract data points for correlation
        overall_ratings = []
        high_impact_ratings = []
        weighted_engagement = []
        high_engagement_pct = []

        for sem, data in semester_data.items():
            if "error" in data:
                continue

            overall_ratings.append(data["course_rating"]["overall_score"])
            high_impact_ratings.append(data["course_rating"]["high_impact_score"])
            weighted_engagement.append(data["engagement"]["weighted_score"])
            high_engagement_pct.append(data["engagement"]["distribution"]["high"])

        # Using utility function instead of custom implementation
        corr_overall_weighted = calculate_correlation(
            overall_ratings, weighted_engagement
        )
        corr_high_impact_weighted = calculate_correlation(
            high_impact_ratings, weighted_engagement
        )
        corr_overall_high_engagement = calculate_correlation(
            overall_ratings, high_engagement_pct
        )
        corr_high_impact_high_engagement = calculate_correlation(
            high_impact_ratings, high_engagement_pct
        )

        return {
            "overall_rating_weighted_engagement": corr_overall_weighted,
            "high_impact_rating_weighted_engagement": corr_high_impact_weighted,
            "overall_rating_high_engagement_pct": corr_overall_high_engagement,
            "high_impact_rating_high_engagement_pct": corr_high_impact_high_engagement,
        }

    def compare_learning_outcomes_by_cohort(
        self,
        pre_tool_semester: Union[str, Semester] = Semester.FALL_2023,
        post_tool_semesters: Optional[List[Union[str, Semester]]] = None,
    ) -> Dict[str, Any]:
        """
        Compare learning outcomes between pre-tool and post-tool cohorts.

        Args:
            pre_tool_semester: Semester before tool introduction
            post_tool_semesters: List of semesters after tool introduction

        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Default to all post-tool semesters if not specified
            if post_tool_semesters is None:
                post_tool_semesters = [
                    Semester.SPRING_2024,
                    Semester.FALL_2024,
                    Semester.SPRING_2025,
                ]

            # Get pre-tool evaluation
            pre_tool_eval = self._data_repo.courses.find_by_semester(pre_tool_semester)
            if not pre_tool_eval:
                return {"error": f"No course evaluation found for {pre_tool_semester}"}

            # Get post-tool evaluations
            post_tool_evals = []
            for semester in post_tool_semesters:
                eval = self._data_repo.courses.find_by_semester(semester)
                if eval:
                    post_tool_evals.append(eval)

            if not post_tool_evals:
                return {"error": "No course evaluations found for post-tool semesters"}

            # Get high-impact questions from pre-tool evaluation
            pre_tool_questions = {}
            pre_tool_sections = {}

            for metric in pre_tool_eval.evaluation_metrics:
                section = metric.section
                if not section:
                    continue

                # Get section average
                pre_tool_sections[section] = metric.get_average_score()

                # Get question scores
                for question in metric.questions:
                    if question.question and question.avg is not None:
                        pre_tool_questions[question.question] = question.avg

            # Collect post-tool metrics
            post_tool_metrics = {
                "by_semester": {},
                "aggregated": {
                    "questions": {},
                    "sections": {},
                    "overall": 0.0,
                    "high_impact_overall": 0.0,
                },
            }

            # Track all sections and questions for alignment
            all_sections = set(pre_tool_sections.keys())
            all_questions = set(pre_tool_questions.keys())

            # Process each post-tool evaluation
            for eval in post_tool_evals:
                sem = (
                    eval.semester.get_semester_enum().value
                    if eval.semester
                    else "Unknown"
                )
                post_tool_metrics["by_semester"][sem] = {
                    "questions": {},
                    "sections": {},
                    "overall": eval.get_overall_score(),
                    "high_impact_overall": eval.get_overall_score(
                        high_impact_only=True
                    ),
                    "tool_version": eval.get_tool_version().value,
                }

                # Process metrics
                for metric in eval.evaluation_metrics:
                    section = metric.section
                    if not section:
                        continue

                    # Add to tracked sections
                    all_sections.add(section)

                    # Get section average
                    post_tool_metrics["by_semester"][sem]["sections"][
                        section
                    ] = metric.get_average_score()

                    # Aggregate to overall post-tool metrics
                    if section not in post_tool_metrics["aggregated"]["sections"]:
                        post_tool_metrics["aggregated"]["sections"][section] = []
                    post_tool_metrics["aggregated"]["sections"][section].append(
                        metric.get_average_score()
                    )

                    # Get question scores
                    for question in metric.questions:
                        if question.question and question.avg is not None:
                            # Add to tracked questions
                            all_questions.add(question.question)

                            # Add to semester metrics
                            post_tool_metrics["by_semester"][sem]["questions"][
                                question.question
                            ] = question.avg

                            # Aggregate to overall post-tool metrics
                            if (
                                question.question
                                not in post_tool_metrics["aggregated"]["questions"]
                            ):
                                post_tool_metrics["aggregated"]["questions"][
                                    question.question
                                ] = []
                            post_tool_metrics["aggregated"]["questions"][
                                question.question
                            ].append(question.avg)

                # Add to overall aggregated scores
                if "overall" not in post_tool_metrics["aggregated"]:
                    post_tool_metrics["aggregated"]["overall"] = []
                    post_tool_metrics["aggregated"]["high_impact_overall"] = []

                post_tool_metrics["aggregated"]["overall"].append(
                    eval.get_overall_score()
                )
                post_tool_metrics["aggregated"]["high_impact_overall"].append(
                    eval.get_overall_score(high_impact_only=True)
                )

            # Calculate average for aggregated metrics
            for section, scores in post_tool_metrics["aggregated"]["sections"].items():
                post_tool_metrics["aggregated"]["sections"][section] = sum(
                    scores
                ) / len(scores)

            for question, scores in post_tool_metrics["aggregated"][
                "questions"
            ].items():
                post_tool_metrics["aggregated"]["questions"][question] = sum(
                    scores
                ) / len(scores)

            post_tool_metrics["aggregated"]["overall"] = sum(
                post_tool_metrics["aggregated"]["overall"]
            ) / len(post_tool_metrics["aggregated"]["overall"])

            post_tool_metrics["aggregated"]["high_impact_overall"] = sum(
                post_tool_metrics["aggregated"]["high_impact_overall"]
            ) / len(post_tool_metrics["aggregated"]["high_impact_overall"])

            # Calculate differences between pre-tool and post-tool
            differences = {
                "sections": {},
                "questions": {},
                "overall": post_tool_metrics["aggregated"]["overall"]
                - pre_tool_eval.get_overall_score(),
                "high_impact_overall": (
                    post_tool_metrics["aggregated"]["high_impact_overall"]
                    - pre_tool_eval.get_overall_score(high_impact_only=True)
                ),
            }

            # Calculate section differences
            for section in all_sections:
                pre_score = pre_tool_sections.get(section, 0.0)
                post_score = post_tool_metrics["aggregated"]["sections"].get(
                    section, 0.0
                )

                if pre_score > 0 and post_score > 0:
                    differences["sections"][section] = {
                        "pre_tool": pre_score,
                        "post_tool": post_score,
                        "difference": post_score - pre_score,
                        "percent_change": (
                            (post_score - pre_score) / pre_score * 100
                            if pre_score > 0
                            else 0.0
                        ),
                    }

            # Calculate question differences
            for question in all_questions:
                pre_score = pre_tool_questions.get(question, 0.0)
                post_score = post_tool_metrics["aggregated"]["questions"].get(
                    question, 0.0
                )

                if pre_score > 0 and post_score > 0:
                    differences["questions"][question] = {
                        "pre_tool": pre_score,
                        "post_tool": post_score,
                        "difference": post_score - pre_score,
                        "percent_change": (
                            (post_score - pre_score) / pre_score * 100
                            if pre_score > 0
                            else 0.0
                        ),
                    }

            # Return complete comparison
            return {
                "pre_tool": {
                    "semester": (
                        pre_tool_semester.value
                        if isinstance(pre_tool_semester, Semester)
                        else pre_tool_semester
                    ),
                    "overall": pre_tool_eval.get_overall_score(),
                    "high_impact_overall": pre_tool_eval.get_overall_score(
                        high_impact_only=True
                    ),
                    "sections": pre_tool_sections,
                    "questions": pre_tool_questions,
                },
                "post_tool": post_tool_metrics,
                "differences": differences,
            }

        except Exception as e:
            self._logger.error(f"Error comparing learning outcomes by cohort: {e}")
            return {"error": str(e)}

    def analyze_tool_version_impact(self) -> Dict[str, Any]:
        """
        Compare the impact of different tool versions on learning outcomes.

        Returns:
            Dict[str, Any]: Tool version impact analysis
        """
        try:
            # Get semesters by tool version
            v1_semesters = []
            v2_semesters = []

            semester_tool_map = self._data_repo.courses.get_tool_version_by_semester()
            for semester, tool_version in semester_tool_map.items():
                if tool_version == ToolVersion.V1:
                    v1_semesters.append(semester)
                elif tool_version == ToolVersion.V2:
                    v2_semesters.append(semester)

            # Get evaluations for each version
            v1_evals = []
            for semester in v1_semesters:
                eval = self._data_repo.courses.find_by_semester(semester)
                if eval:
                    v1_evals.append(eval)

            v2_evals = []
            for semester in v2_semesters:
                eval = self._data_repo.courses.find_by_semester(semester)
                if eval:
                    v2_evals.append(eval)

            if not v1_evals or not v2_evals:
                return {"error": "Insufficient data for version comparison"}

            # Calculate aggregate metrics for each version
            v1_metrics = self._aggregate_evaluation_metrics(v1_evals)
            v2_metrics = self._aggregate_evaluation_metrics(v2_evals)

            # Get user engagement for each version
            v1_engagement = self._get_engagement_by_tool_version(ToolVersion.V1)
            v2_engagement = self._get_engagement_by_tool_version(ToolVersion.V2)

            # Calculate differences between versions
            differences = {
                "overall": v2_metrics["overall"] - v1_metrics["overall"],
                "high_impact_overall": v2_metrics["high_impact_overall"]
                - v1_metrics["high_impact_overall"],
                "sections": {},
                "questions": {},
                "engagement": {
                    "high_pct": v2_engagement["distribution"]["high"]
                    - v1_engagement["distribution"]["high"],
                    "medium_pct": v2_engagement["distribution"]["medium"]
                    - v1_engagement["distribution"]["medium"],
                    "low_pct": v2_engagement["distribution"]["low"]
                    - v1_engagement["distribution"]["low"],
                    "weighted_score": v2_engagement["weighted_score"]
                    - v1_engagement["weighted_score"],
                    "avg_content_score": v2_engagement["avg_content_score"]
                    - v1_engagement["avg_content_score"],
                    "avg_completion_score": v2_engagement["avg_completion_score"]
                    - v1_engagement["avg_completion_score"],
                },
            }

            # Calculate section differences
            all_sections = set(v1_metrics["sections"].keys()) | set(
                v2_metrics["sections"].keys()
            )
            for section in all_sections:
                v1_score = v1_metrics["sections"].get(section, 0.0)
                v2_score = v2_metrics["sections"].get(section, 0.0)

                if v1_score > 0 and v2_score > 0:
                    differences["sections"][section] = {
                        "v1": v1_score,
                        "v2": v2_score,
                        "difference": v2_score - v1_score,
                        "percent_change": (
                            (v2_score - v1_score) / v1_score * 100
                            if v1_score > 0
                            else 0.0
                        ),
                    }

            # Calculate question differences
            all_questions = set(v1_metrics["questions"].keys()) | set(
                v2_metrics["questions"].keys()
            )
            for question in all_questions:
                v1_score = v1_metrics["questions"].get(question, 0.0)
                v2_score = v2_metrics["questions"].get(question, 0.0)

                if v1_score > 0 and v2_score > 0:
                    differences["questions"][question] = {
                        "v1": v1_score,
                        "v2": v2_score,
                        "difference": v2_score - v1_score,
                        "percent_change": (
                            (v2_score - v1_score) / v1_score * 100
                            if v1_score > 0
                            else 0.0
                        ),
                    }

            # Return comparison results
            return {
                "v1": {
                    "semesters": v1_semesters,
                    "metrics": v1_metrics,
                    "engagement": v1_engagement,
                },
                "v2": {
                    "semesters": v2_semesters,
                    "metrics": v2_metrics,
                    "engagement": v2_engagement,
                },
                "differences": differences,
            }

        except Exception as e:
            self._logger.error(f"Error analyzing tool version impact: {e}")
            return {"error": str(e)}

    def _aggregate_evaluation_metrics(
        self, evaluations: List[CourseEvaluation]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple course evaluations.

        Args:
            evaluations: List of course evaluations

        Returns:
            Dict[str, Any]: Aggregated metrics
        """
        if not evaluations:
            return {
                "overall": 0.0,
                "high_impact_overall": 0.0,
                "sections": {},
                "questions": {},
            }

        # Collect metrics
        overall_scores = []
        high_impact_scores = []
        section_scores = defaultdict(list)
        question_scores = defaultdict(list)

        for eval in evaluations:
            overall_scores.append(eval.get_overall_score())
            high_impact_scores.append(eval.get_overall_score(high_impact_only=True))

            # Process metrics
            for metric in eval.evaluation_metrics:
                section = metric.section
                if not section:
                    continue

                # Get section average
                section_scores[section].append(metric.get_average_score())

                # Get question scores
                for question in metric.questions:
                    if question.question and question.avg is not None:
                        question_scores[question.question].append(question.avg)

        # Calculate averages
        aggregated = {
            "overall": (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            ),
            "high_impact_overall": (
                sum(high_impact_scores) / len(high_impact_scores)
                if high_impact_scores
                else 0.0
            ),
            "sections": {},
            "questions": {},
        }

        for section, scores in section_scores.items():
            aggregated["sections"][section] = sum(scores) / len(scores)

        for question, scores in question_scores.items():
            aggregated["questions"][question] = sum(scores) / len(scores)

        return aggregated

    def _get_engagement_by_tool_version(
        self, tool_version: ToolVersion
    ) -> Dict[str, Any]:
        """
        Get user engagement metrics for a specific tool version.

        Args:
            tool_version: Tool version to analyze

        Returns:
            Dict[str, Any]: Engagement metrics
        """
        # Get semesters for this tool version
        semesters = []
        semester_tool_map = self._data_repo.courses.get_tool_version_by_semester()
        for semester, version in semester_tool_map.items():
            if version == tool_version:
                semesters.append(semester)

        # Get all users for these semesters
        all_users = []
        for semester in semesters:
            sem_obj = None
            for sem_enum in Semester:
                if sem_enum.value == semester:
                    sem_obj = sem_enum
                    break

            if sem_obj:
                users = self._data_repo.users.get_users_by_semester(
                    sem_obj, course_id=self._course_id
                )
                all_users.extend(users)

        # Count users by engagement level
        high_users = []
        medium_users = []
        low_users = []

        for user in all_users:
            level = user.get_engagement_level()
            if level == UserEngagementLevel.HIGH:
                high_users.append(user)
            elif level == UserEngagementLevel.MEDIUM:
                medium_users.append(user)
            else:
                low_users.append(user)

        total_users = len(all_users)

        # Calculate engagement metrics
        if total_users > 0:
            distribution = {
                "high": len(high_users) / total_users,
                "medium": len(medium_users) / total_users,
                "low": len(low_users) / total_users,
            }
        else:
            distribution = {"high": 0.0, "medium": 0.0, "low": 0.0}

        # Calculate weighted engagement score
        weighted_score = (
            3 * distribution["high"]
            + 2 * distribution["medium"]
            + 1 * distribution["low"]
        ) / 3

        # Calculate average scores
        avg_content_score = self._get_avg_score(all_users, "content")
        avg_completion_score = self._get_avg_score(all_users, "completion")

        return {
            "distribution": distribution,
            "weighted_score": weighted_score,
            "avg_content_score": avg_content_score,
            "avg_completion_score": avg_completion_score,
            "user_count": total_users,
        }

    def get_learning_outcome_metrics(
        self, outcome_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Measure impact on specific learning objectives.

        Args:
            outcome_keywords: Optional list of keywords for learning outcomes

        Returns:
            Dict[str, Any]: Learning outcome metrics
        """
        try:
            # Default learning outcome keywords if not provided
            if outcome_keywords is None:
                outcome_keywords = [
                    "objectives",
                    "learn",
                    "understand",
                    "knowledge",
                    "skill",
                    "concept",
                    "application",
                    "entrepreneurial",
                    "framework",
                ]

            # Get evaluations by semester
            evaluations = self._data_repo.courses.get_all()

            # Group by tool version
            results_by_version = {
                "none": {"questions": {}, "count": 0, "score": 0.0},
                "v1": {"questions": {}, "count": 0, "score": 0.0},
                "v2": {"questions": {}, "count": 0, "score": 0.0},
            }

            results_by_semester = {}

            # Track which questions match learning outcomes
            learning_outcome_questions = set()

            # Process each evaluation
            for eval in evaluations:
                version = eval.get_tool_version().value
                semester = (
                    eval.semester.get_semester_enum().value
                    if eval.semester
                    else "Unknown"
                )

                # Initialize semester results
                results_by_semester[semester] = {
                    "tool_version": version,
                    "questions": {},
                    "score": 0.0,
                    "count": 0,
                }

                # Find questions related to learning outcomes
                outcome_scores = []

                for metric in eval.evaluation_metrics:
                    for question in metric.questions:
                        if not question.question or question.avg is None:
                            continue

                        # Check if question relates to learning outcomes
                        is_outcome_question = question.is_high_impact or any(
                            keyword.lower() in question.question.lower()
                            for keyword in outcome_keywords
                        )

                        if is_outcome_question:
                            # Add to tracked learning outcome questions
                            learning_outcome_questions.add(question.question)

                            # Add to version results
                            if (
                                question.question
                                not in results_by_version[version]["questions"]
                            ):
                                results_by_version[version]["questions"][
                                    question.question
                                ] = []

                            results_by_version[version]["questions"][
                                question.question
                            ].append(question.avg)
                            results_by_version[version]["count"] += 1

                            # Add to semester results
                            results_by_semester[semester]["questions"][
                                question.question
                            ] = question.avg
                            results_by_semester[semester]["count"] += 1

                            # Add to outcome scores for this semester
                            outcome_scores.append(question.avg)

                # Calculate average score for learning outcome questions
                if outcome_scores:
                    avg_outcome_score = sum(outcome_scores) / len(outcome_scores)
                    results_by_semester[semester]["score"] = avg_outcome_score

            # Calculate averages for version results
            for version, data in results_by_version.items():
                for question, scores in data["questions"].items():
                    data["questions"][question] = sum(scores) / len(scores)

                all_scores = []
                for scores in data["questions"].values():
                    if isinstance(scores, list):
                        all_scores.extend(scores)
                    else:
                        all_scores.append(scores)

                if all_scores:
                    data["score"] = sum(all_scores) / len(all_scores)

            # Group semester results by cohort (pre-tool, v1, v2)
            cohort_results = {"pre_tool": {}, "v1": {}, "v2": {}}

            for semester, data in results_by_semester.items():
                version = data["tool_version"]
                if version == "none":
                    cohort_results["pre_tool"][semester] = data
                elif version == "v1":
                    cohort_results["v1"][semester] = data
                elif version == "v2":
                    cohort_results["v2"][semester] = data

            # Calculate cohort aggregates
            cohort_aggregates = {}
            for cohort, semesters in cohort_results.items():
                all_scores = []
                for semester_data in semesters.values():
                    if semester_data["score"] > 0:
                        all_scores.append(semester_data["score"])

                cohort_aggregates[cohort] = {
                    "score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                    "semester_count": len(semesters),
                    "question_count": sum(data["count"] for data in semesters.values()),
                }

            # Calculate impact metrics
            impact_metrics = {}

            # V1 vs. Pre-tool
            if (
                cohort_aggregates["pre_tool"]["score"] > 0
                and cohort_aggregates["v1"]["score"] > 0
            ):
                impact_metrics["v1_vs_pre"] = {
                    "absolute_diff": cohort_aggregates["v1"]["score"]
                    - cohort_aggregates["pre_tool"]["score"],
                    "percent_change": (
                        (
                            (
                                cohort_aggregates["v1"]["score"]
                                - cohort_aggregates["pre_tool"]["score"]
                            )
                            / cohort_aggregates["pre_tool"]["score"]
                            * 100
                        )
                        if cohort_aggregates["pre_tool"]["score"] > 0
                        else 0.0
                    ),
                }

            # V2 vs. V1
            if (
                cohort_aggregates["v1"]["score"] > 0
                and cohort_aggregates["v2"]["score"] > 0
            ):
                impact_metrics["v2_vs_v1"] = {
                    "absolute_diff": cohort_aggregates["v2"]["score"]
                    - cohort_aggregates["v1"]["score"],
                    "percent_change": (
                        (
                            (
                                cohort_aggregates["v2"]["score"]
                                - cohort_aggregates["v1"]["score"]
                            )
                            / cohort_aggregates["v1"]["score"]
                            * 100
                        )
                        if cohort_aggregates["v1"]["score"] > 0
                        else 0.0
                    ),
                }

            # V2 vs. Pre-tool
            if (
                cohort_aggregates["pre_tool"]["score"] > 0
                and cohort_aggregates["v2"]["score"] > 0
            ):
                impact_metrics["v2_vs_pre"] = {
                    "absolute_diff": cohort_aggregates["v2"]["score"]
                    - cohort_aggregates["pre_tool"]["score"],
                    "percent_change": (
                        (
                            (
                                cohort_aggregates["v2"]["score"]
                                - cohort_aggregates["pre_tool"]["score"]
                            )
                            / cohort_aggregates["pre_tool"]["score"]
                            * 100
                        )
                        if cohort_aggregates["pre_tool"]["score"] > 0
                        else 0.0
                    ),
                }

            return {
                "learning_outcome_questions": list(learning_outcome_questions),
                "by_version": results_by_version,
                "by_semester": results_by_semester,
                "cohort_aggregates": cohort_aggregates,
                "impact_metrics": impact_metrics,
            }

        except Exception as e:
            self._logger.error(f"Error calculating learning outcome metrics: {e}")
            return {"error": str(e)}

    def analyze_learning_outcomes_vs_engagement(self) -> Dict[str, Any]:
        """
        Analyze the relationship between tool engagement and learning outcomes.

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get course evaluations
            evaluations = self._data_repo.courses.get_all()

            # Filter out semesters with no tool (need tool usage to correlate)
            evaluations = [
                eval
                for eval in evaluations
                if eval.get_tool_version() != ToolVersion.NONE
            ]

            if not evaluations:
                return {"error": "No course evaluations found with tool usage"}

            results = {}

            for eval in evaluations:
                sem = (
                    eval.semester.get_semester_enum().value
                    if eval.semester
                    else "Unknown"
                )

                # Get users for this semester
                users = self._data_repo.users.get_users_by_semester(
                    eval.semester.get_semester_enum(), course_id=self._course_id
                )

                if not users:
                    results[sem] = {"error": "No users found for this semester"}
                    continue

                # Group users by engagement level
                engagement_groups = {
                    UserEngagementLevel.HIGH: [],
                    UserEngagementLevel.MEDIUM: [],
                    UserEngagementLevel.LOW: [],
                }

                for user in users:
                    level = user.get_engagement_level()
                    engagement_groups[level].append(user)

                # Get users' idea and step counts by engagement level
                idea_counts = {}
                step_counts = {}

                for level, level_users in engagement_groups.items():
                    level_name = level.value
                    idea_counts[level_name] = 0
                    step_counts[level_name] = 0

                    for user in level_users:
                        if user.email:
                            ideas = self._data_repo.ideas.find_by_owner(user.email)
                            idea_counts[level_name] += len(ideas)

                            for idea in ideas:
                                if idea.id:
                                    idea_id = (
                                        idea.id.oid
                                        if hasattr(idea.id, "oid")
                                        else str(idea.id)
                                    )
                                    steps = self._data_repo.steps.find_by_idea_id(
                                        idea_id
                                    )
                                    step_counts[level_name] += len(steps)

                # Calculate per-user averages
                avg_ideas = {}
                avg_steps = {}

                for level, count in idea_counts.items():
                    level_users = len(engagement_groups[UserEngagementLevel(level)])
                    avg_ideas[level] = count / level_users if level_users > 0 else 0

                for level, count in step_counts.items():
                    level_users = len(engagement_groups[UserEngagementLevel(level)])
                    avg_steps[level] = count / level_users if level_users > 0 else 0

                # Get learning outcome scores
                outcome_scores = []

                for metric in eval.evaluation_metrics:
                    for question in metric.questions:
                        if question.is_high_impact and question.avg is not None:
                            outcome_scores.append(question.avg)

                learning_outcome_score = (
                    sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0
                )

                # Store results for this semester
                results[sem] = {
                    "learning_outcome_score": learning_outcome_score,
                    "tool_version": eval.get_tool_version().value,
                    "engagement": {
                        "distribution": {
                            "high": len(engagement_groups[UserEngagementLevel.HIGH]),
                            "medium": len(
                                engagement_groups[UserEngagementLevel.MEDIUM]
                            ),
                            "low": len(engagement_groups[UserEngagementLevel.LOW]),
                        },
                        "avg_ideas_per_user": avg_ideas,
                        "avg_steps_per_user": avg_steps,
                        "total_ideas": sum(idea_counts.values()),
                        "total_steps": sum(step_counts.values()),
                    },
                }

            # Calculate correlations between engagement and outcomes
            correlation_data = {
                "high_engagement_vs_outcome": [],
                "avg_ideas_vs_outcome": [],
                "avg_steps_vs_outcome": [],
            }

            for sem, data in results.items():
                if "error" in data:
                    continue

                total_users = sum(data["engagement"]["distribution"].values())
                if total_users > 0:
                    high_pct = data["engagement"]["distribution"]["high"] / total_users

                    correlation_data["high_engagement_vs_outcome"].append(
                        (high_pct, data["learning_outcome_score"])
                    )

                    avg_ideas = (
                        data["engagement"]["total_ideas"] / total_users
                        if total_users > 0
                        else 0
                    )

                    correlation_data["avg_ideas_vs_outcome"].append(
                        (avg_ideas, data["learning_outcome_score"])
                    )

                    avg_steps = (
                        data["engagement"]["total_steps"] / total_users
                        if total_users > 0
                        else 0
                    )

                    correlation_data["avg_steps_vs_outcome"].append(
                        (avg_steps, data["learning_outcome_score"])
                    )

            # Calculate correlation coefficients
            correlations = {}

            for metric, data_points in correlation_data.items():
                if data_points:
                    x_values = [point[0] for point in data_points]
                    y_values = [point[1] for point in data_points]

                    correlations[metric] = calculate_correlation(x_values, y_values)

            return {"semester_data": results, "correlations": correlations}

        except Exception as e:
            self._logger.error(f"Error analyzing learning outcomes vs engagement: {e}")
            return {"error": str(e)}

    def analyze_idea_quality_vs_learning(self) -> Dict[str, Any]:
        """
        Analyze the relationship between idea quality and learning outcomes.

        This method assesses whether better learning outcomes correlate
        with higher quality entrepreneurial ideas.

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # For each semester with the tool, analyze:
            # 1. Learning outcome metrics
            # 2. Idea quality metrics (step completion, content scores)
            # 3. Correlation between the two

            results = {}

            # Get semesters with tool usage
            tool_semesters = []
            semester_tool_map = self._data_repo.courses.get_tool_version_by_semester()
            for semester, version in semester_tool_map.items():
                if version != ToolVersion.NONE:
                    tool_semesters.append(semester)

            # Calculate metrics for each semester
            for semester in tool_semesters:
                # Get course evaluation for learning outcomes
                evaluation = self._data_repo.courses.find_by_semester(semester)
                if not evaluation:
                    continue

                # Get learning outcome metrics
                learning_outcome_score = evaluation.get_overall_score(
                    high_impact_only=True
                )

                # Get users for this semester
                sem_obj = None
                for sem_enum in Semester:
                    if sem_enum.value == semester:
                        sem_obj = sem_enum
                        break

                if not sem_obj:
                    continue

                users = self._data_repo.users.get_users_by_semester(
                    sem_obj, course_id=self._course_id
                )

                if not users:
                    continue

                # Get ideas for these users
                all_ideas = []
                idea_quality_scores = []

                for user in users:
                    if user.email:
                        ideas = self._data_repo.ideas.find_by_owner(user.email)
                        all_ideas.extend(ideas)

                        # Calculate quality metrics for each idea
                        for idea in ideas:
                            # Quality factors:
                            # 1. Number of steps completed
                            # 2. Content score
                            # 3. Completion score
                            de_steps_count = idea.get_de_steps_count()
                            st_steps_count = idea.get_st_steps_count()
                            total_steps = de_steps_count + st_steps_count

                            # Get framework progress
                            de_progress = idea.get_framework_progress(
                                FrameworkType.DISCIPLINED_ENTREPRENEURSHIP
                            )

                            # Use content/completion scores if available
                            content_score = user.scores.content if user.scores else 0.0
                            completion_score = (
                                user.scores.completion if user.scores else 0.0
                            )

                            # Calculate combined quality score
                            quality_score = (
                                0.4
                                * (
                                    total_steps / 24 if total_steps <= 24 else 1.0
                                )  # Steps count (max 24)
                                + 0.3 * de_progress  # Framework progress
                                + 0.15 * content_score  # Content score
                                + 0.15 * completion_score  # Completion score
                            )

                            idea_quality_scores.append(quality_score)

                # Calculate average idea quality
                avg_quality = (
                    sum(idea_quality_scores) / len(idea_quality_scores)
                    if idea_quality_scores
                    else 0.0
                )

                # Get step completion rates
                completion_rates = self._data_repo.ideas.get_step_completion_rates(
                    FrameworkType.DISCIPLINED_ENTREPRENEURSHIP
                )

                # Calculate average step completion rate
                avg_completion_rate = (
                    sum(completion_rates.values()) / len(completion_rates)
                    if completion_rates
                    else 0.0
                )

                # Using utility function to create a distribution
                quality_data = {}
                for i in range(5):
                    start = i * 0.2
                    end = (i + 1) * 0.2
                    key = f"{start:.1f}-{end:.1f}"
                    quality_data[key] = sum(
                        1 for score in idea_quality_scores if start <= score < end
                    )

                quality_distribution = calculate_distribution_percentages(quality_data)

                # Store results for this semester
                results[semester] = {
                    "learning_outcome_score": learning_outcome_score,
                    "idea_metrics": {
                        "count": len(all_ideas),
                        "avg_quality_score": avg_quality,
                        "avg_step_completion_rate": avg_completion_rate,
                        "quality_distribution": quality_distribution,
                    },
                    "tool_version": evaluation.get_tool_version().value,
                }

            # Calculate correlation between learning outcomes and idea quality
            learning_scores = []
            quality_scores = []
            completion_rates = []

            for semester, data in results.items():
                learning_scores.append(data["learning_outcome_score"])
                quality_scores.append(data["idea_metrics"]["avg_quality_score"])
                completion_rates.append(
                    data["idea_metrics"]["avg_step_completion_rate"]
                )

            correlation_learning_quality = calculate_correlation(
                learning_scores, quality_scores
            )
            correlation_learning_completion = calculate_correlation(
                learning_scores, completion_rates
            )

            return {
                "semester_data": results,
                "correlations": {
                    "learning_outcome_vs_idea_quality": correlation_learning_quality,
                    "learning_outcome_vs_step_completion": correlation_learning_completion,
                },
            }

        except Exception as e:
            self._logger.error(f"Error analyzing idea quality vs learning: {e}")
            return {"error": str(e)}

    def analyze_learning_objectives_by_step(self) -> Dict[str, Any]:
        """
        Analyze how different framework steps contribute to learning objectives.

        This method assesses which entrepreneurial framework steps are most
        strongly associated with achievement of learning objectives.

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get steps data
            all_steps = self._data_repo.steps.get_all()

            # Get step completion rates - using utility function
            ideas = self._data_repo.ideas.get_all()
            framework_steps = [step.value for step in DisciplinedEntrepreneurshipStep]

            completion_data = get_step_completion_data(ideas, framework_steps)
            completion_rates = {
                step: data["rate"] for step, data in completion_data.items()
            }

            # Get learning outcome metrics
            learning_outcomes = self.get_learning_outcome_metrics()

            # Calculate step impact by comparing engagement with steps to learning outcomes
            # Get user engagement scores
            user_engagement_scores = {}
            for user in self._data_repo.users.get_all():
                if user.scores and user.scores.content is not None:
                    user_engagement_scores[user.email] = user.scores.content

            # Get step sequences for each user
            step_sequences = []
            for user_id, _ in user_engagement_scores.items():
                user_ideas = self._data_repo.ideas.find_by_owner(user_id)
                user_sequences = []

                for idea in user_ideas:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    steps = self._data_repo.steps.find_by_idea_id(idea_id)
                    if steps:
                        completed_steps = {step.step for step in steps if step.step}
                        if completed_steps:
                            user_sequences.extend(list(completed_steps))

                if user_sequences:
                    step_sequences.append(user_sequences)

            # Using the framework utility to identify high impact steps
            step_impact_data = identify_high_impact_steps(
                completion_rates,
                user_engagement_scores,
                step_sequences,
                framework_steps,
            )

            # Using the framework utility to analyze step relationships
            step_relationship_data = analyze_step_relationships(step_sequences)

            # Using the framework utility to analyze bottlenecks
            step_numbers = {
                step.value: step.step_number for step in DisciplinedEntrepreneurshipStep
            }

            bottleneck_analysis = analyze_framework_bottlenecks(
                completion_rates, step_relationship_data, framework_steps, step_numbers
            )

            return {
                "step_metrics": step_impact_data["step_impact_scores"],
                "steps_by_impact": [
                    {"step": step, "metrics": data}
                    for step, data in sorted(
                        step_impact_data["step_impact_scores"].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )
                ],
                "impact_groups": {
                    "high_impact": [
                        step["step"] for step in step_impact_data["high_impact_steps"]
                    ],
                    "medium_impact": [],  # This needs custom logic
                    "low_impact": [
                        step["step"] for step in step_impact_data["low_impact_steps"]
                    ],
                },
                "bottlenecks": bottleneck_analysis,
                "step_relationships": step_relationship_data,
                "learning_outcomes": learning_outcomes,
            }

        except Exception as e:
            self._logger.error(f"Error analyzing learning objectives by step: {e}")
            return {"error": str(e)}

    def calculate_framework_engagement_metrics(self) -> Dict[str, Any]:
        """
        Calculate engagement metrics for the entrepreneurial frameworks.

        Returns:
            Dict[str, Any]: Framework engagement metrics
        """
        try:
            # Get all steps
            all_steps = self._data_repo.steps.get_all()

            # Count steps by framework using utility function
            steps_by_framework = {}
            for framework in FrameworkType:
                framework_steps = filter_steps_by_framework(all_steps, framework.value)
                steps_by_framework[framework.value] = len(framework_steps)

            # Get user counts by semester
            semester_user_counts = {}

            for semester in Semester:
                users = self._data_repo.users.get_users_by_semester(
                    semester, course_id=self._course_id
                )
                semester_user_counts[semester.value] = len(users)

            # Get engagement by framework and semester
            framework_semester_metrics = {}

            for framework in FrameworkType:
                framework_semester_metrics[framework.value] = {}

                for semester in Semester:
                    # Get users for this semester
                    users = self._data_repo.users.get_users_by_semester(
                        semester, course_id=self._course_id
                    )

                    if not users:
                        continue

                    # Count framework steps for these users
                    step_count = 0
                    user_count = 0

                    for user in users:
                        if user.email:
                            # Get steps by this user in this framework using utility function
                            user_steps = [
                                step
                                for step in filter_steps_by_framework(
                                    all_steps, framework.value
                                )
                                if step.owner == user.email
                            ]

                            if user_steps:
                                step_count += len(user_steps)
                                user_count += 1

                    # Calculate average steps per user
                    avg_steps = step_count / user_count if user_count > 0 else 0

                    # Calculate engagement rate (users engaging with framework / total users)
                    engagement_rate = user_count / len(users) if users else 0

                    framework_semester_metrics[framework.value][semester.value] = {
                        "step_count": step_count,
                        "user_count": user_count,
                        "total_users": len(users),
                        "avg_steps_per_user": avg_steps,
                        "engagement_rate": engagement_rate,
                    }

            # Calculate correlations with learning outcomes
            learning_correlations = {}

            # Get course evaluations by semester
            evaluations = self._data_repo.courses.get_all()
            semester_learning_scores = {}

            for eval in evaluations:
                if eval.semester and eval.semester.get_semester_enum():
                    semester = eval.semester.get_semester_enum().value
                    semester_learning_scores[semester] = eval.get_overall_score(
                        high_impact_only=True
                    )

            # Calculate correlation for each framework
            for framework in FrameworkType:
                x_values = []  # Framework engagement rates
                y_values = []  # Learning outcome scores

                for semester, score in semester_learning_scores.items():
                    if (
                        framework.value in framework_semester_metrics
                        and semester in framework_semester_metrics[framework.value]
                    ):
                        metrics = framework_semester_metrics[framework.value][semester]
                        x_values.append(metrics["engagement_rate"])
                        y_values.append(score)

                correlation = calculate_correlation(x_values, y_values)
                learning_correlations[framework.value] = correlation

            return {
                "steps_by_framework": steps_by_framework,
                "framework_semester_metrics": framework_semester_metrics,
                "learning_outcome_correlations": learning_correlations,
            }

        except Exception as e:
            self._logger.error(f"Error calculating framework engagement metrics: {e}")
            return {"error": str(e)}

    def analyze_tool_impact_on_time_allocation(self) -> Dict[str, Any]:
        """
        Analyze how the tool impacts time allocation for students.

        Examines whether the tool helps students complete the entrepreneurial
        frameworks more efficiently, potentially freeing time for other activities
        like customer interaction.

        Returns:
            Dict[str, Any]: Impact analysis results
        """
        try:
            # Compare time metrics between tool and non-tool semesters

            # Get time between steps distribution for each tool version
            time_metrics_by_version = {"none": {}, "v1": {}, "v2": {}}

            # Get all steps
            all_steps = self._data_repo.steps.get_all()

            # Group steps by idea_id
            steps_by_idea = {}
            for step in all_steps:
                if step.idea_id:
                    idea_id = (
                        step.idea_id.oid
                        if hasattr(step.idea_id, "oid")
                        else str(step.idea_id)
                    )
                    if idea_id not in steps_by_idea:
                        steps_by_idea[idea_id] = []
                    steps_by_idea[idea_id].append(step)

            # Get semester for each idea based on first step creation date
            idea_semesters = {}
            semester_date_ranges = {
                Semester.FALL_2023: (datetime(2023, 9, 1), datetime(2023, 12, 31)),
                Semester.SPRING_2024: (datetime(2024, 1, 1), datetime(2024, 5, 31)),
                Semester.FALL_2024: (datetime(2024, 9, 1), datetime(2024, 12, 31)),
                Semester.SPRING_2025: (datetime(2025, 1, 1), datetime(2025, 5, 31)),
            }

            for idea_id, steps in steps_by_idea.items():
                # Get creation dates using utility function
                creation_dates = extract_timestamps_from_steps(steps)
                if not creation_dates:
                    continue

                # Use earliest creation date to determine semester
                earliest_date = min(creation_dates)

                for semester, (start_date, end_date) in semester_date_ranges.items():
                    if start_date <= earliest_date <= end_date:
                        idea_semesters[idea_id] = semester
                        break

            # Calculate time metrics for each idea
            idea_time_metrics = {}

            for idea_id, steps in steps_by_idea.items():
                if idea_id not in idea_semesters:
                    continue

                semester = idea_semesters[idea_id]
                tool_version = Semester.get_tool_version(semester)

                # Using utility function to group steps into sessions for better time analysis
                sessions = group_steps_into_sessions(steps)

                # Skip ideas with no valid sessions
                if not sessions:
                    continue

                # Sort steps by creation date
                sorted_steps = sorted(
                    [s for s in steps if s.get_creation_date()],
                    key=lambda s: s.get_creation_date(),
                )

                # Skip ideas with too few steps
                if len(sorted_steps) < 2:
                    continue

                # Get activity patterns using utility function
                activity_metrics = get_activity_metrics_by_time(sorted_steps)

                # Calculate time metrics
                first_step = sorted_steps[0]
                last_step = sorted_steps[-1]

                # Total completion time (first to last step)
                total_time = (
                    last_step.get_creation_date() - first_step.get_creation_date()
                )
                total_hours = total_time.total_seconds() / 3600

                # Number of steps
                step_count = len(sorted_steps)

                # Using utility function for time difference calculation
                creation_dates = [step.get_creation_date() for step in sorted_steps]
                time_intervals = calculate_time_differences(
                    creation_dates, unit="hours"
                )

                avg_interval = (
                    sum(time_intervals) / len(time_intervals) if time_intervals else 0
                )

                # Store metrics
                idea_time_metrics[idea_id] = {
                    "semester": semester.value,
                    "tool_version": tool_version.value,
                    "step_count": step_count,
                    "total_hours": total_hours,
                    "avg_hours_between_steps": avg_interval,
                    "hours_per_step": total_hours / step_count if step_count > 0 else 0,
                    "session_count": len(sessions),
                    "activity_patterns": activity_metrics,
                }

            # Aggregate metrics by tool version
            version_metrics = {
                "none": {
                    "idea_count": 0,
                    "avg_steps_per_idea": 0,
                    "avg_total_hours": 0,
                    "avg_hours_between_steps": 0,
                    "avg_hours_per_step": 0,
                    "ideas": [],
                },
                "v1": {
                    "idea_count": 0,
                    "avg_steps_per_idea": 0,
                    "avg_total_hours": 0,
                    "avg_hours_between_steps": 0,
                    "avg_hours_per_step": 0,
                    "ideas": [],
                },
                "v2": {
                    "idea_count": 0,
                    "avg_steps_per_idea": 0,
                    "avg_total_hours": 0,
                    "avg_hours_between_steps": 0,
                    "avg_hours_per_step": 0,
                    "ideas": [],
                },
            }

            for idea_id, metrics in idea_time_metrics.items():
                version = metrics["tool_version"]

                version_metrics[version]["idea_count"] += 1
                version_metrics[version]["avg_steps_per_idea"] += metrics["step_count"]
                version_metrics[version]["avg_total_hours"] += metrics["total_hours"]
                version_metrics[version]["avg_hours_between_steps"] += metrics[
                    "avg_hours_between_steps"
                ]
                version_metrics[version]["avg_hours_per_step"] += metrics[
                    "hours_per_step"
                ]
                version_metrics[version]["ideas"].append(idea_id)

            # Calculate averages
            for version, metrics in version_metrics.items():
                idea_count = metrics["idea_count"]
                if idea_count > 0:
                    metrics["avg_steps_per_idea"] /= idea_count
                    metrics["avg_total_hours"] /= idea_count
                    metrics["avg_hours_between_steps"] /= idea_count
                    metrics["avg_hours_per_step"] /= idea_count

            # Calculate time savings
            time_savings = {}

            # V1 vs. No tool
            if (
                version_metrics["none"]["idea_count"] > 0
                and version_metrics["v1"]["idea_count"] > 0
            ):
                time_savings["v1_vs_none"] = {
                    "hours_per_step_diff": version_metrics["v1"]["avg_hours_per_step"]
                    - version_metrics["none"]["avg_hours_per_step"],
                    "hours_per_step_pct": (
                        (
                            (
                                version_metrics["v1"]["avg_hours_per_step"]
                                - version_metrics["none"]["avg_hours_per_step"]
                            )
                            / version_metrics["none"]["avg_hours_per_step"]
                            * 100
                        )
                        if version_metrics["none"]["avg_hours_per_step"] > 0
                        else 0
                    ),
                    "total_hours_diff": version_metrics["v1"]["avg_total_hours"]
                    - version_metrics["none"]["avg_total_hours"],
                    "total_hours_pct": (
                        (
                            (
                                version_metrics["v1"]["avg_total_hours"]
                                - version_metrics["none"]["avg_total_hours"]
                            )
                            / version_metrics["none"]["avg_total_hours"]
                            * 100
                        )
                        if version_metrics["none"]["avg_total_hours"] > 0
                        else 0
                    ),
                }

            # V2 vs. V1
            if (
                version_metrics["v1"]["idea_count"] > 0
                and version_metrics["v2"]["idea_count"] > 0
            ):
                time_savings["v2_vs_v1"] = {
                    "hours_per_step_diff": version_metrics["v2"]["avg_hours_per_step"]
                    - version_metrics["v1"]["avg_hours_per_step"],
                    "hours_per_step_pct": (
                        (
                            (
                                version_metrics["v2"]["avg_hours_per_step"]
                                - version_metrics["v1"]["avg_hours_per_step"]
                            )
                            / version_metrics["v1"]["avg_hours_per_step"]
                            * 100
                        )
                        if version_metrics["v1"]["avg_hours_per_step"] > 0
                        else 0
                    ),
                    "total_hours_diff": version_metrics["v2"]["avg_total_hours"]
                    - version_metrics["v1"]["avg_total_hours"],
                    "total_hours_pct": (
                        (
                            (
                                version_metrics["v2"]["avg_total_hours"]
                                - version_metrics["v1"]["avg_total_hours"]
                            )
                            / version_metrics["v1"]["avg_total_hours"]
                            * 100
                        )
                        if version_metrics["v1"]["avg_total_hours"] > 0
                        else 0
                    ),
                }

            # V2 vs. No tool
            if (
                version_metrics["none"]["idea_count"] > 0
                and version_metrics["v2"]["idea_count"] > 0
            ):
                time_savings["v2_vs_none"] = {
                    "hours_per_step_diff": version_metrics["v2"]["avg_hours_per_step"]
                    - version_metrics["none"]["avg_hours_per_step"],
                    "hours_per_step_pct": (
                        (
                            (
                                version_metrics["v2"]["avg_hours_per_step"]
                                - version_metrics["none"]["avg_hours_per_step"]
                            )
                            / version_metrics["none"]["avg_hours_per_step"]
                            * 100
                        )
                        if version_metrics["none"]["avg_hours_per_step"] > 0
                        else 0
                    ),
                    "total_hours_diff": version_metrics["v2"]["avg_total_hours"]
                    - version_metrics["none"]["avg_total_hours"],
                    "total_hours_pct": (
                        (
                            (
                                version_metrics["v2"]["avg_total_hours"]
                                - version_metrics["none"]["avg_total_hours"]
                            )
                            / version_metrics["none"]["avg_total_hours"]
                            * 100
                        )
                        if version_metrics["none"]["avg_total_hours"] > 0
                        else 0
                    ),
                }

            # Analyze step completion consistency using utility functions
            step_consistency = {}

            for version in ["none", "v1", "v2"]:
                if not version_metrics[version]["ideas"]:
                    continue

                # Get step completion patterns
                completion_patterns = []

                for idea_id in version_metrics[version]["ideas"]:
                    if idea_id not in steps_by_idea:
                        continue

                    idea_steps = steps_by_idea[idea_id]
                    completed_steps = set()

                    for step in idea_steps:
                        if step.step:
                            completed_steps.add(step.step)

                    # Add pattern to list
                    completion_patterns.append(completed_steps)

                # Calculate consistency metrics
                if completion_patterns:
                    # Average number of completed steps
                    avg_completed = sum(
                        len(pattern) for pattern in completion_patterns
                    ) / len(completion_patterns)

                    # Step completion consistency (% of steps that appear in >50% of patterns)
                    all_steps = set()
                    for pattern in completion_patterns:
                        all_steps.update(pattern)

                    consistent_steps = set()
                    for step in all_steps:
                        count = sum(
                            1 for pattern in completion_patterns if step in pattern
                        )
                        if count / len(completion_patterns) > 0.5:
                            consistent_steps.add(step)

                    consistency_score = (
                        len(consistent_steps) / len(all_steps) if all_steps else 0
                    )

                    # Categorize progression patterns using utility function
                    progression_types = {}
                    framework_steps_list = [
                        step.value for step in DisciplinedEntrepreneurshipStep
                    ]
                    step_numbers = {
                        step.value: step.step_number
                        for step in DisciplinedEntrepreneurshipStep
                    }

                    for pattern in completion_patterns:
                        pattern_list = list(pattern)
                        if pattern_list:
                            prog_type = classify_progression_pattern(
                                pattern_list, framework_steps_list, step_numbers
                            )
                            if prog_type not in progression_types:
                                progression_types[prog_type] = 0
                            progression_types[prog_type] += 1

                    # Calculate pattern distribution
                    pattern_distribution = calculate_distribution_percentages(
                        progression_types
                    )

                    step_consistency[version] = {
                        "avg_completed_steps": avg_completed,
                        "consistency_score": consistency_score,
                        "consistent_steps": list(consistent_steps),
                        "progression_patterns": pattern_distribution,
                    }

            return {
                "idea_time_metrics": idea_time_metrics,
                "version_metrics": version_metrics,
                "time_savings": time_savings,
                "step_consistency": step_consistency,
            }

        except Exception as e:
            self._logger.error(f"Error analyzing tool impact on time allocation: {e}")
            return {"error": str(e)}

    def analyze_demographic_learning_impact(self) -> Dict[str, Any]:
        """
        Analyze how the tool impacts learning outcomes across different demographics.

        Examines whether the tool benefits certain demographic groups more than others,
        based on factors like academic background, experience level, etc.

        Returns:
            Dict[str, Any]: Demographic impact analysis
        """
        try:
            # Get tool users grouped by demographic factors

            # 1. Group by user type (undergraduate, graduate, etc.)
            users_by_type = {
                user_type.value: self._data_repo.users.get_users_by_type(user_type)
                for user_type in UserType
            }

            # 2. Group by department
            # Get all departments
            departments = set()
            for user in self._data_repo.users.get_all():
                dept = user.get_department()
                if dept:
                    departments.add(dept)

            users_by_department = {
                dept: self._data_repo.users.get_users_by_department(dept)
                for dept in departments
            }

            # 3. Group by experience level (from orbit_profile.experience)
            experience_levels = set()
            users_by_experience = {}

            for user in self._data_repo.users.get_all():
                if user.orbit_profile and user.orbit_profile.experience:
                    experience = user.orbit_profile.experience
                    experience_levels.add(experience)

                    if experience not in users_by_experience:
                        users_by_experience[experience] = []
                    users_by_experience[experience].append(user)

            # Calculate engagement metrics by demographic group
            type_metrics = self._calculate_demographic_metrics(users_by_type)
            department_metrics = self._calculate_demographic_metrics(
                users_by_department
            )
            experience_metrics = self._calculate_demographic_metrics(
                users_by_experience
            )

            # Compare learning improvements across demographics
            # For each group, compare pre-tool vs. post-tool learning outcomes

            # Get course evaluations
            evaluations = self._data_repo.courses.get_all()

            # Group by tool version
            evals_by_version = {"none": [], "v1": [], "v2": []}

            for eval in evaluations:
                version = eval.get_tool_version().value
                evals_by_version[version].append(eval)

            # Get pre-tool and post-tool learning scores
            pre_tool_score = 0
            if evals_by_version["none"]:
                pre_tool_scores = [
                    eval.get_overall_score(high_impact_only=True)
                    for eval in evals_by_version["none"]
                ]
                pre_tool_score = sum(pre_tool_scores) / len(pre_tool_scores)

            post_tool_scores = []
            for version in ["v1", "v2"]:
                post_tool_scores.extend(
                    [
                        eval.get_overall_score(high_impact_only=True)
                        for eval in evals_by_version[version]
                    ]
                )

            post_tool_score = (
                sum(post_tool_scores) / len(post_tool_scores) if post_tool_scores else 0
            )

            # Calculate improvement
            overall_improvement = post_tool_score - pre_tool_score

            # Calculate engagement improvement correlation for each demographic group
            type_improvement = self._calculate_improvement_correlation(
                type_metrics, overall_improvement
            )
            department_improvement = self._calculate_improvement_correlation(
                department_metrics, overall_improvement
            )
            experience_improvement = self._calculate_improvement_correlation(
                experience_metrics, overall_improvement
            )

            return {
                "user_type_analysis": {
                    "engagement_metrics": type_metrics,
                    "improvement_correlation": type_improvement,
                },
                "department_analysis": {
                    "engagement_metrics": department_metrics,
                    "improvement_correlation": department_improvement,
                },
                "experience_analysis": {
                    "engagement_metrics": experience_metrics,
                    "improvement_correlation": experience_improvement,
                },
                "overall_improvement": overall_improvement,
            }

        except Exception as e:
            self._logger.error(f"Error analyzing demographic learning impact: {e}")
            return {"error": str(e)}

    def _calculate_demographic_metrics(
        self, users_by_group: Dict[str, List[Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate engagement metrics for each demographic group.

        Args:
            users_by_group: Dictionary mapping group name to list of users

        Returns:
            Dict[str, Dict[str, Any]]: Metrics by demographic group
        """
        metrics = {}

        for group, users in users_by_group.items():
            if not users:
                continue

            # Calculate engagement metrics

            # 1. Engagement distribution
            engagement_counts = {"high": 0, "medium": 0, "low": 0}

            for user in users:
                level = user.get_engagement_level().value
                engagement_counts[level] += 1

            total_users = len(users)
            engagement_dist = {
                level: count / total_users if total_users > 0 else 0
                for level, count in engagement_counts.items()
            }

            # 2. Average scores
            content_scores = []
            completion_scores = []

            for user in users:
                if user.scores:
                    if user.scores.content is not None:
                        content_scores.append(user.scores.content)
                    if user.scores.completion is not None:
                        completion_scores.append(user.scores.completion)

            # Using utility function to calculate summary statistics
            content_stats = (
                calculate_summary_statistics(content_scores)
                if content_scores
                else {"mean": 0}
            )
            completion_stats = (
                calculate_summary_statistics(completion_scores)
                if completion_scores
                else {"mean": 0}
            )

            # 3. Idea and step metrics
            idea_count = 0
            step_count = 0

            for user in users:
                if user.email:
                    ideas = self._data_repo.ideas.find_by_owner(user.email)
                    idea_count += len(ideas)

                    for idea in ideas:
                        if idea.id:
                            idea_id = (
                                idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                            )
                            steps = self._data_repo.steps.find_by_idea_id(idea_id)
                            step_count += len(steps)

            avg_ideas = idea_count / total_users if total_users > 0 else 0
            avg_steps = step_count / total_users if total_users > 0 else 0

            # Store metrics
            metrics[group] = {
                "user_count": total_users,
                "engagement_distribution": engagement_dist,
                "weighted_engagement": (
                    3 * engagement_dist["high"]
                    + 2 * engagement_dist["medium"]
                    + 1 * engagement_dist["low"]
                )
                / 3,
                "content_score_stats": content_stats,
                "completion_score_stats": completion_stats,
                "avg_content_score": content_stats["mean"],
                "avg_completion_score": completion_stats["mean"],
                "avg_ideas_per_user": avg_ideas,
                "avg_steps_per_user": avg_steps,
                "total_ideas": idea_count,
                "total_steps": step_count,
            }

        return metrics

    def _calculate_improvement_correlation(
        self, demographic_metrics: Dict[str, Dict[str, Any]], overall_improvement: float
    ) -> Dict[str, Any]:
        """
        Calculate correlation between demographic engagement and improvement.

        Args:
            demographic_metrics: Metrics by demographic group
            overall_improvement: Overall learning outcome improvement

        Returns:
            Dict[str, Any]: Improvement correlation metrics
        """
        # Calculate relative improvement for each group based on engagement
        relative_improvement = {}

        for group, metrics in demographic_metrics.items():
            # Use weighted engagement as a factor
            engagement_factor = metrics["weighted_engagement"]

            # Calculate expected improvement
            expected_improvement = overall_improvement * engagement_factor

            # Calculate improvement per unit of engagement
            improvement_efficiency = (
                overall_improvement / engagement_factor if engagement_factor > 0 else 0
            )

            relative_improvement[group] = {
                "engagement_factor": engagement_factor,
                "expected_improvement": expected_improvement,
                "improvement_efficiency": improvement_efficiency,
            }

        # Identify groups with highest and lowest improvement efficiency
        sorted_groups = sorted(
            relative_improvement.items(),
            key=lambda x: (
                x[1]["improvement_efficiency"]
                if x[1]["improvement_efficiency"] != float("inf")
                else 0
            ),
            reverse=True,
        )

        highest_groups = [
            group for group, _ in sorted_groups[:3] if len(sorted_groups) >= 3
        ]
        lowest_groups = [
            group for group, _ in sorted_groups[-3:] if len(sorted_groups) >= 3
        ]

        return {
            "by_group": relative_improvement,
            "highest_efficiency_groups": highest_groups,
            "lowest_efficiency_groups": lowest_groups,
        }

    def analyze_combined_learning_metrics(self) -> Dict[str, Any]:
        """
        Comprehensive analysis combining multiple learning metrics.

        Provides a holistic view of the tool's impact on learning by
        combining multiple analytical approaches.

        Returns:
            Dict[str, Any]: Combined analysis results
        """
        try:
            # Run key analyses
            tool_version_impact = self.analyze_tool_version_impact()
            learning_outcomes = self.compare_learning_outcomes_by_cohort()
            engagement_correlation = self.correlate_tool_usage_with_course_ratings()
            time_impact = self.analyze_tool_impact_on_time_allocation()

            # Extract key metrics

            # 1. Overall learning outcome improvement
            learning_improvement = 0
            if (
                "differences" in learning_outcomes
                and "high_impact_overall" in learning_outcomes["differences"]
            ):
                learning_improvement = learning_outcomes["differences"][
                    "high_impact_overall"
                ]

            # 2. Tool version comparison
            version_comparison = {}
            if "differences" in tool_version_impact:
                if "overall" in tool_version_impact["differences"]:
                    version_comparison["overall_score_diff"] = tool_version_impact[
                        "differences"
                    ]["overall"]
                if "high_impact_overall" in tool_version_impact["differences"]:
                    version_comparison["high_impact_score_diff"] = tool_version_impact[
                        "differences"
                    ]["high_impact_overall"]

            # 3. Time savings
            time_savings = {}
            if "time_savings" in time_impact:
                if "v2_vs_none" in time_impact["time_savings"]:
                    time_savings["total_time_saved_pct"] = abs(
                        time_impact["time_savings"]["v2_vs_none"]["total_hours_pct"]
                    )
                    time_savings["per_step_time_saved_pct"] = abs(
                        time_impact["time_savings"]["v2_vs_none"]["hours_per_step_pct"]
                    )

            # 4. Engagement correlation
            engagement_score = 0
            if "correlation" in engagement_correlation:
                if (
                    "overall_rating_weighted_engagement"
                    in engagement_correlation["correlation"]
                ):
                    engagement_score = engagement_correlation["correlation"][
                        "overall_rating_weighted_engagement"
                    ]

            # Calculate composite effectiveness score using a weighted average
            # Combine multiple metrics into a single score
            metrics_to_combine = [
                learning_improvement,  # Weight: 0.4
                engagement_score,  # Weight: 0.3
                time_savings.get("total_time_saved_pct", 0)
                / 100,  # Weight: 0.3 (divide by 100 to convert percentage to fraction)
            ]

            composite_score = (
                0.4 * metrics_to_combine[0]
                + 0.3 * metrics_to_combine[1]
                + 0.3 * metrics_to_combine[2]
            )

            # Interpret effectiveness score
            effectiveness_interpretation = "Neutral"

            if composite_score > 0.5:
                effectiveness_interpretation = "Highly Effective"
            elif composite_score > 0.2:
                effectiveness_interpretation = "Moderately Effective"
            elif composite_score > 0:
                effectiveness_interpretation = "Slightly Effective"
            elif composite_score > -0.2:
                effectiveness_interpretation = "Slightly Ineffective"
            else:
                effectiveness_interpretation = "Ineffective"

            # Generate recommendations based on analysis
            recommendations = self._generate_recommendations(
                learning_improvement,
                engagement_score,
                time_savings.get("total_time_saved_pct", 0),
                version_comparison,
            )

            return {
                "composite_effectiveness_score": composite_score,
                "effectiveness_interpretation": effectiveness_interpretation,
                "key_metrics": {
                    "learning_improvement": learning_improvement,
                    "engagement_correlation": engagement_score,
                    "time_savings_pct": time_savings.get("total_time_saved_pct", 0),
                    "version_improvement": version_comparison,
                },
                "recommendations": recommendations,
            }

        except Exception as e:
            self._logger.error(f"Error analyzing combined learning metrics: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        learning_improvement: float,
        engagement_correlation: float,
        time_savings_pct: float,
        version_comparison: Dict[str, float],
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on analysis results.

        Args:
            learning_improvement: Learning outcome improvement
            engagement_correlation: Engagement-learning correlation
            time_savings_pct: Time savings percentage
            version_comparison: Tool version comparison metrics

        Returns:
            List[Dict[str, str]]: List of recommendation objects
        """
        recommendations = []

        # Learning improvement recommendations
        if learning_improvement <= 0:
            recommendations.append(
                {
                    "category": "Learning Outcomes",
                    "finding": "The tool has not demonstrated measurable improvement in learning outcomes.",
                    "recommendation": "Review the tool's alignment with course learning objectives. Consider redesigning the tool to better support specific educational goals.",
                }
            )
        elif learning_improvement < 0.2:
            recommendations.append(
                {
                    "category": "Learning Outcomes",
                    "finding": "The tool shows modest improvements in learning outcomes.",
                    "recommendation": "Identify the most effective components and strengthen their implementation. Consider surveying students to understand which features they find most educationally valuable.",
                }
            )
        else:
            recommendations.append(
                {
                    "category": "Learning Outcomes",
                    "finding": "The tool demonstrates significant positive impact on learning outcomes.",
                    "recommendation": "Document successful aspects for broader implementation. Consider expanding to other courses or frameworks.",
                }
            )

        # Engagement correlation recommendations
        if engagement_correlation < 0.3:
            recommendations.append(
                {
                    "category": "Engagement",
                    "finding": "There is a weak correlation between tool engagement and learning outcomes.",
                    "recommendation": "Redesign engagement metrics to better align with learning objectives. Focus on quality of engagement rather than quantity.",
                }
            )
        else:
            recommendations.append(
                {
                    "category": "Engagement",
                    "finding": "Strong correlation between tool engagement and learning outcomes.",
                    "recommendation": "Implement features that encourage the specific types of engagement that correlate with better outcomes. Consider gamification elements that reward productive engagement patterns.",
                }
            )

        # Time savings recommendations
        if time_savings_pct < 10:
            recommendations.append(
                {
                    "category": "Time Efficiency",
                    "finding": "The tool provides minimal time savings in framework completion.",
                    "recommendation": "Identify bottlenecks in the user workflow and optimize the most time-consuming steps. Consider adding templates or AI-generated starter content for complex steps.",
                }
            )
        else:
            recommendations.append(
                {
                    "category": "Time Efficiency",
                    "finding": f"The tool provides significant time savings ({time_savings_pct:.1f}%) in framework completion.",
                    "recommendation": "Verify that time savings translate to more customer interaction or deeper learning. Add features to explicitly encourage redirection of saved time to higher-value activities.",
                }
            )

        # Version improvement recommendations
        if (
            "overall_score_diff" in version_comparison
            and version_comparison["overall_score_diff"] <= 0
        ):
            recommendations.append(
                {
                    "category": "Tool Iteration",
                    "finding": "Newer versions of the tool have not shown improved effectiveness.",
                    "recommendation": "Conduct user testing to understand why new features aren't translating to better outcomes. Consider reverting some changes or taking a different approach to feature development.",
                }
            )
        else:
            recommendations.append(
                {
                    "category": "Tool Iteration",
                    "finding": "Newer versions show improvements over earlier versions.",
                    "recommendation": "Continue the current development trajectory with an emphasis on features that showed the most positive impact between versions.",
                }
            )

        # Add general recommendation
        recommendations.append(
            {
                "category": "Long-term Development",
                "finding": "Current analysis is based on limited time-series data across cohorts.",
                "recommendation": "Establish a continuous assessment framework to monitor tool impact over time. Implement A/B testing for new features to measure specific impacts.",
            }
        )

        return recommendations
