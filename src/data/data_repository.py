"""
Main data repository for the data analysis system.

This module provides the DataRepository class that serves as the primary entry point
for accessing all entity-specific repositories and managing relationships between them.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Future implementation for settings
from config.settings import Settings
from src.data.repositories.user_repository import UserRepository
from src.data.repositories.idea_repository import IdeaRepository
from src.data.repositories.step_repository import StepRepository
from src.data.repositories.team_repository import TeamRepository
from src.data.repositories.course_repository import CourseRepository
from src.data.repositories.user_repository import InMemoryDatabase
from src.data.models.enums import DisciplinedEntrepreneurshipStep, IdeaCategory


class DataRepository:
    """
    Main data repository for coordinating access to all entity-specific repositories.

    This class serves as a facade for accessing all repository classes and
    provides methods for cross-entity operations that involve multiple repositories.
    """

    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize the data repository.

        Args:
            config: Optional settings configuration
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = config or Settings()
        self._db = InMemoryDatabase()  # Shared in-memory database for all repositories

        # Initialize entity-specific repositories
        self._user_repo = UserRepository(db=self._db, config=self._config)
        self._idea_repo = IdeaRepository(db=self._db, config=self._config)
        self._step_repo = StepRepository(db=self._db, config=self._config)
        self._team_repo = TeamRepository(db=self._db, config=self._config)
        self._course_repo = CourseRepository(db=self._db, config=self._config)

        # Track whether data has been loaded
        self._data_loaded = False

    def connect(self) -> None:
        """
        Connect to all data sources.
        """
        if self._data_loaded:
            return

        try:
            # Connect to each repository
            self._user_repo.connect()
            self._idea_repo.connect()
            self._step_repo.connect()
            self._team_repo.connect()
            self._course_repo.connect()

            self._data_loaded = True
            self._logger.info("Successfully connected to all data sources")
        except Exception as e:
            self._logger.error(f"Error connecting to data sources: {e}")
            raise

    @property
    def users(self) -> UserRepository:
        """Get the user repository."""
        self._ensure_connected()
        return self._user_repo

    @property
    def ideas(self) -> IdeaRepository:
        """Get the idea repository."""
        self._ensure_connected()
        return self._idea_repo

    @property
    def steps(self) -> StepRepository:
        """Get the step repository."""
        self._ensure_connected()
        return self._step_repo

    @property
    def teams(self) -> TeamRepository:
        """Get the team repository."""
        self._ensure_connected()
        return self._team_repo

    @property
    def courses(self) -> CourseRepository:
        """Get the course repository."""
        self._ensure_connected()
        return self._course_repo

    def _ensure_connected(self) -> None:
        """
        Ensure connection to data sources.

        Raises:
            RuntimeError: If not connected
        """
        if not self._data_loaded:
            self.connect()

    def load_data_from_directory(self, directory_path: str) -> Dict[str, int]:
        """
        Load data from all JSON files in a directory.

        Args:
            directory_path: Path to directory containing JSON data files

        Returns:
            Dict[str, int]: Summary of loaded documents
        """
        results = {}
        path = Path(directory_path)

        # Define file mappings for each repository
        file_mappings = {
            "users.json": (self._user_repo, "users"),
            "ideas.json": (self._idea_repo, "ideas"),
            "steps.json": (self._step_repo, "steps"),
            "de_team_user.json": (self._team_repo, "teams"),
            "course_evaluations.json": (self._course_repo, "course_evaluations"),
        }

        # Load each file if it exists
        for filename, (repo, collection_name) in file_mappings.items():
            file_path = path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)

                    if isinstance(data, list):
                        self._db.insert_many(collection_name, data)
                        count = len(data)
                    else:
                        self._db.insert_one(collection_name, data)
                        count = 1

                    results[filename] = count
                    self._logger.info(f"Loaded {count} documents from {file_path}")
                except Exception as e:
                    self._logger.error(f"Error loading data from {file_path}: {e}")
                    results[filename] = 0

        # Load categorized ideas if available
        categorized_ideas_path = path / "categorized_ideas_latest.json"
        if categorized_ideas_path.exists():
            try:
                self._idea_repo._load_categorized_ideas()
                results["categorized_ideas_latest.json"] = "loaded"
            except Exception as e:
                self._logger.error(f"Error loading categorized ideas: {e}")

        # Mark as loaded
        self._data_loaded = True

        return results

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded data.

        Returns:
            Dict[str, Any]: Data summary statistics
        """
        self._ensure_connected()

        try:

            # user_count = len(self._user_repo.get_all())
            # by_type = self._user_repo.get_user_type_distribution()
            # by_engagement = self._user_repo.get_engagement_distribution()
            # idea_count = len(self._idea_repo.get_all())
            # by_category = self._idea_repo.get_idea_count_by_category()
            # avg_steps_per_idea = self._idea_repo.get_average_steps_per_idea()
            # step_count = len(self._step_repo.get_all())
            # by_framework = self._step_repo.get_step_count_by_framework()
            # team_count = len(self._team_repo.get_all())
            # avg_team_size = self._team_repo.get_avg_team_size()
            # team_size_distribution = self._team_repo.get_team_sizes()
            # course_count = len(self._course_repo.get_all())
            # semesters = self._course_repo.get_semester_order()

            return {
                "users": {
                    "count": len(self._user_repo.get_all()),
                    "by_type": self._user_repo.get_user_type_distribution(),
                    "by_engagement": self._user_repo.get_engagement_distribution(),
                },
                "ideas": {
                    "count": len(self._idea_repo.get_all()),
                    "by_category": self._idea_repo.get_idea_count_by_category(),
                    "avg_steps_per_idea": self._idea_repo.get_average_steps_per_idea(),
                },
                "steps": {
                    "count": len(self._step_repo.get_all()),
                    "by_framework": self._step_repo.get_step_count_by_framework(),
                },
                "teams": {
                    "count": len(self._team_repo.get_all()),
                    "avg_team_size": self._team_repo.get_avg_team_size(),
                    "team_size_distribution": self._team_repo.get_team_sizes(),
                },
                "course_evaluations": {
                    "count": len(self._course_repo.get_all()),
                    "semesters": self._course_repo.get_semester_order(),
                },
            }

        except Exception as e:
            self._logger.error(f"Error creating data summary: {e}")
            return {"error": str(e)}

    def get_user_ideas_steps(self, email: str) -> Dict[str, Any]:
        """
        Get all ideas and their associated steps for a specific user.

        Args:
            email: User email address

        Returns:
            Dict[str, Any]: User's ideas and steps
        """
        self._ensure_connected()

        try:
            # Get user
            user = self._user_repo.find_by_email(email)
            if not user:
                return {"error": f"User not found: {email}"}

            # Get user's ideas
            ideas = self._idea_repo.find_by_owner(email)

            # Get steps for each idea
            result = {
                "user": {
                    "email": user.email,
                    "name": user.name,
                    "engagement_level": user.get_engagement_level().value,
                    "user_type": user.get_user_type().value,
                },
                "ideas": [],
            }

            for idea in ideas:
                idea_data = {
                    "id": idea.id.oid if idea.id else None,
                    "title": idea.title,
                    "description": idea.description,
                    "created": (
                        idea.get_creation_date().isoformat()
                        if idea.get_creation_date()
                        else None
                    ),
                    "ranking": idea.ranking,
                    "category": idea.get_idea_category().value,
                    "steps": [],
                }

                # Get steps for this idea
                if idea.id:
                    steps = self._step_repo.find_by_idea_id(
                        idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    )

                    for step in steps:
                        step_data = {
                            "id": step.id.oid if step.id else None,
                            "framework": step.framework,
                            "step": step.step,
                            "created_at": (
                                step.get_creation_date().isoformat()
                                if step.get_creation_date()
                                else None
                            ),
                            "version": step.get_version(),
                            "has_user_input": step.has_user_input(),
                        }
                        idea_data["steps"].append(step_data)

                result["ideas"].append(idea_data)

            return result
        except Exception as e:
            self._logger.error(f"Error getting user ideas and steps: {e}")
            return {"error": str(e)}

    def get_team_engagement(self, team_id: int) -> Dict[str, Any]:
        """
        Get engagement metrics for a specific team.

        Args:
            team_id: Team ID

        Returns:
            Dict[str, Any]: Team engagement metrics
        """
        self._ensure_connected()

        try:
            # Get team
            team = self._team_repo.find_by_team_id(team_id)
            if not team:
                return {"error": f"Team not found: {team_id}"}

            # Get members
            member_emails = team.get_member_emails()
            members = []

            for email in member_emails:
                user = self._user_repo.find_by_email(email)
                if not user:
                    continue

                # Get ideas and steps for this user
                ideas = self._idea_repo.find_by_owner(email)
                idea_count = len(ideas)

                total_steps = 0
                framework_steps = {
                    "Disciplined Entrepreneurship": 0,
                    "Startup Tactics": 0,
                    "My Journey": 0,
                    "Product Management": 0,
                }

                for idea in ideas:
                    if idea.id:
                        steps = self._step_repo.find_by_idea_id(
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        total_steps += len(steps)

                        # Count steps by framework
                        for step in steps:
                            if step.framework in framework_steps:
                                framework_steps[step.framework] += 1

                member_data = {
                    "email": user.email,
                    "name": user.name,
                    "engagement_level": user.get_engagement_level().value,
                    "idea_count": idea_count,
                    "step_count": total_steps,
                    "framework_steps": framework_steps,
                }
                members.append(member_data)

            # Calculate team averages
            avg_ideas = (
                sum(m["idea_count"] for m in members) / len(members) if members else 0
            )
            avg_steps = (
                sum(m["step_count"] for m in members) / len(members) if members else 0
            )

            # Count unique ideas (some may be shared)
            all_ideas = set()
            for email in member_emails:
                ideas = self._idea_repo.find_by_owner(email)
                for idea in ideas:
                    if idea.id:
                        all_ideas.add(
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )

            return {
                "team": {
                    "id": team.team_id,
                    "name": team.team_name,
                    "member_count": team.get_member_count(),
                },
                "engagement": {
                    "unique_ideas": len(all_ideas),
                    "avg_ideas_per_member": avg_ideas,
                    "avg_steps_per_member": avg_steps,
                    "high_engagement_members": sum(
                        1 for m in members if m["engagement_level"] == "high"
                    ),
                    "medium_engagement_members": sum(
                        1 for m in members if m["engagement_level"] == "medium"
                    ),
                    "low_engagement_members": sum(
                        1 for m in members if m["engagement_level"] == "low"
                    ),
                },
                "members": members,
            }
        except Exception as e:
            self._logger.error(f"Error getting team engagement: {e}")
            return {"error": str(e)}

    def get_semester_comparison(self, semester1: str, semester2: str) -> Dict[str, Any]:
        """
        Compare data between two semesters.

        Args:
            semester1: First semester name
            semester2: Second semester name

        Returns:
            Dict[str, Any]: Comparison results
        """
        self._ensure_connected()

        try:
            # Get course evaluations
            eval_comparison = self._course_repo.get_evaluation_comparison(
                semester1, semester2
            )

            # TODO: Compare user engagement, idea creation, step completion, etc.
            # This would involve filtering users by creation date or registration semester

            return {
                "course_evaluations": eval_comparison,
                # Additional comparison metrics would go here
            }
        except Exception as e:
            self._logger.error(f"Error comparing semesters: {e}")
            return {"error": str(e)}

    def get_course_student_engagement(self, course_id: str) -> Dict[str, Any]:
        """
        Get engagement metrics for students in a specific course.

        Args:
            course_id: Course ID (e.g., "15.390")

        Returns:
            Dict[str, Any]: Course student engagement metrics
        """
        self._ensure_connected()

        try:
            # Get course students
            students = self._user_repo.find_by_course(course_id)

            # Get teams for these students
            student_emails = [s.email for s in students if s.email]
            teams_by_student = self._team_repo.match_users_to_teams(student_emails)

            # Collect engagement metrics
            student_metrics = []
            for student in students:
                if not student.email:
                    continue

                # Get ideas
                ideas = self._idea_repo.find_by_owner(student.email)

                # Get steps
                total_steps = 0
                for idea in ideas:
                    if idea.id:
                        steps = self._step_repo.find_by_idea_id(
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        total_steps += len(steps)

                # Get team info
                teams = teams_by_student.get(student.email, [])

                student_metrics.append(
                    {
                        "email": student.email,
                        "name": student.name,
                        "engagement_level": student.get_engagement_level().value,
                        "user_type": student.get_user_type().value,
                        "idea_count": len(ideas),
                        "step_count": total_steps,
                        "on_team": len(teams) > 0,
                        "team_count": len(teams),
                        "team_names": [t.team_name for t in teams if t.team_name],
                    }
                )

            # Calculate overall metrics
            engagement_levels = {
                "high": sum(
                    1 for s in student_metrics if s["engagement_level"] == "high"
                ),
                "medium": sum(
                    1 for s in student_metrics if s["engagement_level"] == "medium"
                ),
                "low": sum(
                    1 for s in student_metrics if s["engagement_level"] == "low"
                ),
            }

            idea_percentiles = self._calculate_percentiles(
                [s["idea_count"] for s in student_metrics]
            )
            step_percentiles = self._calculate_percentiles(
                [s["step_count"] for s in student_metrics]
            )

            return {
                "course_id": course_id,
                "student_count": len(students),
                "engagement_levels": engagement_levels,
                "idea_metrics": {
                    "total": sum(s["idea_count"] for s in student_metrics),
                    "avg_per_student": (
                        sum(s["idea_count"] for s in student_metrics) / len(students)
                        if students
                        else 0
                    ),
                    "percentiles": idea_percentiles,
                },
                "step_metrics": {
                    "total": sum(s["step_count"] for s in student_metrics),
                    "avg_per_student": (
                        sum(s["step_count"] for s in student_metrics) / len(students)
                        if students
                        else 0
                    ),
                    "percentiles": step_percentiles,
                },
                "team_metrics": {
                    "students_on_teams": sum(
                        1 for s in student_metrics if s["on_team"]
                    ),
                    "students_not_on_teams": sum(
                        1 for s in student_metrics if not s["on_team"]
                    ),
                    "team_engagement_correlation": self._calculate_team_engagement_correlation(
                        student_metrics
                    ),
                },
                "students": student_metrics,
            }
        except Exception as e:
            self._logger.error(f"Error getting course student engagement: {e}")
            return {"error": str(e)}

    def _calculate_percentiles(self, values: List[int]) -> Dict[str, float]:
        """
        Calculate percentiles for a list of values.

        Args:
            values: List of values

        Returns:
            Dict[str, float]: Percentile values
        """
        if not values:
            return {"p25": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}

        values = sorted(values)
        n = len(values)

        return {
            "p25": values[int(n * 0.25)] if n > 0 else 0,
            "p50": values[int(n * 0.5)] if n > 0 else 0,
            "p75": values[int(n * 0.75)] if n > 0 else 0,
            "p90": values[int(n * 0.9)] if n > 0 else 0,
            "max": values[-1] if n > 0 else 0,
        }

    def _calculate_team_engagement_correlation(
        self, student_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate correlation between team membership and engagement metrics.

        Args:
            student_metrics: List of student metric dictionaries

        Returns:
            Dict[str, Any]: Correlation metrics
        """
        team_students = [s for s in student_metrics if s["on_team"]]
        non_team_students = [s for s in student_metrics if not s["on_team"]]

        if not team_students or not non_team_students:
            return {"no_comparison_available": True}

        # Calculate averages
        team_avg_ideas = sum(s["idea_count"] for s in team_students) / len(
            team_students
        )
        non_team_avg_ideas = sum(s["idea_count"] for s in non_team_students) / len(
            non_team_students
        )

        team_avg_steps = sum(s["step_count"] for s in team_students) / len(
            team_students
        )
        non_team_avg_steps = sum(s["step_count"] for s in non_team_students) / len(
            non_team_students
        )

        # Count high engagement students
        team_high_engagement = sum(
            1 for s in team_students if s["engagement_level"] == "high"
        )
        non_team_high_engagement = sum(
            1 for s in non_team_students if s["engagement_level"] == "high"
        )

        team_high_engagement_pct = (
            team_high_engagement / len(team_students) if team_students else 0
        )
        non_team_high_engagement_pct = (
            non_team_high_engagement / len(non_team_students)
            if non_team_students
            else 0
        )

        return {
            "team_students": {
                "count": len(team_students),
                "avg_ideas": team_avg_ideas,
                "avg_steps": team_avg_steps,
                "high_engagement_pct": team_high_engagement_pct,
            },
            "non_team_students": {
                "count": len(non_team_students),
                "avg_ideas": non_team_avg_ideas,
                "avg_steps": non_team_avg_steps,
                "high_engagement_pct": non_team_high_engagement_pct,
            },
            "comparison": {
                "idea_diff": team_avg_ideas - non_team_avg_ideas,
                "step_diff": team_avg_steps - non_team_avg_steps,
                "high_engagement_diff": team_high_engagement_pct
                - non_team_high_engagement_pct,
            },
        }

    def get_step_progression_analysis(self) -> Dict[str, Any]:
        """
        Analyze the progression through framework steps.

        Returns:
            Dict[str, Any]: Step progression analysis
        """
        self._ensure_connected()

        try:
            # Get dropout analysis
            dropout_analysis = self._step_repo.get_dropout_analysis()

            # Get step completion rates
            completion_rates = self._idea_repo.get_step_completion_rates(
                framework="Disciplined Entrepreneurship"
            )

            # Sort steps by number
            sorted_steps = sorted(
                completion_rates.items(),
                key=lambda x: DisciplinedEntrepreneurshipStep.get_step_number(x[0]),
            )

            # Calculate common step sequences
            popular_sequences = self._step_repo.get_popular_step_patterns(
                max_patterns=5
            )

            return {
                "completion_rates": dict(sorted_steps),
                "dropout_analysis": dropout_analysis,
                "popular_sequences": popular_sequences,
                "time_between_steps": self._step_repo.get_time_between_steps_distribution(),
                "session_durations": self._step_repo.get_session_duration_distribution(),
            }
        except Exception as e:
            self._logger.error(f"Error getting step progression analysis: {e}")
            return {"error": str(e)}

    def get_idea_category_analysis(self) -> Dict[str, Any]:
        """
        Analyze ideas by category.

        Returns:
            Dict[str, Any]: Idea category analysis
        """
        self._ensure_connected()

        try:
            # Get category distribution
            category_counts = self._idea_repo.get_idea_count_by_category()

            # Get step completion by category
            category_step_completion = (
                self._idea_repo.get_category_step_completion_correlation()
            )

            # Get average steps per idea by category
            ideas_by_category = {}
            avg_steps_by_category = {}

            for category in IdeaCategory:
                cat_ideas = self._idea_repo.find_by_category(category)
                ideas_by_category[category.value] = cat_ideas

                if cat_ideas:
                    total_steps = 0
                    for idea in cat_ideas:
                        total_steps += (
                            idea.get_de_steps_count() + idea.get_st_steps_count()
                        )
                    avg_steps_by_category[category.value] = total_steps / len(cat_ideas)
                else:
                    avg_steps_by_category[category.value] = 0

            return {
                "category_counts": category_counts,
                "avg_steps_by_category": avg_steps_by_category,
                "category_step_completion": category_step_completion,
            }
        except Exception as e:
            self._logger.error(f"Error getting idea category analysis: {e}")
            return {"error": str(e)}
