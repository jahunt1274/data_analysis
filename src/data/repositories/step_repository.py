"""
Step repository for the data analysis system.

This module provides data access and query methods for Step entities,
including framework-specific step analysis and progression tracking.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from src.data.models.step_model import Step
from src.data.models.enums import (
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
    StartupTacticsStep,
)
from src.data.repositories.base_repository import BaseRepository
from src.data.db import InMemoryDatabase


class StepRepository(BaseRepository[Step]):
    """
    Repository for Step data access and analysis.

    This repository provides methods for querying and analyzing step data,
    including completion rates, progression analysis, and engagement metrics.
    """

    def __init__(self, db=None, config=None):
        """
        Initialize the step repository.

        Args:
            db: Optional database connection
            config: Optional configuration object
        """
        super().__init__("steps", Step)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._config = config
        # In-memory indexes for faster lookups
        self._owner_index = {}  # Maps user email to list of steps
        self._idea_index = {}  # Maps idea_id to list of steps
        self._framework_index = {}  # Maps framework to list of steps
        self._session_index = {}  # Maps session_id to list of steps

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
            if self._config and hasattr(self._config, "STEP_DATA_PATH"):
                file_path = self._config.STEP_DATA_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/steps.json"

            # Create in-memory DB-like structure if needed
            if not isinstance(self._db, InMemoryDatabase):
                self._db = InMemoryDatabase()

            self._load_data_from_json(file_path)

            # Build indexes for faster lookups
            self._build_indexes()
        except Exception as e:
            self._logger.error(f"Error connecting to step data: {e}")
            raise

    def _load_data_from_json(self, file_path: str) -> None:
        """
        Load step data from a JSON file.

        Args:
            file_path: Path to the JSON file containing step data
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
                f"Loaded {self._db.count(self._collection_name, {})} steps from {file_path}"
            )
        except Exception as e:
            self._logger.error(f"Error loading step data from {file_path}: {e}")
            raise

    def _build_indexes(self) -> None:
        """
        Build in-memory indexes for faster lookups.
        """
        try:
            steps = self.get_all()

            # Build owner index
            for step in steps:
                if step.owner:
                    if step.owner not in self._owner_index:
                        self._owner_index[step.owner] = []
                    self._owner_index[step.owner].append(step)

            # Build idea_id index
            for step in steps:
                if step.idea_id:
                    idea_id = (
                        step.idea_id.oid
                        if hasattr(step.idea_id, "oid")
                        else str(step.idea_id)
                    )
                    if idea_id not in self._idea_index:
                        self._idea_index[idea_id] = []
                    self._idea_index[idea_id].append(step)

            # Build framework index
            for step in steps:
                if step.framework:
                    if step.framework not in self._framework_index:
                        self._framework_index[step.framework] = []
                    self._framework_index[step.framework].append(step)

            # Build session index
            for step in steps:
                if step.session_id:
                    session_id = (
                        step.session_id.oid
                        if hasattr(step.session_id, "oid")
                        else str(step.session_id)
                    )
                    if session_id not in self._session_index:
                        self._session_index[session_id] = []
                    self._session_index[session_id].append(step)
        except Exception as e:
            self._logger.error(f"Error building indexes: {e}")

    def find_by_owner(self, email: str) -> List[Step]:
        """
        Find all steps created by a specific user.

        Args:
            email: User's email address

        Returns:
            List[Step]: List of steps by the user
        """
        # Check in-memory index first
        if email in self._owner_index:
            return self._owner_index[email]

        # Fall back to DB query
        return self.find_many({"owner": email})

    def find_by_idea_id(self, idea_id: str) -> List[Step]:
        """
        Find all steps associated with a specific idea.

        Args:
            idea_id: Idea ID

        Returns:
            List[Step]: List of steps for the idea
        """
        # Check in-memory index first
        if idea_id in self._idea_index:
            return self._idea_index[idea_id]

        # Fall back to DB query
        return self.find_many({"idea_id._id": idea_id})

    def find_by_framework(self, framework: Union[str, FrameworkType]) -> List[Step]:
        """
        Find all steps in a specific framework.

        Args:
            framework: Framework name or enum

        Returns:
            List[Step]: List of steps in the framework
        """
        # Convert to string framework name if an enum is provided
        if isinstance(framework, FrameworkType):
            framework_name = framework.value
        else:
            framework_name = framework

        # Check in-memory index first
        if framework_name in self._framework_index:
            return self._framework_index[framework_name]

        # Fall back to DB query
        return self.find_many({"framework": framework_name})

    def find_by_session_id(self, session_id: str) -> List[Step]:
        """
        Find all steps created in a specific session.

        Args:
            session_id: Session ID

        Returns:
            List[Step]: List of steps in the session
        """
        # Check in-memory index first
        if session_id in self._session_index:
            return self._session_index[session_id]

        # Fall back to DB query
        return self.find_many({"session_id._id": session_id})

    def find_by_step_type(
        self,
        step_type: Union[str, DisciplinedEntrepreneurshipStep, StartupTacticsStep],
        framework: Optional[Union[str, FrameworkType]] = None,
    ) -> List[Step]:
        """
        Find all steps of a specific type.

        Args:
            step_type: Step type name or enum
            framework: Optional framework filter

        Returns:
            List[Step]: List of steps of the specified type
        """
        # Convert to string step name if an enum is provided
        if isinstance(step_type, (DisciplinedEntrepreneurshipStep, StartupTacticsStep)):
            step_name = step_type.value
        else:
            step_name = step_type

        # Apply framework filter if provided
        framework_name = None
        if framework:
            if isinstance(framework, FrameworkType):
                framework_name = framework.value
            else:
                framework_name = framework

        # Get all steps by framework first if specified
        if framework_name:
            steps = self.find_by_framework(framework_name)
        else:
            steps = self.get_all()

        # Filter by step type
        return [step for step in steps if step.step == step_name]

    def find_by_creation_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Step]:
        """
        Find steps created within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List[Step]: List of steps created within the date range
        """
        steps = self.get_all()

        # Filter by creation date
        return [
            step
            for step in steps
            if step.get_creation_date()
            and start_date <= step.get_creation_date() <= end_date
        ]

    def find_by_owner_and_idea(self, email: str, idea_id: str) -> List[Step]:
        """
        Find steps by a specific user for a specific idea.

        Args:
            email: User's email address
            idea_id: Idea ID

        Returns:
            List[Step]: List of matching steps
        """
        # Get steps by owner first (typically fewer)
        owner_steps = self.find_by_owner(email)

        # Filter by idea_id
        return [
            step
            for step in owner_steps
            if step.idea_id
            and (
                (hasattr(step.idea_id, "oid") and step.idea_id.oid == idea_id)
                or str(step.idea_id) == idea_id
            )
        ]

    def get_step_creation_timeline(
        self,
        owner: Optional[str] = None,
        idea_id: Optional[str] = None,
        framework: Optional[Union[str, FrameworkType]] = None,
    ) -> List[Tuple[datetime, Step]]:
        """
        Get a timeline of step creations, optionally filtered.

        Args:
            owner: Optional user email filter
            idea_id: Optional idea ID filter
            framework: Optional framework filter

        Returns:
            List[Tuple[datetime, Step]]: List of (creation date, step) tuples
        """
        # Apply filters
        if owner:
            steps = self.find_by_owner(owner)
        elif idea_id:
            steps = self.find_by_idea_id(idea_id)
        elif framework:
            steps = self.find_by_framework(framework)
        else:
            steps = self.get_all()

        # Create timeline with datetime objects
        timeline = []
        for step in steps:
            creation_date = step.get_creation_date()
            if creation_date:
                timeline.append((creation_date, step))

        # Sort by creation date
        return sorted(timeline, key=lambda x: x[0])

    def get_steps_by_version(self, version: int) -> List[Step]:
        """
        Get steps by version number.

        Args:
            version: Version number (1 for first version, 2 for updates, etc.)

        Returns:
            List[Step]: List of steps with the specified version
        """
        return [step for step in self.get_all() if step.get_version() == version]

    def get_first_steps_timeline(
        self, owner: Optional[str] = None
    ) -> List[Tuple[datetime, Step]]:
        """
        Get a timeline of first version steps, optionally filtered by owner.

        This helps analyze the initial engagement pattern.

        Args:
            owner: Optional user email filter

        Returns:
            List[Tuple[datetime, Step]]: List of (creation date, step) tuples
        """
        # Get first version steps
        if owner:
            steps = [
                step for step in self.find_by_owner(owner) if step.is_first_version()
            ]
        else:
            steps = self.get_steps_by_version(1)

        # Create timeline
        timeline = []
        for step in steps:
            creation_date = step.get_creation_date()
            if creation_date:
                timeline.append((creation_date, step))

        # Sort by creation date
        return sorted(timeline, key=lambda x: x[0])

    def get_step_completion_count_by_owner(
        self, framework: Optional[Union[str, FrameworkType]] = None
    ) -> Dict[str, int]:
        """
        Get the number of steps completed by each user.

        Args:
            framework: Optional framework filter

        Returns:
            Dict[str, int]: Mapping from user email to step count
        """
        result = {}

        # Get steps, optionally filtered by framework
        if framework:
            steps = self.find_by_framework(framework)
        else:
            steps = self.get_all()

        # Count steps by owner
        for step in steps:
            if step.owner:
                if step.owner in result:
                    result[step.owner] += 1
                else:
                    result[step.owner] = 1

        return result

    def get_step_count_by_framework(self) -> Dict[str, int]:
        """
        Get the number of steps in each framework.

        Returns:
            Dict[str, int]: Mapping from framework name to step count
        """
        result = {framework.value: 0 for framework in FrameworkType}

        for step in self.get_all():
            if step.framework in result:
                result[step.framework] += 1

        return result

    def get_step_count_by_type(
        self, framework: Optional[Union[str, FrameworkType]] = None
    ) -> Dict[str, int]:
        """
        Get the number of steps of each type.

        Args:
            framework: Optional framework filter

        Returns:
            Dict[str, int]: Mapping from step type to count
        """
        if framework:
            if isinstance(framework, FrameworkType):
                framework_name = framework.value
            else:
                framework_name = framework

            # Initialize result based on framework
            if framework_name == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP.value:
                result = {step.value: 0 for step in DisciplinedEntrepreneurshipStep}
            elif framework_name == FrameworkType.STARTUP_TACTICS.value:
                result = {step.value: 0 for step in StartupTacticsStep}
            else:
                result = {}

            # Count steps in the specified framework
            steps = self.find_by_framework(framework_name)
        else:
            # Initialize result with all possible step types
            result = {}
            for step in DisciplinedEntrepreneurshipStep:
                result[step.value] = 0
            for step in StartupTacticsStep:
                result[step.value] = 0

            # Count all steps
            steps = self.get_all()

        # Count steps by type
        for step in steps:
            if step.step in result:
                result[step.step] += 1

        return result

    def get_step_progression_by_idea(self, idea_id: str) -> List[Tuple[datetime, str]]:
        """
        Get the progression of steps for a specific idea.

        Args:
            idea_id: Idea ID

        Returns:
            List[Tuple[datetime, str]]: List of (creation date, step name) tuples
        """
        steps = self.find_by_idea_id(idea_id)

        # Create timeline of first versions only
        timeline = []
        for step in steps:
            if step.is_first_version():
                creation_date = step.get_creation_date()
                if creation_date and step.step:
                    timeline.append((creation_date, step.step))

        # Sort by creation date
        return sorted(timeline, key=lambda x: x[0])

    def get_dropout_analysis(
        self,
        framework: Union[
            str, FrameworkType
        ] = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
    ) -> Dict[str, int]:
        """
        Analyze where users tend to drop out of the framework.

        Dropout is defined as the last step taken for an idea.

        Args:
            framework: Framework to analyze

        Returns:
            Dict[str, int]: Mapping from step name to dropout count
        """
        if isinstance(framework, FrameworkType):
            framework_name = framework.value
        else:
            framework_name = framework

        # For now, only support Disciplined Entrepreneurship
        if framework_name != FrameworkType.DISCIPLINED_ENTREPRENEURSHIP.value:
            return {}

        # Initialize result
        result = {step.value: 0 for step in DisciplinedEntrepreneurshipStep}

        # Get all steps in the framework
        all_steps = self.find_by_framework(framework_name)

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

        # Find the last step for each idea
        for idea_id, idea_steps in steps_by_idea.items():
            # Sort steps by creation date
            sorted_steps = sorted(
                [s for s in idea_steps if s.get_creation_date()],
                key=lambda s: s.get_creation_date(),
            )

            if sorted_steps:
                last_step = sorted_steps[-1]
                if last_step.step in result:
                    result[last_step.step] += 1

        return result

    def get_session_duration_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of session durations.

        Returns:
            Dict[str, int]: Mapping from duration range to count
        """
        # Define duration ranges (in minutes)
        ranges = ["0-5", "5-15", "15-30", "30-60", "60-120", "120+"]
        result = {r: 0 for r in ranges}

        # Get all sessions
        all_sessions = {}
        for step in self.get_all():
            if step.session_id:
                session_id = (
                    step.session_id.oid
                    if hasattr(step.session_id, "oid")
                    else str(step.session_id)
                )
                if session_id not in all_sessions:
                    all_sessions[session_id] = []
                all_sessions[session_id].append(step)

        # Calculate duration for each session
        for session_id, session_steps in all_sessions.items():
            # Filter steps with valid creation dates
            valid_steps = [s for s in session_steps if s.get_creation_date()]

            if len(valid_steps) > 0:
                # Sort steps by creation date
                sorted_steps = sorted(valid_steps, key=lambda s: s.get_creation_date())

                # Calculate duration in minutes
                start = sorted_steps[0].get_creation_date()
                end = sorted_steps[-1].get_creation_date()
                duration_minutes = (end - start).total_seconds() / 60

                # Assign to range
                if duration_minutes <= 5:
                    result["0-5"] += 1
                elif duration_minutes <= 15:
                    result["5-15"] += 1
                elif duration_minutes <= 30:
                    result["15-30"] += 1
                elif duration_minutes <= 60:
                    result["30-60"] += 1
                elif duration_minutes <= 120:
                    result["60-120"] += 1
                else:
                    result["120+"] += 1

        return result

    def get_popular_step_patterns(
        self, max_patterns: int = 10
    ) -> List[Tuple[List[str], int]]:
        """
        Get the most popular step patterns (sequences of steps).

        Args:
            max_patterns: Maximum number of patterns to return

        Returns:
            List[Tuple[List[str], int]]: List of (step pattern, count) tuples
        """
        pattern_counts = {}

        # Get all steps grouped by idea
        all_steps = self.get_all()
        steps_by_idea = {}
        for step in all_steps:
            if step.idea_id and step.is_first_version():
                idea_id = (
                    step.idea_id.oid
                    if hasattr(step.idea_id, "oid")
                    else str(step.idea_id)
                )
                if idea_id not in steps_by_idea:
                    steps_by_idea[idea_id] = []
                steps_by_idea[idea_id].append(step)

        # Extract patterns
        for idea_id, idea_steps in steps_by_idea.items():
            # Sort steps by creation date
            sorted_steps = sorted(
                [s for s in idea_steps if s.get_creation_date()],
                key=lambda s: s.get_creation_date(),
            )

            if len(sorted_steps) >= 2:  # Need at least 2 steps for a pattern
                # Create pattern tuples
                pattern = tuple(s.step for s in sorted_steps)
                if pattern in pattern_counts:
                    pattern_counts[pattern] += 1
                else:
                    pattern_counts[pattern] = 1

        # Sort by count (descending)
        sorted_patterns = sorted(
            [(list(pattern), count) for pattern, count in pattern_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_patterns[:max_patterns]

    def get_time_between_steps_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of time intervals between consecutive steps.

        Returns:
            Dict[str, int]: Mapping from time interval range to count
        """
        # Define time ranges (in minutes)
        ranges = ["0-1", "1-5", "5-15", "15-30", "30-60", "60-120", "120-720", "720+"]
        result = {r: 0 for r in ranges}

        # Get all steps grouped by idea
        all_steps = self.get_all()
        steps_by_idea = {}
        for step in all_steps:
            if step.idea_id and step.is_first_version():
                idea_id = (
                    step.idea_id.oid
                    if hasattr(step.idea_id, "oid")
                    else str(step.idea_id)
                )
                if idea_id not in steps_by_idea:
                    steps_by_idea[idea_id] = []
                steps_by_idea[idea_id].append(step)

        # Calculate intervals
        for idea_id, idea_steps in steps_by_idea.items():
            # Sort steps by creation date
            sorted_steps = sorted(
                [s for s in idea_steps if s.get_creation_date()],
                key=lambda s: s.get_creation_date(),
            )

            # Calculate intervals between consecutive steps
            for i in range(1, len(sorted_steps)):
                prev_time = sorted_steps[i - 1].get_creation_date()
                curr_time = sorted_steps[i].get_creation_date()
                interval_minutes = (curr_time - prev_time).total_seconds() / 60

                # Assign to range
                if interval_minutes <= 1:
                    result["0-1"] += 1
                elif interval_minutes <= 5:
                    result["1-5"] += 1
                elif interval_minutes <= 15:
                    result["5-15"] += 1
                elif interval_minutes <= 30:
                    result["15-30"] += 1
                elif interval_minutes <= 60:
                    result["30-60"] += 1
                elif interval_minutes <= 120:
                    result["60-120"] += 1
                elif interval_minutes <= 720:
                    result["120-720"] += 1
                else:
                    result["720+"] += 1

        return result

    def get_hourly_activity_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of step creation activity by hour of day.

        Returns:
            Dict[int, int]: Mapping from hour (0-23) to step count
        """
        result = {hour: 0 for hour in range(24)}

        for step in self.get_all():
            creation_date = step.get_creation_date()
            if creation_date:
                hour = creation_date.hour
                result[hour] += 1

        return result

    def get_daily_activity_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of step creation activity by day of week.

        Returns:
            Dict[str, int]: Mapping from day name to step count
        """
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        result = {day: 0 for day in days}

        for step in self.get_all():
            creation_date = step.get_creation_date()
            if creation_date:
                day_name = days[creation_date.weekday()]
                result[day_name] += 1

        return result

    def get_steps_with_user_input(self) -> List[Step]:
        """
        Get steps that have user input in the message field.

        Returns:
            List[Step]: List of steps with user input
        """
        return [step for step in self.get_all() if step.has_user_input()]

    def get_user_input_rate_by_step_type(
        self,
        framework: Union[
            str, FrameworkType
        ] = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
    ) -> Dict[str, float]:
        """
        Get the rate of user input for each step type.

        Args:
            framework: Framework to analyze

        Returns:
            Dict[str, float]: Mapping from step type to user input rate
        """
        if isinstance(framework, FrameworkType):
            framework_name = framework.value
        else:
            framework_name = framework

        # Initialize result based on framework
        if framework_name == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP.value:
            result = {step.value: 0.0 for step in DisciplinedEntrepreneurshipStep}
            steps_by_type = {step.value: [] for step in DisciplinedEntrepreneurshipStep}
        elif framework_name == FrameworkType.STARTUP_TACTICS.value:
            result = {step.value: 0.0 for step in StartupTacticsStep}
            steps_by_type = {step.value: [] for step in StartupTacticsStep}
        else:
            return {}

        # Group steps by type
        framework_steps = self.find_by_framework(framework_name)
        for step in framework_steps:
            if step.step in steps_by_type:
                steps_by_type[step.step].append(step)

        # Calculate input rates
        for step_type, steps in steps_by_type.items():
            if steps:
                input_count = sum(1 for s in steps if s.has_user_input())
                result[step_type] = input_count / len(steps)

        return result

    def get_step_completion_timeline_by_semester(
        self,
        framework: Union[
            str, FrameworkType
        ] = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
    ) -> Dict[str, List[Tuple[datetime, int]]]:
        """
        Get the timeline of step completions by semester.

        Args:
            framework: Framework to analyze

        Returns:
            Dict[str, List[Tuple[datetime, int]]]: Mapping from semester to (date, count) tuples
        """
        if isinstance(framework, FrameworkType):
            framework_name = framework.value
        else:
            framework_name = framework

        # Define semesters
        semesters = {
            "Fall 2023": (datetime(2023, 9, 1), datetime(2023, 12, 31)),
            "Spring 2024": (datetime(2024, 1, 1), datetime(2024, 5, 31)),
            "Fall 2024": (datetime(2024, 9, 1), datetime(2024, 12, 31)),
            "Spring 2025": (datetime(2025, 1, 1), datetime(2025, 5, 31)),
        }
        result = {semester: [] for semester in semesters.keys()}

        # Get steps in the framework
        framework_steps = self.find_by_framework(framework_name)

        # Group steps by creation date
        steps_by_date = defaultdict(int)
        for step in framework_steps:
            creation_date = step.get_creation_date()
            if creation_date:
                # Truncate to day
                day = creation_date.replace(hour=0, minute=0, second=0, microsecond=0)
                steps_by_date[day] += 1

        # Assign to semesters
        for day, count in steps_by_date.items():
            for semester, (start, end) in semesters.items():
                if start <= day <= end:
                    result[semester].append((day, count))
                    break

        # Sort each semester's timeline
        for semester in result:
            result[semester].sort(key=lambda x: x[0])

        return result
