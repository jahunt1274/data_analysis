"""
User repository for the data analysis system.

This module provides data access and query methods for User entities,
including engagement analysis and demographic segmentation.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from src.data.models.user_model import User
from src.data.models.enums import UserEngagementLevel, UserType, Semester
from src.data.repositories.base_repository import BaseRepository
from src.data.db import InMemoryDatabase

from src.utils.safe_ops import safe_lower


class UserRepository(BaseRepository[User]):
    """
    Repository for User data access and analysis.

    This repository provides methods for querying and analyzing user data,
    including engagement levels, demographics, and course enrollments.
    """

    def __init__(self, db=None, config=None):
        """
        Initialize the user repository.

        Args:
            db: Optional database connection
            config: Optional configuration object
        """
        super().__init__("users", User)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._config = config
        self._email_index = {}  # In-memory index for quick user lookup by email

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
            if self._config and hasattr(self._config, "USER_DATA_PATH"):
                file_path = self._config.USER_DATA_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/user.json"

            # Create in-memory DB-like structure
            self._db = InMemoryDatabase()
            self._load_data_from_json(file_path)

            # Build email index for faster lookups
            self._build_email_index()
        except Exception as e:
            self._logger.error(f"Error connecting to user data: {e}")
            raise

    def _load_data_from_json(self, file_path: str) -> None:
        """
        Load user data from a JSON file.

        Args:
            file_path: Path to the JSON file containing user data
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
                f"Loaded {self._db.count(self._collection_name, {})} users from {file_path}"
            )
        except Exception as e:
            self._logger.error(f"Error loading user data from {file_path}: {e}")
            raise

    def _build_email_index(self) -> None:
        """
        Build an in-memory index for email lookups.
        """
        try:
            users = self.get_all()
            for user in users:
                if user.email:
                    self._email_index[user.email] = user
        except Exception as e:
            self._logger.error(f"Error building email index: {e}")

    def find_by_email(self, email: str) -> Optional[User]:
        """
        Find a user by email address.

        Args:
            email: User's email address

        Returns:
            Optional[User]: User model or None if not found
        """
        # Check in-memory index first
        if email in self._email_index:
            return self._email_index[email]

        # Fall back to DB query
        return self.find_one({"email": email})

    def find_by_course(self, course_id: str) -> List[User]:
        """
        Find all users enrolled in a specific course.

        Args:
            course_id: Course identifier (e.g., "15.390")

        Returns:
            List[User]: List of enrolled users
        """
        return self.find_many({"enrollments": course_id})

    def get_users_by_engagement_level(
        self, level: UserEngagementLevel, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users filtered by engagement level.

        Engagement level is determined by combining content and completion scores.

        Args:
            level: Engagement level enum
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users at the specified engagement level
        """
        # Get all users (potentially filtered by course)
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by engagement level
        return [user for user in users if user.get_engagement_level() == level]

    def get_engagement_distribution(
        self, course_id: Optional[str] = None
    ) -> Dict[UserEngagementLevel, int]:
        """
        Get the distribution of users across engagement levels.

        Args:
            course_id: Optional course ID to filter by

        Returns:
            Dict[UserEngagementLevel, int]: Mapping from engagement level to count
        """
        distribution = {
            UserEngagementLevel.HIGH: 0,
            UserEngagementLevel.MEDIUM: 0,
            UserEngagementLevel.LOW: 0,
        }

        # Get all users (potentially filtered by course)
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Count users at each engagement level
        for user in users:
            level = user.get_engagement_level()
            distribution[level] += 1

        return distribution

    def get_users_by_type(
        self, user_type: UserType, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users filtered by type (undergraduate, graduate, etc.).

        Args:
            user_type: User type enum
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users of the specified type
        """
        # Get all users (potentially filtered by course)
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by user type
        return [user for user in users if user.get_user_type() == user_type]

    def get_user_type_distribution(
        self, course_id: Optional[str] = None
    ) -> Dict[UserType, int]:
        """
        Get the distribution of users across user types.

        Args:
            course_id: Optional course ID to filter by

        Returns:
            Dict[UserType, int]: Mapping from user type to count
        """
        distribution = {user_type: 0 for user_type in UserType}

        # Get all users (potentially filtered by course)
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Count users of each type
        for user in users:
            user_type = user.get_user_type()
            distribution[user_type] += 1

        return distribution

    def get_users_by_department(
        self, department: str, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users filtered by department.

        Args:
            department: Department name
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users in the specified department
        """
        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by department
        return [
            user
            for user in users
            if user.get_department()
            and safe_lower(department) in safe_lower(user.get_department())
        ]

    def get_users_by_creation_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[User]:
        """
        Get users created within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List[User]: List of users created within the date range
        """
        users = self.get_all()

        # Filter by creation date
        return [
            user
            for user in users
            if user.get_creation_date()
            and start_date <= user.get_creation_date() <= end_date
        ]

    def get_users_by_interest(
        self, interest: str, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users who have expressed a specific interest.

        Args:
            interest: Interest keyword
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users with the specified interest
        """
        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by interest
        return [
            user
            for user in users
            if user.orbit_profile
            and user.orbit_profile.interest
            and any(
                safe_lower(interest) in safe_lower(i)
                for i in user.orbit_profile.interest
            )
        ]

    def get_users_by_experience(
        self, experience: str, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users with a specific experience level.

        Args:
            experience: Experience level
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users with the specified experience
        """
        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by experience
        return [
            user
            for user in users
            if user.orbit_profile
            and user.orbit_profile.experience
            and safe_lower(experience) in safe_lower(user.orbit_profile.experience)
        ]

    def get_users_in_teams(self, team_emails: Set[str]) -> List[User]:
        """
        Get users who are members of specified teams.

        Args:
            team_emails: Set of team member emails

        Returns:
            List[User]: List of users who are team members
        """
        return [user for user in self.get_all() if user.email in team_emails]

    def get_users_by_semester(
        self, semester: Semester, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users from a specific semester cohort.

        Uses creation date to determine which semester a user belongs to.

        Args:
            semester: Semester enum
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of users from the specified semester
        """
        # Get date ranges for semesters
        semester_ranges = {
            Semester.FALL_2023: (datetime(2023, 9, 1), datetime(2023, 12, 31)),
            Semester.SPRING_2024: (datetime(2024, 1, 1), datetime(2024, 5, 31)),
            Semester.FALL_2024: (datetime(2024, 9, 1), datetime(2024, 12, 31)),
            Semester.SPRING_2025: (datetime(2025, 1, 1), datetime(2025, 5, 31)),
        }

        if semester not in semester_ranges:
            return []

        start_date, end_date = semester_ranges[semester]

        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by creation date
        return [
            user
            for user in users
            if user.get_creation_date()
            and start_date <= user.get_creation_date() <= end_date
        ]

    def get_active_users(
        self, since_date: datetime, course_id: Optional[str] = None
    ) -> List[User]:
        """
        Get users who have been active since a specific date.

        Args:
            since_date: Activity threshold date
            course_id: Optional course ID to filter by

        Returns:
            List[User]: List of active users
        """
        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Filter by last login date
        return [
            user
            for user in users
            if user.get_last_login_date() and user.get_last_login_date() >= since_date
        ]

    def get_top_users_by_score(
        self, score_type: str, limit: int = 10, course_id: Optional[str] = None
    ) -> List[Tuple[User, float]]:
        """
        Get top users by a specific score type.

        Args:
            score_type: Score type ("content" or "completion")
            limit: Maximum number of users to return
            course_id: Optional course ID to filter by

        Returns:
            List[Tuple[User, float]]: List of (user, score) tuples sorted by score
        """
        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Create list of (user, score) tuples
        user_scores = []
        for user in users:
            if user.scores:
                if score_type == "content" and user.scores.content is not None:
                    user_scores.append((user, user.scores.content))
                elif score_type == "completion" and user.scores.completion is not None:
                    user_scores.append((user, user.scores.completion))

        # Sort by score (descending) and take top N
        return sorted(user_scores, key=lambda x: x[1], reverse=True)[:limit]

    def get_interest_distribution(
        self, course_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Get the distribution of user interests.

        Args:
            course_id: Optional course ID to filter by

        Returns:
            Dict[str, int]: Mapping from interest to count
        """
        distribution = {}

        # First filter by course if specified
        if course_id:
            users = self.find_by_course(course_id)
        else:
            users = self.get_all()

        # Count interests
        for user in users:
            if user.orbit_profile and user.orbit_profile.interest:
                for interest in user.orbit_profile.interest:
                    if interest in distribution:
                        distribution[interest] += 1
                    else:
                        distribution[interest] = 1

        return distribution
