"""
User repository for the data analysis system.

This module provides data access and query methods for User entities,
including engagement analysis and demographic segmentation.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime
from pathlib import Path

from src.data.models.user_model import User
from src.data.models.enums import UserEngagementLevel, UserType, Semester
from src.data.repositories.base_repository import BaseRepository

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


class InMemoryDatabase:
    """
    Simple in-memory database implementation.

    Provides a MongoDB-like interface for storing and querying JSON data.
    """

    def __init__(self):
        """Initialize the in-memory database."""
        self.collections = {}

    def __getitem__(self, collection_name):
        """
        Get a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = InMemoryCollection(collection_name)
        return self.collections[collection_name]

    def list_collection_names(self):
        """
        Get a list of all collection names.

        Returns:
            List[str]: List of collection names
        """
        return list(self.collections.keys())

    def drop_collection(self, collection_name):
        """
        Drop a collection.

        Args:
            collection_name: Name of the collection to drop
        """
        if collection_name in self.collections:
            del self.collections[collection_name]

    def insert_one(self, collection_name, document):
        """
        Insert a single document into a collection.

        Args:
            collection_name: Collection name
            document: Document to insert
        """
        self[collection_name].insert_one(document)

    def insert_many(self, collection_name, documents):
        """
        Insert multiple documents into a collection.

        Args:
            collection_name: Collection name
            documents: List of documents to insert
        """
        self[collection_name].insert_many(documents)

    def count(self, collection_name, query):
        """
        Count documents in a collection.

        Args:
            collection_name: Collection name
            query: Query to filter documents

        Returns:
            int: Number of matching documents
        """
        return self[collection_name].count_documents(query)


class InMemoryCollection:
    """
    In-memory collection for the in-memory database.

    Provides MongoDB-like query methods for in-memory data.
    """

    def __init__(self, name):
        """
        Initialize the collection.

        Args:
            name: Collection name
        """
        self.name = name
        self.documents = []

    def insert_one(self, document):
        """
        Insert a single document.

        Args:
            document: Document to insert
        """
        self.documents.append(document)

    def insert_many(self, documents):
        """
        Insert multiple documents.

        Args:
            documents: List of documents to insert
        """
        self.documents.extend(documents)

    def find_one(self, query):
        """
        Find a single document matching the query.

        Args:
            query: Query to filter documents

        Returns:
            dict: Matching document or None
        """
        for doc in self.documents:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query=None):
        """
        Find all documents matching the query.

        Args:
            query: Query to filter documents

        Returns:
            InMemoryCursor: Cursor for the matching documents
        """
        if query is None:
            query = {}

        matches = [doc for doc in self.documents if self._matches(doc, query)]
        return InMemoryCursor(matches)

    def count_documents(self, query):
        """
        Count documents matching the query.

        Args:
            query: Query to filter documents

        Returns:
            int: Number of matching documents
        """
        matches = [doc for doc in self.documents if self._matches(doc, query)]
        return len(matches)

    def distinct(self, field, query=None):
        """
        Get distinct values for a field.

        Args:
            field: Field name
            query: Optional query to filter documents

        Returns:
            List: List of distinct values
        """
        if query is None:
            query = {}

        matches = [doc for doc in self.documents if self._matches(doc, query)]

        # Extract field values
        values = []
        for doc in matches:
            value = self._get_field_value(doc, field)
            if value is not None and value not in values:
                values.append(value)

        return values

    def create_index(self, field, unique=False):
        """
        Create an index (no-op in this implementation).

        Args:
            field: Field to index
            unique: Whether values should be unique
        """
        # No-op in this simple implementation
        pass

    def aggregate(self, pipeline):
        """
        Perform an aggregation operation.

        This is a simplified implementation that supports only
        basic $match, $group, $sort, and $project operations.

        Args:
            pipeline: Aggregation pipeline

        Returns:
            List: Results of the aggregation
        """
        results = self.documents

        for stage in pipeline:
            if "$match" in stage:
                results = [
                    doc for doc in results if self._matches(doc, stage["$match"])
                ]
            elif "$group" in stage:
                results = self._group(results, stage["$group"])
            elif "$sort" in stage:
                results = self._sort(results, stage["$sort"])
            elif "$project" in stage:
                results = self._project(results, stage["$project"])

        return results

    def _get_field_value(self, doc, field):
        """
        Get a field value from a document.

        Supports dot notation for nested fields.

        Args:
            doc: Document
            field: Field name

        Returns:
            Field value or None
        """
        if "." in field:
            parts = field.split(".")
            value = doc
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        elif field in doc:
            return doc[field]
        return None

    def _matches(self, doc, query):
        """
        Check if a document matches a query.

        Args:
            doc: Document
            query: Query

        Returns:
            bool: True if the document matches
        """
        for key, value in query.items():
            if key.startswith("$"):
                # Logical operators
                if key == "$and":
                    if not all(self._matches(doc, q) for q in value):
                        return False
                elif key == "$or":
                    if not any(self._matches(doc, q) for q in value):
                        return False
            else:
                # Field match
                if isinstance(value, dict) and any(
                    k.startswith("$") for k in value.keys()
                ):
                    # Operator match
                    field_value = self._get_field_value(doc, key)
                    if not self._matches_operator(field_value, value):
                        return False
                else:
                    # Exact match
                    field_value = self._get_field_value(doc, key)
                    if field_value != value:
                        return False

        return True

    def _matches_operator(self, field_value, operators):
        """
        Check if a field value matches operator conditions.

        Args:
            field_value: Field value
            operators: Operator conditions

        Returns:
            bool: True if the field value matches
        """
        for op, value in operators.items():
            if op == "$eq" and field_value != value:
                return False
            elif op == "$ne" and field_value == value:
                return False
            elif op == "$gt" and (field_value is None or field_value <= value):
                return False
            elif op == "$gte" and (field_value is None or field_value < value):
                return False
            elif op == "$lt" and (field_value is None or field_value >= value):
                return False
            elif op == "$lte" and (field_value is None or field_value > value):
                return False
            elif op == "$in" and field_value not in value:
                return False
            elif op == "$nin" and field_value in value:
                return False

        return True

    def _group(self, documents, group_spec):
        """
        Perform a group operation.

        Args:
            documents: Documents to group
            group_spec: Group specification

        Returns:
            List: Grouped results
        """
        # Simple implementation for basic grouping
        groups = {}
        id_spec = group_spec.pop("_id")

        for doc in documents:
            # Determine group key
            if id_spec is None:
                key = None
            elif isinstance(id_spec, str) and id_spec.startswith("$"):
                # Group by field
                field = id_spec[1:]
                key = self._get_field_value(doc, field)
            else:
                # Complex grouping (not fully implemented)
                key = str(id_spec)

            # Create group if not exists
            if key not in groups:
                groups[key] = {"_id": key}
                for field, spec in group_spec.items():
                    if isinstance(spec, dict):
                        if "$sum" in spec:
                            groups[key][field] = 0
                        elif "$avg" in spec:
                            groups[key][field] = {"sum": 0, "count": 0}
                        elif "$min" in spec:
                            groups[key][field] = None
                        elif "$max" in spec:
                            groups[key][field] = None

            # Accumulate values
            for field, spec in group_spec.items():
                if isinstance(spec, dict):
                    if "$sum" in spec:
                        if spec["$sum"] == 1:
                            groups[key][field] += 1
                        else:
                            value = self._get_field_value(doc, spec["$sum"][1:])
                            if value is not None:
                                groups[key][field] += value
                    elif "$avg" in spec:
                        value = self._get_field_value(doc, spec["$avg"][1:])
                        if value is not None:
                            groups[key][field]["sum"] += value
                            groups[key][field]["count"] += 1
                    elif "$min" in spec:
                        value = self._get_field_value(doc, spec["$min"][1:])
                        if value is not None and (
                            groups[key][field] is None or value < groups[key][field]
                        ):
                            groups[key][field] = value
                    elif "$max" in spec:
                        value = self._get_field_value(doc, spec["$max"][1:])
                        if value is not None and (
                            groups[key][field] is None or value > groups[key][field]
                        ):
                            groups[key][field] = value

        # Finalize averages
        for group in groups.values():
            for field, value in group.items():
                if isinstance(value, dict) and "sum" in value and "count" in value:
                    if value["count"] > 0:
                        group[field] = value["sum"] / value["count"]
                    else:
                        group[field] = 0

        return list(groups.values())

    def _sort(self, documents, sort_spec):
        """
        Sort documents.

        Args:
            documents: Documents to sort
            sort_spec: Sort specification

        Returns:
            List: Sorted documents
        """
        if isinstance(sort_spec, str):
            return sorted(
                documents, key=lambda doc: self._get_field_value(doc, sort_spec)
            )

        # Multiple sort fields
        def sort_key(doc):
            key = []
            for field, direction in sort_spec.items():
                value = self._get_field_value(doc, field)
                key.append((value, direction))
            return key

        return sorted(documents, key=sort_key)

    def _project(self, documents, project_spec):
        """
        Project documents.

        Args:
            documents: Documents to project
            project_spec: Projection specification

        Returns:
            List: Projected documents
        """
        result = []
        for doc in documents:
            projected = {}
            for field, include in project_spec.items():
                if include:
                    projected[field] = self._get_field_value(doc, field)
            result.append(projected)
        return result


class InMemoryCursor:
    """
    Cursor for in-memory query results.

    Provides MongoDB-like cursor methods.
    """

    def __init__(self, documents):
        """
        Initialize the cursor.

        Args:
            documents: Query result documents
        """
        self.documents = documents
        self.position = 0

    def __iter__(self):
        """
        Iterator protocol implementation.

        Returns:
            self: Iterator
        """
        return self

    def __next__(self):
        """
        Get the next document.

        Returns:
            dict: Next document

        Raises:
            StopIteration: When no more documents
        """
        if self.position >= len(self.documents):
            raise StopIteration

        document = self.documents[self.position]
        self.position += 1
        return document

    def skip(self, count):
        """
        Skip a number of documents.

        Args:
            count: Number of documents to skip

        Returns:
            self: Cursor
        """
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        """
        Limit the number of documents.

        Args:
            count: Maximum number of documents

        Returns:
            self: Cursor
        """
        self.documents = self.documents[:count]
        return self

    def sort(self, spec, direction=None):
        """
        Sort the documents.

        Args:
            spec: Sort specification or field name
            direction: Sort direction (1=ascending, -1=descending)

        Returns:
            self: Cursor
        """
        if isinstance(spec, str):
            reverse = direction == -1
            self.documents.sort(key=lambda doc: doc.get(spec), reverse=reverse)
        else:
            # Multiple sort fields
            def sort_key(doc):
                if isinstance(spec, list):
                    return tuple(doc.get(field) for field, _ in spec)
                return tuple(doc.get(field) for field in spec.keys())

            reverse = False  # Not fully implemented for multi-field sort
            self.documents.sort(key=sort_key, reverse=reverse)

        return self
