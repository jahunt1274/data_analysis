"""
Team repository for the data analysis system.

This module provides data access and query methods for Team entities,
including team composition analysis and comparison of team vs. individual usage patterns.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path

from ..models.team_model import Team
from ..models.enums import Semester
from .base_repository import BaseRepository
from .user_repository import InMemoryDatabase


class TeamRepository(BaseRepository[Team]):
    """
    Repository for Team data access and analysis.

    This repository provides methods for querying and analyzing team data,
    including team composition and member relationships.
    """

    def __init__(self, db=None, config=None):
        """
        Initialize the team repository.

        Args:
            db: Optional database connection
            config: Optional configuration object
        """
        super().__init__("teams", Team)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._config = config
        # In-memory indexes for faster lookups
        self._member_index = {}  # Maps member email to list of teams
        self._semester_index = {}  # Maps semester to list of teams

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
            if self._config and hasattr(self._config, "TEAM_DATA_PATH"):
                file_path = self._config.TEAM_DATA_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/de_teams.json"

            # Create in-memory DB-like structure if needed
            if not isinstance(self._db, InMemoryDatabase):
                self._db = InMemoryDatabase()

            self._load_data_from_json(file_path)

            # Build indexes for faster lookups
            self._build_indexes()
        except Exception as e:
            self._logger.error(f"Error connecting to team data: {e}")
            raise

    def _load_data_from_json(self, file_path: str) -> None:
        """
        Load team data from a JSON file.

        Args:
            file_path: Path to the JSON file containing team data
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
                f"Loaded {self._db.count(self._collection_name, {})} teams from {file_path}"
            )
        except Exception as e:
            self._logger.error(f"Error loading team data from {file_path}: {e}")
            raise

    def _build_indexes(self) -> None:
        """
        Build in-memory indexes for faster lookups.
        """
        try:
            teams = self.get_all()

            # Build member index
            for team in teams:
                for member in team.team_members:
                    if member.email:
                        if member.email not in self._member_index:
                            self._member_index[member.email] = []
                        self._member_index[member.email].append(team)

            # Build semester index (assuming teams have a semester attribute or we infer it)
            # This would need to be populated with actual team semester data
            # For now, we'll leave this empty
        except Exception as e:
            self._logger.error(f"Error building indexes: {e}")

    def find_by_member_email(self, email: str) -> List[Team]:
        """
        Find all teams that have a specific member.

        Args:
            email: Member's email address

        Returns:
            List[Team]: List of teams with the member
        """
        # Check in-memory index first
        if email in self._member_index:
            return self._member_index[email]

        # Fall back to filtering all teams
        return [team for team in self.get_all() if team.has_member(email)]

    def find_by_team_name(self, team_name: str) -> Optional[Team]:
        """
        Find a team by its name.

        Args:
            team_name: Team name to search for

        Returns:
            Optional[Team]: Team or None if not found
        """
        team_name = team_name.lower()
        for team in self.get_all():
            if team.team_name and team.team_name.lower() == team_name:
                return team
        return None

    def find_by_team_id(self, team_id: int) -> Optional[Team]:
        """
        Find a team by its ID.

        Args:
            team_id: Team ID

        Returns:
            Optional[Team]: Team or None if not found
        """
        return self.find_one({"teamId": team_id})

    def find_by_semester(self, semester: Union[str, Semester]) -> List[Team]:
        """
        Find teams from a specific semester.

        Note: This assumes there is semester information in the team data,
        which may need to be inferred or added during processing.

        Args:
            semester: Semester name or enum

        Returns:
            List[Team]: List of teams from the semester
        """
        # Convert to string semester name if an enum is provided
        if isinstance(semester, Semester):
            semester_name = semester.value
        else:
            semester_name = semester

        # Check in-memory index first (if populated)
        if semester_name in self._semester_index:
            return self._semester_index[semester_name]

        # If no semester info available, return empty list
        return []

    def get_team_sizes(self) -> Dict[int, int]:
        """
        Get the distribution of team sizes.

        Returns:
            Dict[int, int]: Mapping from team size to count
        """
        result = {}

        for team in self.get_all():
            size = team.get_member_count()
            if size in result:
                result[size] += 1
            else:
                result[size] = 1

        return result

    def get_all_team_members(self) -> Set[str]:
        """
        Get the set of all team member emails.

        Returns:
            Set[str]: Set of all member emails
        """
        member_emails = set()

        for team in self.get_all():
            member_emails.update(team.get_member_emails())

        return member_emails

    def get_teams_with_min_size(self, min_size: int) -> List[Team]:
        """
        Get teams with at least a minimum number of members.

        Args:
            min_size: Minimum team size

        Returns:
            List[Team]: List of teams with sufficient size
        """
        return [team for team in self.get_all() if team.get_member_count() >= min_size]

    def get_teams_with_max_size(self, max_size: int) -> List[Team]:
        """
        Get teams with at most a maximum number of members.

        Args:
            max_size: Maximum team size

        Returns:
            List[Team]: List of teams with limited size
        """
        return [team for team in self.get_all() if team.get_member_count() <= max_size]

    def get_avg_team_size(self) -> float:
        """
        Get the average team size.

        Returns:
            float: Average number of members per team
        """
        teams = self.get_all()
        if not teams:
            return 0.0

        total_members = sum(team.get_member_count() for team in teams)
        return total_members / len(teams)

    def get_member_overlap(self, team1_id: int, team2_id: int) -> List[str]:
        """
        Get the list of members who belong to both teams.

        Args:
            team1_id: First team ID
            team2_id: Second team ID

        Returns:
            List[str]: List of common member emails
        """
        team1 = self.find_by_team_id(team1_id)
        team2 = self.find_by_team_id(team2_id)

        if not team1 or not team2:
            return []

        team1_emails = set(team1.get_member_emails())
        team2_emails = set(team2.get_member_emails())

        return list(team1_emails.intersection(team2_emails))

    def get_members_in_multiple_teams(self) -> Dict[str, int]:
        """
        Get members who belong to multiple teams.

        Returns:
            Dict[str, int]: Mapping from member email to team count
        """
        member_team_counts = {}

        for team in self.get_all():
            for email in team.get_member_emails():
                if email in member_team_counts:
                    member_team_counts[email] += 1
                else:
                    member_team_counts[email] = 1

        # Filter to only members in multiple teams
        return {
            email: count for email, count in member_team_counts.items() if count > 1
        }

    def get_team_distribution_by_semester(self) -> Dict[str, int]:
        """
        Get the number of teams in each semester.

        Returns:
            Dict[str, int]: Mapping from semester to team count
        """
        # This assumes teams have semester information
        # If not available, this would return an empty dictionary
        result = {semester.value: 0 for semester in Semester}

        # If we had semester info, we would count teams by semester
        # For now, this is a placeholder
        return result

    def get_teams_by_department(self) -> Dict[str, List[Team]]:
        """
        Group teams by primary department.

        This might need to be inferred from member departments.

        Returns:
            Dict[str, List[Team]]: Mapping from department to team list
        """
        # This would require department information for teams or members
        # For now, this is a placeholder
        return {}

    def get_team_member_emails(self, team_id: int) -> List[str]:
        """
        Get the list of email addresses for a team's members.

        Args:
            team_id: Team ID

        Returns:
            List[str]: List of member emails
        """
        team = self.find_by_team_id(team_id)
        if team:
            return team.get_member_emails()
        return []

    def get_all_team_ids(self) -> List[int]:
        """
        Get the list of all team IDs.

        Returns:
            List[int]: List of team IDs
        """
        return [team.team_id for team in self.get_all() if team.team_id is not None]

    def get_team_with_needed_members(self) -> List[Team]:
        """
        Get teams that still need members.

        Returns:
            List[Team]: List of teams needing members
        """
        return [
            team
            for team in self.get_all()
            if team.teammates_needed and team.teammates_needed > 0
        ]

    def match_users_to_teams(self, user_emails: List[str]) -> Dict[str, List[Team]]:
        """
        Match users to their teams.

        Args:
            user_emails: List of user emails

        Returns:
            Dict[str, List[Team]]: Mapping from user email to team list
        """
        result = {email: [] for email in user_emails}

        for email in user_emails:
            if email in self._member_index:
                result[email] = self._member_index[email]

        return result

    def get_users_not_in_teams(self, user_emails: List[str]) -> List[str]:
        """
        Find users who are not in any team.

        Args:
            user_emails: List of user emails

        Returns:
            List[str]: List of user emails not in any team
        """
        team_members = self.get_all_team_members()
        return [email for email in user_emails if email not in team_members]

    def get_team_member_count_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of team member counts.

        Returns:
            Dict[int, int]: Mapping from member count to team count
        """
        result = {}

        for team in self.get_all():
            count = team.get_member_count()
            if count in result:
                result[count] += 1
            else:
                result[count] = 1

        return result

    def get_team_descriptions(self) -> Dict[int, str]:
        """
        Get the descriptions for all teams.

        Returns:
            Dict[int, str]: Mapping from team ID to description
        """
        return {
            team.team_id: team.description
            for team in self.get_all()
            if team.team_id is not None and team.description
        }

    def get_teams_with_description_keywords(self, keywords: List[str]) -> List[Team]:
        """
        Find teams with specific keywords in their descriptions.

        Args:
            keywords: List of keywords to search for

        Returns:
            List[Team]: List of matching teams
        """
        result = []
        keywords = [kw.lower() for kw in keywords]

        for team in self.get_all():
            if not team.description:
                continue

            desc_lower = team.description.lower()
            if any(kw in desc_lower for kw in keywords):
                result.append(team)

        return result
