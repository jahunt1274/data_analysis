"""
Team model for the data analysis system.

This module defines the data models for teams in the 15.390 course,
including team composition and member details.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TeamMember(BaseModel):
    """Team member information."""

    email: str  # Primary identifier, links to User.email
    name: Optional[str] = None
    alt_email: Optional[str] = Field(default=None, alias="alt_email")
    phone: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class Team(BaseModel):
    """
    Team model for 15.390 course teams.

    This represents a team in the course, including all team members
    and their contact information.
    """

    matches: Optional[str] = None  # Purpose unclear
    team_id: Optional[int] = Field(default=None, alias="teamId")
    team_name: Optional[str] = Field(default=None, alias="teamName")
    description: Optional[str] = None
    team_members: List[TeamMember] = Field(default_factory=list, alias="teamMembers")
    teammates_needed: Optional[int] = Field(default=0, alias="teammatesNeeded")

    class Config:
        allow_population_by_field_name = True

    def get_member_emails(self) -> List[str]:
        """
        Get a list of all team member emails.

        Returns:
            List[str]: Email addresses
        """
        return [member.email for member in self.team_members]

    def get_member_count(self) -> int:
        """
        Get the number of team members.

        Returns:
            int: Member count
        """
        return len(self.team_members)

    def has_member(self, email: str) -> bool:
        """
        Check if a specific user is a member of this team.

        Args:
            email: User email to check

        Returns:
            bool: True if the user is a member
        """
        return email in self.get_member_emails()

    def get_member_by_email(self, email: str) -> Optional[TeamMember]:
        """
        Get a team member by email.

        Args:
            email: User email to find

        Returns:
            Optional[TeamMember]: Team member or None if not found
        """
        for member in self.team_members:
            if member.email == email:
                return member
        return None

    @field_validator("team_members", mode="before")
    def validate_team_members(cls, v):
        """Validate team members, ensuring they have valid emails."""
        if not v:
            return []

        # Filter out any team members without email
        valid_members = []
        for member in v:
            if isinstance(member, dict) and "email" in member and member["email"]:
                valid_members.append(member)

        return valid_members
