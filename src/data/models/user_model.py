"""
User model for the data analysis system.

This module defines the data models for users of the JetPack/Orbit tool,
including their affiliations, profiles, and engagement metrics.
"""

from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.models.base_model import ObjectId, DateField
from src.data.models.enums import UserType, UserEngagementLevel
from src.utils.safe_ops import safe_lower


class Department(BaseModel):
    """Department information for a user."""

    name: Optional[str] = None
    code: Optional[str] = None
    org_unit_id: Optional[str] = Field(default=None, alias="orgUnitId")

    model_config = ConfigDict(populate_by_name=True)


class Course(BaseModel):
    """Course information for a user."""

    department_code: Optional[str] = Field(default=None, alias="departmentCode")
    name: Optional[str] = None
    degree_status: Optional[str] = Field(default=None, alias="degreeStatus")
    course_option: Optional[str] = Field(default=None, alias="courseOption")
    primary: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True)


class StudentAffiliation(BaseModel):
    """Student affiliation information for a user."""

    courses: Optional[List[Course]] = []
    departments: Optional[List[Department]] = []
    student_type: Optional[str] = Field(default=None, alias="student_type")
    yog: Optional[Union[int, str]] = None  # Year of graduation
    advisors: Optional[List[Any]] = []
    office: Optional[str] = None
    type: Optional[str] = None
    class_year: Optional[str] = Field(default=None, alias="classYear")
    title: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    def get_user_type(self) -> UserType:
        """Extract the user type from the affiliation data."""
        affiliation_dict = self.model_dump(by_alias=True)
        return UserType.from_affiliation(affiliation_dict)


class Institution(BaseModel):
    """Institution information for a user."""

    affiliation: Optional[StudentAffiliation] = None
    name: Optional[str] = None


class OrbitProfilePermissions(BaseModel):
    """Permission settings for the user's Orbit profile."""

    share_resources: Optional[bool] = Field(default=False, alias="share_resources")
    share_events: Optional[bool] = Field(default=False, alias="share_events")
    share_classes: Optional[bool] = Field(default=False, alias="share_classes")

    model_config = ConfigDict(populate_by_name=True)


class OrbitProfile(BaseModel):
    """Orbit profile information for a user."""

    share_resources: Optional[bool] = Field(default=False, alias="share_resources")
    interest: Optional[List[str]] = []
    has_image: Optional[bool] = Field(default=False, alias="has_image")
    terms_and_conditions: Optional[bool] = Field(
        default=False, alias="termsAndConditions"
    )
    twitter: Optional[str] = None
    student_affiliation: Optional[StudentAffiliation] = None
    departments: Optional[List[str]] = []
    facebook: Optional[str] = None
    share_events: Optional[bool] = Field(default=False, alias="share_events")
    mtc_newsletter: Optional[bool] = Field(default=False, alias="MTCNewsletter")
    persona: Optional[List[str]] = []
    linkedin: Optional[str] = None
    share_classes: Optional[bool] = Field(default=False, alias="share_classes")
    need: Optional[List[str]] = []  # What the user needs/expects from the tool
    bio: Optional[str] = None
    permissions: Optional[OrbitProfilePermissions] = None
    experience: Optional[str] = None  # Entrepreneurial experience level

    model_config = ConfigDict(populate_by_name=True)


class UserScores(BaseModel):
    """Engagement and completion scores for a user."""

    content: Optional[float] = 0.0  # Content creation score
    completion: Optional[float] = 0.0  # Framework completion score

    def get_engagement_level(self) -> UserEngagementLevel:
        """
        Determine the user's engagement level based on scores.

        This is a simplified version that may need refinement based on
        actual score distributions in the data.
        """
        # Combined score for general engagement assessment
        combined = (self.content + self.completion) / 2

        if combined >= 0.7:
            return UserEngagementLevel.HIGH
        elif combined >= 0.3:
            return UserEngagementLevel.MEDIUM
        else:
            return UserEngagementLevel.LOW


class User(BaseModel):
    """
    User model for JetPack/Orbit tool users.

    This represents a user in the system, including their
    personal information, affiliations, and engagement metrics.
    """

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    website: Optional[str] = None
    tags: Optional[List[str]] = []
    middle_name: Optional[str] = None
    updated: Optional[Union[DateField, datetime]] = None
    created: Optional[Union[DateField, datetime]] = None
    token: Optional[str] = None
    views: Optional[List[int]] = []
    saved_classes: Optional[List[str]] = []
    enrollments: Optional[List[str]] = []  # Course enrollments like "15.390"
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    student_affiliation: Optional[StudentAffiliation] = None
    synced_linked: Optional[bool] = False
    aerospace_innovation_certificate: Optional[bool] = False
    name: Optional[str] = None
    orbit_profile: Optional[OrbitProfile] = Field(default=None, alias="orbitProfile")
    saved_opportunities: Optional[List[str]] = []
    has_onboarded: Optional[bool] = False
    personal_email: Optional[str] = None
    kerberos: Optional[str] = None  # MIT Kerberos username
    saved_contacts: Optional[List[str]] = []
    title: Optional[str] = None
    random: Optional[str] = None  # Purpose unclear
    verified: Optional[bool] = False
    departments: Optional[List[str]] = []
    additional_email: Optional[str] = None
    newsletter: Optional[bool] = False
    logins: Optional[List[Any]] = []
    affiliations: Optional[List[Any]] = []
    scores: Optional[UserScores] = None
    email: Optional[str] = None  # Primary identifier
    role: Optional[str] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    last_login: Optional[Union[DateField, datetime, int]] = None
    profile_photo: Optional[str] = None
    # following_ideas: Optional[List[Union[ObjectId, Dict]]] = []
    type: Optional[str] = None
    applications: Optional[List[str]] = []
    institution: Optional[Institution] = None
    resource_tags: Optional[List[str]] = []

    model_config = ConfigDict(populate_by_name=True)

    def get_user_type(self) -> UserType:
        """
        Determine the user type from various sources of information.

        Returns:
            UserType: The determined user type enum
        """
        # Try student_affiliation first
        if self.student_affiliation:
            return self.student_affiliation.get_user_type()

        # Try institution.affiliation
        if self.institution and self.institution.affiliation:
            return UserType.from_affiliation(
                self.institution.affiliation.model_dump(by_alias=True)
            )

        # Try type field
        if self.type:
            for member in UserType:
                if safe_lower(member.value) == safe_lower(self.type):
                    return member

        # Default
        return UserType.OTHER

    def get_engagement_level(self) -> UserEngagementLevel:
        """
        Determine the user's engagement level.

        Returns:
            UserEngagementLevel: HIGH, MEDIUM, or LOW
        """
        if self.scores:
            return self.scores.get_engagement_level()
        return UserEngagementLevel.LOW

    def is_in_course(self, course_id: str) -> bool:
        """
        Check if the user is enrolled in a specific course.

        Args:
            course_id: Course identifier (e.g., "15.390")

        Returns:
            bool: True if enrolled, False otherwise
        """
        if self.enrollments:
            return course_id in self.enrollments
        return False

    def is_student(self) -> bool:
        """
        Check if the user is a student.

        Returns:
            bool: True if student, False otherwise
        """
        user_type = self.get_user_type()
        return user_type in [
            UserType.UNDERGRADUATE,
            UserType.GRADUATE,
            UserType.MBA,
            UserType.PHD,
        ]

    def get_department(self) -> Optional[str]:
        """
        Get the user's primary department.

        Returns:
            Optional[str]: Department name if available
        """
        # Check student_affiliation
        if self.student_affiliation and self.student_affiliation.departments:
            for dept in self.student_affiliation.departments:
                if dept.name:
                    return dept.name

        # Check departments list
        if self.departments and len(self.departments) > 0:
            return self.departments[0]

        return None

    def get_creation_date(self) -> Optional[datetime]:
        """
        Get the user creation date as a datetime object.

        Returns:
            Optional[datetime]: Creation date
        """
        if isinstance(self.created, DateField):
            return self.created.to_datetime()
        elif isinstance(self.created, datetime):
            return self.created
        return None

    def get_last_login_date(self) -> Optional[datetime]:
        """
        Get the last login date as a datetime object.

        Returns:
            Optional[datetime]: Last login date
        """
        if isinstance(self.last_login, DateField):
            return self.last_login.to_datetime()
        elif isinstance(self.last_login, datetime):
            return self.last_login
        return None

    @field_validator("orbit_profile", mode="before")
    @classmethod
    def validate_orbit_profile(cls, v):
        """Validate the orbit profile field."""
        if v is None:
            return OrbitProfile()
        return v

    @field_validator("scores", mode="before")
    @classmethod
    def validate_scores(cls, v):
        """Validate the scores field."""
        if v is None:
            return UserScores()
        return v
