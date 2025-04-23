"""
Step model for the data analysis system.

This module defines the data models for steps in the entrepreneurial frameworks,
including their relationships to ideas and users.
"""

from typing import Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

from .base_model import ObjectId, DateField
from .enums import FrameworkType, DisciplinedEntrepreneurshipStep, StartupTacticsStep


class Step(BaseModel):
    """
    Step model for entrepreneurial framework steps in the JetPack/Orbit tool.

    This represents a step taken by a user for a specific idea within
    one of the entrepreneurial frameworks.
    """

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    framework: Optional[str] = (
        None  # Framework type (matches FrameworkType enum values)
    )
    owner: Optional[str] = None  # User email
    step: Optional[str] = None  # Step name (without prefix)
    created_at: Optional[Union[DateField, datetime]] = None
    active: Optional[bool] = True  # Indicates if the step is active/valid
    session_id: Optional[ObjectId] = None  # ID of the user session
    name: Optional[str] = None  # Version information (e.g., "Version 1", "Version 2")
    content: Optional[str] = None  # Step content (AI-generated or user-modified)
    message: Optional[str] = None  # User input for the step
    idea_id: Optional[ObjectId] = None  # ID of the associated idea

    class Config:
        allow_population_by_field_name = True

    def get_creation_date(self) -> Optional[datetime]:
        """
        Get the step creation date as a datetime object.

        Returns:
            Optional[datetime]: Creation date
        """
        if isinstance(self.created_at, DateField):
            return self.created_at.to_datetime()
        elif isinstance(self.created_at, datetime):
            return self.created_at
        return None

    def get_framework_type(self) -> Optional[FrameworkType]:
        """
        Get the framework type as an enum.

        Returns:
            Optional[FrameworkType]: Framework type enum
        """
        if not self.framework:
            return None

        # Try to match with defined frameworks
        for framework in FrameworkType:
            if framework.value == self.framework:
                return framework

        return None

    def get_step_type(
        self,
    ) -> Union[DisciplinedEntrepreneurshipStep, StartupTacticsStep, None]:
        """
        Get the step type as an enum based on the framework.

        Returns:
            Union[DisciplinedEntrepreneurshipStep, StartupTacticsStep, None]: Step type enum
        """
        if not self.step:
            return None

        framework = self.get_framework_type()
        if not framework:
            return None

        # Check appropriate step type based on framework
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            for step in DisciplinedEntrepreneurshipStep:
                if step.value == self.step:
                    return step
        elif framework == FrameworkType.STARTUP_TACTICS:
            for step in StartupTacticsStep:
                if step.value == self.step:
                    return step

        return None

    def get_version(self) -> int:
        """
        Get the version number from the name field.

        Returns:
            int: Version number (1, 2, ...) or 0 if not found
        """
        if not self.name:
            return 0

        # Extract version number using regex
        match = re.search(r"Version\s+(\d+)", self.name)
        if match:
            return int(match.group(1))

        return 0

    def is_first_version(self) -> bool:
        """
        Check if this is the first version of the step.

        Returns:
            bool: True if this is the first version
        """
        return self.get_version() == 1

    def has_user_input(self) -> bool:
        """
        Check if this step has user input in the message field.

        Returns:
            bool: True if there's user input
        """
        return bool(self.message and self.message.strip())

    def has_content(self) -> bool:
        """
        Check if this step has content.

        Returns:
            bool: True if there's content
        """
        return bool(self.content and self.content.strip())

    @field_validator("framework")
    def validate_framework(cls, v):
        """Validate that the framework is one of the recognized types."""
        if v and v not in FrameworkType.get_all_values():
            raise ValueError(f"Framework {v} is not recognized")
        return v

    @field_validator("step")
    def validate_step(cls, v, values):
        """Validate that the step is appropriate for the framework."""
        if not v:
            return v

        framework = values.get("framework")
        if not framework:
            return v

        # Check if step is valid for the framework
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            if v not in DisciplinedEntrepreneurshipStep.get_all_step_values():
                raise ValueError(f"Step {v} is not valid for {framework}")
        elif framework == FrameworkType.STARTUP_TACTICS:
            if v not in StartupTacticsStep.get_all_step_values():
                raise ValueError(f"Step {v} is not valid for {framework}")

        return v
