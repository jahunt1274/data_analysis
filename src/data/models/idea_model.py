"""
Idea model for the data analysis system.

This module defines the data models for ideas in the JetPack/Orbit tool,
including framework progress, step completion, and ownership information.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.models.base_model import ObjectId, DateField
from src.data.models.enums import (
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
    StartupTacticsStep,
    StepPrefix,
    IdeaCategory,
)
from src.utils.safe_ops import safe_lower


class FrameworkProgress(BaseModel):
    """Progress in a particular entrepreneurial framework."""

    disciplined_entrepreneurship: Optional[float] = Field(
        default=0.0, alias="Disciplined Entrepreneurship"
    )
    startup_tactics: Optional[float] = Field(default=0.0, alias="Startup Tactics")
    my_journey: Optional[float] = Field(default=0.0, alias="My Journey")
    product_management: Optional[float] = Field(default=0.0, alias="Product Management")

    model_config = ConfigDict(populate_by_name=True)


class Idea(BaseModel):
    """
    Idea model for entrepreneurial ideas in the JetPack/Orbit tool.

    This represents an idea created by a user, including all of its
    associated steps across entrepreneurial frameworks.
    """

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    title: Optional[str] = None
    slug: Optional[str] = None
    description: Optional[str] = None
    created: Optional[Union[DateField, datetime]] = None
    # Sequential number for a user's ideas (1st, 2nd, etc.)
    ranking: Optional[int] = None
    owner: Optional[str] = None
    progress: Optional[FrameworkProgress] = None
    total_progress: Optional[int] = 0
    completeness: Optional[float] = 0.0
    from_tactics: Optional[bool] = Field(default=False, alias="from_tactics")
    language: Optional[str] = None
    created_ago: Optional[str] = Field(default=None, alias="created_ago")
    de_progress: Optional[str] = Field(default=None, alias="DE_progress")
    st_progress: Optional[str] = Field(default=None, alias="ST_progress")
    category: Optional[str] = None  # From categorized ideas data

    # Dictionary to hold step fields
    # This will store all fields from the idea document regardless of prefix
    steps: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def extract_step_fields(cls, data: Dict) -> Dict:
        """
        Extract all step-related fields into the steps dictionary.

        This allows us to access all step fields (including those with
        ai- and selected- prefixes) through the steps attribute.
        """
        if not isinstance(data, dict):
            return data

        steps = {}

        # Get all step field values for Disciplined Entrepreneurship steps
        de_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        for base_step in de_steps:
            # Check all prefix variations
            for prefix in ["", "ai-", "selected-"]:
                field_name = f"{prefix}{base_step}"
                if field_name in data:
                    steps[field_name] = data[field_name]

        # Get all step field values for Startup Tactics steps
        st_steps = StartupTacticsStep.get_all_step_values()
        for base_step in st_steps:
            # Check all prefix variations
            for prefix in ["", "ai-", "selected-"]:
                field_name = f"{prefix}{base_step}"
                if field_name in data:
                    steps[field_name] = data[field_name]

        # Store in the steps field
        data["steps"] = steps
        return data

    def get_creation_date(self) -> Optional[datetime]:
        """
        Get the idea creation date as a datetime object.

        Returns:
            Optional[datetime]: Creation date
        """
        if isinstance(self.created, DateField):
            return self.created.to_datetime()
        elif isinstance(self.created, datetime):
            return self.created
        return None

    def get_framework_progress(self, framework: FrameworkType) -> float:
        """
        Get the progress percentage for a specific framework.

        Args:
            framework: Framework type enum

        Returns:
            float: Progress percentage (0.0 to 1.0)
        """
        if not self.progress:
            return 0.0

        framework_value = framework.value

        # Access using getattr to handle the spaces in the field names
        progress_value = getattr(
            self.progress, safe_lower(framework_value).replace(" ", "_"), 0.0
        )

        # Ensure it's a decimal between 0 and 1
        if isinstance(progress_value, (int, float)) and progress_value > 1.0:
            return progress_value / 100.0

        return progress_value or 0.0

    def has_step(self, step_name: str, prefix: Optional[StepPrefix] = None) -> bool:
        """
        Check if this idea has a specific step.

        Args:
            step_name: Base step name (without prefix)
            prefix: Optional step prefix

        Returns:
            bool: True if the step exists, False otherwise
        """
        if prefix:
            field_name = f"{prefix.value}{step_name}"
        else:
            field_name = step_name

        return field_name in self.steps and self.steps[field_name] is not None

    def get_step_value(
        self, step_name: str, prefix: Optional[StepPrefix] = None
    ) -> Any:
        """
        Get the value of a specific step.

        Args:
            step_name: Base step name (without prefix)
            prefix: Optional step prefix

        Returns:
            Any: Step value or None if not found
        """
        if prefix:
            field_name = f"{prefix.value}{step_name}"
        else:
            field_name = step_name

        return self.steps.get(field_name)

    def get_all_step_names(self, with_prefix: bool = False) -> List[str]:
        """
        Get all step names available in this idea.

        Args:
            with_prefix: If True, include the prefix in the step names

        Returns:
            List[str]: List of step names
        """
        if with_prefix:
            return list(self.steps.keys())

        # Return unique base step names without prefix
        base_steps = set()
        for step_field in self.steps.keys():
            base_step = StepPrefix.extract_base_step(step_field)
            base_steps.add(base_step)

        return list(base_steps)

    def get_de_steps_count(self) -> int:
        """
        Count the number of Disciplined Entrepreneurship steps in this idea.

        Returns:
            int: Number of DE steps
        """
        count = 0
        for step in DisciplinedEntrepreneurshipStep:
            if self.has_step(step.value):
                count += 1
        return count

    def get_st_steps_count(self) -> int:
        """
        Count the number of Startup Tactics steps in this idea.

        Returns:
            int: Number of ST steps
        """
        count = 0
        for step in StartupTacticsStep:
            if self.has_step(step.value):
                count += 1
        return count

    def get_idea_category(self) -> IdeaCategory:
        """
        Get the category of this idea.

        Returns:
            IdeaCategory: Category enum
        """
        if not self.category:
            return IdeaCategory.OTHER

        # Try to match with defined categories
        for cat in IdeaCategory:
            if safe_lower(cat.value) == safe_lower(self.category):
                return cat

        return IdeaCategory.OTHER
