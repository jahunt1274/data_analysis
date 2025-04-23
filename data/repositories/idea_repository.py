"""
Idea repository for the data analysis system.

This module provides data access and query methods for Idea entities,
including idea categorization and framework progress analysis.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime
from pathlib import Path

from ..models.idea_model import Idea
from ..models.enums import (
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
    StepPrefix,
    IdeaCategory,
)
from .base_repository import BaseRepository
from .user_repository import InMemoryDatabase


class IdeaRepository(BaseRepository[Idea]):
    """
    Repository for Idea data access and analysis.

    This repository provides methods for querying and analyzing idea data,
    including categorization, framework progress, and step completion.
    """

    def __init__(self, db=None, config=None):
        """
        Initialize the idea repository.

        Args:
            db: Optional database connection
            config: Optional configuration object
        """
        super().__init__("ideas", Idea)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._config = config
        # In-memory indexes for faster lookups
        self._owner_index = {}  # Maps user email to list of ideas
        self._category_index = {}  # Maps category to list of ideas
        self._categorized_ideas = {}  # Additional categorization data

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
            if self._config and hasattr(self._config, "IDEA_DATA_PATH"):
                file_path = self._config.IDEA_DATA_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/idea.json"

            # Create in-memory DB-like structure if needed
            if not isinstance(self._db, InMemoryDatabase):
                self._db = InMemoryDatabase()

            self._load_data_from_json(file_path)

            # Load categorized ideas data if available
            self._load_categorized_ideas()

            # Build indexes for faster lookups
            self._build_indexes()
        except Exception as e:
            self._logger.error(f"Error connecting to idea data: {e}")
            raise

    def _load_data_from_json(self, file_path: str) -> None:
        """
        Load idea data from a JSON file.

        Args:
            file_path: Path to the JSON file containing idea data
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
                f"Loaded {self._db.count(self._collection_name, {})} ideas from {file_path}"
            )
        except Exception as e:
            self._logger.error(f"Error loading idea data from {file_path}: {e}")
            raise

    def _load_categorized_ideas(self) -> None:
        """
        Load categorized ideas data from a JSON file.

        This data contains AI-generated categorizations for ideas.
        """
        try:
            if self._config and hasattr(self._config, "CATEGORIZED_IDEAS_PATH"):
                file_path = self._config.CATEGORIZED_IDEAS_PATH
            else:
                # Default path if not specified in config
                file_path = "input/raw/categorized_ideas.json"

            if not Path(file_path).exists():
                self._logger.warning(f"Categorized ideas file not found: {file_path}")
                return

            with open(file_path, "r", encoding="utf-8") as file:
                categorized_data = json.load(file)

            # Create a mapping from idea ID to category
            for item in categorized_data:
                if "idea_id" in item and "category" in item:
                    self._categorized_ideas[item["idea_id"]] = item["category"]

            self._logger.info(
                f"Loaded {len(self._categorized_ideas)} categorized ideas"
            )

            # Update category for existing ideas
            ideas = self.get_all()
            updated_count = 0
            for idea in ideas:
                if idea.id and idea.id.oid in self._categorized_ideas:
                    # Update the model with category data
                    category = self._categorized_ideas[idea.id.oid]
                    idea.category = category
                    updated_count += 1

            self._logger.info(
                f"Updated {updated_count} ideas with category information"
            )
        except Exception as e:
            self._logger.error(f"Error loading categorized ideas data: {e}")

    def _build_indexes(self) -> None:
        """
        Build in-memory indexes for faster lookups.
        """
        try:
            ideas = self.get_all()

            # Build owner index
            for idea in ideas:
                if idea.owner:
                    if idea.owner not in self._owner_index:
                        self._owner_index[idea.owner] = []
                    self._owner_index[idea.owner].append(idea)

            # Build category index
            for idea in ideas:
                category = idea.get_idea_category().value
                if category not in self._category_index:
                    self._category_index[category] = []
                self._category_index[category].append(idea)
        except Exception as e:
            self._logger.error(f"Error building indexes: {e}")

    def find_by_owner(self, email: str) -> List[Idea]:
        """
        Find all ideas created by a specific user.

        Args:
            email: User's email address

        Returns:
            List[Idea]: List of ideas owned by the user
        """
        # Check in-memory index first
        if email in self._owner_index:
            return self._owner_index[email]

        # Fall back to DB query
        return self.find_many({"owner": email})

    def find_by_owner_and_ranking(self, email: str, ranking: int) -> Optional[Idea]:
        """
        Find a specific idea by owner and ranking.

        The ranking indicates the order in which ideas were created by a user.

        Args:
            email: User's email address
            ranking: Idea ranking (1st, 2nd, etc.)

        Returns:
            Optional[Idea]: Idea or None if not found
        """
        ideas = self.find_by_owner(email)
        for idea in ideas:
            if idea.ranking == ranking:
                return idea
        return None

    def find_by_category(self, category: Union[str, IdeaCategory]) -> List[Idea]:
        """
        Find ideas by category.

        Args:
            category: Category name or enum

        Returns:
            List[Idea]: List of ideas in the category
        """
        # Convert to string category name if an enum is provided
        if isinstance(category, IdeaCategory):
            category_name = category.value
        else:
            category_name = category

        # Check in-memory index first
        if category_name in self._category_index:
            return self._category_index[category_name]

        # Fall back to DB query
        return [
            idea
            for idea in self.get_all()
            if idea.category and idea.category.lower() == category_name.lower()
        ]

    def find_by_creation_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Idea]:
        """
        Find ideas created within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List[Idea]: List of ideas created within the date range
        """
        ideas = self.get_all()

        # Filter by creation date
        return [
            idea
            for idea in ideas
            if idea.get_creation_date()
            and start_date <= idea.get_creation_date() <= end_date
        ]

    def find_with_step(
        self, step_name: str, prefix: Optional[StepPrefix] = None
    ) -> List[Idea]:
        """
        Find ideas that have a specific step.

        Args:
            step_name: Step name (e.g., "market-segmentation")
            prefix: Optional step prefix (e.g., StepPrefix.AI)

        Returns:
            List[Idea]: List of ideas with the specified step
        """
        return [idea for idea in self.get_all() if idea.has_step(step_name, prefix)]

    def get_ideas_with_min_progress(
        self, framework: FrameworkType, min_progress: float
    ) -> List[Idea]:
        """
        Get ideas with at least a minimum progress percentage in a framework.

        Args:
            framework: Framework type
            min_progress: Minimum progress percentage (0.0 to 1.0)

        Returns:
            List[Idea]: List of ideas with sufficient progress
        """
        return [
            idea
            for idea in self.get_all()
            if idea.get_framework_progress(framework) >= min_progress
        ]

    def get_ideas_by_progress_level(
        self, framework: FrameworkType, progress_levels: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[Idea]]:
        """
        Group ideas by progress level.

        Args:
            framework: Framework type
            progress_levels: Dict mapping level names to (min, max) progress percentages

        Returns:
            Dict[str, List[Idea]]: Mapping from level name to ideas list
        """
        result = {level: [] for level in progress_levels.keys()}

        for idea in self.get_all():
            progress = idea.get_framework_progress(framework)

            for level, (min_val, max_val) in progress_levels.items():
                if min_val <= progress <= max_val:
                    result[level].append(idea)
                    break

        return result

    def get_ideas_with_completed_step(
        self, step: Union[str, DisciplinedEntrepreneurshipStep]
    ) -> List[Idea]:
        """
        Get ideas that have completed a specific step.

        Args:
            step: Step name or enum

        Returns:
            List[Idea]: List of ideas with the step completed
        """
        if isinstance(step, DisciplinedEntrepreneurshipStep):
            step_name = step.value
        else:
            step_name = step

        # An idea has completed a step if there's an entry for the step (not None)
        return [idea for idea in self.get_all() if idea.has_step(step_name)]

    def get_ideas_by_step_count(
        self, min_steps: int = 0, max_steps: Optional[int] = None
    ) -> List[Idea]:
        """
        Get ideas with a certain number of completed steps.

        Args:
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps (None for unlimited)

        Returns:
            List[Idea]: List of ideas with the specified step count
        """
        result = []

        for idea in self.get_all():
            de_count = idea.get_de_steps_count()
            st_count = idea.get_st_steps_count()
            total_count = de_count + st_count

            if total_count >= min_steps and (
                max_steps is None or total_count <= max_steps
            ):
                result.append(idea)

        return result

    def get_idea_count_by_owner(self) -> Dict[str, int]:
        """
        Get the number of ideas created by each user.

        Returns:
            Dict[str, int]: Mapping from user email to idea count
        """
        result = {}

        for idea in self.get_all():
            if idea.owner:
                if idea.owner in result:
                    result[idea.owner] += 1
                else:
                    result[idea.owner] = 1

        return result

    def get_idea_count_by_category(self) -> Dict[str, int]:
        """
        Get the number of ideas in each category.

        Returns:
            Dict[str, int]: Mapping from category to idea count
        """
        result = {category.value: 0 for category in IdeaCategory}

        for idea in self.get_all():
            category = idea.get_idea_category().value
            result[category] += 1

        return result

    def get_framework_progress_distribution(
        self, framework: FrameworkType, num_bins: int = 10
    ) -> Dict[str, int]:
        """
        Get the distribution of ideas across progress bins.

        Args:
            framework: Framework type
            num_bins: Number of bins (default: 10)

        Returns:
            Dict[str, int]: Mapping from bin label to idea count
        """
        bin_size = 1.0 / num_bins
        bins = {
            f"{int(i*100)}%-{int((i+bin_size)*100)}%": 0
            for i in [j * bin_size for j in range(num_bins)]
        }

        for idea in self.get_all():
            progress = idea.get_framework_progress(framework)

            # Determine bin
            bin_index = min(int(progress / bin_size), num_bins - 1)
            bin_label = list(bins.keys())[bin_index]
            bins[bin_label] += 1

        return bins

    def get_step_completion_rates(self, framework: FrameworkType) -> Dict[str, float]:
        """
        Get the completion rate for each step in a framework.

        Args:
            framework: Framework type

        Returns:
            Dict[str, float]: Mapping from step name to completion rate
        """
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            steps = [step.value for step in DisciplinedEntrepreneurshipStep]
        else:
            # Not implemented for other frameworks
            return {}

        total_ideas = len(self.get_all())
        if total_ideas == 0:
            return {step: 0.0 for step in steps}

        completion_counts = {step: 0 for step in steps}

        for idea in self.get_all():
            for step in steps:
                if idea.has_step(step):
                    completion_counts[step] += 1

        # Calculate rates
        return {step: count / total_ideas for step, count in completion_counts.items()}

    def get_average_steps_per_idea(self) -> float:
        """
        Get the average number of steps completed per idea.

        Returns:
            float: Average number of steps
        """
        ideas = self.get_all()
        if not ideas:
            return 0.0

        total_steps = sum(
            idea.get_de_steps_count() + idea.get_st_steps_count() for idea in ideas
        )

        return total_steps / len(ideas)

    def get_ideas_with_ai_generated_steps(self) -> List[Idea]:
        """
        Get ideas that have AI-generated steps.

        Returns:
            List[Idea]: List of ideas with AI-generated steps
        """
        result = []

        for idea in self.get_all():
            # Check for any step with AI prefix
            has_ai_step = any(
                step.startswith(StepPrefix.AI.value)
                for step in idea.get_all_step_names(with_prefix=True)
            )

            if has_ai_step:
                result.append(idea)

        return result

    def get_popular_step_sequences(
        self, max_sequences: int = 10
    ) -> List[Tuple[List[str], int]]:
        """
        Get the most popular step sequences.

        Args:
            max_sequences: Maximum number of sequences to return

        Returns:
            List[Tuple[List[str], int]]: List of (step sequence, count) tuples
        """
        sequence_counts = {}

        for idea in self.get_all():
            # Get steps in order
            steps = []
            for step in DisciplinedEntrepreneurshipStep:
                if idea.has_step(step.value):
                    steps.append(step.value)

            if steps:
                seq_tuple = tuple(steps)
                if seq_tuple in sequence_counts:
                    sequence_counts[seq_tuple] += 1
                else:
                    sequence_counts[seq_tuple] = 1

        # Sort by count (descending)
        sorted_sequences = sorted(
            [(list(seq), count) for seq, count in sequence_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_sequences[:max_sequences]

    def get_idea_category_by_id(self, idea_id: str) -> Optional[str]:
        """
        Get the category for a specific idea by ID.

        Args:
            idea_id: Idea ID

        Returns:
            Optional[str]: Category name or None
        """
        if idea_id in self._categorized_ideas:
            return self._categorized_ideas[idea_id]
        return None

    def get_idea_owner_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of idea counts per owner.

        Returns:
            Dict[int, int]: Mapping from idea count to owner count
        """
        idea_counts = self.get_idea_count_by_owner()
        distribution = {}

        for count in idea_counts.values():
            if count in distribution:
                distribution[count] += 1
            else:
                distribution[count] = 1

        return distribution

    def get_ideas_by_keyword(self, keyword: str) -> List[Idea]:
        """
        Get ideas containing a specific keyword in title or description.

        Args:
            keyword: Keyword to search for

        Returns:
            List[Idea]: List of matching ideas
        """
        keyword = keyword.lower()
        return [
            idea
            for idea in self.get_all()
            if (idea.title and keyword in idea.title.lower())
            or (idea.description and keyword in idea.description.lower())
        ]

    def get_category_step_completion_correlation(self) -> Dict[str, Dict[str, float]]:
        """
        Get the correlation between idea categories and step completion rates.

        Returns:
            Dict[str, Dict[str, float]]: Mapping from category to step completion rates
        """
        result = {category.value: {} for category in IdeaCategory}

        # Group ideas by category
        ideas_by_category = {
            category: self.find_by_category(category)
            for category in [cat.value for cat in IdeaCategory]
        }

        # Calculate step completion rates for each category
        for category, ideas in ideas_by_category.items():
            if not ideas:
                continue

            for step in DisciplinedEntrepreneurshipStep:
                step_name = step.value
                completed_count = sum(1 for idea in ideas if idea.has_step(step_name))
                result[category][step_name] = completed_count / len(ideas)

        return result
