"""
Framework analyzer for the data analysis system.

This module provides functionality for analyzing framework progression
through the JetPack/Orbit tool, including step completion rates, common
pathways, progression patterns, and framework effectiveness.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import itertools
import math
from enum import Enum

from ..data.data_repository import DataRepository
from ..data.models.enums import (
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
    StartupTacticsStep,
    Semester,
    ToolVersion,
)


class ProgressionType(str, Enum):
    """Types of progression patterns through frameworks."""

    LINEAR = "linear"  # Steps completed in order (1, 2, 3, 4...)
    JUMPING = "jumping"  # Skipping steps (1, 3, 5...)
    NONLINEAR = "nonlinear"  # Out of order steps (3, 1, 4, 2...)
    FOCUSED = "focused"  # Concentrating on specific areas
    COMPREHENSIVE = "comprehensive"  # Completing most steps
    MINIMAL = "minimal"  # Completing very few steps


class FrameworkAnalyzer:
    """
    Analyzer for framework progression in the JetPack/Orbit tool.

    This class provides methods for analyzing how users progress through
    entrepreneurial frameworks, identifying patterns, bottlenecks, and
    effectiveness of different steps.
    """

    def __init__(self, data_repository: DataRepository):
        """
        Initialize the framework analyzer.

        Args:
            data_repository: Data repository for accessing all entity repositories
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._data_repo = data_repository

        # Ensure data is loaded
        self._data_repo.connect()

    def get_framework_completion_metrics(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate framework step completion metrics.

        This method analyzes how many users complete each step in the framework
        and the overall completion rates.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users (e.g., "15.390")
            include_ideas_without_steps: Whether to include ideas with no steps in calculations

        Returns:
            Dict with framework completion metrics
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "step_completion_rates": {},
            "overall_metrics": {},
            "user_completion_distribution": {},
            "idea_completion_distribution": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Initialize step completion counts
        step_counts = {step: 0 for step in framework_steps}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Filter out users without email
        valid_users = [user for user in users if user.email]

        # Track user and idea completion
        user_completion_counts = defaultdict(int)  # Maps user to completed step count
        idea_completion_counts = defaultdict(
            int
        )  # Maps idea_id to completed step count

        # Track total ideas and ideas with steps
        total_ideas = 0
        ideas_with_steps = 0

        # Analyze each user's ideas
        for user in valid_users:
            user_ideas = self._data_repo.ideas.find_by_owner(user.email)
            total_ideas += len(user_ideas)

            # Track user's max step completion across all ideas
            user_max_completed = 0

            for idea in user_ideas:
                # Count completed steps for this idea
                completed_steps = 0
                completed_step_names = set()

                for step_name in framework_steps:
                    if idea.has_step(step_name):
                        step_counts[step_name] += 1
                        completed_steps += 1
                        completed_step_names.add(step_name)

                # Update idea completion count
                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    idea_completion_counts[idea_id] = completed_steps

                # Count idea with steps
                if completed_steps > 0:
                    ideas_with_steps += 1

                # Update user's max completion
                user_max_completed = max(user_max_completed, completed_steps)

            # Update user completion count
            user_completion_counts[user.email] = user_max_completed

        # Calculate step completion rates
        total_ideas_for_rate = (
            total_ideas if include_ideas_without_steps else ideas_with_steps
        )
        if total_ideas_for_rate > 0:
            for step_name in framework_steps:
                completion_rate = step_counts[step_name] / total_ideas_for_rate
                result["step_completion_rates"][step_name] = {
                    "count": step_counts[step_name],
                    "rate": completion_rate,
                    "step_number": self._get_step_number(step_name, framework),
                }

        # Calculate overall metrics
        result["overall_metrics"] = {
            "total_users": len(valid_users),
            "total_ideas": total_ideas,
            "ideas_with_steps": ideas_with_steps,
            "idea_to_step_conversion_rate": (
                ideas_with_steps / total_ideas if total_ideas > 0 else 0
            ),
            "avg_steps_per_idea": (
                sum(step_counts.values()) / total_ideas_for_rate
                if total_ideas_for_rate > 0
                else 0
            ),
            "avg_completion_percentage": (
                sum(step_counts.values())
                / (total_ideas_for_rate * len(framework_steps))
                if total_ideas_for_rate > 0 and framework_steps
                else 0
            )
            * 100,
        }

        # Group user completion into ranges
        user_ranges = self._group_into_ranges(
            user_completion_counts.values(),
            max_value=len(framework_steps),
            num_ranges=5,
        )
        result["user_completion_distribution"] = user_ranges

        # Group idea completion into ranges
        idea_ranges = self._group_into_ranges(
            idea_completion_counts.values(),
            max_value=len(framework_steps),
            num_ranges=5,
        )
        result["idea_completion_distribution"] = idea_ranges

        return result

    def identify_common_progression_patterns(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_steps: int = 3,
        max_patterns: int = 5,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Identify common progression patterns through framework steps.

        This method analyzes the order in which users complete steps and
        identifies frequently occurring patterns.

        Args:
            framework: The framework to analyze
            min_steps: Minimum number of steps to consider a pattern
            max_patterns: Maximum number of patterns to return
            course_id: Optional course ID to filter users

        Returns:
            Dict with common progression patterns
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "common_patterns": [],
            "progression_types": {},
            "starting_points": {},
            "ending_points": {},
        }

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]
        else:
            user_emails = None

        # Get all step sequences for this framework
        step_sequences = self._get_step_sequences(framework, user_emails)

        # Filter sequences by minimum length
        step_sequences = [seq for seq in step_sequences if len(seq) >= min_steps]

        if not step_sequences:
            return {
                "framework": framework.value,
                "error": f"No progression patterns found with at least {min_steps} steps",
            }

        # Find common patterns
        pattern_counts = defaultdict(int)

        for sequence in step_sequences:
            # Consider all subsequences of appropriate length
            for i in range(len(sequence) - min_steps + 1):
                for length in range(
                    min_steps, min(len(sequence) - i + 1, min_steps + 3)
                ):
                    subseq = tuple(sequence[i : i + length])
                    pattern_counts[subseq] += 1

        # Sort by frequency
        sorted_patterns = sorted(
            [(list(pattern), count) for pattern, count in pattern_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Prepare common patterns data
        for pattern, count in sorted_patterns[:max_patterns]:
            # Calculate coverage (percentage of sequences containing this pattern)
            coverage = count / len(step_sequences)

            # Determine pattern type
            pattern_type = self._classify_progression_pattern(pattern, framework)

            result["common_patterns"].append(
                {
                    "steps": pattern,
                    "step_names": [
                        self._get_step_display_name(step, framework) for step in pattern
                    ],
                    "count": count,
                    "coverage": coverage,
                    "pattern_type": pattern_type,
                }
            )

        # Analyze progression types across all sequences
        progression_types = {
            progression_type.value: 0 for progression_type in ProgressionType
        }

        for sequence in step_sequences:
            pattern_type = self._classify_progression_pattern(sequence, framework)
            progression_types[pattern_type] += 1

        # Calculate percentages
        total_sequences = len(step_sequences)
        result["progression_types"] = {
            prog_type: {
                "count": count,
                "percentage": count / total_sequences if total_sequences > 0 else 0,
            }
            for prog_type, count in progression_types.items()
        }

        # Analyze common starting points
        starting_points = Counter(seq[0] for seq in step_sequences if seq)
        result["starting_points"] = {
            step: {
                "count": count,
                "percentage": count / total_sequences if total_sequences > 0 else 0,
                "step_name": self._get_step_display_name(step, framework),
            }
            for step, count in starting_points.most_common(5)
        }

        # Analyze common ending points
        ending_points = Counter(seq[-1] for seq in step_sequences if seq)
        result["ending_points"] = {
            step: {
                "count": count,
                "percentage": count / total_sequences if total_sequences > 0 else 0,
                "step_name": self._get_step_display_name(step, framework),
            }
            for step, count in ending_points.most_common(5)
        }

        return result

    def analyze_step_dependencies(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_correlation: float = 0.3,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze dependencies between framework steps.

        This method identifies which steps tend to be completed together
        and which steps tend to follow others.

        Args:
            framework: The framework to analyze
            min_correlation: Minimum correlation coefficient to include in results
            course_id: Optional course ID to filter users

        Returns:
            Dict with step dependency analysis
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "step_correlations": {},
            "sequential_patterns": {},
            "prerequisites": {},
            "dependent_steps": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]
        else:
            user_emails = None

        # Get all ideas with at least one step in this framework
        ideas = self._get_ideas_with_framework_steps(framework, user_emails)

        if not ideas:
            return {
                "framework": framework.value,
                "error": "No ideas found with steps in this framework",
            }

        # Build step completion matrix
        # For each idea, track which steps were completed
        step_completion = {step: [] for step in framework_steps}

        for idea in ideas:
            for step in framework_steps:
                step_completion[step].append(1 if idea.has_step(step) else 0)

        # Calculate correlations between steps
        step_correlations = {}

        for step1, step2 in itertools.combinations(framework_steps, 2):
            correlation = self._calculate_correlation(
                step_completion[step1], step_completion[step2]
            )

            if abs(correlation) >= min_correlation:
                if step1 not in step_correlations:
                    step_correlations[step1] = {}

                step_correlations[step1][step2] = correlation

        # Format correlations for output
        formatted_correlations = {}
        for step1, correlations in step_correlations.items():
            step1_name = self._get_step_display_name(step1, framework)
            formatted_correlations[step1_name] = {
                self._get_step_display_name(step2, framework): correlation
                for step2, correlation in correlations.items()
            }

        result["step_correlations"] = formatted_correlations

        # Analyze step sequences to find prerequisites and dependencies
        step_sequences = self._get_step_sequences(framework, user_emails)

        # Count how often step2 follows step1
        sequence_counts = defaultdict(lambda: defaultdict(int))
        total_sequences = len(step_sequences)

        for sequence in step_sequences:
            for i in range(len(sequence) - 1):
                step1 = sequence[i]
                step2 = sequence[i + 1]
                sequence_counts[step1][step2] += 1

        # Calculate probability that step2 follows step1
        sequential_patterns = {}

        for step1, followers in sequence_counts.items():
            step1_count = sum(1 for seq in step_sequences if step1 in seq)

            if step1_count > 0:
                sequential_patterns[step1] = {
                    step2: count / step1_count
                    for step2, count in followers.items()
                    if count / step1_count >= min_correlation
                }

        # Format sequential patterns for output
        formatted_patterns = {}
        for step1, patterns in sequential_patterns.items():
            step1_name = self._get_step_display_name(step1, framework)
            formatted_patterns[step1_name] = {
                self._get_step_display_name(step2, framework): probability
                for step2, probability in patterns.items()
            }

        result["sequential_patterns"] = formatted_patterns

        # Identify prerequisites (steps that typically come before a given step)
        prerequisites = {}

        for step in framework_steps:
            step_prerequisites = defaultdict(int)
            step_occurrences = 0

            for sequence in step_sequences:
                if step in sequence:
                    step_index = sequence.index(step)
                    step_occurrences += 1

                    # Count all steps that come before this step
                    for i in range(step_index):
                        step_prerequisites[sequence[i]] += 1

            if step_occurrences > 0:
                prerequisites[step] = {
                    prereq: count / step_occurrences
                    for prereq, count in step_prerequisites.items()
                    if count / step_occurrences >= min_correlation
                }

        # Format prerequisites for output
        formatted_prerequisites = {}
        for step, prereqs in prerequisites.items():
            if prereqs:  # Only include steps with prerequisites
                step_name = self._get_step_display_name(step, framework)
                formatted_prerequisites[step_name] = {
                    self._get_step_display_name(prereq, framework): probability
                    for prereq, probability in prereqs.items()
                }

        result["prerequisites"] = formatted_prerequisites

        # Identify dependent steps (steps that typically come after a given step)
        dependent_steps = {}

        for step in framework_steps:
            step_dependents = defaultdict(int)
            step_occurrences = 0

            for sequence in step_sequences:
                if step in sequence:
                    step_index = sequence.index(step)
                    step_occurrences += 1

                    # Count all steps that come after this step
                    for i in range(step_index + 1, len(sequence)):
                        step_dependents[sequence[i]] += 1

            if step_occurrences > 0:
                dependent_steps[step] = {
                    dependent: count / step_occurrences
                    for dependent, count in step_dependents.items()
                    if count / step_occurrences >= min_correlation
                }

        # Format dependent steps for output
        formatted_dependents = {}
        for step, dependents in dependent_steps.items():
            if dependents:  # Only include steps with dependents
                step_name = self._get_step_display_name(step, framework)
                formatted_dependents[step_name] = {
                    self._get_step_display_name(dependent, framework): probability
                    for dependent, probability in dependents.items()
                }

        result["dependent_steps"] = formatted_dependents

        return result

    def get_framework_dropout_analysis(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        min_idea_count: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze where users tend to stop progressing through the framework.

        This method identifies common dropout points and factors that
        may contribute to users abandoning the framework.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            min_idea_count: Minimum number of ideas needed for analysis

        Returns:
            Dict with framework dropout analysis
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "dropout_by_step": {},
            "completion_bottlenecks": [],
            "abandonment_factors": {},
            "recommendation": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
            # Sort steps by their number
            framework_steps.sort(
                key=lambda s: DisciplinedEntrepreneurshipStep.get_step_number(s)
            )
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]
        else:
            user_emails = None

        # Get all ideas with at least one step in this framework
        ideas = self._get_ideas_with_framework_steps(framework, user_emails)

        if len(ideas) < min_idea_count:
            return {
                "framework": framework.value,
                "error": f"Not enough ideas for analysis (found {len(ideas)}, need at least {min_idea_count})",
            }

        # Analyze where users stop in the framework
        dropout_counts = defaultdict(int)
        total_ideas = len(ideas)

        for idea in ideas:
            # Find the last step completed
            last_step = None
            last_step_number = -1

            for step in framework_steps:
                if idea.has_step(step):
                    step_number = self._get_step_number(step, framework)
                    if step_number > last_step_number:
                        last_step = step
                        last_step_number = step_number

            if last_step:
                dropout_counts[last_step] += 1

        # Calculate dropout rates
        for step in framework_steps:
            step_display_name = self._get_step_display_name(step, framework)
            step_number = self._get_step_number(step, framework)

            result["dropout_by_step"][step_display_name] = {
                "count": dropout_counts.get(step, 0),
                "percentage": (
                    dropout_counts.get(step, 0) / total_ideas if total_ideas > 0 else 0
                ),
                "step_number": step_number,
            }

        # Find completion bottlenecks (steps with high completion drop-off)
        bottlenecks = []

        # Get completion rates for each step
        completion_rates = {
            step: sum(1 for idea in ideas if idea.has_step(step)) / total_ideas
            for step in framework_steps
        }

        # Look for significant drops in completion rates between consecutive steps
        for i in range(1, len(framework_steps)):
            prev_step = framework_steps[i - 1]
            curr_step = framework_steps[i]

            prev_rate = completion_rates[prev_step]
            curr_rate = completion_rates[curr_step]

            # Calculate drop-off
            drop_off = prev_rate - curr_rate

            if drop_off > 0.15:  # Significant drop-off threshold
                prev_name = self._get_step_display_name(prev_step, framework)
                curr_name = self._get_step_display_name(curr_step, framework)

                bottlenecks.append(
                    {
                        "from_step": prev_name,
                        "to_step": curr_name,
                        "drop_off_rate": drop_off,
                        "from_completion_rate": prev_rate,
                        "to_completion_rate": curr_rate,
                        "severity": "high" if drop_off > 0.3 else "medium",
                    }
                )

        # Sort bottlenecks by drop-off rate (descending)
        bottlenecks.sort(key=lambda x: x["drop_off_rate"], reverse=True)
        result["completion_bottlenecks"] = bottlenecks

        # Analyze potential abandonment factors
        result["abandonment_factors"] = self._analyze_abandonment_factors(
            ideas, framework, framework_steps, dropout_counts
        )

        # Generate recommendations for improving completion
        result["recommendation"] = self._generate_framework_recommendations(
            bottlenecks, completion_rates, framework_steps, framework
        )

        return result

    def calculate_step_time_intervals(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_version_comparison: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate time intervals between framework steps.

        This method analyzes how long users take between completing different
        steps and identifies patterns in progression timing.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            include_version_comparison: Whether to include comparison between tool versions

        Returns:
            Dict with step time interval analysis
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "overall_metrics": {},
            "step_interval_metrics": {},
            "interval_distribution": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]
        else:
            user_emails = None

        # Collect all step creation timestamps by idea
        idea_step_timestamps = {}

        # Get all ideas with steps in this framework
        ideas = self._get_ideas_with_framework_steps(framework, user_emails)

        for idea in ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

            # Get all steps for this idea
            steps = self._data_repo.steps.find_by_idea_id(idea_id)

            # Filter to steps in this framework with valid creation dates
            framework_steps_with_dates = [
                step
                for step in steps
                if step.framework == framework.value and step.get_creation_date()
            ]

            if not framework_steps_with_dates:
                continue

            # Create timestamp dictionary for this idea
            idea_step_timestamps[idea_id] = {}

            for step in framework_steps_with_dates:
                if step.step:
                    # Use the earliest timestamp if there are multiple versions of the same step
                    current_timestamp = idea_step_timestamps[idea_id].get(step.step)
                    step_timestamp = step.get_creation_date()

                    if current_timestamp is None or step_timestamp < current_timestamp:
                        idea_step_timestamps[idea_id][step.step] = step_timestamp

        # Calculate intervals between consecutive steps
        all_intervals = []  # List of all intervals in minutes
        step_pair_intervals = defaultdict(
            list
        )  # Maps (step1, step2) to list of intervals

        for idea_id, timestamps in idea_step_timestamps.items():
            if len(timestamps) < 2:
                continue

            # Sort steps by timestamp
            sorted_steps = sorted(timestamps.items(), key=lambda x: x[1])

            # Calculate intervals between consecutive steps
            for i in range(1, len(sorted_steps)):
                prev_step, prev_time = sorted_steps[i - 1]
                curr_step, curr_time = sorted_steps[i]

                # Calculate interval in minutes
                interval_minutes = (curr_time - prev_time).total_seconds() / 60

                all_intervals.append(interval_minutes)
                step_pair_intervals[(prev_step, curr_step)].append(interval_minutes)

        # Calculate overall metrics
        if all_intervals:
            result["overall_metrics"] = {
                "total_intervals": len(all_intervals),
                "avg_interval_minutes": sum(all_intervals) / len(all_intervals),
                "median_interval_minutes": sorted(all_intervals)[
                    len(all_intervals) // 2
                ],
                "min_interval_minutes": min(all_intervals),
                "max_interval_minutes": max(all_intervals),
            }

            # Group intervals into categories
            interval_categories = {
                "under_5min": sum(1 for i in all_intervals if i < 5),
                "5_15min": sum(1 for i in all_intervals if 5 <= i < 15),
                "15_30min": sum(1 for i in all_intervals if 15 <= i < 30),
                "30_60min": sum(1 for i in all_intervals if 30 <= i < 60),
                "1_3hr": sum(1 for i in all_intervals if 60 <= i < 180),
                "3_24hr": sum(1 for i in all_intervals if 180 <= i < 1440),
                "over_24hr": sum(1 for i in all_intervals if i >= 1440),
            }

            # Calculate percentages
            interval_distribution = {
                category: {"count": count, "percentage": count / len(all_intervals)}
                for category, count in interval_categories.items()
            }

            result["interval_distribution"] = interval_distribution

        # Calculate metrics for each step pair
        step_interval_metrics = {}

        for (step1, step2), intervals in step_pair_intervals.items():
            if not intervals:
                continue

            step1_name = self._get_step_display_name(step1, framework)
            step2_name = self._get_step_display_name(step2, framework)

            metrics = {
                "count": len(intervals),
                "avg_minutes": sum(intervals) / len(intervals),
                "median_minutes": sorted(intervals)[len(intervals) // 2],
                "min_minutes": min(intervals),
                "max_minutes": max(intervals),
            }

            # Add speed classification
            if metrics["avg_minutes"] < 15:
                metrics["speed"] = "very_fast"
            elif metrics["avg_minutes"] < 60:
                metrics["speed"] = "fast"
            elif metrics["avg_minutes"] < 360:  # 6 hours
                metrics["speed"] = "moderate"
            elif metrics["avg_minutes"] < 1440:  # 24 hours
                metrics["speed"] = "slow"
            else:
                metrics["speed"] = "very_slow"

            step_interval_metrics[f"{step1_name} → {step2_name}"] = metrics

        result["step_interval_metrics"] = step_interval_metrics

        # Add version comparison if requested
        if include_version_comparison:
            result["version_comparison"] = self._compare_step_intervals_by_version(
                framework, user_emails
            )

        return result

    def compare_framework_progression_by_cohort(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        cohorts: Optional[List[Semester]] = None,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare framework progression patterns between different cohorts.

        This method analyzes how different cohorts of users progress through
        the framework and identifies trends or improvements over time.

        Args:
            framework: The framework to analyze
            cohorts: Optional list of semesters to compare (defaults to all)
            course_id: Optional course ID to filter users

        Returns:
            Dict with cohort comparison analysis
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "cohort_metrics": {},
            "completion_trends": {},
            "tool_version_impact": {},
        }

        # Use all semesters if none specified
        if not cohorts:
            cohorts = list(Semester)

        # Get semester date ranges
        semester_ranges = {
            Semester.FALL_2023: (datetime(2023, 9, 1), datetime(2023, 12, 31)),
            Semester.SPRING_2024: (datetime(2024, 1, 1), datetime(2024, 5, 31)),
            Semester.FALL_2024: (datetime(2024, 9, 1), datetime(2024, 12, 31)),
            Semester.SPRING_2025: (datetime(2025, 1, 1), datetime(2025, 5, 31)),
        }

        # Get all steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get all ideas with steps (to be filtered by cohort)
        all_ideas = self._get_ideas_with_framework_steps(framework)

        # Analyze each cohort
        for semester in cohorts:
            semester_name = semester.value
            tool_version = Semester.get_tool_version(semester).value
            date_range = semester_ranges.get(semester)

            if not date_range:
                continue

            # Filter ideas by creation date within this semester
            semester_ideas = [
                idea
                for idea in all_ideas
                if idea.get_creation_date()
                and date_range[0] <= idea.get_creation_date() <= date_range[1]
            ]

            # Skip if no ideas for this semester
            if not semester_ideas:
                continue

            # Get user emails for this semester
            semester_user_emails = set(
                idea.owner for idea in semester_ideas if idea.owner
            )

            # Filter by course if specified
            if course_id:
                course_users = set(
                    user.email
                    for user in self._data_repo.users.find_by_course(course_id)
                    if user.email
                )
                semester_user_emails = semester_user_emails.intersection(course_users)

                # Re-filter ideas by course users
                semester_ideas = [
                    idea
                    for idea in semester_ideas
                    if idea.owner and idea.owner in semester_user_emails
                ]

            # Calculate step completion rates for this semester
            step_completion = {}
            for step in framework_steps:
                completed_count = sum(
                    1 for idea in semester_ideas if idea.has_step(step)
                )
                step_completion[step] = {
                    "count": completed_count,
                    "rate": (
                        completed_count / len(semester_ideas) if semester_ideas else 0
                    ),
                    "step_name": self._get_step_display_name(step, framework),
                }

            # Calculate overall metrics
            total_steps_completed = sum(
                data["count"] for data in step_completion.values()
            )
            avg_steps_per_idea = (
                total_steps_completed / len(semester_ideas) if semester_ideas else 0
            )
            completion_percentage = (
                total_steps_completed / (len(semester_ideas) * len(framework_steps))
                if semester_ideas and framework_steps
                else 0
            ) * 100

            # Get step sequences for this semester
            step_sequences = []
            for idea in semester_ideas:
                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

                    # Filter to framework steps with creation dates
                    valid_steps = [
                        step
                        for step in idea_steps
                        if step.framework == framework.value
                        and step.step
                        and step.get_creation_date()
                    ]

                    if valid_steps:
                        # Sort steps by creation date
                        sorted_steps = sorted(
                            valid_steps, key=lambda s: s.get_creation_date()
                        )

                        # Extract step names
                        step_sequence = [step.step for step in sorted_steps]
                        step_sequences.append(step_sequence)

            # Calculate progression type distribution
            progression_types = {
                progression_type.value: 0 for progression_type in ProgressionType
            }

            for sequence in step_sequences:
                if sequence:
                    pattern_type = self._classify_progression_pattern(
                        sequence, framework
                    )
                    progression_types[pattern_type] += 1

            # Calculate percentages
            progression_distribution = {}
            total_sequences = len(step_sequences)

            if total_sequences > 0:
                progression_distribution = {
                    prog_type: {"count": count, "percentage": count / total_sequences}
                    for prog_type, count in progression_types.items()
                }

            # Add cohort metrics to result
            result["cohort_metrics"][semester_name] = {
                "tool_version": tool_version,
                "user_count": len(semester_user_emails),
                "idea_count": len(semester_ideas),
                "avg_steps_per_idea": avg_steps_per_idea,
                "completion_percentage": completion_percentage,
                "step_completion": step_completion,
                "progression_distribution": progression_distribution,
            }

        # Skip trend analysis if fewer than 2 cohorts
        if len(result["cohort_metrics"]) < 2:
            return result

        # Analyze completion trends over time
        completion_trends = {}

        # Sort cohorts by time
        sorted_cohorts = sorted(
            result["cohort_metrics"].keys(),
            key=lambda x: (
                int(x.split()[1]),  # Year
                0 if x.split()[0] == "Spring" else 1,  # Term (Spring before Fall)
            ),
        )

        # Calculate trends for each step
        for step in framework_steps:
            step_name = self._get_step_display_name(step, framework)
            completion_trend = []

            for cohort in sorted_cohorts:
                cohort_metrics = result["cohort_metrics"].get(cohort)
                if not cohort_metrics:
                    continue

                step_data = cohort_metrics.get("step_completion", {}).get(step)
                if not step_data:
                    continue

                completion_trend.append(
                    {
                        "cohort": cohort,
                        "completion_rate": step_data.get("rate", 0),
                        "count": step_data.get("count", 0),
                    }
                )

            # Only include steps with data from multiple cohorts
            if len(completion_trend) >= 2:
                # Calculate trend direction
                first_rate = completion_trend[0]["completion_rate"]
                last_rate = completion_trend[-1]["completion_rate"]

                trend_data = {
                    "trend_points": completion_trend,
                    "overall_change": last_rate - first_rate,
                    "trend_direction": (
                        "increasing"
                        if last_rate > first_rate
                        else "stable" if last_rate == first_rate else "decreasing"
                    ),
                }

                completion_trends[step_name] = trend_data

        result["completion_trends"] = completion_trends

        # Analyze tool version impact
        tool_version_impact = self._analyze_tool_version_impact(
            framework, result["cohort_metrics"]
        )
        result["tool_version_impact"] = tool_version_impact

        return result

    def get_framework_effectiveness_metrics(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_category_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate metrics to assess the effectiveness of the framework.

        This method analyzes factors like step utility, user satisfaction,
        and correlation with idea quality.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            include_category_analysis: Whether to include analysis by idea category

        Returns:
            Dict with framework effectiveness metrics
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "step_utility_metrics": {},
            "user_progression_metrics": {},
            "framework_impact": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = DisciplinedEntrepreneurshipStep.get_all_step_values()
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = StartupTacticsStep.get_all_step_values()
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]
        else:
            user_emails = None

        # Get ideas with steps in the framework
        ideas_with_steps = self._get_ideas_with_framework_steps(framework, user_emails)

        if not ideas_with_steps:
            return {
                "framework": framework.value,
                "error": "No ideas found with steps in this framework",
            }

        # Analyze step user input rates
        step_user_input = self._data_repo.steps.get_user_input_rate_by_step_type(
            framework
        )

        # Prepare step utility metrics
        step_utility_metrics = {}

        for step in framework_steps:
            # Count ideas with this step
            step_count = sum(1 for idea in ideas_with_steps if idea.has_step(step))

            # Skip steps with no data
            if step_count == 0:
                continue

            step_name = self._get_step_display_name(step, framework)

            step_utility_metrics[step_name] = {
                "usage_count": step_count,
                "usage_rate": step_count / len(ideas_with_steps),
                "user_input_rate": step_user_input.get(step, 0),
            }

        result["step_utility_metrics"] = step_utility_metrics

        # Analyze user progression metrics
        user_progression = {}

        # Group ideas by user
        ideas_by_user = defaultdict(list)
        for idea in ideas_with_steps:
            if idea.owner:
                ideas_by_user[idea.owner].append(idea)

        for user_email, user_ideas in ideas_by_user.items():
            # Skip users with only one idea
            if len(user_ideas) < 2:
                continue

            # Sort ideas by creation date
            sorted_ideas = sorted(
                [idea for idea in user_ideas if idea.get_creation_date()],
                key=lambda idea: idea.get_creation_date(),
            )

            # Analyze progression between consecutive ideas
            for i in range(1, len(sorted_ideas)):
                prev_idea = sorted_ideas[i - 1]
                curr_idea = sorted_ideas[i]

                # Count completed steps for each idea
                prev_steps = sum(
                    1 for step in framework_steps if prev_idea.has_step(step)
                )
                curr_steps = sum(
                    1 for step in framework_steps if curr_idea.has_step(step)
                )

                # Calculate progression metrics
                step_change = curr_steps - prev_steps
                progression_rate = (
                    curr_steps / prev_steps if prev_steps > 0 else float("inf")
                )

                # Add to user progression data
                if user_email not in user_progression:
                    user_progression[user_email] = []

                user_progression[user_email].append(
                    {
                        "prev_idea_steps": prev_steps,
                        "curr_idea_steps": curr_steps,
                        "step_change": step_change,
                        "progression_rate": (
                            progression_rate
                            if progression_rate != float("inf")
                            else None
                        ),
                    }
                )

        # Calculate aggregate progression metrics
        progression_metrics = {
            "users_with_multiple_ideas": len(user_progression),
            "avg_step_change": 0,
            "improvement_rate": 0,
        }

        step_changes = []
        improvement_count = 0
        total_comparisons = 0

        for user_email, progressions in user_progression.items():
            for prog in progressions:
                if prog["step_change"] is not None:
                    step_changes.append(prog["step_change"])

                    if prog["step_change"] > 0:
                        improvement_count += 1

                    total_comparisons += 1

        if step_changes:
            progression_metrics["avg_step_change"] = sum(step_changes) / len(
                step_changes
            )

        if total_comparisons > 0:
            progression_metrics["improvement_rate"] = (
                improvement_count / total_comparisons
            )

        result["user_progression_metrics"] = progression_metrics

        # Analyze framework impact
        # This would ideally use more concrete outcome metrics, but we'll use proxy measures

        # 1. Correlation between step count and content/completion scores
        step_score_correlation = self._calculate_step_score_correlation(
            ideas_with_steps, framework, framework_steps
        )

        # 2. User engagement levels for users with different step counts
        engagement_by_step_count = self._group_users_by_step_completion(
            ideas_with_steps, framework, framework_steps, user_emails
        )

        result["framework_impact"] = {
            "step_score_correlation": step_score_correlation,
            "engagement_by_step_count": engagement_by_step_count,
        }

        # Add category analysis if requested
        if include_category_analysis:
            result["category_analysis"] = self._analyze_framework_by_category(
                ideas_with_steps, framework, framework_steps
            )

        return result

    def _get_ideas_with_framework_steps(
        self,
        framework: FrameworkType,
        user_emails: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Get all ideas with at least one step in the specified framework.

        Args:
            framework: The framework to filter by
            user_emails: Optional list of user emails to filter by

        Returns:
            List of ideas with steps in the framework
        """
        # Get all ideas
        if user_emails:
            # Get ideas for specific users
            all_ideas = []
            for email in user_emails:
                user_ideas = self._data_repo.ideas.find_by_owner(email)
                all_ideas.extend(user_ideas)
        else:
            # Get all ideas
            all_ideas = self._data_repo.ideas.get_all()

        # Filter to ideas with at least one step in the framework
        result = []

        for idea in all_ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

            # Get steps for this idea
            steps = self._data_repo.steps.find_by_idea_id(idea_id)

            # Check if any steps belong to the specified framework
            has_framework_step = any(
                step.framework == framework.value for step in steps
            )

            if has_framework_step:
                result.append(idea)

        return result

    def _get_step_sequences(
        self,
        framework: FrameworkType,
        user_emails: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Get sequences of steps completed for all ideas in the framework.

        Args:
            framework: The framework to analyze
            user_emails: Optional list of user emails to filter by

        Returns:
            List of step sequences (each a list of step names in order of completion)
        """
        # Get ideas with steps in the framework
        ideas = self._get_ideas_with_framework_steps(framework, user_emails)

        # Extract step sequences
        sequences = []

        for idea in ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

            # Get steps for this idea
            steps = self._data_repo.steps.find_by_idea_id(idea_id)

            # Filter to framework steps with creation dates
            framework_steps = [
                step
                for step in steps
                if step.framework == framework.value
                and step.step
                and step.get_creation_date()
            ]

            if not framework_steps:
                continue

            # Sort steps by creation date
            sorted_steps = sorted(framework_steps, key=lambda s: s.get_creation_date())

            # Extract step names
            step_sequence = [step.step for step in sorted_steps]

            # Add to sequences
            if step_sequence:
                sequences.append(step_sequence)

        return sequences

    def _get_step_number(self, step_name: str, framework: FrameworkType) -> int:
        """
        Get the numeric position of a step in the framework.

        Args:
            step_name: The step name
            framework: The framework

        Returns:
            int: The step number (1-based) or 0 if not found
        """
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            return DisciplinedEntrepreneurshipStep.get_step_number(step_name)

        # For other frameworks, use alphabetical order for now
        if framework == FrameworkType.STARTUP_TACTICS:
            steps = StartupTacticsStep.get_all_step_values()
            steps.sort()
            try:
                return steps.index(step_name) + 1
            except ValueError:
                return 0

        return 0

    def _get_step_display_name(self, step_name: str, framework: FrameworkType) -> str:
        """
        Convert a step name to a more readable display name.

        Args:
            step_name: The step name (e.g., "market-segmentation")
            framework: The framework

        Returns:
            str: Display name (e.g., "1. Market Segmentation")
        """
        # Get step number
        step_number = self._get_step_number(step_name, framework)

        # Format display name
        display_name = step_name.replace("-", " ").title()

        if step_number > 0:
            return f"{step_number}. {display_name}"

        return display_name

    def _classify_progression_pattern(
        self,
        sequence: List[str],
        framework: FrameworkType,
    ) -> str:
        """
        Classify a step sequence into a progression pattern type.

        Args:
            sequence: List of step names in order of completion
            framework: The framework

        Returns:
            str: Progression pattern type
        """
        if not sequence:
            return ProgressionType.MINIMAL.value

        # Get step numbers for the sequence
        step_numbers = [self._get_step_number(step, framework) for step in sequence]

        # Filter out any invalid step numbers
        valid_numbers = [num for num in step_numbers if num > 0]

        if not valid_numbers:
            return ProgressionType.MINIMAL.value

        # Check if comprehensive (completing most steps)
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            total_steps = 24
            if len(valid_numbers) >= total_steps * 0.75:  # 75% or more of steps
                return ProgressionType.COMPREHENSIVE.value
        elif framework == FrameworkType.STARTUP_TACTICS:
            total_steps = len(StartupTacticsStep)
            if len(valid_numbers) >= total_steps * 0.75:  # 75% or more of steps
                return ProgressionType.COMPREHENSIVE.value

        # Check if minimal (very few steps)
        if len(valid_numbers) <= 2:
            return ProgressionType.MINIMAL.value

        # Check if linear (steps in order)
        is_sorted = all(
            valid_numbers[i] <= valid_numbers[i + 1]
            for i in range(len(valid_numbers) - 1)
        )

        if is_sorted:
            # Check if jumping (skipping steps)
            total_range = valid_numbers[-1] - valid_numbers[0] + 1
            if total_range > len(valid_numbers) * 1.5:  # Significant gaps
                return ProgressionType.JUMPING.value
            else:
                return ProgressionType.LINEAR.value

        # Check if focused (steps clustered in specific areas)
        step_groups = []
        current_group = [valid_numbers[0]]

        for i in range(1, len(valid_numbers)):
            if (
                abs(valid_numbers[i] - valid_numbers[i - 1]) <= 3
            ):  # Close to previous step
                current_group.append(valid_numbers[i])
            else:
                step_groups.append(current_group)
                current_group = [valid_numbers[i]]

        if current_group:
            step_groups.append(current_group)

        if len(step_groups) >= 2 and all(len(group) >= 2 for group in step_groups):
            return ProgressionType.FOCUSED.value

        # Default to nonlinear
        return ProgressionType.NONLINEAR.value

    def _calculate_correlation(self, values1: List[int], values2: List[int]) -> float:
        """
        Calculate correlation coefficient between two lists of values.

        Args:
            values1: First list of values
            values2: Second list of values

        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        if len(values1) != len(values2) or len(values1) == 0:
            return 0.0

        n = len(values1)

        # Calculate means
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n

        # Calculate variances and covariance
        var1 = sum((x - mean1) ** 2 for x in values1)
        var2 = sum((x - mean2) ** 2 for x in values2)

        if var1 == 0 or var2 == 0:
            return 0.0  # No correlation if one variable doesn't vary

        cov = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))

        # Calculate correlation coefficient
        return cov / math.sqrt(var1 * var2)

    def _group_into_ranges(
        self,
        values: List[int],
        max_value: int,
        num_ranges: int = 5,
    ) -> Dict[str, int]:
        """
        Group values into ranges and count occurrences.

        Args:
            values: List of values to group
            max_value: Maximum possible value
            num_ranges: Number of ranges to create

        Returns:
            Dict mapping range labels to counts
        """
        if not values:
            return {}

        # Calculate range size
        range_size = max(1, max_value / num_ranges)

        # Create ranges
        ranges = {}
        for i in range(num_ranges):
            start = int(i * range_size)
            end = int((i + 1) * range_size - 1) if i < num_ranges - 1 else max_value

            if start == end:
                label = f"{start}"
            else:
                label = f"{start}-{end}"

            ranges[label] = 0

        # Count values in each range
        for value in values:
            range_index = min(int(value / range_size), num_ranges - 1)

            # Get range label
            start = int(range_index * range_size)
            end = (
                int((range_index + 1) * range_size - 1)
                if range_index < num_ranges - 1
                else max_value
            )

            if start == end:
                label = f"{start}"
            else:
                label = f"{start}-{end}"

            ranges[label] += 1

        return ranges

    def _analyze_abandonment_factors(
        self,
        ideas: List[Any],
        framework: FrameworkType,
        framework_steps: List[str],
        dropout_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Analyze factors that may contribute to users abandoning the framework.

        Args:
            ideas: List of ideas with framework steps
            framework: The framework
            framework_steps: List of steps in the framework
            dropout_counts: Dict mapping steps to dropout counts

        Returns:
            Dict with abandonment factor analysis
        """
        result = {
            "complexity_factors": {},
            "time_factors": {},
            "user_factors": {},
        }

        # Identify high-dropout steps
        total_ideas = len(ideas)
        high_dropout_steps = [
            step
            for step in framework_steps
            if dropout_counts.get(step, 0) / total_ideas
            >= 0.1  # At least 10% dropout rate
        ]

        # Skip analysis if no significant dropout steps
        if not high_dropout_steps:
            return result

        # Analyze complexity factors
        for step in high_dropout_steps:
            # Get steps with this step
            ideas_with_step = [idea for idea in ideas if idea.has_step(step)]

            # Get steps that include user input for this step
            user_input_count = 0
            for idea in ideas_with_step:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

                # Get steps for this idea
                idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

                # Check for user input
                for s in idea_steps:
                    if s.step == step and s.has_user_input():
                        user_input_count += 1
                        break

            # Calculate user input rate
            user_input_rate = (
                user_input_count / len(ideas_with_step) if ideas_with_step else 0
            )

            # Add to complexity factors
            step_name = self._get_step_display_name(step, framework)
            result["complexity_factors"][step_name] = {
                "dropout_count": dropout_counts.get(step, 0),
                "dropout_rate": dropout_counts.get(step, 0) / total_ideas,
                "user_input_rate": user_input_rate,
                "complexity_level": "high" if user_input_rate < 0.5 else "medium",
            }

        # Analyze time factors
        # Get step completion timestamps
        step_timestamps = {}

        for idea in ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

            # Get steps for this idea
            idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

            # Filter to framework steps with creation dates
            framework_idea_steps = [
                s
                for s in idea_steps
                if s.framework == framework.value and s.step and s.get_creation_date()
            ]

            # Group by step
            for step in framework_idea_steps:
                if step.step not in step_timestamps:
                    step_timestamps[step.step] = []

                step_timestamps[step.step].append(step.get_creation_date())

        # Calculate average time spent on each step
        for step in high_dropout_steps:
            if step not in step_timestamps or len(step_timestamps[step]) < 2:
                continue

            # Sort timestamps
            sorted_times = sorted(step_timestamps[step])

            # Calculate time intervals
            intervals = []
            for i in range(1, len(sorted_times)):
                interval_minutes = (
                    sorted_times[i] - sorted_times[i - 1]
                ).total_seconds() / 60

                # Only include reasonable intervals (< 24 hours)
                if interval_minutes < 1440:
                    intervals.append(interval_minutes)

            if not intervals:
                continue

            # Calculate average interval
            avg_interval = sum(intervals) / len(intervals)

            # Add to time factors
            step_name = self._get_step_display_name(step, framework)
            result["time_factors"][step_name] = {
                "avg_time_minutes": avg_interval,
                "time_intensity": (
                    "high"
                    if avg_interval > 60
                    else "medium" if avg_interval > 30 else "low"
                ),
            }

        # Analyze user factors
        # Group ideas by user
        ideas_by_user = defaultdict(list)
        for idea in ideas:
            if idea.owner:
                ideas_by_user[idea.owner].append(idea)

        # Count users who drop out at each step
        user_dropout_counts = defaultdict(int)

        for user_email, user_ideas in ideas_by_user.items():
            # Get the latest step for each idea
            latest_steps = []

            for idea in user_ideas:
                completed_steps = [
                    step for step in framework_steps if idea.has_step(step)
                ]

                if completed_steps:
                    # Get the step with the highest number
                    latest_step = max(
                        completed_steps,
                        key=lambda s: self._get_step_number(s, framework),
                    )
                    latest_steps.append(latest_step)

            if latest_steps:
                # Get the step that appears most frequently as the latest step
                common_dropout = Counter(latest_steps).most_common(1)[0][0]
                user_dropout_counts[common_dropout] += 1

        # Add to user factors
        for step in high_dropout_steps:
            step_name = self._get_step_display_name(step, framework)

            result["user_factors"][step_name] = {
                "user_dropout_count": user_dropout_counts.get(step, 0),
                "user_dropout_rate": (
                    user_dropout_counts.get(step, 0) / len(ideas_by_user)
                    if ideas_by_user
                    else 0
                ),
            }

        return result

    def _generate_framework_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        completion_rates: Dict[str, float],
        framework_steps: List[str],
        framework: FrameworkType,
    ) -> Dict[str, Any]:
        """
        Generate recommendations for improving framework completion.

        Args:
            bottlenecks: List of identified bottlenecks
            completion_rates: Dict mapping steps to completion rates
            framework_steps: List of steps in the framework
            framework: The framework

        Returns:
            Dict with recommendations
        """
        result = {
            "primary_bottlenecks": [],
            "improvement_areas": [],
            "workflow_recommendations": [],
        }

        # Skip if no bottlenecks
        if not bottlenecks:
            return result

        # Add primary bottlenecks
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            result["primary_bottlenecks"].append(
                {
                    "from_step": bottleneck["from_step"],
                    "to_step": bottleneck["to_step"],
                    "drop_off_rate": bottleneck["drop_off_rate"],
                    "severity": bottleneck["severity"],
                }
            )

        # Identify improvement areas
        low_completion_steps = [
            step
            for step in framework_steps
            if completion_rates.get(step, 0) < 0.3  # Less than 30% completion
        ]

        for step in low_completion_steps:
            step_name = self._get_step_display_name(step, framework)

            # Check if it's part of a bottleneck
            is_bottleneck = any(
                bottleneck["to_step"] == step_name for bottleneck in bottlenecks
            )

            result["improvement_areas"].append(
                {
                    "step": step_name,
                    "completion_rate": completion_rates.get(step, 0),
                    "is_bottleneck": is_bottleneck,
                    "priority": "high" if is_bottleneck else "medium",
                }
            )

        # Generate workflow recommendations
        # 1. Address major bottlenecks
        if bottlenecks:
            result["workflow_recommendations"].append(
                {
                    "focus": "Address Major Bottlenecks",
                    "description": "Provide additional guidance or simplify the transition between consecutive steps with high drop-off rates.",
                    "target_steps": [b["to_step"] for b in bottlenecks[:2]],
                }
            )

        # 2. Low completion steps
        if low_completion_steps:
            low_step_names = [
                self._get_step_display_name(s, framework)
                for s in low_completion_steps[:3]
            ]

            result["workflow_recommendations"].append(
                {
                    "focus": "Improve Low Completion Steps",
                    "description": "Enhance user experience for steps with consistently low completion rates.",
                    "target_steps": low_step_names,
                }
            )

        # 3. Early vs. late framework steps
        early_completion = self._calculate_avg_completion(
            completion_rates, framework_steps[: len(framework_steps) // 3], framework
        )
        late_completion = self._calculate_avg_completion(
            completion_rates, framework_steps[-len(framework_steps) // 3 :], framework
        )

        if (
            early_completion > late_completion * 1.5
        ):  # Significantly higher early completion
            result["workflow_recommendations"].append(
                {
                    "focus": "Address Late Framework Drop-off",
                    "description": "Users are much more likely to complete early steps than later ones, suggesting potential engagement issues in later steps.",
                    "completion_difference": early_completion - late_completion,
                }
            )

        return result

    def _calculate_avg_completion(
        self,
        completion_rates: Dict[str, float],
        steps: List[str],
        framework: FrameworkType,
    ) -> float:
        """
        Calculate average completion rate for a list of steps.

        Args:
            completion_rates: Dict mapping steps to completion rates
            steps: List of steps to average
            framework: The framework

        Returns:
            float: Average completion rate
        """
        if not steps:
            return 0.0

        rates = [completion_rates.get(step, 0) for step in steps]
        return sum(rates) / len(rates)

    def _compare_step_intervals_by_version(
        self,
        framework: FrameworkType,
        user_emails: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare step time intervals between different tool versions.

        Args:
            framework: The framework to analyze
            user_emails: Optional list of user emails to filter by

        Returns:
            Dict with version comparison results
        """
        result = {}

        # Define tool versions to compare
        tool_versions = [ToolVersion.V1, ToolVersion.V2]

        # Get semester date ranges
        semester_ranges = {
            ToolVersion.V1: (
                datetime(2024, 1, 1),
                datetime(2024, 5, 31),
            ),  # Spring 2024
            ToolVersion.V2: (
                datetime(2024, 9, 1),
                datetime(2025, 5, 31),
            ),  # Fall 2024 + Spring 2025
        }

        # Get all ideas with steps in this framework
        all_ideas = self._get_ideas_with_framework_steps(framework, user_emails)

        # Analyze each tool version
        for version in tool_versions:
            date_range = semester_ranges.get(version)

            if not date_range:
                continue

            # Filter ideas by creation date
            version_ideas = [
                idea
                for idea in all_ideas
                if idea.get_creation_date()
                and date_range[0] <= idea.get_creation_date() <= date_range[1]
            ]

            # Skip if no ideas for this version
            if not version_ideas:
                continue

            # Collect step creation timestamps by idea
            idea_step_timestamps = {}

            for idea in version_ideas:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

                # Get all steps for this idea
                steps = self._data_repo.steps.find_by_idea_id(idea_id)

                # Filter to steps in this framework with valid creation dates
                framework_steps_with_dates = [
                    step
                    for step in steps
                    if step.framework == framework.value and step.get_creation_date()
                ]

                if not framework_steps_with_dates:
                    continue

                # Create timestamp dictionary for this idea
                idea_step_timestamps[idea_id] = {}

                for step in framework_steps_with_dates:
                    if step.step:
                        # Use the earliest timestamp if there are multiple versions of the same step
                        current_timestamp = idea_step_timestamps[idea_id].get(step.step)
                        step_timestamp = step.get_creation_date()

                        if (
                            current_timestamp is None
                            or step_timestamp < current_timestamp
                        ):
                            idea_step_timestamps[idea_id][step.step] = step_timestamp

            # Calculate intervals between consecutive steps
            all_intervals = []

            for idea_id, timestamps in idea_step_timestamps.items():
                if len(timestamps) < 2:
                    continue

                # Sort steps by timestamp
                sorted_steps = sorted(timestamps.items(), key=lambda x: x[1])

                # Calculate intervals between consecutive steps
                for i in range(1, len(sorted_steps)):
                    prev_step, prev_time = sorted_steps[i - 1]
                    curr_step, curr_time = sorted_steps[i]

                    # Calculate interval in minutes
                    interval_minutes = (curr_time - prev_time).total_seconds() / 60
                    all_intervals.append(interval_minutes)

            # Calculate metrics
            if all_intervals:
                version_metrics = {
                    "version": version.value,
                    "sample_size": len(all_intervals),
                    "avg_interval_minutes": sum(all_intervals) / len(all_intervals),
                    "median_interval_minutes": sorted(all_intervals)[
                        len(all_intervals) // 2
                    ],
                    "min_interval_minutes": min(all_intervals),
                    "max_interval_minutes": max(all_intervals),
                }

                # Group intervals into categories
                interval_categories = {
                    "under_5min": sum(1 for i in all_intervals if i < 5),
                    "5_15min": sum(1 for i in all_intervals if 5 <= i < 15),
                    "15_30min": sum(1 for i in all_intervals if 15 <= i < 30),
                    "30_60min": sum(1 for i in all_intervals if 30 <= i < 60),
                    "1_3hr": sum(1 for i in all_intervals if 60 <= i < 180),
                    "3_24hr": sum(1 for i in all_intervals if 180 <= i < 1440),
                    "over_24hr": sum(1 for i in all_intervals if i >= 1440),
                }

                # Calculate percentages
                interval_distribution = {
                    category: {"count": count, "percentage": count / len(all_intervals)}
                    for category, count in interval_categories.items()
                }

                version_metrics["interval_distribution"] = interval_distribution
                result[version.value] = version_metrics

        # Skip comparison if not enough data
        if len(result) < 2:
            return result

        # Add direct comparison
        v1_metrics = result.get(ToolVersion.V1.value)
        v2_metrics = result.get(ToolVersion.V2.value)

        if v1_metrics and v2_metrics:
            v1_avg = v1_metrics["avg_interval_minutes"]
            v2_avg = v2_metrics["avg_interval_minutes"]

            result["comparison"] = {
                "avg_interval_difference": v2_avg - v1_avg,
                "percent_change": (
                    ((v2_avg - v1_avg) / v1_avg) * 100 if v1_avg > 0 else 0
                ),
                "time_savings": "improved" if v2_avg < v1_avg else "worsened",
                "rapid_completion_change": (
                    v2_metrics["interval_distribution"]["under_5min"]["percentage"]
                    - v1_metrics["interval_distribution"]["under_5min"]["percentage"]
                ),
            }

        return result

    def _analyze_tool_version_impact(
        self,
        framework: FrameworkType,
        cohort_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze the impact of different tool versions on framework progression.

        Args:
            framework: The framework
            cohort_metrics: Dict with metrics for each cohort

        Returns:
            Dict with tool version impact analysis
        """
        result = {
            "version_metrics": {},
            "version_comparison": {},
        }

        # Group cohorts by tool version
        version_cohorts = {
            ToolVersion.NONE.value: [],
            ToolVersion.V1.value: [],
            ToolVersion.V2.value: [],
        }

        for cohort, metrics in cohort_metrics.items():
            tool_version = metrics.get("tool_version")

            if tool_version:
                version_cohorts[tool_version].append(cohort)

        # Calculate aggregate metrics for each version
        for version, cohorts in version_cohorts.items():
            # Skip if no cohorts for this version
            if not cohorts:
                continue

            # Aggregate metrics
            cohort_data = [cohort_metrics[cohort] for cohort in cohorts]

            # Calculate averages
            avg_completion = sum(
                data.get("completion_percentage", 0) for data in cohort_data
            ) / len(cohort_data)

            avg_steps_per_idea = sum(
                data.get("avg_steps_per_idea", 0) for data in cohort_data
            ) / len(cohort_data)

            # Count total users and ideas
            total_users = sum(data.get("user_count", 0) for data in cohort_data)
            total_ideas = sum(data.get("idea_count", 0) for data in cohort_data)

            # Add to result
            result["version_metrics"][version] = {
                "cohorts": cohorts,
                "total_users": total_users,
                "total_ideas": total_ideas,
                "avg_completion_percentage": avg_completion,
                "avg_steps_per_idea": avg_steps_per_idea,
            }

        # Compare versions
        # Skip comparison if not enough data
        versions_to_compare = [
            v for v, m in result["version_metrics"].items() if m["total_ideas"] > 0
        ]

        if len(versions_to_compare) >= 2:
            # Compare v1 vs. no tool
            if (
                ToolVersion.NONE.value in versions_to_compare
                and ToolVersion.V1.value in versions_to_compare
            ):
                v0_metrics = result["version_metrics"][ToolVersion.NONE.value]
                v1_metrics = result["version_metrics"][ToolVersion.V1.value]

                result["version_comparison"]["v1_vs_no_tool"] = {
                    "completion_difference": v1_metrics["avg_completion_percentage"]
                    - v0_metrics["avg_completion_percentage"],
                    "steps_per_idea_difference": v1_metrics["avg_steps_per_idea"]
                    - v0_metrics["avg_steps_per_idea"],
                    "percent_improvement": (
                        (
                            v1_metrics["avg_completion_percentage"]
                            - v0_metrics["avg_completion_percentage"]
                        )
                        / v0_metrics["avg_completion_percentage"]
                        * 100
                        if v0_metrics["avg_completion_percentage"] > 0
                        else 0
                    ),
                }

            # Compare v2 vs. v1
            if (
                ToolVersion.V1.value in versions_to_compare
                and ToolVersion.V2.value in versions_to_compare
            ):
                v1_metrics = result["version_metrics"][ToolVersion.V1.value]
                v2_metrics = result["version_metrics"][ToolVersion.V2.value]

                result["version_comparison"]["v2_vs_v1"] = {
                    "completion_difference": v2_metrics["avg_completion_percentage"]
                    - v1_metrics["avg_completion_percentage"],
                    "steps_per_idea_difference": v2_metrics["avg_steps_per_idea"]
                    - v1_metrics["avg_steps_per_idea"],
                    "percent_improvement": (
                        (
                            v2_metrics["avg_completion_percentage"]
                            - v1_metrics["avg_completion_percentage"]
                        )
                        / v1_metrics["avg_completion_percentage"]
                        * 100
                        if v1_metrics["avg_completion_percentage"] > 0
                        else 0
                    ),
                }

        return result

    def _calculate_step_score_correlation(
        self,
        ideas: List[Any],
        framework: FrameworkType,
        framework_steps: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate correlation between step counts and user scores.

        Args:
            ideas: List of ideas with framework steps
            framework: The framework
            framework_steps: List of steps in the framework

        Returns:
            Dict with correlation analysis
        """
        result = {
            "content_score_correlation": 0,
            "completion_score_correlation": 0,
            "step_count_by_score_level": {},
        }

        # Group ideas by user
        ideas_by_user = defaultdict(list)
        for idea in ideas:
            if idea.owner:
                ideas_by_user[idea.owner].append(idea)

        # Calculate step counts and get scores
        user_data = []

        for user_email, user_ideas in ideas_by_user.items():
            # Get user
            user = self._data_repo.users.find_by_email(user_email)

            if not user or not user.scores:
                continue

            # Count steps across all user ideas
            step_count = 0

            for idea in user_ideas:
                step_count += sum(1 for step in framework_steps if idea.has_step(step))

            # Add to user data
            user_data.append(
                {
                    "email": user_email,
                    "step_count": step_count,
                    "content_score": user.scores.content or 0,
                    "completion_score": user.scores.completion or 0,
                }
            )

        # Calculate correlations
        if user_data:
            step_counts = [user["step_count"] for user in user_data]
            content_scores = [user["content_score"] for user in user_data]
            completion_scores = [user["completion_score"] for user in user_data]

            content_correlation = self._calculate_correlation(
                step_counts, content_scores
            )
            completion_correlation = self._calculate_correlation(
                step_counts, completion_scores
            )

            result["content_score_correlation"] = content_correlation
            result["completion_score_correlation"] = completion_correlation

        # Group users by score level
        score_levels = {
            "high": [],
            "medium": [],
            "low": [],
        }

        for user_data_item in user_data:
            avg_score = (
                user_data_item["content_score"] + user_data_item["completion_score"]
            ) / 2

            if avg_score >= 0.7:
                score_levels["high"].append(user_data_item["step_count"])
            elif avg_score >= 0.3:
                score_levels["medium"].append(user_data_item["step_count"])
            else:
                score_levels["low"].append(user_data_item["step_count"])

        # Calculate average steps per score level
        for level, step_counts in score_levels.items():
            if step_counts:
                result["step_count_by_score_level"][level] = {
                    "avg_steps": sum(step_counts) / len(step_counts),
                    "user_count": len(step_counts),
                }

        return result

    def _group_users_by_step_completion(
        self,
        ideas: List[Any],
        framework: FrameworkType,
        framework_steps: List[str],
        user_emails: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Group users by their step completion levels and analyze engagement.

        Args:
            ideas: List of ideas with framework steps
            framework: The framework
            framework_steps: List of steps in the framework
            user_emails: Optional list of user emails to filter by

        Returns:
            Dict with user grouping analysis
        """
        result = {
            "engagement_levels": {},
            "step_completion_groups": {},
        }

        # Group ideas by user
        ideas_by_user = defaultdict(list)
        for idea in ideas:
            if idea.owner:
                ideas_by_user[idea.owner].append(idea)

        # Calculate total step count for each user
        user_step_counts = {}

        for user_email, user_ideas in ideas_by_user.items():
            # Count steps across all user ideas
            step_count = 0

            for idea in user_ideas:
                step_count += sum(1 for step in framework_steps if idea.has_step(step))

            user_step_counts[user_email] = step_count

        # Skip if no data
        if not user_step_counts:
            return result

        # Define step completion groups
        total_possible_steps = len(framework_steps)
        completion_groups = {
            "high": [],  # >50% of steps completed
            "medium": [],  # 20-50% of steps completed
            "low": [],  # <20% of steps completed
        }

        for user_email, step_count in user_step_counts.items():
            completion_percentage = step_count / total_possible_steps

            if completion_percentage > 0.5:
                completion_groups["high"].append(user_email)
            elif completion_percentage >= 0.2:
                completion_groups["medium"].append(user_email)
            else:
                completion_groups["low"].append(user_email)

        # Get engagement levels for each group
        for group_name, group_emails in completion_groups.items():
            engagement_distribution = {
                "high": 0,
                "medium": 0,
                "low": 0,
            }

            for email in group_emails:
                user = self._data_repo.users.find_by_email(email)

                if user:
                    engagement_level = user.get_engagement_level().value
                    engagement_distribution[engagement_level] += 1

            # Calculate percentages
            total_users = len(group_emails)

            if total_users > 0:
                result["engagement_levels"][group_name] = {
                    "total_users": total_users,
                    "high_engagement_count": engagement_distribution["high"],
                    "medium_engagement_count": engagement_distribution["medium"],
                    "low_engagement_count": engagement_distribution["low"],
                    "high_engagement_percentage": engagement_distribution["high"]
                    / total_users,
                    "medium_engagement_percentage": engagement_distribution["medium"]
                    / total_users,
                    "low_engagement_percentage": engagement_distribution["low"]
                    / total_users,
                }

        # Calculate metrics for each completion group
        for group_name, group_emails in completion_groups.items():
            if not group_emails:
                continue

            # Get ideas for this group
            group_ideas = []
            for email in group_emails:
                group_ideas.extend(ideas_by_user.get(email, []))

            # Calculate metrics
            result["step_completion_groups"][group_name] = {
                "user_count": len(group_emails),
                "avg_ideas_per_user": len(group_ideas) / len(group_emails),
                "percentage_of_users": len(group_emails) / len(user_step_counts),
            }

        return result

    def _analyze_framework_by_category(
        self,
        ideas: List[Any],
        framework: FrameworkType,
        framework_steps: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze framework effectiveness by idea category.

        Args:
            ideas: List of ideas with framework steps
            framework: The framework
            framework_steps: List of steps in the framework

        Returns:
            Dict with category analysis
        """
        result = {
            "categories": {},
            "step_effectiveness_by_category": {},
        }

        # Group ideas by category
        ideas_by_category = defaultdict(list)

        for idea in ideas:
            category = idea.get_idea_category().value
            ideas_by_category[category].append(idea)

        # Calculate metrics for each category
        for category, category_ideas in ideas_by_category.items():
            # Calculate step completion counts
            step_counts = {}

            for step in framework_steps:
                completed_count = sum(
                    1 for idea in category_ideas if idea.has_step(step)
                )
                step_counts[step] = completed_count

            # Calculate average steps per idea
            total_steps = sum(step_counts.values())
            avg_steps = total_steps / len(category_ideas) if category_ideas else 0

            # Calculate completion percentage
            completion_percentage = (
                total_steps / (len(category_ideas) * len(framework_steps))
                if category_ideas and framework_steps
                else 0
            ) * 100

            # Add to result
            result["categories"][category] = {
                "idea_count": len(category_ideas),
                "avg_steps_per_idea": avg_steps,
                "completion_percentage": completion_percentage,
                "most_completed_steps": self._get_top_steps(
                    step_counts, framework, n=3
                ),
                "least_completed_steps": self._get_bottom_steps(
                    step_counts, framework, n=3
                ),
            }

        # Find steps that are particularly effective or ineffective for specific categories
        for step in framework_steps:
            step_name = self._get_step_display_name(step, framework)
            step_effectiveness = {}

            # Calculate completion rate for each category
            overall_rate = (
                sum(1 for idea in ideas if idea.has_step(step)) / len(ideas)
                if ideas
                else 0
            )

            category_rates = {}
            for category, category_ideas in ideas_by_category.items():
                if not category_ideas:
                    continue

                completion_rate = sum(
                    1 for idea in category_ideas if idea.has_step(step)
                ) / len(category_ideas)

                # Calculate difference from overall
                difference = completion_rate - overall_rate

                category_rates[category] = {
                    "completion_rate": completion_rate,
                    "difference_from_overall": difference,
                    "relative_effectiveness": (
                        "much_higher"
                        if difference > 0.25
                        else (
                            "higher"
                            if difference > 0.1
                            else (
                                "lower"
                                if difference < -0.1
                                else "much_lower" if difference < -0.25 else "average"
                            )
                        )
                    ),
                }

            # Only include steps with significant differences
            significant_categories = {
                category: data
                for category, data in category_rates.items()
                if abs(data["difference_from_overall"]) > 0.1
            }

            if significant_categories:
                result["step_effectiveness_by_category"][
                    step_name
                ] = significant_categories

        return result

    def _get_top_steps(
        self,
        step_counts: Dict[str, int],
        framework: FrameworkType,
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get the top most completed steps.

        Args:
            step_counts: Dict mapping steps to completion counts
            framework: The framework
            n: Number of top steps to return

        Returns:
            List of top step data
        """
        # Sort steps by count (descending)
        sorted_steps = sorted(step_counts.items(), key=lambda x: x[1], reverse=True)

        # Return top n
        result = []
        for step, count in sorted_steps[:n]:
            step_name = self._get_step_display_name(step, framework)
            result.append(
                {
                    "step": step_name,
                    "count": count,
                }
            )

        return result

    def _get_bottom_steps(
        self,
        step_counts: Dict[str, int],
        framework: FrameworkType,
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get the bottom least completed steps.

        Args:
            step_counts: Dict mapping steps to completion counts
            framework: The framework
            n: Number of bottom steps to return

        Returns:
            List of bottom step data
        """
        # Sort steps by count (ascending)
        sorted_steps = sorted(step_counts.items(), key=lambda x: x[1])

        # Return bottom n
        result = []
        for step, count in sorted_steps[:n]:
            step_name = self._get_step_display_name(step, framework)
            result.append(
                {
                    "step": step_name,
                    "count": count,
                }
            )

        return result
