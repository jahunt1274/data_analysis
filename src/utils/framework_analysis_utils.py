"""
Framework analysis utilities for the data analysis system.

This module provides utilities for analyzing entrepreneurial frameworks,
step progressions, and framework-related metrics across the JetPack/Orbit tool.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
from enum import Enum


class ProgressionType(str, Enum):
    """Types of progression patterns through frameworks."""

    LINEAR = "linear"  # Steps completed in order (1, 2, 3, 4...)
    JUMPING = "jumping"  # Skipping steps (1, 3, 5...)
    NONLINEAR = "nonlinear"  # Out of order steps (3, 1, 4, 2...)
    FOCUSED = "focused"  # Concentrating on specific areas
    COMPREHENSIVE = "comprehensive"  # Completing most steps
    MINIMAL = "minimal"  # Completing very few steps


def classify_progression_pattern(
    sequence: List[str],
    framework_steps: List[str],
    step_numbers: Optional[Dict[str, int]] = None,
) -> str:
    """
    Classify a step sequence into a progression pattern type.

    Args:
        sequence: List of step names in order of completion
        framework_steps: List of all steps in the framework
        step_numbers: Optional dict mapping step names to their position number

    Returns:
        str: Progression pattern type from ProgressionType enum
    """
    if not sequence:
        return ProgressionType.MINIMAL.value

    # Get step numbers for the sequence
    if step_numbers:
        step_numbers_dict = step_numbers
    else:
        # Create step numbers based on position in framework_steps
        step_numbers_dict = {step: i + 1 for i, step in enumerate(framework_steps)}

    # Get step numbers for the sequence
    step_nums = [step_numbers_dict.get(step, 0) for step in sequence]

    # Filter out any invalid step numbers
    valid_numbers = [num for num in step_nums if num > 0]

    if not valid_numbers:
        return ProgressionType.MINIMAL.value

    # Check if comprehensive (completing most steps)
    total_steps = len(framework_steps)
    if len(valid_numbers) >= total_steps * 0.75:  # 75% or more of steps
        return ProgressionType.COMPREHENSIVE.value

    # Check if minimal (very few steps)
    if len(valid_numbers) <= 2:
        return ProgressionType.MINIMAL.value

    # Check if linear (steps in order)
    is_sorted = all(
        valid_numbers[i] <= valid_numbers[i + 1] for i in range(len(valid_numbers) - 1)
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
        if abs(valid_numbers[i] - valid_numbers[i - 1]) <= 3:  # Close to previous step
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


def get_step_sequences(
    ideas: List[Any], framework: str, steps_repository: Any
) -> List[List[str]]:
    """
    Get sequences of steps completed for ideas in the framework.

    Args:
        ideas: List of ideas to analyze
        framework: Framework identifier
        steps_repository: Repository for accessing steps

    Returns:
        List of step sequences (each a list of step names in order of completion)
    """
    sequences = []

    for idea in ideas:
        if not idea.id:
            continue

        idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

        # Get steps for this idea
        steps = steps_repository.find_by_idea_id(idea_id)

        # Filter to framework steps with creation dates
        framework_steps = [
            step
            for step in steps
            if step.framework == framework and step.step and step.get_creation_date()
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


def analyze_step_relationships(
    step_sequences: List[List[str]], min_support: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze relationships between steps (prerequisites, dependents, co-occurrences).

    Args:
        step_sequences: List of step sequences in order of completion
        min_support: Minimum support threshold for relationships (0-1)

    Returns:
        Dict with step relationship analysis
    """
    result = {
        "prerequisites": defaultdict(dict),
        "dependents": defaultdict(dict),
        "co_occurrences": defaultdict(dict),
    }

    if not step_sequences:
        return {
            "prerequisites": {},
            "dependents": {},
            "co_occurrences": {},
        }

    # Get all unique steps
    all_steps = set()
    for sequence in step_sequences:
        all_steps.update(sequence)

    # Calculate prerequisites (steps that come before)
    for sequence in step_sequences:
        for i, step in enumerate(sequence):
            # Calculate prerequisites (steps that come before this step)
            for j in range(i):
                prereq = sequence[j]
                if prereq != step:  # Skip self-relationships
                    if prereq not in result["prerequisites"][step]:
                        result["prerequisites"][step][prereq] = 0
                    result["prerequisites"][step][prereq] += 1

            # Calculate dependents (steps that come after this step)
            for j in range(i + 1, len(sequence)):
                dependent = sequence[j]
                if dependent != step:  # Skip self-relationships
                    if dependent not in result["dependents"][step]:
                        result["dependents"][step][dependent] = 0
                    result["dependents"][step][dependent] += 1

    # Calculate co-occurrences (steps that appear in the same sequence)
    for sequence in step_sequences:
        unique_steps = set(sequence)
        for step1 in unique_steps:
            for step2 in unique_steps:
                if step1 != step2:  # Skip self-relationships
                    if step2 not in result["co_occurrences"][step1]:
                        result["co_occurrences"][step1][step2] = 0
                    result["co_occurrences"][step1][step2] += 1

    # Calculate percentages and filter by minimum support
    total_sequences = len(step_sequences)

    # Process prerequisites
    for step, prereqs in list(result["prerequisites"].items()):
        step_occurrences = sum(1 for seq in step_sequences if step in seq)
        if step_occurrences == 0:
            del result["prerequisites"][step]
            continue

        for prereq, count in list(prereqs.items()):
            percentage = count / step_occurrences
            if percentage < min_support:
                del result["prerequisites"][step][prereq]
            else:
                result["prerequisites"][step][prereq] = percentage

    # Process dependents
    for step, deps in list(result["dependents"].items()):
        step_occurrences = sum(1 for seq in step_sequences if step in seq)
        if step_occurrences == 0:
            del result["dependents"][step]
            continue

        for dep, count in list(deps.items()):
            percentage = count / step_occurrences
            if percentage < min_support:
                del result["dependents"][step][dep]
            else:
                result["dependents"][step][dep] = percentage

    # Process co-occurrences
    for step, coocs in list(result["co_occurrences"].items()):
        step_occurrences = sum(1 for seq in step_sequences if step in seq)
        if step_occurrences == 0:
            del result["co_occurrences"][step]
            continue

        for cooc, count in list(coocs.items()):
            percentage = count / step_occurrences
            if percentage < min_support:
                del result["co_occurrences"][step][cooc]
            else:
                result["co_occurrences"][step][cooc] = percentage

    # Convert defaultdicts to regular dicts for serialization
    result["prerequisites"] = dict(result["prerequisites"])
    result["dependents"] = dict(result["dependents"])
    result["co_occurrences"] = dict(result["co_occurrences"])

    return result


def calculate_step_time_intervals(
    step_sequences_with_dates: List[List[Tuple[str, datetime]]],
) -> Dict[str, Any]:
    """
    Calculate time intervals between framework steps.

    Args:
        step_sequences_with_dates: List of step sequences with timestamps
            Format: [[(step1, timestamp1), (step2, timestamp2), ...], ...]

    Returns:
        Dict with step interval analysis
    """
    result = {
        "step_pair_intervals": defaultdict(list),
        "overall_intervals": [],
    }

    for sequence in step_sequences_with_dates:
        if len(sequence) < 2:
            continue

        # Calculate intervals between consecutive steps
        for i in range(1, len(sequence)):
            prev_step, prev_time = sequence[i - 1]
            curr_step, curr_time = sequence[i]

            # Calculate interval in minutes
            interval_minutes = (curr_time - prev_time).total_seconds() / 60

            # Add to step pair intervals
            result["step_pair_intervals"][(prev_step, curr_step)].append(
                interval_minutes
            )

            # Add to overall intervals
            result["overall_intervals"].append(interval_minutes)

    # Calculate statistics for each step pair
    step_pair_metrics = {}

    for (step1, step2), intervals in result["step_pair_intervals"].items():
        if not intervals:
            continue

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

        step_pair_metrics[f"{step1} → {step2}"] = metrics

    # Calculate overall interval statistics
    if result["overall_intervals"]:
        result["interval_statistics"] = {
            "count": len(result["overall_intervals"]),
            "avg_minutes": sum(result["overall_intervals"])
            / len(result["overall_intervals"]),
            "median_minutes": sorted(result["overall_intervals"])[
                len(result["overall_intervals"]) // 2
            ],
            "min_minutes": min(result["overall_intervals"]),
            "max_minutes": max(result["overall_intervals"]),
        }

        # Group intervals into categories
        interval_categories = {
            "under_5min": 0,
            "5_15min": 0,
            "15_30min": 0,
            "30_60min": 0,
            "1_3hr": 0,
            "3_24hr": 0,
            "over_24hr": 0,
        }

        for interval in result["overall_intervals"]:
            if interval < 5:
                interval_categories["under_5min"] += 1
            elif interval < 15:
                interval_categories["5_15min"] += 1
            elif interval < 30:
                interval_categories["15_30min"] += 1
            elif interval < 60:
                interval_categories["30_60min"] += 1
            elif interval < 180:
                interval_categories["1_3hr"] += 1
            elif interval < 1440:
                interval_categories["3_24hr"] += 1
            else:
                interval_categories["over_24hr"] += 1

        # Calculate percentages
        result["interval_distribution"] = {
            category: {
                "count": count,
                "percentage": count / len(result["overall_intervals"]),
            }
            for category, count in interval_categories.items()
        }
    else:
        result["interval_statistics"] = {
            "count": 0,
            "avg_minutes": 0,
            "median_minutes": 0,
            "min_minutes": 0,
            "max_minutes": 0,
        }
        result["interval_distribution"] = {}

    # Set step pair metrics
    result["step_pair_metrics"] = step_pair_metrics

    # Remove raw data if large
    if len(result["overall_intervals"]) > 100:
        result["overall_intervals"] = result["overall_intervals"][
            :100
        ]  # Keep just a sample

    # Convert defaultdicts to regular dicts for serialization
    result["step_pair_intervals"] = dict(result["step_pair_intervals"])

    return result


def analyze_framework_bottlenecks(
    step_completion_rates: Dict[str, float],
    step_relationship_data: Dict[str, Any],
    framework_steps: List[str],
    step_numbers: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Identify bottlenecks in framework progression.

    Args:
        step_completion_rates: Dict mapping step names to completion rates
        step_relationship_data: Step relationship analysis from analyze_step_relationships
        framework_steps: List of all steps in the framework
        step_numbers: Optional dict mapping step names to their position number

    Returns:
        Dict with bottleneck analysis
    """
    result = {
        "completion_bottlenecks": [],
        "progression_bottlenecks": [],
        "recommendation": [],
    }

    # Get step numbers if not provided
    if not step_numbers:
        step_numbers = {step: i + 1 for i, step in enumerate(framework_steps)}

    # Find significant drops in completion rates between consecutive steps
    for i in range(1, len(framework_steps)):
        prev_step = framework_steps[i - 1]
        curr_step = framework_steps[i]

        prev_rate = step_completion_rates.get(prev_step, 0)
        curr_rate = step_completion_rates.get(curr_step, 0)

        # Calculate drop-off
        drop_off = prev_rate - curr_rate

        if drop_off > 0.15:  # Significant drop-off threshold
            result["completion_bottlenecks"].append(
                {
                    "from_step": prev_step,
                    "to_step": curr_step,
                    "from_step_number": step_numbers.get(prev_step, 0),
                    "to_step_number": step_numbers.get(curr_step, 0),
                    "drop_off_rate": drop_off,
                    "from_completion_rate": prev_rate,
                    "to_completion_rate": curr_rate,
                    "severity": "high" if drop_off > 0.3 else "medium",
                }
            )

    # Sort bottlenecks by drop-off rate (descending)
    result["completion_bottlenecks"].sort(
        key=lambda x: x["drop_off_rate"], reverse=True
    )

    # Find progression bottlenecks using relationship data
    prerequisites = step_relationship_data.get("prerequisites", {})

    for step, prereqs in prerequisites.items():
        # Skip if no prerequisites
        if not prereqs:
            continue

        # Find the most common prerequisite
        most_common_prereq = max(prereqs.items(), key=lambda x: x[1])
        prereq_step, prereq_rate = most_common_prereq

        # Check if this step has a low completion rate
        step_rate = step_completion_rates.get(step, 0)
        prereq_completion = step_completion_rates.get(prereq_step, 0)

        if prereq_completion > step_rate * 2 and prereq_rate > 0.5:
            # This step is often preceded by a specific step but has much lower completion
            result["progression_bottlenecks"].append(
                {
                    "step": step,
                    "step_number": step_numbers.get(step, 0),
                    "prerequisite": prereq_step,
                    "prerequisite_number": step_numbers.get(prereq_step, 0),
                    "prerequisite_rate": prereq_rate,
                    "completion_rate": step_rate,
                    "completion_drop": prereq_completion - step_rate,
                }
            )

    # Sort progression bottlenecks by completion drop (descending)
    result["progression_bottlenecks"].sort(
        key=lambda x: x["completion_drop"], reverse=True
    )

    # Generate recommendations
    if result["completion_bottlenecks"]:
        for bottleneck in result["completion_bottlenecks"][:3]:  # Top 3 bottlenecks
            result["recommendation"].append(
                {
                    "focus": f"Address drop-off after step {bottleneck['from_step_number']}",
                    "description": f"There is a significant drop-off ({bottleneck['drop_off_rate']:.2f}) between {bottleneck['from_step']} and {bottleneck['to_step']}.",
                    "suggestion": "Provide additional guidance or simplify the transition between these steps.",
                }
            )

    if result["progression_bottlenecks"]:
        for bottleneck in result["progression_bottlenecks"][:3]:  # Top 3 bottlenecks
            result["recommendation"].append(
                {
                    "focus": f"Improve step {bottleneck['step_number']} completion",
                    "description": f"Step {bottleneck['step']} has a low completion rate despite following from {bottleneck['prerequisite']}.",
                    "suggestion": "Review this step's complexity and provide additional resources or simplification.",
                }
            )

    return result

def identify_high_impact_steps(
    step_completion_rates: Dict[str, float],
    user_engagement_scores: Dict[str, float],
    step_sequences: List[List[str]],
    framework_steps: List[str],
) -> Dict[str, Any]:
    """
    Identify which framework steps have the highest impact on outcomes.

    Args:
        step_completion_rates: Dict mapping step names to completion rates
        user_engagement_scores: Dict mapping user identifiers to engagement scores
        step_sequences: List of step sequences by user
        framework_steps: List of all steps in the framework

    Returns:
        Dict with high impact step analysis
    """
    result = {
        "step_impact_scores": {},
        "high_impact_steps": [],
        "low_impact_steps": [],
    }

    # Skip if insufficient data
    if not user_engagement_scores or not step_sequences:
        return result

    # Map step sequences to users
    if len(step_sequences) != len(user_engagement_scores):
        # Sequences must correspond to users with engagement scores
        return result

    # Calculate correlation between step completion and engagement scores
    user_step_completion = {}

    for i, sequence in enumerate(step_sequences):
        user_id = list(user_engagement_scores.keys())[i]

        # Mark which steps this user completed
        user_step_completion[user_id] = {
            step: 1 if step in sequence else 0 for step in framework_steps
        }

    # Calculate point-biserial correlation for each step
    step_correlations = {}

    for step in framework_steps:
        # Get completion values (0 or 1) for this step across users
        step_values = [
            user_step_completion[user_id][step]
            for user_id in user_engagement_scores.keys()
        ]

        # Get engagement scores
        engagement_values = list(user_engagement_scores.values())

        # Calculate correlation
        biserial_correlation = calculate_point_biserial_correlation(
            step_values, engagement_values
        )
        step_correlations[step] = biserial_correlation

    # Set impact scores
    result["step_impact_scores"] = step_correlations

    # Identify high and low impact steps
    sorted_steps = sorted(step_correlations.items(), key=lambda x: x[1], reverse=True)

    # High impact steps (top 25% of positive correlations)
    high_impact_threshold = max(
        0.3, sorted_steps[0][1] * 0.75
    )  # At least 0.3 correlation
    result["high_impact_steps"] = [
        {
            "step": step,
            "impact_score": score,
            "completion_rate": step_completion_rates.get(step, 0),
        }
        for step, score in sorted_steps
        if score >= high_impact_threshold
    ]

    # Low impact steps (negative correlations)
    result["low_impact_steps"] = [
        {
            "step": step,
            "impact_score": score,
            "completion_rate": step_completion_rates.get(step, 0),
        }
        for step, score in sorted_steps
        if score < 0
    ]

    return result


def calculate_point_biserial_correlation(
    binary_values: List[int], continuous_values: List[float]
) -> float:
    """
    Calculate point-biserial correlation between binary and continuous variables.

    Args:
        binary_values: List of binary values (0 or 1)
        continuous_values: List of continuous values

    Returns:
        float: Point-biserial correlation coefficient (-1 to 1)
    """
    if len(binary_values) != len(continuous_values) or len(binary_values) < 2:
        return 0.0

    try:
        # Count occurrences of each binary value
        n0 = sum(1 for x in binary_values if x == 0)
        n1 = sum(1 for x in binary_values if x == 1)

        # Skip if all values are the same
        if n0 == 0 or n1 == 0:
            return 0.0

        n = n0 + n1

        # Calculate mean for each group
        mean0 = sum(y for x, y in zip(binary_values, continuous_values) if x == 0) / n0
        mean1 = sum(y for x, y in zip(binary_values, continuous_values) if x == 1) / n1

        # Calculate overall mean and standard deviation
        mean = sum(continuous_values) / n
        var = sum((y - mean) ** 2 for y in continuous_values) / n

        # Skip if no variance
        if var == 0:
            return 0.0

        std = var**0.5

        # Calculate point-biserial correlation
        rpb = ((mean1 - mean0) / std) * ((n0 * n1 / n**2) ** 0.5)

        return rpb
    except Exception:
        return 0.0
