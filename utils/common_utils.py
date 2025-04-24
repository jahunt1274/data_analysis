"""
Utility functions for data analysis.

This module provides general-purpose utility functions for the data analysis system,
including statistical calculations, data transformations, and time-based operations.
These utilities are designed to be reusable across all analyzer modules.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import math


# Statistical Utilities


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two lists of values.

    Args:
        x: First list of values
        y: Second list of values

    Returns:
        float: Correlation coefficient (-1 to 1) or 0 if calculation fails
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    try:
        # Calculate means
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        # Calculate covariance and standard deviations
        covariance = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
        stdev_x = math.sqrt(sum((x_i - mean_x) ** 2 for x_i in x))
        stdev_y = math.sqrt(sum((y_i - mean_y) ** 2 for y_i in y))

        # Calculate correlation
        if stdev_x > 0 and stdev_y > 0:
            return covariance / (stdev_x * stdev_y)
        else:
            return 0.0
    except Exception:
        return 0.0


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate the Gini coefficient of inequality.

    A Gini coefficient of 0 represents perfect equality,
    while 1 represents perfect inequality.

    Args:
        values: List of values

    Returns:
        float: Gini coefficient
    """
    # Handle edge cases
    if not values or all(x == 0 for x in values):
        return 0.0

    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate cumulative sum
    cum_values = [sum(sorted_values[: i + 1]) for i in range(n)]
    total = cum_values[-1]

    # Calculate Gini coefficient
    if total == 0:
        return 0.0

    fair_area = sum(range(1, n + 1)) / n / n
    actual_area = sum((n - i) * sorted_values[i] for i in range(n)) / n / total

    return 2 * (fair_area - actual_area)


def calculate_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a list of values.

    Args:
        values: List of values to analyze

    Returns:
        Dict with summary statistics
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
            "coefficient_of_variation": 0.0,
        }

    n = len(values)
    mean = sum(values) / n

    # Sort for median and range
    sorted_values = sorted(values)
    median = (
        sorted_values[n // 2]
        if n % 2 == 1
        else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )

    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)

    # Calculate coefficient of variation
    cv = std_dev / mean if mean != 0 else 0.0

    return {
        "count": n,
        "mean": mean,
        "median": median,
        "min": min(values),
        "max": max(values),
        "std_dev": std_dev,
        "coefficient_of_variation": cv,
    }


def group_values_into_ranges(
    values: List[float], max_value: float, num_ranges: int = 5
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


def calculate_distribution_percentages(
    counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate percentage distribution from count data.

    Args:
        counts: Dict mapping categories to counts

    Returns:
        Dict with counts and percentages
    """
    total = sum(counts.values())
    if total == 0:
        return {
            category: {"count": count, "percentage": 0.0}
            for category, count in counts.items()
        }

    return {
        category: {"count": count, "percentage": count / total}
        for category, count in counts.items()
    }


# Time-based Utilities


def generate_time_periods(
    start_date: datetime, end_date: datetime, interval: str = "day"
) -> List[Tuple[datetime, datetime, str]]:
    """
    Generate time periods between start and end dates.

    Args:
        start_date: Start date
        end_date: End date
        interval: Time interval ('day', 'week', 'month')

    Returns:
        List of (period_start, period_end, period_label) tuples
    """
    periods = []
    current = start_date

    if interval == "day":
        while current <= end_date:
            period_end = datetime(current.year, current.month, current.day, 23, 59, 59)
            label = current.strftime("%Y-%m-%d")
            periods.append((current, period_end, label))
            current = current + timedelta(days=1)
            current = datetime(current.year, current.month, current.day, 0, 0, 0)

    elif interval == "week":
        # Start from the beginning of the week
        current = current - timedelta(days=current.weekday())
        current = datetime(current.year, current.month, current.day, 0, 0, 0)

        while current <= end_date:
            period_end = current + timedelta(days=6, hours=23, minutes=59, seconds=59)
            label = (
                f'{current.strftime("%Y-%m-%d")} to {period_end.strftime("%Y-%m-%d")}'
            )
            periods.append((current, period_end, label))
            current = current + timedelta(days=7)

    elif interval == "month":
        # Start from the beginning of the month
        current = datetime(current.year, current.month, 1, 0, 0, 0)

        while current <= end_date:
            # Find the last day of the month
            if current.month == 12:
                next_month = datetime(current.year + 1, 1, 1)
            else:
                next_month = datetime(current.year, current.month + 1, 1)

            period_end = next_month - timedelta(seconds=1)
            label = current.strftime("%Y-%m")
            periods.append((current, period_end, label))
            current = next_month

    return periods


def calculate_time_differences(
    datetimes: List[datetime], unit: str = "hours"
) -> List[float]:
    """
    Calculate time differences between consecutive datetimes.

    Args:
        datetimes: List of datetime objects in chronological order
        unit: Unit for time differences ('seconds', 'minutes', 'hours', 'days')

    Returns:
        List of time differences in the specified unit
    """
    if len(datetimes) < 2:
        return []

    # Sort to ensure chronological order
    sorted_times = sorted(datetimes)

    # Calculate differences
    differences = []
    for i in range(1, len(sorted_times)):
        diff = sorted_times[i] - sorted_times[i - 1]

        # Convert to requested unit
        if unit == "seconds":
            differences.append(diff.total_seconds())
        elif unit == "minutes":
            differences.append(diff.total_seconds() / 60)
        elif unit == "hours":
            differences.append(diff.total_seconds() / 3600)
        elif unit == "days":
            differences.append(diff.total_seconds() / 86400)
        else:
            differences.append(diff.total_seconds())  # Default to seconds

    return differences


def get_date_range(
    datetimes: List[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the minimum and maximum dates from a list of datetimes.

    Args:
        datetimes: List of datetime objects

    Returns:
        Tuple of (min_date, max_date), or (None, None) if the list is empty
    """
    if not datetimes:
        return None, None

    return min(datetimes), max(datetimes)


def group_by_time_period(
    items: List[Dict[str, Any]], timestamp_field: str, period: str = "day"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group items by time period based on a timestamp field.

    Args:
        items: List of dictionaries with timestamp fields
        timestamp_field: Field name containing the datetime object
        period: Time period for grouping ('day', 'week', 'month', 'hour')

    Returns:
        Dict mapping period labels to lists of items
    """
    result = defaultdict(list)

    for item in items:
        if timestamp_field not in item:
            continue

        timestamp = item[timestamp_field]

        if period == "day":
            key = timestamp.strftime("%Y-%m-%d")
        elif period == "week":
            # ISO week format: YYYY-Www
            key = f"{timestamp.isocalendar()[0]}-W{timestamp.isocalendar()[1]:02d}"
        elif period == "month":
            key = timestamp.strftime("%Y-%m")
        elif period == "hour":
            key = timestamp.strftime("%Y-%m-%d %H:00")
        else:
            # Default to day
            key = timestamp.strftime("%Y-%m-%d")

        result[key].append(item)

    return dict(result)


# Data Transformation Utilities


def get_step_display_name(
    step_name: str,
    framework_type: str,
    step_number_lookup: Optional[Dict[str, int]] = None,
) -> str:
    """
    Convert a step name to a more readable display name.

    Args:
        step_name: The step name (e.g., "market-segmentation")
        framework_type: The framework type identifier
        step_number_lookup: Optional dict mapping step names to numbers

    Returns:
        str: Display name (e.g., "1. Market Segmentation")
    """
    # Format display name
    display_name = step_name.replace("-", " ").title()

    # Get step number if available
    step_number = None
    if step_number_lookup and step_name in step_number_lookup:
        step_number = step_number_lookup[step_name]

    # Format with step number if available
    if step_number is not None:
        return f"{step_number}. {display_name}"

    return display_name


def is_linear_progression(step_sequence: List[str], framework_steps: List[str]) -> bool:
    """
    Determine if a sequence of steps follows a linear progression through a framework.

    Args:
        step_sequence: List of steps in order of completion
        framework_steps: List of all steps in the framework in standard order

    Returns:
        bool: True if progression is linear, False otherwise
    """
    if not step_sequence:
        return False

    # Get indices of steps in framework order
    try:
        indices = [framework_steps.index(step) for step in step_sequence]
    except ValueError:
        # Step not in framework_steps
        return False

    # Check if indices are in non-decreasing order
    return all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))


def find_common_subsequences(
    sequences: List[List[str]], min_length: int = 2, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Find common subsequences across multiple sequences.

    Args:
        sequences: List of sequences to analyze
        min_length: Minimum subsequence length to consider
        max_results: Maximum number of results to return

    Returns:
        List of common subsequence data, sorted by frequency
    """
    if not sequences:
        return []

    # Count subsequences
    subsequence_counts = defaultdict(int)

    for sequence in sequences:
        # Generate all subsequences of minimum length
        for i in range(len(sequence) - min_length + 1):
            for j in range(i + min_length, len(sequence) + 1):
                subsequence = tuple(sequence[i:j])
                subsequence_counts[subsequence] += 1

    # Find common subsequences (appearing in at least 2 sequences)
    common_subseqs = [
        {"steps": list(subseq), "count": count, "frequency": count / len(sequences)}
        for subseq, count in subsequence_counts.items()
        if count >= 2
    ]

    # Sort by frequency (descending)
    common_subseqs.sort(key=lambda x: x["frequency"], reverse=True)

    # Return top results
    return common_subseqs[:max_results]


def classify_engagement_level(
    metrics: Dict[str, Any], thresholds: Dict[str, Tuple[float, float]] = None
) -> str:
    """
    Classify a user's engagement level based on activity metrics.

    Args:
        metrics: Dict with user metrics (idea_count, step_count, etc.)
        thresholds: Optional custom thresholds for classification
            Format: {'ideas': (min_high, min_medium), 'steps': (min_high, min_medium)}

    Returns:
        str: Engagement level ('high', 'medium', 'low')
    """
    # Default thresholds
    if not thresholds:
        # Format: (high_threshold, medium_threshold)
        thresholds = {
            "ideas": (3, 1),  # High: 3+ ideas, Medium: 1-2 ideas
            "steps": (10, 3),  # High: 10+ steps, Medium: 3-9 steps
        }

    idea_count = metrics.get("idea_count", 0)
    step_count = metrics.get("step_count", 0)

    # Determine engagement level
    if idea_count >= thresholds["ideas"][0] or step_count >= thresholds["steps"][0]:
        return "high"
    elif idea_count >= thresholds["ideas"][1] or step_count >= thresholds["steps"][1]:
        return "medium"
    else:
        return "low"
