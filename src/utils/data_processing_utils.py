"""
Data processing utilities for the data analysis system.

This module provides utilities for processing and transforming data specific to
the JetPack/Orbit data structures, helping with common data preparation tasks
across analyzer modules.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime


def extract_timestamps_from_steps(steps: List[Any]) -> List[datetime]:
    """
    Extract creation timestamps from a list of steps.

    Args:
        steps: List of step objects

    Returns:
        List of datetime objects representing creation timestamps
    """
    return [step.get_creation_date() for step in steps if step.get_creation_date()]


def extract_user_emails_from_team(team: Any) -> List[str]:
    """
    Extract member email addresses from a team.

    Args:
        team: Team object

    Returns:
        List of email addresses for team members
    """
    return [email for email in team.get_member_emails() if email]


def filter_steps_by_framework(steps: List[Any], framework: str) -> List[Any]:
    """
    Filter a list of steps to include only those for a specific framework.

    Args:
        steps: List of step objects
        framework: Framework identifier

    Returns:
        List of steps that belong to the specified framework
    """
    return [step for step in steps if step.framework == framework]


def identify_shared_ideas(
    member_emails: List[str], ideas: List[Any], steps_repository: Any
) -> List[Dict[str, Any]]:
    """
    Identify ideas that have contributions from multiple team members.

    Args:
        member_emails: List of email addresses for team members
        ideas: List of idea objects
        steps_repository: Repository for accessing steps

    Returns:
        List of shared idea data with contributors
    """
    shared_ideas = []

    for idea in ideas:
        if not idea.id:
            continue

        idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

        # Get all steps for this idea
        idea_steps = steps_repository.find_by_idea_id(idea_id)

        # Get unique contributors to this idea
        contributors = set()
        for step in idea_steps:
            if step.owner and step.owner in member_emails:
                contributors.add(step.owner)

        # Check if multiple team members contributed
        if len(contributors) > 1:
            shared_ideas.append(
                {
                    "idea_id": idea_id,
                    "contributors": list(contributors),
                    "steps": idea_steps,
                }
            )

    return shared_ideas

def get_step_completion_data(
    ideas: List[Any], framework_steps: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate step completion rates across a set of ideas.

    Args:
        ideas: List of idea objects
        framework_steps: List of steps in the framework

    Returns:
        Dict mapping step names to completion metrics
    """
    # Initialize step completion counts
    step_counts = {step: 0 for step in framework_steps}
    total_ideas = len(ideas)

    # Skip if no ideas
    if total_ideas == 0:
        return {step: {"count": 0, "rate": 0.0} for step in framework_steps}

    # Count completed steps
    for idea in ideas:
        for step in framework_steps:
            if idea.has_step(step):
                step_counts[step] += 1

    # Calculate rates
    return {
        step: {"count": count, "rate": count / total_ideas}
        for step, count in step_counts.items()
    }


def get_activity_metrics_by_time(steps: List[Any]) -> Dict[str, Any]:
    """
    Calculate activity metrics by time of day and day of week.

    Args:
        steps: List of step objects with creation dates

    Returns:
        Dict with activity metrics by time
    """
    # Initialize metrics
    metrics = {
        "daily_distribution": {},
        "hourly_distribution": {},
    }

    # Initialize with zeros
    for day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]:
        metrics["daily_distribution"][day] = 0

    for hour in range(24):
        metrics["hourly_distribution"][str(hour)] = 0

    # Filter steps with valid creation dates
    steps_with_dates = [step for step in steps if step.get_creation_date()]

    # Skip if no valid steps
    if not steps_with_dates:
        return metrics

    # Process step timestamps
    for step in steps_with_dates:
        # Track day of week
        day_name = step.get_creation_date().strftime("%A")
        metrics["daily_distribution"][day_name] += 1

        # Track hour of day
        hour = step.get_creation_date().hour
        metrics["hourly_distribution"][str(hour)] += 1

    # Calculate percentages
    total_steps = len(steps_with_dates)

    metrics["daily_distribution"] = {
        day: count / total_steps for day, count in metrics["daily_distribution"].items()
    }

    metrics["hourly_distribution"] = {
        hour: count / total_steps
        for hour, count in metrics["hourly_distribution"].items()
    }

    return metrics


def group_steps_into_sessions(
    steps: List[Any], gap_threshold_minutes: int = 30
) -> List[Dict[str, Any]]:
    """
    Group steps into sessions based on time gaps.

    Args:
        steps: List of step objects with creation dates
        gap_threshold_minutes: Maximum time gap in minutes to consider steps part of the same session

    Returns:
        List of session data
    """
    # Skip if no steps
    if not steps:
        return []

    # Filter steps with creation dates
    steps_with_dates = [step for step in steps if step.get_creation_date()]

    # Skip if no steps with dates
    if not steps_with_dates:
        return []

    # Sort steps by creation date
    sorted_steps = sorted(steps_with_dates, key=lambda s: s.get_creation_date())

    # Group into sessions
    sessions = []
    current_session = {
        "start_time": sorted_steps[0].get_creation_date(),
        "end_time": sorted_steps[0].get_creation_date(),
        "steps": [sorted_steps[0]],
    }

    for step in sorted_steps[1:]:
        # Calculate time gap
        time_gap = (
            step.get_creation_date() - current_session["end_time"]
        ).total_seconds() / 60

        if time_gap <= gap_threshold_minutes:
            # Add to current session
            current_session["steps"].append(step)
            current_session["end_time"] = step.get_creation_date()
        else:
            # End current session and start a new one
            duration_minutes = (
                current_session["end_time"] - current_session["start_time"]
            ).total_seconds() / 60
            current_session["duration_minutes"] = duration_minutes
            sessions.append(current_session)

            # Start new session
            current_session = {
                "start_time": step.get_creation_date(),
                "end_time": step.get_creation_date(),
                "steps": [step],
            }

    # Add the last session
    duration_minutes = (
        current_session["end_time"] - current_session["start_time"]
    ).total_seconds() / 60
    current_session["duration_minutes"] = duration_minutes
    sessions.append(current_session)

    return sessions


def categorize_session_lengths(sessions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Categorize sessions by duration.

    Args:
        sessions: List of session data with duration_minutes field

    Returns:
        Dict mapping duration categories to counts
    """
    categories = {
        "under_5min": 0,
        "5_15min": 0,
        "15_30min": 0,
        "30_60min": 0,
        "1_3hr": 0,
        "over_3hr": 0,
    }

    for session in sessions:
        duration = session["duration_minutes"]

        if duration < 5:
            categories["under_5min"] += 1
        elif duration < 15:
            categories["5_15min"] += 1
        elif duration < 30:
            categories["15_30min"] += 1
        elif duration < 60:
            categories["30_60min"] += 1
        elif duration < 180:  # 3 hours
            categories["1_3hr"] += 1
        else:
            categories["over_3hr"] += 1

    return categories

"""
Utility functions for datetime standardization.

This module provides helper functions to ensure all datetime objects
are consistently either timezone-aware or timezone-naive.
"""

from datetime import datetime, timezone

def standardize_datetime(dt):
    """
    Standardize a datetime object to be timezone-naive in UTC.
    
    If the datetime is timezone-aware, convert to UTC and remove timezone info.
    If the datetime is already timezone-naive, return as is.
    
    Args:
        dt: A datetime object or None
        
    Returns:
        datetime: A timezone-naive datetime in UTC or None if input is None
    """
    if dt is None:
        return None
    
    if not isinstance(dt, datetime):
        return dt
        
    # If timezone aware, convert to UTC and make naive
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    
    # Already naive, return as is
    return dt

def compare_datetimes(dt1, dt2):
    """
    Safely compare two datetime objects.
    
    Ensures both datetimes are standardized before comparison.
    
    Args:
        dt1: First datetime object
        dt2: Second datetime object
        
    Returns:
        int: -1 if dt1 < dt2, 0 if dt1 == dt2, 1 if dt1 > dt2
        or None if either input is None
    """
    if dt1 is None or dt2 is None:
        return None
        
    std_dt1 = standardize_datetime(dt1)
    std_dt2 = standardize_datetime(dt2)
    
    if std_dt1 < std_dt2:
        return -1
    elif std_dt1 > std_dt2:
        return 1
    else:
        return 0

def date_in_range(dt, start, end):
    """
    Check if a date is within a range.
    
    All datetimes are standardized before comparison.
    
    Args:
        dt: The datetime to check
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
        
    Returns:
        bool: True if dt is within the range, False otherwise
        or None if dt is None
    """
    if dt is None:
        return None
        
    std_dt = standardize_datetime(dt)
    std_start = standardize_datetime(start)
    std_end = standardize_datetime(end)
    
    return std_start <= std_dt <= std_end