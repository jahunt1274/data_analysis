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


def merge_user_data_with_metrics(
    users: List[Any], ideas_repository: Any, steps_repository: Any
) -> List[Dict[str, Any]]:
    """
    Merge user objects with their activity metrics.

    Args:
        users: List of user objects
        ideas_repository: Repository for accessing ideas
        steps_repository: Repository for accessing steps

    Returns:
        List of user data enriched with activity metrics
    """
    enriched_users = []

    for user in users:
        if not user.email:
            continue

        # Get user's ideas
        ideas = ideas_repository.find_by_owner(user.email)

        # Count ideas with steps
        ideas_with_steps = 0
        total_steps = 0

        for idea in ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
            steps = steps_repository.find_by_idea_id(idea_id)

            if steps:
                ideas_with_steps += 1
                total_steps += len(steps)

        # Create enriched user data
        user_data = {
            "email": user.email,
            "name": user.name,
            "user_type": (
                user.get_user_type().value if user.get_user_type() else "unknown"
            ),
            "department": user.get_department() or "unknown",
            "experience": (
                user.orbit_profile.experience
                if user.orbit_profile and user.orbit_profile.experience
                else "unknown"
            ),
            "idea_count": len(ideas),
            "ideas_with_steps": ideas_with_steps,
            "step_count": total_steps,
            "engagement_level": user.get_engagement_level().value,
            "content_score": user.scores.content if user.scores else 0,
            "completion_score": user.scores.completion if user.scores else 0,
        }

        enriched_users.append(user_data)

    return enriched_users


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


def identify_team_roles(team_members: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Identify potential team roles based on activity patterns.

    Args:
        team_members: List of team member data with activity metrics

    Returns:
        Dict mapping role types to lists of member emails
    """
    roles = {
        "primary_contributor": [],
        "idea_creator": [],
        "implementer": [],
        "occasional_contributor": [],
        "inactive": [],
    }

    # Skip if no team members
    if not team_members:
        return roles

    # Calculate team totals
    total_ideas = sum(member["idea_count"] for member in team_members)
    total_steps = sum(member["step_count"] for member in team_members)

    # Ideal equal distribution
    avg_ideas_per_member = (
        total_ideas / len(team_members) if len(team_members) > 0 else 0
    )
    avg_steps_per_member = (
        total_steps / len(team_members) if len(team_members) > 0 else 0
    )

    # Categorize members
    for member in team_members:
        if member["idea_count"] == 0 and member["step_count"] == 0:
            # Inactive member
            roles["inactive"].append(member["email"])
        elif (
            member["idea_count"] > avg_ideas_per_member * 1.5
            and member["step_count"] > avg_steps_per_member * 1.5
        ):
            # Primary contributor (high on both ideas and steps)
            roles["primary_contributor"].append(member["email"])
        elif (
            member["idea_count"] > avg_ideas_per_member * 1.5
            and member["step_count"] <= avg_steps_per_member
        ):
            # Idea creator but not implementer
            roles["idea_creator"].append(member["email"])
        elif (
            member["idea_count"] <= avg_ideas_per_member
            and member["step_count"] > avg_steps_per_member * 1.5
        ):
            # Implementer but not creator
            roles["implementer"].append(member["email"])
        else:
            # Occasional contributor
            roles["occasional_contributor"].append(member["email"])

    return roles
