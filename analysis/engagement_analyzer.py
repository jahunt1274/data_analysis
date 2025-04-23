"""
Engagement analyzer for the data analysis system.

This module provides functionality for analyzing user engagement with the
JetPack/Orbit tool, including engagement levels, usage patterns, and
demographic breakdowns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

from ..data.data_repository import DataRepository
from ..data.models.enums import (
    UserEngagementLevel,
    UserType,
    FrameworkType,
    Semester,
    ToolVersion,
    MetricType,
)


class EngagementAnalyzer:
    """
    Analyzer for user engagement with the JetPack/Orbit tool.

    This class provides methods for analyzing engagement patterns, classifying
    users by engagement level, and identifying trends and correlations in tool usage.
    """

    def __init__(self, data_repository: DataRepository):
        """
        Initialize the engagement analyzer.

        Args:
            data_repository: Data repository for accessing all entity repositories
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._data_repo = data_repository

        # Ensure data is loaded
        self._data_repo.connect()

    def classify_users_by_engagement(
        self,
        course_id: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[UserEngagementLevel, List[Dict[str, Any]]]:
        """
        Classify users into engagement level categories.

        This method segments users into high, medium, and low engagement levels
        based on tool activity metrics like idea creation and step completion.

        Args:
            course_id: Optional course ID to filter users (e.g., "15.390")
            custom_thresholds: Optional custom thresholds for classification
                Format: {'ideas': (min_high, min_medium), 'steps': (min_high, min_medium)}

        Returns:
            Dict mapping engagement levels to lists of user data
        """
        # Default thresholds
        if not custom_thresholds:
            # Format: (high_threshold, medium_threshold)
            custom_thresholds = {
                "ideas": (3, 1),  # High: 3+ ideas, Medium: 1-2 ideas
                "steps": (10, 3),  # High: 10+ steps, Medium: 3-9 steps
            }

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Prepare result structure
        result = {
            UserEngagementLevel.HIGH: [],
            UserEngagementLevel.MEDIUM: [],
            UserEngagementLevel.LOW: [],
        }

        # Process each user
        for user in users:
            if not user.email:
                continue

            # Get user's ideas and calculate metrics
            ideas = self._data_repo.ideas.find_by_owner(user.email)
            idea_count = len(ideas)

            # Initialize step count
            step_count = 0

            # Get steps for each idea
            ideas_with_steps = 0
            for idea in ideas:
                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    steps = self._data_repo.steps.find_by_idea_id(idea_id)
                    if steps:
                        ideas_with_steps += 1
                    step_count += len(steps)

            # Calculate engagement score (simplified version)
            idea_score = min(1.0, idea_count / custom_thresholds["ideas"][0])
            step_score = min(1.0, step_count / custom_thresholds["steps"][0])

            # Combined engagement score (weighted average)
            engagement_score = (idea_score * 0.4) + (step_score * 0.6)

            # Determine engagement level
            if (
                idea_count >= custom_thresholds["ideas"][0]
                or step_count >= custom_thresholds["steps"][0]
            ):
                level = UserEngagementLevel.HIGH
            elif (
                idea_count >= custom_thresholds["ideas"][1]
                or step_count >= custom_thresholds["steps"][1]
            ):
                level = UserEngagementLevel.MEDIUM
            else:
                level = UserEngagementLevel.LOW

            # Create user summary
            user_summary = {
                "email": user.email,
                "name": user.name,
                "user_type": user.get_user_type().value,
                "idea_count": idea_count,
                "step_count": step_count,
                "ideas_with_steps": ideas_with_steps,
                "engagement_score": engagement_score,
                "score_components": {
                    "idea_score": idea_score,
                    "step_score": step_score,
                },
                "department": user.get_department(),
                "created_at": user.get_creation_date(),
                "last_login": user.get_last_login_date(),
            }

            # Add to appropriate category
            result[level].append(user_summary)

        return result

    def get_engagement_metrics_over_time(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "week",
        metric_type: str = "activity",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate engagement metrics over time.

        Args:
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            interval: Time interval for grouping ('day', 'week', 'month')
            metric_type: Type of metric ('activity', 'users', 'ideas', 'steps')

        Returns:
            Dict with time series data for specified metrics
        """
        # Get date bounds if not specified
        if not start_date or not end_date:
            # Find earliest and latest dates in data
            idea_min, idea_max = self._data_repo.ideas.get_date_range("created")
            step_min, step_max = self._data_repo.steps.get_date_range("created_at")
            user_min, user_max = self._data_repo.users.get_date_range("created")

            # Combine to find overall range
            date_mins = [d for d in [idea_min, step_min, user_min] if d]
            date_maxs = [d for d in [idea_max, step_max, user_max] if d]

            if date_mins and not start_date:
                start_date = min(date_mins)
            if date_maxs and not end_date:
                end_date = max(date_maxs)

        # Default to last 6 months if still no dates
        if not start_date:
            start_date = datetime.now() - timedelta(days=180)
        if not end_date:
            end_date = datetime.now()

        # Initialize time periods based on interval
        time_periods = self._generate_time_periods(start_date, end_date, interval)

        # Initialize result data structure
        result = {
            "timeline": [],
            "new_users": [],
            "active_users": [],
            "new_ideas": [],
            "new_steps": [],
            "cumulative_users": [],
            "cumulative_ideas": [],
            "cumulative_steps": [],
        }

        # Get all users, ideas, and steps
        all_users = self._data_repo.users.get_all()
        all_ideas = self._data_repo.ideas.get_all()
        all_steps = self._data_repo.steps.get_all()

        # Track cumulative totals
        cumulative_users = 0
        cumulative_ideas = 0
        cumulative_steps = 0

        # Process each time period
        for period_start, period_end, period_label in time_periods:
            # Count new users in this period
            new_users = sum(
                1
                for user in all_users
                if user.get_creation_date()
                and period_start <= user.get_creation_date() < period_end
            )

            # Count active users in this period (with login or idea/step creation)
            active_users = set()

            # Users with login activity
            for user in all_users:
                if (
                    user.get_last_login_date()
                    and period_start <= user.get_last_login_date() < period_end
                ):
                    if user.email:
                        active_users.add(user.email)

            # Count new ideas in this period
            new_ideas = []
            for idea in all_ideas:
                if (
                    idea.get_creation_date()
                    and period_start <= idea.get_creation_date() < period_end
                ):
                    new_ideas.append(idea)
                    # Add idea owner to active users
                    if idea.owner:
                        active_users.add(idea.owner)

            # Count new steps in this period
            new_steps = []
            for step in all_steps:
                if (
                    step.get_creation_date()
                    and period_start <= step.get_creation_date() < period_end
                ):
                    new_steps.append(step)
                    # Add step owner to active users
                    if step.owner:
                        active_users.add(step.owner)

            # Update cumulative totals
            cumulative_users += new_users
            cumulative_ideas += len(new_ideas)
            cumulative_steps += len(new_steps)

            # Add data for this period
            result["timeline"].append(period_label)
            result["new_users"].append(new_users)
            result["active_users"].append(len(active_users))
            result["new_ideas"].append(len(new_ideas))
            result["new_steps"].append(len(new_steps))
            result["cumulative_users"].append(cumulative_users)
            result["cumulative_ideas"].append(cumulative_ideas)
            result["cumulative_steps"].append(cumulative_steps)

        return result

    def analyze_dropout_patterns(
        self,
        course_id: Optional[str] = None,
        inactivity_threshold: int = 30,  # Days of inactivity to consider as dropout
    ) -> Dict[str, Any]:
        """
        Analyze patterns of user dropout from the tool.

        Identifies when users stop using the tool and what factors might contribute.

        Args:
            course_id: Optional course ID to filter users
            inactivity_threshold: Days of inactivity to consider as dropout

        Returns:
            Dict with dropout analysis results
        """
        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Initialize result structure
        result = {
            "dropout_rate": 0.0,
            "dropout_by_stage": {},
            "avg_days_active": 0,
            "avg_days_to_dropout": 0,
            "dropout_by_user_type": {},
            "retention_factors": [],
        }

        # Track user activity timelines
        user_timelines = []
        dropout_count = 0
        active_days_sum = 0
        dropout_days_sum = 0

        # Current date reference (for inactivity calculation)
        today = datetime.now()

        # Process each user
        for user in users:
            if not user.email:
                continue

            # Get user's ideas and steps
            ideas = self._data_repo.ideas.find_by_owner(user.email)

            # Collect all activity timestamps
            activity_dates = []

            # Add user creation date
            if user.get_creation_date():
                activity_dates.append(user.get_creation_date())

            # Add last login date
            if user.get_last_login_date():
                activity_dates.append(user.get_last_login_date())

            # Add idea creation dates
            for idea in ideas:
                if idea.get_creation_date():
                    activity_dates.append(idea.get_creation_date())

                # Add step creation dates
                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    steps = self._data_repo.steps.find_by_idea_id(idea_id)

                    for step in steps:
                        if step.get_creation_date():
                            activity_dates.append(step.get_creation_date())

            # Calculate activity metrics
            if activity_dates:
                # Sort activity dates
                activity_dates.sort()

                # First and last activity
                first_activity = activity_dates[0]
                last_activity = activity_dates[-1]

                # Calculate days active (between first and last activity)
                days_active = (last_activity - first_activity).days + 1

                # Determine if user has dropped out (inactive for threshold period)
                days_since_last_activity = (today - last_activity).days
                is_dropout = days_since_last_activity >= inactivity_threshold

                # Update running totals
                active_days_sum += days_active

                if is_dropout:
                    dropout_count += 1
                    dropout_days_sum += days_since_last_activity

                # Determine dropout stage (if applicable)
                dropout_stage = None
                if is_dropout:
                    # Check if user created any ideas
                    if not ideas:
                        dropout_stage = "pre_idea"
                    else:
                        # Check how far they got in the framework
                        max_steps = 0
                        for idea in ideas:
                            if idea.id:
                                idea_id = (
                                    idea.id.oid
                                    if hasattr(idea.id, "oid")
                                    else str(idea.id)
                                )
                                steps = self._data_repo.steps.find_by_idea_id(idea_id)
                                if len(steps) > max_steps:
                                    max_steps = len(steps)

                        if max_steps == 0:
                            dropout_stage = "idea_created_no_steps"
                        elif max_steps < 5:
                            dropout_stage = "early_steps"
                        elif max_steps < 15:
                            dropout_stage = "mid_framework"
                        else:
                            dropout_stage = "late_framework"

                # Track dropout stage
                if dropout_stage:
                    if dropout_stage in result["dropout_by_stage"]:
                        result["dropout_by_stage"][dropout_stage] += 1
                    else:
                        result["dropout_by_stage"][dropout_stage] = 1

                # Track user type dropout
                user_type = user.get_user_type().value
                if is_dropout:
                    if user_type in result["dropout_by_user_type"]:
                        result["dropout_by_user_type"][user_type] += 1
                    else:
                        result["dropout_by_user_type"][user_type] = 1

                # Add to user timelines for pattern analysis
                user_timelines.append(
                    {
                        "user_email": user.email,
                        "user_type": user_type,
                        "first_activity": first_activity,
                        "last_activity": last_activity,
                        "days_active": days_active,
                        "activity_count": len(activity_dates),
                        "is_dropout": is_dropout,
                        "days_since_last": days_since_last_activity,
                        "dropout_stage": dropout_stage,
                    }
                )

        # Calculate aggregate metrics
        total_users = len(users)
        if total_users > 0:
            result["dropout_rate"] = dropout_count / total_users

        if user_timelines:
            result["avg_days_active"] = active_days_sum / len(user_timelines)

        if dropout_count > 0:
            result["avg_days_to_dropout"] = dropout_days_sum / dropout_count

        # Analyze potential retention factors
        # (correlation between user attributes and longer retention)

        # Analyze if team membership correlates with retention
        team_members = set()
        for team in self._data_repo.teams.get_all():
            team_members.update(team.get_member_emails())

        team_member_retention = {
            "in_team": {"count": 0, "dropouts": 0},
            "not_in_team": {"count": 0, "dropouts": 0},
        }

        for timeline in user_timelines:
            if timeline["user_email"] in team_members:
                team_member_retention["in_team"]["count"] += 1
                if timeline["is_dropout"]:
                    team_member_retention["in_team"]["dropouts"] += 1
            else:
                team_member_retention["not_in_team"]["count"] += 1
                if timeline["is_dropout"]:
                    team_member_retention["not_in_team"]["dropouts"] += 1

        # Calculate dropout rates
        for category, data in team_member_retention.items():
            if data["count"] > 0:
                data["dropout_rate"] = data["dropouts"] / data["count"]
            else:
                data["dropout_rate"] = 0

        # Add to retention factors
        result["retention_factors"].append(
            {"factor": "team_membership", "data": team_member_retention}
        )

        return result

    def get_engagement_by_demographic(
        self,
        course_id: Optional[str] = None,
        include_types: bool = True,
        include_departments: bool = True,
        include_experience: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze engagement patterns by user demographics.

        Args:
            course_id: Optional course ID to filter users
            include_types: Whether to include user type analysis
            include_departments: Whether to include department analysis
            include_experience: Whether to include experience level analysis

        Returns:
            Dict with engagement metrics by demographic categories
        """
        # Initialize result structure
        result = {}

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Analyze by user type if requested
        if include_types:
            user_type_metrics = {
                user_type.value: {
                    "count": 0,
                    "idea_count": 0,
                    "step_count": 0,
                    "avg_ideas_per_user": 0,
                    "avg_steps_per_user": 0,
                    "engagement_levels": {
                        UserEngagementLevel.HIGH.value: 0,
                        UserEngagementLevel.MEDIUM.value: 0,
                        UserEngagementLevel.LOW.value: 0,
                    },
                }
                for user_type in UserType
            }

            # Process each user
            for user in users:
                if not user.email:
                    continue

                user_type = user.get_user_type().value

                # Get user's ideas and steps
                ideas = self._data_repo.ideas.find_by_owner(user.email)
                step_count = 0

                for idea in ideas:
                    if idea.id:
                        idea_id = (
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        steps = self._data_repo.steps.find_by_idea_id(idea_id)
                        step_count += len(steps)

                # Update metrics
                user_type_metrics[user_type]["count"] += 1
                user_type_metrics[user_type]["idea_count"] += len(ideas)
                user_type_metrics[user_type]["step_count"] += step_count

                # Track engagement level
                engagement_level = user.get_engagement_level().value
                user_type_metrics[user_type]["engagement_levels"][engagement_level] += 1

            # Calculate averages
            for user_type, metrics in user_type_metrics.items():
                if metrics["count"] > 0:
                    metrics["avg_ideas_per_user"] = (
                        metrics["idea_count"] / metrics["count"]
                    )
                    metrics["avg_steps_per_user"] = (
                        metrics["step_count"] / metrics["count"]
                    )

            # Add to result
            result["user_type"] = user_type_metrics

        # Analyze by department if requested
        if include_departments:
            department_metrics = defaultdict(
                lambda: {
                    "count": 0,
                    "idea_count": 0,
                    "step_count": 0,
                    "avg_ideas_per_user": 0,
                    "avg_steps_per_user": 0,
                    "engagement_levels": {
                        UserEngagementLevel.HIGH.value: 0,
                        UserEngagementLevel.MEDIUM.value: 0,
                        UserEngagementLevel.LOW.value: 0,
                    },
                }
            )

            # Process each user
            for user in users:
                if not user.email:
                    continue

                department = user.get_department()
                if not department:
                    department = "Unknown"

                # Get user's ideas and steps
                ideas = self._data_repo.ideas.find_by_owner(user.email)
                step_count = 0

                for idea in ideas:
                    if idea.id:
                        idea_id = (
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        steps = self._data_repo.steps.find_by_idea_id(idea_id)
                        step_count += len(steps)

                # Update metrics
                department_metrics[department]["count"] += 1
                department_metrics[department]["idea_count"] += len(ideas)
                department_metrics[department]["step_count"] += step_count

                # Track engagement level
                engagement_level = user.get_engagement_level().value
                department_metrics[department]["engagement_levels"][
                    engagement_level
                ] += 1

            # Calculate averages
            for department, metrics in department_metrics.items():
                if metrics["count"] > 0:
                    metrics["avg_ideas_per_user"] = (
                        metrics["idea_count"] / metrics["count"]
                    )
                    metrics["avg_steps_per_user"] = (
                        metrics["step_count"] / metrics["count"]
                    )

            # Add to result
            result["department"] = dict(department_metrics)

        # Analyze by experience level if requested
        if include_experience:
            experience_metrics = defaultdict(
                lambda: {
                    "count": 0,
                    "idea_count": 0,
                    "step_count": 0,
                    "avg_ideas_per_user": 0,
                    "avg_steps_per_user": 0,
                    "engagement_levels": {
                        UserEngagementLevel.HIGH.value: 0,
                        UserEngagementLevel.MEDIUM.value: 0,
                        UserEngagementLevel.LOW.value: 0,
                    },
                }
            )

            # Process each user
            for user in users:
                if not user.email:
                    continue

                experience = "Unknown"
                if user.orbit_profile and user.orbit_profile.experience:
                    experience = user.orbit_profile.experience

                # Get user's ideas and steps
                ideas = self._data_repo.ideas.find_by_owner(user.email)
                step_count = 0

                for idea in ideas:
                    if idea.id:
                        idea_id = (
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        steps = self._data_repo.steps.find_by_idea_id(idea_id)
                        step_count += len(steps)

                # Update metrics
                experience_metrics[experience]["count"] += 1
                experience_metrics[experience]["idea_count"] += len(ideas)
                experience_metrics[experience]["step_count"] += step_count

                # Track engagement level
                engagement_level = user.get_engagement_level().value
                experience_metrics[experience]["engagement_levels"][
                    engagement_level
                ] += 1

            # Calculate averages
            for experience, metrics in experience_metrics.items():
                if metrics["count"] > 0:
                    metrics["avg_ideas_per_user"] = (
                        metrics["idea_count"] / metrics["count"]
                    )
                    metrics["avg_steps_per_user"] = (
                        metrics["step_count"] / metrics["count"]
                    )

            # Add to result
            result["experience"] = dict(experience_metrics)

        return result

    def calculate_user_burst_activity(
        self,
        burst_window: int = 24,  # Hours
        min_activity_count: int = 3,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze burst patterns of user activity.

        Identifies periods of concentrated activity ("bursts") where users
        engage intensively with the tool over a short period.

        Args:
            burst_window: Time window in hours for defining a burst
            min_activity_count: Minimum number of activities to qualify as a burst
            course_id: Optional course ID to filter users

        Returns:
            Dict with burst activity analysis results
        """
        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Initialize result structure
        result = {
            "burst_patterns": [],
            "user_bursts": {},
            "avg_bursts_per_user": 0,
            "avg_activities_per_burst": 0,
            "burst_hour_distribution": defaultdict(int),
            "burst_day_distribution": defaultdict(int),
            "burst_length_distribution": defaultdict(int),
        }

        # Convert burst window to timedelta
        burst_window_delta = timedelta(hours=burst_window)

        # Track total burst counts
        total_bursts = 0
        total_burst_activities = 0

        # Process each user
        for user in users:
            if not user.email:
                continue

            # Get all activity timestamps for this user
            activities = []

            # Get user's ideas
            ideas = self._data_repo.ideas.find_by_owner(user.email)

            # Add idea creation timestamps
            for idea in ideas:
                if idea.get_creation_date():
                    activities.append(("idea_creation", idea.get_creation_date(), idea))

            # Add step creation timestamps
            for idea in ideas:
                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    steps = self._data_repo.steps.find_by_idea_id(idea_id)

                    for step in steps:
                        if step.get_creation_date():
                            activities.append(
                                ("step_creation", step.get_creation_date(), step)
                            )

            # Sort activities by timestamp
            activities.sort(key=lambda x: x[1])

            # Skip users with insufficient activity
            if len(activities) < min_activity_count:
                continue

            # Identify burst periods
            user_bursts = []
            current_burst = []

            for i, (activity_type, timestamp, entity) in enumerate(activities):
                if not current_burst:
                    # Start a new burst
                    current_burst = [(activity_type, timestamp, entity)]
                elif (timestamp - current_burst[-1][1]) <= burst_window_delta:
                    # Add to current burst
                    current_burst.append((activity_type, timestamp, entity))
                else:
                    # End current burst if it has enough activities
                    if len(current_burst) >= min_activity_count:
                        burst_start = current_burst[0][1]
                        burst_end = current_burst[-1][1]
                        burst_duration = (
                            burst_end - burst_start
                        ).total_seconds() / 3600  # Hours

                        user_bursts.append(
                            {
                                "start_time": burst_start,
                                "end_time": burst_end,
                                "duration_hours": burst_duration,
                                "activity_count": len(current_burst),
                                "activities": current_burst,
                            }
                        )

                        # Update distribution counts
                        result["burst_hour_distribution"][burst_start.hour] += 1
                        result["burst_day_distribution"][
                            burst_start.strftime("%A")
                        ] += 1

                        # Categorize burst length
                        if burst_duration <= 1:
                            length_category = "<1h"
                        elif burst_duration <= 3:
                            length_category = "1-3h"
                        elif burst_duration <= 6:
                            length_category = "3-6h"
                        elif burst_duration <= 12:
                            length_category = "6-12h"
                        else:
                            length_category = ">12h"

                        result["burst_length_distribution"][length_category] += 1

                        # Update totals
                        total_bursts += 1
                        total_burst_activities += len(current_burst)

                    # Start a new burst
                    current_burst = [(activity_type, timestamp, entity)]

            # Check if the last burst is valid
            if len(current_burst) >= min_activity_count:
                burst_start = current_burst[0][1]
                burst_end = current_burst[-1][1]
                burst_duration = (
                    burst_end - burst_start
                ).total_seconds() / 3600  # Hours

                user_bursts.append(
                    {
                        "start_time": burst_start,
                        "end_time": burst_end,
                        "duration_hours": burst_duration,
                        "activity_count": len(current_burst),
                        "activities": current_burst,
                    }
                )

                # Update distribution counts
                result["burst_hour_distribution"][burst_start.hour] += 1
                result["burst_day_distribution"][burst_start.strftime("%A")] += 1

                # Categorize burst length
                if burst_duration <= 1:
                    length_category = "<1h"
                elif burst_duration <= 3:
                    length_category = "1-3h"
                elif burst_duration <= 6:
                    length_category = "3-6h"
                elif burst_duration <= 12:
                    length_category = "6-12h"
                else:
                    length_category = ">12h"

                result["burst_length_distribution"][length_category] += 1

                # Update totals
                total_bursts += 1
                total_burst_activities += len(current_burst)

            # Add user burst data if any bursts were found
            if user_bursts:
                result["user_bursts"][user.email] = {
                    "count": len(user_bursts),
                    "bursts": user_bursts,
                }

        # Calculate summary metrics
        if total_bursts > 0:
            result["avg_activities_per_burst"] = total_burst_activities / total_bursts

        user_with_bursts = len(result["user_bursts"])
        if user_with_bursts > 0:
            result["avg_bursts_per_user"] = total_bursts / user_with_bursts

        # Extract general burst patterns
        burst_patterns = self._analyze_burst_patterns(result["user_bursts"])
        result["burst_patterns"] = burst_patterns

        # Convert defaultdicts to regular dicts for serialization
        result["burst_hour_distribution"] = dict(result["burst_hour_distribution"])
        result["burst_day_distribution"] = dict(result["burst_day_distribution"])
        result["burst_length_distribution"] = dict(result["burst_length_distribution"])

        return result

    def compare_semester_engagement(
        self, semester1: Semester, semester2: Semester, course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare user engagement between two semesters.

        Args:
            semester1: First semester to compare
            semester2: Second semester to compare
            course_id: Optional course ID to filter users

        Returns:
            Dict with semester comparison results
        """
        # Initialize result structure
        result = {
            "user_counts": {},
            "engagement_metrics": {},
            "tool_version_impact": {},
            "retention_metrics": {},
        }

        # Get semester date ranges
        semester_ranges = {
            Semester.FALL_2023: (datetime(2023, 9, 1), datetime(2023, 12, 31)),
            Semester.SPRING_2024: (datetime(2024, 1, 1), datetime(2024, 5, 31)),
            Semester.FALL_2024: (datetime(2024, 9, 1), datetime(2024, 12, 31)),
            Semester.SPRING_2025: (datetime(2025, 1, 1), datetime(2025, 5, 31)),
        }

        # Get users for each semester
        semester1_users = []
        semester2_users = []

        semester1_range = semester_ranges.get(semester1)
        semester2_range = semester_ranges.get(semester2)

        if not semester1_range or not semester2_range:
            self._logger.error(f"Invalid semester provided: {semester1} or {semester2}")
            return {"error": "Invalid semester provided"}

        # Filter users by semester date range and course if specified
        all_users = self._data_repo.users.get_all()

        for user in all_users:
            creation_date = user.get_creation_date()
            if not creation_date:
                continue

            # Check course filter if specified
            if course_id and not user.is_in_course(course_id):
                continue

            # Check semester 1
            if semester1_range[0] <= creation_date <= semester1_range[1]:
                semester1_users.append(user)

            # Check semester 2
            if semester2_range[0] <= creation_date <= semester2_range[1]:
                semester2_users.append(user)

        # Get tool versions for each semester
        semester1_tool = Semester.get_tool_version(semester1)
        semester2_tool = Semester.get_tool_version(semester2)

        # Add basic user counts
        result["user_counts"] = {
            semester1.value: len(semester1_users),
            semester2.value: len(semester2_users),
        }

        # Add tool version info
        result["tool_version_impact"] = {
            semester1.value: {"tool_version": semester1_tool.value},
            semester2.value: {"tool_version": semester2_tool.value},
        }

        # Calculate engagement metrics for each semester
        for semester, users in [
            (semester1, semester1_users),
            (semester2, semester2_users),
        ]:
            if not users:
                continue

            semester_email_list = [user.email for user in users if user.email]

            # Count ideas created by these users
            idea_count = 0
            idea_with_steps_count = 0
            step_count = 0

            for email in semester_email_list:
                ideas = self._data_repo.ideas.find_by_owner(email)
                idea_count += len(ideas)

                for idea in ideas:
                    if idea.id:
                        idea_id = (
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        steps = self._data_repo.steps.find_by_idea_id(idea_id)

                        if steps:
                            idea_with_steps_count += 1

                        step_count += len(steps)

            # Calculate metrics
            result["engagement_metrics"][semester.value] = {
                "total_ideas": idea_count,
                "total_steps": step_count,
                "ideas_with_steps": idea_with_steps_count,
                "avg_ideas_per_user": idea_count / len(users) if users else 0,
                "avg_steps_per_user": step_count / len(users) if users else 0,
                "avg_steps_per_idea": step_count / idea_count if idea_count else 0,
                "idea_to_step_conversion": (
                    idea_with_steps_count / idea_count if idea_count else 0
                ),
            }

            # Calculate engagement levels
            user_by_engagement = self._categorize_users_by_engagement(users)

            result["engagement_metrics"][semester.value]["engagement_levels"] = {
                "high": len(user_by_engagement.get(UserEngagementLevel.HIGH, [])),
                "medium": len(user_by_engagement.get(UserEngagementLevel.MEDIUM, [])),
                "low": len(user_by_engagement.get(UserEngagementLevel.LOW, [])),
            }

            # Calculate retention metrics (e.g., how long users stayed active)
            retention_data = self._calculate_retention_metrics(users)
            result["retention_metrics"][semester.value] = retention_data

        # Calculate direct comparisons
        if (
            semester1.value in result["engagement_metrics"]
            and semester2.value in result["engagement_metrics"]
        ):
            s1_metrics = result["engagement_metrics"][semester1.value]
            s2_metrics = result["engagement_metrics"][semester2.value]

            # Add comparison section
            result["comparison"] = {
                "idea_difference": {
                    "total": s2_metrics["total_ideas"] - s1_metrics["total_ideas"],
                    "percent": (
                        ((s2_metrics["total_ideas"] / s1_metrics["total_ideas"]) - 1)
                        * 100
                        if s1_metrics["total_ideas"]
                        else float("inf")
                    ),
                },
                "step_difference": {
                    "total": s2_metrics["total_steps"] - s1_metrics["total_steps"],
                    "percent": (
                        ((s2_metrics["total_steps"] / s1_metrics["total_steps"]) - 1)
                        * 100
                        if s1_metrics["total_steps"]
                        else float("inf")
                    ),
                },
                "per_user": {
                    "ideas": s2_metrics["avg_ideas_per_user"]
                    - s1_metrics["avg_ideas_per_user"],
                    "steps": s2_metrics["avg_steps_per_user"]
                    - s1_metrics["avg_steps_per_user"],
                },
                "conversion_difference": s2_metrics["idea_to_step_conversion"]
                - s1_metrics["idea_to_step_conversion"],
            }

        return result

    def analyze_tool_version_impact(
        self, course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the impact of different tool versions on user engagement.

        Args:
            course_id: Optional course ID to filter users

        Returns:
            Dict with tool version impact analysis
        """
        # Initialize result structure
        result = {"version_comparison": {}, "feature_impact": {}, "cohort_metrics": {}}

        # Get semester date ranges and tool versions
        semester_tool_versions = {
            Semester.FALL_2023.value: ToolVersion.NONE,
            Semester.SPRING_2024.value: ToolVersion.V1,
            Semester.FALL_2024.value: ToolVersion.V2,
            Semester.SPRING_2025.value: ToolVersion.V2,
        }

        semester_ranges = {
            Semester.FALL_2023.value: (datetime(2023, 9, 1), datetime(2023, 12, 31)),
            Semester.SPRING_2024.value: (datetime(2024, 1, 1), datetime(2024, 5, 31)),
            Semester.FALL_2024.value: (datetime(2024, 9, 1), datetime(2024, 12, 31)),
            Semester.SPRING_2025.value: (datetime(2025, 1, 1), datetime(2025, 5, 31)),
        }

        # Group users by semester
        semester_users = {semester: [] for semester in semester_ranges.keys()}

        # Get all users
        all_users = self._data_repo.users.get_all()

        for user in all_users:
            creation_date = user.get_creation_date()
            if not creation_date:
                continue

            # Check course filter if specified
            if course_id and not user.is_in_course(course_id):
                continue

            # Assign user to semester
            for semester, date_range in semester_ranges.items():
                if date_range[0] <= creation_date <= date_range[1]:
                    semester_users[semester].append(user)
                    break

        # Group by tool version
        tool_version_users = {
            ToolVersion.NONE.value: [],
            ToolVersion.V1.value: [],
            ToolVersion.V2.value: [],
        }

        for semester, users in semester_users.items():
            tool_version = semester_tool_versions[semester].value
            tool_version_users[tool_version].extend(users)

        # Calculate engagement metrics for each tool version
        for version, users in tool_version_users.items():
            if not users:
                continue

            # Get unique user emails
            user_emails = set(user.email for user in users if user.email)

            # Calculate aggregated metrics
            idea_count = 0
            idea_with_steps_count = 0
            step_count = 0

            for email in user_emails:
                ideas = self._data_repo.ideas.find_by_owner(email)
                idea_count += len(ideas)

                for idea in ideas:
                    if idea.id:
                        idea_id = (
                            idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                        )
                        steps = self._data_repo.steps.find_by_idea_id(idea_id)

                        if steps:
                            idea_with_steps_count += 1

                        step_count += len(steps)

            # Calculate metrics
            result["version_comparison"][version] = {
                "user_count": len(user_emails),
                "total_ideas": idea_count,
                "total_steps": step_count,
                "ideas_with_steps": idea_with_steps_count,
                "avg_ideas_per_user": (
                    idea_count / len(user_emails) if user_emails else 0
                ),
                "avg_steps_per_user": (
                    step_count / len(user_emails) if user_emails else 0
                ),
                "avg_steps_per_idea": step_count / idea_count if idea_count else 0,
                "idea_to_step_conversion": (
                    idea_with_steps_count / idea_count if idea_count else 0
                ),
            }

            # Calculate engagement levels
            user_by_engagement = self._categorize_users_by_engagement(users)

            result["version_comparison"][version]["engagement_levels"] = {
                "high": len(user_by_engagement.get(UserEngagementLevel.HIGH, [])),
                "medium": len(user_by_engagement.get(UserEngagementLevel.MEDIUM, [])),
                "low": len(user_by_engagement.get(UserEngagementLevel.LOW, [])),
            }

            # Calculate high engagement percentage
            total_users = sum(len(users) for users in user_by_engagement.values())
            if total_users > 0:
                high_engagement_pct = (
                    len(user_by_engagement.get(UserEngagementLevel.HIGH, []))
                    / total_users
                )
                result["version_comparison"][version][
                    "high_engagement_percentage"
                ] = high_engagement_pct

        # Add feature impact analysis if possible
        # This assumes specific features were added in different versions
        # For example, session tracking was added in v2

        # Check if session_id is being used (a v2 feature)
        v2_session_usage = 0
        v2_steps = 0

        for step in self._data_repo.steps.get_all():
            if step.session_id:
                v2_session_usage += 1
                v2_steps += 1
            elif (
                step.get_creation_date()
                and step.get_creation_date()
                > semester_ranges[Semester.SPRING_2024.value][1]
            ):
                # Count v2 steps without session_id
                v2_steps += 1

        result["feature_impact"]["session_tracking"] = {
            "availability": "v2 only",
            "usage_rate": v2_session_usage / v2_steps if v2_steps else 0,
        }

        # Add cohort metrics for semester comparison
        for semester, users in semester_users.items():
            if not users:
                continue

            # Calculate retention metrics
            retention_data = self._calculate_retention_metrics(users)

            # Add to cohort metrics
            result["cohort_metrics"][semester] = {
                "tool_version": semester_tool_versions[semester].value,
                "user_count": len(users),
                "retention": retention_data,
            }

        return result

    def get_user_activity_timeline(
        self, email: str, include_ideas: bool = True, include_steps: bool = True
    ) -> Dict[str, Any]:
        """
        Get a detailed timeline of activity for a specific user.

        Args:
            email: User's email address
            include_ideas: Whether to include idea creation events
            include_steps: Whether to include step creation events

        Returns:
            Dict with user activity timeline data
        """
        # Initialize result structure
        result = {
            "user_info": {},
            "activity_timeline": [],
            "session_data": [],
            "engagement_summary": {},
        }

        # Get user
        user = self._data_repo.users.find_by_email(email)
        if not user:
            return {"error": f"User not found: {email}"}

        # Add user info
        result["user_info"] = {
            "email": user.email,
            "name": user.name,
            "user_type": user.get_user_type().value if user.get_user_type() else None,
            "creation_date": user.get_creation_date(),
            "last_login": user.get_last_login_date(),
            "department": user.get_department(),
        }

        # Collect all activity events
        timeline_events = []

        # Add user creation event
        if user.get_creation_date():
            timeline_events.append(
                {
                    "event_type": "user_creation",
                    "timestamp": user.get_creation_date(),
                    "details": {"user_email": email},
                }
            )

        # Add login events if available
        if user.get_last_login_date():
            timeline_events.append(
                {
                    "event_type": "user_login",
                    "timestamp": user.get_last_login_date(),
                    "details": {"user_email": email},
                }
            )

        # Add idea creation events if requested
        if include_ideas:
            ideas = self._data_repo.ideas.find_by_owner(email)

            for idea in ideas:
                if idea.get_creation_date():
                    timeline_events.append(
                        {
                            "event_type": "idea_creation",
                            "timestamp": idea.get_creation_date(),
                            "details": {
                                "idea_id": (
                                    idea.id.oid
                                    if hasattr(idea.id, "oid")
                                    else str(idea.id)
                                ),
                                "title": idea.title,
                                "ranking": idea.ranking,
                                "category": (
                                    idea.get_idea_category().value
                                    if idea.get_idea_category()
                                    else None
                                ),
                            },
                        }
                    )

        # Add step creation events if requested
        if include_steps:
            steps = self._data_repo.steps.find_by_owner(email)

            for step in steps:
                if step.get_creation_date():
                    timeline_events.append(
                        {
                            "event_type": "step_creation",
                            "timestamp": step.get_creation_date(),
                            "details": {
                                "step_id": (
                                    step.id.oid
                                    if hasattr(step.id, "oid")
                                    else str(step.id)
                                ),
                                "idea_id": (
                                    step.idea_id.oid
                                    if hasattr(step.idea_id, "oid")
                                    else str(step.idea_id) if step.idea_id else None
                                ),
                                "framework": step.framework,
                                "step": step.step,
                                "version": step.get_version(),
                                "has_user_input": step.has_user_input(),
                                "session_id": (
                                    step.session_id.oid
                                    if hasattr(step.session_id, "oid")
                                    else (
                                        str(step.session_id)
                                        if step.session_id
                                        else None
                                    )
                                ),
                            },
                        }
                    )

        # Sort events by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])

        # Add sorted events to result
        result["activity_timeline"] = timeline_events

        # Build session data if session_id is available (v2 feature)
        sessions = {}

        for event in timeline_events:
            if (
                event["event_type"] == "step_creation"
                and "session_id" in event["details"]
                and event["details"]["session_id"]
            ):
                session_id = event["details"]["session_id"]

                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "start_time": event["timestamp"],
                        "end_time": event["timestamp"],
                        "duration_minutes": 0,
                        "step_count": 0,
                        "steps": [],
                    }

                # Update session data
                session = sessions[session_id]
                session["step_count"] += 1
                session["steps"].append(event["details"])

                # Update session end time if this event is later
                if event["timestamp"] > session["end_time"]:
                    session["end_time"] = event["timestamp"]

        # Calculate session durations
        for session_id, session in sessions.items():
            duration = (
                session["end_time"] - session["start_time"]
            ).total_seconds() / 60
            session["duration_minutes"] = duration

        # Add sessions to result
        result["session_data"] = list(sessions.values())

        # Calculate engagement summary
        if timeline_events:
            first_activity = min(event["timestamp"] for event in timeline_events)
            last_activity = max(event["timestamp"] for event in timeline_events)

            result["engagement_summary"] = {
                "first_activity": first_activity,
                "last_activity": last_activity,
                "days_active": (last_activity - first_activity).days + 1,
                "total_events": len(timeline_events),
                "idea_count": sum(
                    1
                    for event in timeline_events
                    if event["event_type"] == "idea_creation"
                ),
                "step_count": sum(
                    1
                    for event in timeline_events
                    if event["event_type"] == "step_creation"
                ),
                "avg_steps_per_day": 0,  # Will be calculated below
            }

            # Calculate activity distribution
            if result["engagement_summary"]["days_active"] > 0:
                result["engagement_summary"]["avg_steps_per_day"] = (
                    result["engagement_summary"]["step_count"]
                    / result["engagement_summary"]["days_active"]
                )

        return result

    def _generate_time_periods(
        self, start_date: datetime, end_date: datetime, interval: str
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
                period_end = datetime(
                    current.year, current.month, current.day, 23, 59, 59
                )
                label = current.strftime("%Y-%m-%d")
                periods.append((current, period_end, label))
                current = current + timedelta(days=1)
                current = datetime(current.year, current.month, current.day, 0, 0, 0)

        elif interval == "week":
            # Start from the beginning of the week
            current = current - timedelta(days=current.weekday())
            current = datetime(current.year, current.month, current.day, 0, 0, 0)

            while current <= end_date:
                period_end = current + timedelta(
                    days=6, hours=23, minutes=59, seconds=59
                )
                label = f'{current.strftime("%Y-%m-%d")} to {period_end.strftime("%Y-%m-%d")}'
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

    def _analyze_burst_patterns(
        self, user_bursts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze burst patterns across users to identify common patterns.

        Args:
            user_bursts: User burst data from calculate_user_burst_activity

        Returns:
            List of identified burst patterns
        """
        # Extract all bursts
        all_bursts = []
        for user_email, data in user_bursts.items():
            for burst in data["bursts"]:
                burst_copy = burst.copy()
                burst_copy["user_email"] = user_email
                all_bursts.append(burst_copy)

        # Skip pattern analysis if not enough bursts
        if len(all_bursts) < 3:
            return []

        # Identify patterns
        patterns = []

        # Pattern 1: Time of day patterns
        hour_counts = defaultdict(int)
        for burst in all_bursts:
            hour = burst["start_time"].hour
            hour_counts[hour] += 1

        # Find hours with more than average bursts
        avg_bursts_per_hour = len(all_bursts) / 24
        peak_hours = [
            hour
            for hour, count in hour_counts.items()
            if count > avg_bursts_per_hour * 1.5
        ]

        if peak_hours:
            patterns.append(
                {
                    "pattern_type": "time_of_day",
                    "description": "Peak activity hours",
                    "peak_hours": peak_hours,
                    "burst_count_by_hour": dict(hour_counts),
                }
            )

        # Pattern 2: Duration patterns
        duration_groups = {
            "short": [b for b in all_bursts if b["duration_hours"] <= 1],
            "medium": [b for b in all_bursts if 1 < b["duration_hours"] <= 3],
            "long": [b for b in all_bursts if b["duration_hours"] > 3],
        }

        patterns.append(
            {
                "pattern_type": "duration",
                "description": "Burst duration patterns",
                "short_bursts": len(duration_groups["short"]),
                "medium_bursts": len(duration_groups["medium"]),
                "long_bursts": len(duration_groups["long"]),
                "avg_duration": sum(b["duration_hours"] for b in all_bursts)
                / len(all_bursts),
            }
        )

        # Pattern 3: Sequential step patterns
        step_sequences = []

        for burst in all_bursts:
            # Collect step names in sequence
            steps = []
            for activity in burst["activities"]:
                if activity[0] == "step_creation":
                    step_entity = activity[2]
                    if hasattr(step_entity, "step") and step_entity.step:
                        steps.append(step_entity.step)

            if len(steps) >= 2:
                step_sequences.append(steps)

        # Find common sub-sequences
        if step_sequences:
            common_sequences = self._find_common_subsequences(step_sequences)

            if common_sequences:
                patterns.append(
                    {
                        "pattern_type": "step_sequence",
                        "description": "Common step sequences",
                        "common_sequences": common_sequences,
                    }
                )

        return patterns

    def _find_common_subsequences(
        self, sequences: List[List[str]], min_length: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find common subsequences in a list of sequences.

        Args:
            sequences: List of sequences to analyze
            min_length: Minimum subsequence length to consider

        Returns:
            List of common subsequence data
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

        # Return top 5 most common
        return common_subseqs[:5]

    def _categorize_users_by_engagement(
        self, users: List[Any]
    ) -> Dict[UserEngagementLevel, List[Any]]:
        """
        Categorize users by engagement level.

        Args:
            users: List of users to categorize

        Returns:
            Dict mapping engagement levels to lists of users
        """
        result = {
            UserEngagementLevel.HIGH: [],
            UserEngagementLevel.MEDIUM: [],
            UserEngagementLevel.LOW: [],
        }

        for user in users:
            # Get user's engagement level
            level = user.get_engagement_level()
            result[level].append(user)

        return result

    def _calculate_retention_metrics(self, users: List[Any]) -> Dict[str, Any]:
        """
        Calculate retention metrics for a group of users.

        Args:
            users: List of users to analyze

        Returns:
            Dict with retention metrics
        """
        # Initialize result
        result = {"avg_active_days": 0, "retention_rate": 0, "engagement_duration": {}}

        # Reference point for calculating if user is still active
        today = datetime.now()
        inactive_threshold = 30  # Days

        # Track user activity periods
        active_days_sum = 0
        active_user_count = 0

        # Track engagement duration distribution
        duration_counts = {
            "1_day": 0,
            "2_7_days": 0,
            "8_30_days": 0,
            "31_90_days": 0,
            "90_plus_days": 0,
        }

        for user in users:
            if not user.email:
                continue

            # Collect all activity dates
            activity_dates = []

            # Add user creation date
            if user.get_creation_date():
                activity_dates.append(user.get_creation_date())

            # Add last login date
            if user.get_last_login_date():
                activity_dates.append(user.get_last_login_date())

            # Add dates from ideas and steps
            ideas = self._data_repo.ideas.find_by_owner(user.email)

            for idea in ideas:
                if idea.get_creation_date():
                    activity_dates.append(idea.get_creation_date())

                if idea.id:
                    idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                    steps = self._data_repo.steps.find_by_idea_id(idea_id)

                    for step in steps:
                        if step.get_creation_date():
                            activity_dates.append(step.get_creation_date())

            # Calculate metrics if we have activity dates
            if activity_dates:
                # Sort dates
                activity_dates.sort()

                # First and last activity
                first_activity = activity_dates[0]
                last_activity = activity_dates[-1]

                # Calculate active days
                days_active = (last_activity - first_activity).days + 1
                active_days_sum += days_active
                active_user_count += 1

                # Determine if user is still active
                days_since_last_activity = (today - last_activity).days
                is_active = days_since_last_activity < inactive_threshold

                # Track engagement duration
                if days_active == 1:
                    duration_counts["1_day"] += 1
                elif 2 <= days_active <= 7:
                    duration_counts["2_7_days"] += 1
                elif 8 <= days_active <= 30:
                    duration_counts["8_30_days"] += 1
                elif 31 <= days_active <= 90:
                    duration_counts["31_90_days"] += 1
                else:
                    duration_counts["90_plus_days"] += 1

        # Calculate aggregate metrics
        if active_user_count > 0:
            result["avg_active_days"] = active_days_sum / active_user_count

        # Add engagement duration distribution
        result["engagement_duration"] = duration_counts

        # Calculate retention rate (percentage of users still active)
        active_count = sum(
            1 for user in users if self._is_user_still_active(user, inactive_threshold)
        )
        if users:
            result["retention_rate"] = active_count / len(users)

        return result

    def _is_user_still_active(self, user: Any, inactive_threshold: int = 30) -> bool:
        """
        Determine if a user is still active based on their recent activity.

        Args:
            user: User to check
            inactive_threshold: Days of inactivity to consider as inactive

        Returns:
            bool: True if the user is still active
        """
        if not user.email:
            return False

        # Collect all activity dates
        activity_dates = []

        # Add user creation date
        if user.get_creation_date():
            activity_dates.append(user.get_creation_date())

        # Add last login date
        if user.get_last_login_date():
            activity_dates.append(user.get_last_login_date())

        # Add dates from ideas and steps
        ideas = self._data_repo.ideas.find_by_owner(user.email)

        for idea in ideas:
            if idea.get_creation_date():
                activity_dates.append(idea.get_creation_date())

            if idea.id:
                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                steps = self._data_repo.steps.find_by_idea_id(idea_id)

                for step in steps:
                    if step.get_creation_date():
                        activity_dates.append(step.get_creation_date())

        # No activity dates means not active
        if not activity_dates:
            return False

        # Find the most recent activity
        last_activity = max(activity_dates)

        # Check if within threshold
        days_since_last = (datetime.now() - last_activity).days
        return days_since_last < inactive_threshold

    def get_usage_pattern_analysis(
        self, course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze usage patterns such as time of day, day of week, and session lengths.

        Args:
            course_id: Optional course ID to filter users

        Returns:
            Dict with usage pattern analysis results
        """
        # Initialize result structure
        result = {
            "time_of_day": {},
            "day_of_week": {},
            "session_length": {},
            "step_intervals": {},
            "usage_consistency": {},
            "common_sequences": [],
        }

        # Get steps, optionally filtered by course
        steps = []
        if course_id:
            # Get users in the course
            users = self._data_repo.users.find_by_course(course_id)
            user_emails = [user.email for user in users if user.email]

            # Get steps from these users
            for email in user_emails:
                user_steps = self._data_repo.steps.find_by_owner(email)
                steps.extend(user_steps)
        else:
            steps = self._data_repo.steps.get_all()

        # Filter steps with creation dates
        steps_with_dates = [step for step in steps if step.get_creation_date()]

        # Skip analysis if not enough data
        if len(steps_with_dates) < 10:
            return {"error": "Insufficient data for pattern analysis"}

        # Time of day analysis
        hour_counts = defaultdict(int)
        for step in steps_with_dates:
            hour = step.get_creation_date().hour
            hour_counts[hour] += 1

        # Group hours into time periods
        time_periods = {
            "early_morning": sum(hour_counts.get(h, 0) for h in range(5, 9)),
            "morning": sum(hour_counts.get(h, 0) for h in range(9, 12)),
            "afternoon": sum(hour_counts.get(h, 0) for h in range(12, 17)),
            "evening": sum(hour_counts.get(h, 0) for h in range(17, 21)),
            "night": sum(hour_counts.get(h, 0) for h in range(21, 24))
            + sum(hour_counts.get(h, 0) for h in range(0, 5)),
        }

        # Add to result
        result["time_of_day"] = {
            "hour_distribution": dict(hour_counts),
            "period_distribution": time_periods,
            "peak_hour": (
                max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
            ),
        }

        # Day of week analysis
        day_counts = defaultdict(int)
        weekend_count = 0
        weekday_count = 0

        for step in steps_with_dates:
            day_name = step.get_creation_date().strftime("%A")
            day_counts[day_name] += 1

            # Track weekend vs weekday
            weekday = step.get_creation_date().weekday()
            if weekday >= 5:  # Saturday (5) and Sunday (6)
                weekend_count += 1
            else:
                weekday_count += 1

        # Add to result
        result["day_of_week"] = {
            "day_distribution": dict(day_counts),
            "weekend_vs_weekday": {
                "weekend": weekend_count,
                "weekday": weekday_count,
                "weekend_percentage": (
                    weekend_count / len(steps_with_dates) if steps_with_dates else 0
                ),
            },
            "busiest_day": (
                max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None
            ),
        }

        # Session length analysis (using session_id if available)
        session_lengths = []
        sessions = {}

        for step in steps_with_dates:
            if step.session_id:
                session_id = (
                    step.session_id.oid
                    if hasattr(step.session_id, "oid")
                    else str(step.session_id)
                )

                if session_id not in sessions:
                    sessions[session_id] = {
                        "steps": [],
                        "start_time": step.get_creation_date(),
                        "end_time": step.get_creation_date(),
                    }

                # Add step to session
                sessions[session_id]["steps"].append(step)

                # Update session end time if this step is later
                if step.get_creation_date() > sessions[session_id]["end_time"]:
                    sessions[session_id]["end_time"] = step.get_creation_date()

        # Calculate session durations
        if sessions:
            for session_id, session in sessions.items():
                duration_minutes = (
                    session["end_time"] - session["start_time"]
                ).total_seconds() / 60
                session_lengths.append(duration_minutes)

            # Group into duration categories
            duration_categories = {
                "under_5min": sum(1 for d in session_lengths if d < 5),
                "5_15min": sum(1 for d in session_lengths if 5 <= d < 15),
                "15_30min": sum(1 for d in session_lengths if 15 <= d < 30),
                "30_60min": sum(1 for d in session_lengths if 30 <= d < 60),
                "over_60min": sum(1 for d in session_lengths if d >= 60),
            }

            result["session_length"] = {
                "avg_duration_minutes": (
                    sum(session_lengths) / len(session_lengths)
                    if session_lengths
                    else 0
                ),
                "duration_categories": duration_categories,
                "session_count": len(sessions),
            }
        else:
            # Alternative: group steps by owner and timestamp proximity
            result["session_length"] = self._estimate_sessions_from_steps(
                steps_with_dates
            )

        # Step intervals (time between consecutive steps)
        step_intervals = self._calculate_step_intervals(steps_with_dates)
        result["step_intervals"] = step_intervals

        # Usage consistency analysis
        result["usage_consistency"] = self._analyze_usage_consistency(steps_with_dates)

        # Common step sequences
        result["common_sequences"] = self._analyze_step_sequences(steps_with_dates)

        return result

    def _estimate_sessions_from_steps(
        self, steps: List[Any], session_gap_threshold: int = 30  # Minutes
    ) -> Dict[str, Any]:
        """
        Estimate sessions from steps when session_id is not available.

        Args:
            steps: List of steps with creation dates
            session_gap_threshold: Minimum gap in minutes to consider a new session

        Returns:
            Dict with session length analysis
        """
        # Group steps by owner
        steps_by_owner = defaultdict(list)

        for step in steps:
            if step.owner:
                steps_by_owner[step.owner].append(step)

        # Identify sessions for each owner
        all_session_lengths = []
        session_count = 0

        for owner, owner_steps in steps_by_owner.items():
            # Sort steps by creation date
            sorted_steps = sorted(owner_steps, key=lambda s: s.get_creation_date())

            # Identify sessions based on time gaps
            sessions = []
            current_session = []

            for step in sorted_steps:
                if not current_session:
                    current_session = [step]
                else:
                    # Check if this step is within the threshold of the last step
                    time_gap = (
                        step.get_creation_date()
                        - current_session[-1].get_creation_date()
                    ).total_seconds() / 60

                    if time_gap <= session_gap_threshold:
                        # Part of the current session
                        current_session.append(step)
                    else:
                        # Start a new session
                        if len(current_session) > 0:
                            sessions.append(current_session)
                        current_session = [step]

            # Add the last session if it exists
            if current_session:
                sessions.append(current_session)

            # Calculate session durations
            for session in sessions:
                if len(session) > 0:
                    start_time = session[0].get_creation_date()
                    end_time = session[-1].get_creation_date()

                    # Add a minimum duration for single-step sessions
                    if start_time == end_time:
                        duration_minutes = 5  # Assume at least 5 minutes
                    else:
                        duration_minutes = (end_time - start_time).total_seconds() / 60

                    all_session_lengths.append(duration_minutes)
                    session_count += 1

        # Group into duration categories
        duration_categories = {
            "under_5min": sum(1 for d in all_session_lengths if d < 5),
            "5_15min": sum(1 for d in all_session_lengths if 5 <= d < 15),
            "15_30min": sum(1 for d in all_session_lengths if 15 <= d < 30),
            "30_60min": sum(1 for d in all_session_lengths if 30 <= d < 60),
            "over_60min": sum(1 for d in all_session_lengths if d >= 60),
        }

        return {
            "avg_duration_minutes": (
                sum(all_session_lengths) / len(all_session_lengths)
                if all_session_lengths
                else 0
            ),
            "duration_categories": duration_categories,
            "session_count": session_count,
            "estimation_method": "time_gap_based",
        }

    def _calculate_step_intervals(self, steps: List[Any]) -> Dict[str, Any]:
        """
        Calculate time intervals between consecutive steps.

        Args:
            steps: List of steps with creation dates

        Returns:
            Dict with step interval analysis
        """
        # Group steps by idea_id
        steps_by_idea = defaultdict(list)

        for step in steps:
            if step.idea_id:
                idea_id = (
                    step.idea_id.oid
                    if hasattr(step.idea_id, "oid")
                    else str(step.idea_id)
                )
                steps_by_idea[idea_id].append(step)

        # Calculate intervals
        intervals = []

        for idea_id, idea_steps in steps_by_idea.items():
            # Skip ideas with only one step
            if len(idea_steps) < 2:
                continue

            # Sort steps by creation date
            sorted_steps = sorted(idea_steps, key=lambda s: s.get_creation_date())

            # Calculate intervals between consecutive steps
            for i in range(1, len(sorted_steps)):
                prev_time = sorted_steps[i - 1].get_creation_date()
                curr_time = sorted_steps[i].get_creation_date()

                # Calculate interval in minutes
                interval_minutes = (curr_time - prev_time).total_seconds() / 60
                intervals.append(interval_minutes)

        # Skip if no intervals
        if not intervals:
            return {"count": 0, "message": "No step intervals found"}

        # Calculate statistics
        avg_interval = sum(intervals) / len(intervals)
        median_interval = sorted(intervals)[len(intervals) // 2]

        # Group into categories
        interval_categories = {
            "immediate": sum(1 for i in intervals if i < 1),
            "under_5min": sum(1 for i in intervals if 1 <= i < 5),
            "5_30min": sum(1 for i in intervals if 5 <= i < 30),
            "30min_2hr": sum(1 for i in intervals if 30 <= i < 120),
            "2hr_1day": sum(1 for i in intervals if 120 <= i < 1440),
            "over_1day": sum(1 for i in intervals if i >= 1440),
        }

        return {
            "avg_interval_minutes": avg_interval,
            "median_interval_minutes": median_interval,
            "interval_categories": interval_categories,
            "count": len(intervals),
        }

    def _analyze_usage_consistency(self, steps: List[Any]) -> Dict[str, Any]:
        """
        Analyze how consistently users engage with the tool over time.

        Args:
            steps: List of steps with creation dates

        Returns:
            Dict with usage consistency analysis
        """
        # Group steps by user and date
        activity_by_user_date = defaultdict(lambda: defaultdict(int))

        for step in steps:
            if step.owner and step.get_creation_date():
                # Get the date (without time)
                date_str = step.get_creation_date().strftime("%Y-%m-%d")

                # Count steps on this date
                activity_by_user_date[step.owner][date_str] += 1

        # Calculate consistency metrics
        user_metrics = {}

        for user, date_activity in activity_by_user_date.items():
            # Skip users with only one day of activity
            if len(date_activity) < 2:
                continue

            # Get dates with activity
            activity_dates = [
                datetime.strptime(date_str, "%Y-%m-%d")
                for date_str in date_activity.keys()
            ]

            # Calculate date range
            start_date = min(activity_dates)
            end_date = max(activity_dates)
            date_range_days = (end_date - start_date).days + 1

            # Calculate consistency metrics
            active_days = len(date_activity)
            activity_rate = active_days / date_range_days if date_range_days > 0 else 0

            # Calculate average steps per active day
            total_steps = sum(date_activity.values())
            avg_steps_per_active_day = (
                total_steps / active_days if active_days > 0 else 0
            )

            # Store metrics
            user_metrics[user] = {
                "date_range_days": date_range_days,
                "active_days": active_days,
                "activity_rate": activity_rate,
                "total_steps": total_steps,
                "avg_steps_per_active_day": avg_steps_per_active_day,
            }

        # Calculate aggregate metrics
        if not user_metrics:
            return {
                "user_count": 0,
                "message": "Insufficient data for consistency analysis",
            }

        avg_activity_rate = sum(
            m["activity_rate"] for m in user_metrics.values()
        ) / len(user_metrics)

        # Group users by consistency
        consistent_users = sum(
            1 for m in user_metrics.values() if m["activity_rate"] >= 0.5
        )
        sporadic_users = sum(
            1 for m in user_metrics.values() if 0.1 <= m["activity_rate"] < 0.5
        )
        infrequent_users = sum(
            1 for m in user_metrics.values() if m["activity_rate"] < 0.1
        )

        return {
            "avg_activity_rate": avg_activity_rate,
            "user_consistency": {
                "consistent_users": consistent_users,
                "sporadic_users": sporadic_users,
                "infrequent_users": infrequent_users,
            },
            "user_count": len(user_metrics),
        }

    def _analyze_step_sequences(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """
        Analyze common sequences of steps that users follow.

        Args:
            steps: List of steps with creation dates

        Returns:
            List of common step sequence data
        """
        # Group steps by idea_id
        steps_by_idea = defaultdict(list)

        for step in steps:
            if step.idea_id and step.step:
                idea_id = (
                    step.idea_id.oid
                    if hasattr(step.idea_id, "oid")
                    else str(step.idea_id)
                )
                steps_by_idea[idea_id].append(step)

        # Extract sequences of steps
        step_sequences = []

        for idea_id, idea_steps in steps_by_idea.items():
            # Skip ideas with too few steps
            if len(idea_steps) < 3:
                continue

            # Sort steps by creation date
            sorted_steps = sorted(idea_steps, key=lambda s: s.get_creation_date())

            # Extract step names
            step_names = [step.step for step in sorted_steps]
            step_sequences.append(step_names)

        # Find common subsequences
        if len(step_sequences) >= 3:  # Need at least 3 sequences to find patterns
            return self._find_common_subsequences(step_sequences, min_length=3)

        return []
