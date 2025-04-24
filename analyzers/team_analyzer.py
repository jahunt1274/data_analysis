"""
Team analyzer for the data analysis system.

This module provides functionality for analyzing team dynamics and collaboration
patterns in the JetPack/Orbit tool, including comparisons between team and
individual usage, intra-team patterns, and the impact of team composition on tool usage.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..data.data_repository import DataRepository
from ..data.models.enums import (
    UserEngagementLevel,
    UserType,
    FrameworkType,
    DisciplinedEntrepreneurshipStep,
    StartupTacticsStep,
)


class TeamAnalyzer:
    """
    Analyzer for team dynamics in the JetPack/Orbit tool.

    This class provides methods for analyzing team collaboration patterns,
    comparing team vs. individual usage, and identifying the impact of
    team composition on tool utilization and outcomes.
    """

    def __init__(self, data_repository: DataRepository):
        """
        Initialize the team analyzer.

        Args:
            data_repository: Data repository for accessing all entity repositories
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._data_repo = data_repository

        # Ensure data is loaded
        self._data_repo.connect()

    def compare_team_vs_individual_engagement(
        self,
        course_id: Optional[str] = None,
        include_demographic_breakdown: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare engagement metrics between team members and individual users.

        This method analyzes differences in tool usage patterns between users
        who are part of teams versus those working individually.

        Args:
            course_id: Optional course ID to filter users (e.g., "15.390")
            include_demographic_breakdown: Whether to include analysis by demographics

        Returns:
            Dict with comparative engagement metrics
        """
        # Initialize result structure
        result = {
            "team_metrics": {},
            "individual_metrics": {},
            "comparison": {},
        }

        # Get users, optionally filtered by course
        if course_id:
            users = self._data_repo.users.find_by_course(course_id)
        else:
            users = self._data_repo.users.get_all()

        # Get all team members
        all_teams = self._data_repo.teams.get_all()
        team_member_emails = set()

        for team in all_teams:
            team_member_emails.update(team.get_member_emails())

        # Split users into team members and individuals
        team_users = [
            user for user in users if user.email and user.email in team_member_emails
        ]
        individual_users = [
            user
            for user in users
            if user.email and user.email not in team_member_emails
        ]

        # Calculate engagement metrics for team members
        if team_users:
            result["team_metrics"] = self._calculate_engagement_metrics(
                team_users, "team_members"
            )

        # Calculate engagement metrics for individuals
        if individual_users:
            result["individual_metrics"] = self._calculate_engagement_metrics(
                individual_users, "individuals"
            )

        # Calculate comparison metrics
        if team_users and individual_users:
            result["comparison"] = self._compare_metrics(
                result["team_metrics"], result["individual_metrics"]
            )

        # Add demographic breakdown if requested
        if include_demographic_breakdown and team_users and individual_users:
            result["demographic_breakdown"] = self._analyze_demographics(
                team_users, individual_users
            )

        return result

    def analyze_team_collaboration_patterns(
        self,
        team_id: Optional[int] = None,
        course_id: Optional[str] = None,
        include_temporal_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze how team members collaborate using the tool.

        This method examines how members of the same team interact with
        the tool, including synchronization and collaboration patterns.

        Args:
            team_id: Optional specific team ID to analyze
            course_id: Optional course ID to filter teams
            include_temporal_analysis: Whether to include time-based analysis

        Returns:
            Dict with team collaboration pattern analysis
        """
        # Initialize result structure
        result = {
            "overview": {},
            "teams_analyzed": [],
            "collaboration_metrics": {},
            "idea_sharing_patterns": {},
        }

        # Get teams to analyze
        teams_to_analyze = []

        if team_id is not None:
            # Get specific team
            team = self._data_repo.teams.find_by_team_id(team_id)
            if team:
                teams_to_analyze = [team]
        else:
            # Get all teams, optionally filtered by course
            if course_id:
                # For simplicity, filter teams based on members being in the course
                all_teams = self._data_repo.teams.get_all()
                course_users = self._data_repo.users.find_by_course(course_id)
                course_emails = {user.email for user in course_users if user.email}

                teams_to_analyze = [
                    team
                    for team in all_teams
                    if any(email in course_emails for email in team.get_member_emails())
                ]
            else:
                teams_to_analyze = self._data_repo.teams.get_all()

        # Skip if no teams to analyze
        if not teams_to_analyze:
            return {"error": "No teams found for analysis"}

        # Track overall statistics
        all_team_metrics = []
        total_teams = len(teams_to_analyze)
        teams_with_collaboration = 0

        # Analyze each team
        for team in teams_to_analyze:
            team_metrics = self._analyze_single_team_collaboration(
                team, include_temporal_analysis
            )

            # Track if this team has collaboration
            if team_metrics.get("has_collaboration", False):
                teams_with_collaboration += 1

            all_team_metrics.append(team_metrics)
            result["teams_analyzed"].append(
                {
                    "team_id": team.team_id,
                    "team_name": team.team_name,
                    "member_count": team.get_member_count(),
                    "collaboration_level": team_metrics.get(
                        "collaboration_level", "none"
                    ),
                }
            )

        # Calculate overall metrics
        if all_team_metrics:
            result["overview"] = {
                "teams_analyzed": total_teams,
                "teams_with_collaboration": teams_with_collaboration,
                "collaboration_rate": (
                    teams_with_collaboration / total_teams if total_teams > 0 else 0
                ),
                "avg_shared_ideas": (
                    sum(m.get("shared_idea_count", 0) for m in all_team_metrics)
                    / total_teams
                    if total_teams > 0
                    else 0
                ),
                "avg_collaboration_score": (
                    sum(m.get("collaboration_score", 0) for m in all_team_metrics)
                    / total_teams
                    if total_teams > 0
                    else 0
                ),
            }

        # Generate collaboration metrics across teams
        result["collaboration_metrics"] = self._summarize_collaboration_metrics(
            all_team_metrics
        )

        # Analyze idea sharing patterns
        result["idea_sharing_patterns"] = self._analyze_idea_sharing_patterns(
            teams_to_analyze
        )

        # Add temporal analysis if requested
        if include_temporal_analysis:
            result["temporal_patterns"] = self._analyze_temporal_collaboration(
                all_team_metrics
            )

        return result

    def get_team_usage_distribution(
        self,
        course_id: Optional[str] = None,
        min_team_size: int = 2,
    ) -> Dict[str, Any]:
        """
        Analyze how tool usage is distributed within teams.

        This method examines if all team members use the tool equally
        or if usage is concentrated among specific members.

        Args:
            course_id: Optional course ID to filter teams
            min_team_size: Minimum number of members for a team to be included

        Returns:
            Dict with team usage distribution analysis
        """
        # Initialize result structure
        result = {
            "teams_analyzed": 0,
            "distribution_metrics": {},
            "usage_concentration": {},
            "role_patterns": {},
        }

        # Get teams to analyze, optionally filtered by course
        teams_to_analyze = []

        if course_id:
            # Filter teams based on members being in the course
            all_teams = self._data_repo.teams.get_all()
            course_users = self._data_repo.users.find_by_course(course_id)
            course_emails = {user.email for user in course_users if user.email}

            teams_to_analyze = [
                team
                for team in all_teams
                if any(email in course_emails for email in team.get_member_emails())
                and team.get_member_count() >= min_team_size
            ]
        else:
            teams_to_analyze = [
                team
                for team in self._data_repo.teams.get_all()
                if team.get_member_count() >= min_team_size
            ]

        # Skip if no teams to analyze
        if not teams_to_analyze:
            return {"error": f"No teams found with at least {min_team_size} members"}

        # Track distribution types
        distribution_types = {
            "uniform": 0,  # All members contribute similarly
            "primary": 0,  # One member does most of the work
            "partial": 0,  # Some members contribute, others don't
            "inactive": 0,  # No team members use the tool
        }

        # Analyze each team's usage distribution
        team_distributions = []

        for team in teams_to_analyze:
            team_distribution = self._analyze_team_usage_distribution(team)
            team_distributions.append(team_distribution)

            # Count distribution type
            dist_type = team_distribution.get("distribution_type", "inactive")
            distribution_types[dist_type] += 1

        # Calculate overall metrics
        result["teams_analyzed"] = len(teams_to_analyze)

        # Calculate distribution metrics
        total_teams = len(teams_to_analyze)
        if total_teams > 0:
            result["distribution_metrics"] = {
                dist_type: {"count": count, "percentage": count / total_teams}
                for dist_type, count in distribution_types.items()
            }

        # Calculate usage concentration metrics
        if team_distributions:
            result["usage_concentration"] = {
                "avg_gini_coefficient": sum(
                    d.get("gini_coefficient", 0) for d in team_distributions
                )
                / len(team_distributions),
                "avg_activity_ratio": sum(
                    d.get("active_member_ratio", 0) for d in team_distributions
                )
                / len(team_distributions),
                "avg_contribution_variation": sum(
                    d.get("coefficient_of_variation", 0) for d in team_distributions
                )
                / len(team_distributions),
            }

        # Analyze role patterns
        result["role_patterns"] = self._analyze_team_roles(team_distributions)

        return result

    def correlate_team_composition_with_usage(
        self,
        course_id: Optional[str] = None,
        include_detailed_breakdown: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze how team composition affects tool usage.

        This method examines whether factors like team size, member backgrounds,
        or diversity influence how the team uses the tool.

        Args:
            course_id: Optional course ID to filter teams
            include_detailed_breakdown: Whether to include detailed breakdowns

        Returns:
            Dict with team composition correlation analysis
        """
        # Initialize result structure
        result = {
            "team_size_correlation": {},
            "diversity_correlation": {},
            "composition_factors": {},
        }

        # Get teams to analyze, optionally filtered by course
        teams_to_analyze = []

        if course_id:
            # Filter teams based on members being in the course
            all_teams = self._data_repo.teams.get_all()
            course_users = self._data_repo.users.find_by_course(course_id)
            course_emails = {user.email for user in course_users if user.email}

            teams_to_analyze = [
                team
                for team in all_teams
                if any(email in course_emails for email in team.get_member_emails())
            ]
        else:
            teams_to_analyze = self._data_repo.teams.get_all()

        # Skip if no teams to analyze
        if not teams_to_analyze:
            return {"error": "No teams found for analysis"}

        # Analyze team composition metrics
        team_composition_metrics = []
        for team in teams_to_analyze:
            metrics = self._calculate_team_composition_metrics(team)
            team_composition_metrics.append(metrics)

        # Analyze team size correlation
        result["team_size_correlation"] = self._analyze_team_size_correlation(
            team_composition_metrics
        )

        # Analyze diversity correlation
        result["diversity_correlation"] = self._analyze_diversity_correlation(
            team_composition_metrics
        )

        # Analyze composition factors
        result["composition_factors"] = self._analyze_composition_factors(
            team_composition_metrics
        )

        # Add detailed breakdown if requested
        if include_detailed_breakdown:
            result["detailed_breakdown"] = {
                "by_size": self._breakdown_by_team_size(team_composition_metrics),
                "by_diversity": self._breakdown_by_diversity(team_composition_metrics),
            }

        return result

    def analyze_team_framework_progression(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        compare_to_individuals: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze how teams progress through entrepreneurial frameworks.

        This method examines how teams navigate through framework steps
        compared to individuals, and identifies team-specific patterns.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter teams
            compare_to_individuals: Whether to compare with individual users

        Returns:
            Dict with team framework progression analysis
        """
        # Initialize result structure
        result = {
            "framework": framework.value,
            "team_progression": {},
            "individual_progression": {},
            "comparison": {},
        }

        # Get steps in the framework
        framework_steps = []
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            framework_steps = [step.value for step in DisciplinedEntrepreneurshipStep]
        elif framework == FrameworkType.STARTUP_TACTICS:
            framework_steps = [step.value for step in StartupTacticsStep]
        else:
            # Not implemented for other frameworks
            return {"error": f"Framework not supported: {framework.value}"}

        # Get teams to analyze, optionally filtered by course
        teams_to_analyze = []

        if course_id:
            # Filter teams based on members being in the course
            all_teams = self._data_repo.teams.get_all()
            course_users = self._data_repo.users.find_by_course(course_id)
            course_emails = {user.email for user in course_users if user.email}

            teams_to_analyze = [
                team
                for team in all_teams
                if any(email in course_emails for email in team.get_member_emails())
            ]
        else:
            teams_to_analyze = self._data_repo.teams.get_all()

        # Get individual users (those not on teams)
        team_member_emails = set()
        for team in teams_to_analyze:
            team_member_emails.update(team.get_member_emails())

        if course_id:
            individual_users = [
                user
                for user in self._data_repo.users.find_by_course(course_id)
                if user.email and user.email not in team_member_emails
            ]
        else:
            individual_users = [
                user
                for user in self._data_repo.users.get_all()
                if user.email and user.email not in team_member_emails
            ]

        # Analyze team progression
        result["team_progression"] = self._analyze_team_framework_progression(
            teams_to_analyze, framework, framework_steps
        )

        # Analyze individual progression if requested
        if compare_to_individuals and individual_users:
            result["individual_progression"] = (
                self._analyze_individual_framework_progression(
                    individual_users, framework, framework_steps
                )
            )

            # Compare team vs. individual progression
            result["comparison"] = self._compare_framework_progression(
                result["team_progression"], result["individual_progression"]
            )

        return result

    def _calculate_engagement_metrics(
        self, users: List[Any], group_name: str
    ) -> Dict[str, Any]:
        """
        Calculate engagement metrics for a group of users.

        Args:
            users: List of users to analyze
            group_name: Name of the user group for labeling

        Returns:
            Dict with engagement metrics
        """
        # Initialize metrics
        metrics = {
            "user_count": len(users),
            "idea_metrics": {},
            "step_metrics": {},
            "engagement_distribution": {},
            "activity_metrics": {},
        }

        # Skip if no users
        if not users:
            return metrics

        # Track ideas and steps
        total_ideas = 0
        total_steps = 0
        ideas_with_steps = 0

        # Track user engagement levels
        engagement_levels = {
            UserEngagementLevel.HIGH.value: 0,
            UserEngagementLevel.MEDIUM.value: 0,
            UserEngagementLevel.LOW.value: 0,
        }

        # Analyze each user
        for user in users:
            if not user.email:
                continue

            # Count ideas
            user_ideas = self._data_repo.ideas.find_by_owner(user.email)
            total_ideas += len(user_ideas)

            # Count steps and ideas with steps
            user_steps = 0
            user_ideas_with_steps = 0

            for idea in user_ideas:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)
                steps = self._data_repo.steps.find_by_idea_id(idea_id)

                if steps:
                    user_ideas_with_steps += 1

                user_steps += len(steps)

            total_steps += user_steps
            ideas_with_steps += user_ideas_with_steps

            # Track engagement level
            engagement_level = user.get_engagement_level().value
            engagement_levels[engagement_level] += 1

        # Calculate idea metrics
        metrics["idea_metrics"] = {
            "total_ideas": total_ideas,
            "avg_ideas_per_user": total_ideas / len(users) if users else 0,
            "ideas_with_steps": ideas_with_steps,
            "idea_to_step_conversion_rate": (
                ideas_with_steps / total_ideas if total_ideas > 0 else 0
            ),
        }

        # Calculate step metrics
        metrics["step_metrics"] = {
            "total_steps": total_steps,
            "avg_steps_per_user": total_steps / len(users) if users else 0,
            "avg_steps_per_idea": total_steps / total_ideas if total_ideas > 0 else 0,
        }

        # Calculate engagement distribution
        metrics["engagement_distribution"] = {
            level: {
                "count": count,
                "percentage": count / len(users) if users else 0,
            }
            for level, count in engagement_levels.items()
        }

        # Calculate activity metrics (e.g., when users are active)
        metrics["activity_metrics"] = self._calculate_activity_metrics(users)

        return metrics

    def _calculate_activity_metrics(self, users: List[Any]) -> Dict[str, Any]:
        """
        Calculate when users are active in the tool.

        Args:
            users: List of users to analyze

        Returns:
            Dict with activity metrics
        """
        # Initialize metrics
        metrics = {
            "daily_distribution": {},
            "hourly_distribution": {},
            "session_length_distribution": {},
        }

        # Skip if no users
        if not users:
            return metrics

        # Track activity by day and hour
        day_counts = defaultdict(int)
        hour_counts = defaultdict(int)

        # Track session durations
        session_durations = []

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
            day_counts[day] = 0

        for hour in range(24):
            hour_counts[hour] = 0

        # Analyze steps for activity patterns
        all_steps = []

        for user in users:
            if not user.email:
                continue

            # Get user's steps
            user_steps = self._data_repo.steps.find_by_owner(user.email)
            all_steps.extend(user_steps)

        # Filter steps with valid creation dates
        steps_with_dates = [step for step in all_steps if step.get_creation_date()]

        # Process step timestamps
        for step in steps_with_dates:
            # Track day of week
            day_name = step.get_creation_date().strftime("%A")
            day_counts[day_name] += 1

            # Track hour of day
            hour = step.get_creation_date().hour
            hour_counts[hour] += 1

        # Calculate session durations if session IDs are available
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
                        "start_time": step.get_creation_date(),
                        "end_time": step.get_creation_date(),
                    }

                # Update session end time if this step is later
                elif step.get_creation_date() > sessions[session_id]["end_time"]:
                    sessions[session_id]["end_time"] = step.get_creation_date()

        # Calculate durations
        for session in sessions.values():
            duration_minutes = (
                session["end_time"] - session["start_time"]
            ).total_seconds() / 60
            session_durations.append(duration_minutes)

        # Normalize distributions
        total_steps = len(steps_with_dates)

        if total_steps > 0:
            metrics["daily_distribution"] = {
                day: count / total_steps for day, count in day_counts.items()
            }

            metrics["hourly_distribution"] = {
                str(hour): count / total_steps for hour, count in hour_counts.items()
            }

        # Calculate session length distribution
        if session_durations:
            session_groups = {
                "under_5min": 0,
                "5_15min": 0,
                "15_30min": 0,
                "30_60min": 0,
                "1_3hr": 0,
                "over_3hr": 0,
            }

            for duration in session_durations:
                if duration < 5:
                    session_groups["under_5min"] += 1
                elif duration < 15:
                    session_groups["5_15min"] += 1
                elif duration < 30:
                    session_groups["15_30min"] += 1
                elif duration < 60:
                    session_groups["30_60min"] += 1
                elif duration < 180:
                    session_groups["1_3hr"] += 1
                else:
                    session_groups["over_3hr"] += 1

            # Normalize
            total_sessions = len(session_durations)

            metrics["session_length_distribution"] = {
                group: count / total_sessions for group, count in session_groups.items()
            }

            # Add average session length
            metrics["avg_session_length_minutes"] = (
                sum(session_durations) / total_sessions
            )

        return metrics

    def _compare_metrics(
        self, team_metrics: Dict[str, Any], individual_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare metrics between team members and individuals.

        Args:
            team_metrics: Metrics for team members
            individual_metrics: Metrics for individual users

        Returns:
            Dict with comparative metrics
        """
        comparison = {}

        # Compare idea metrics
        team_ideas_per_user = team_metrics.get("idea_metrics", {}).get(
            "avg_ideas_per_user", 0
        )
        indiv_ideas_per_user = individual_metrics.get("idea_metrics", {}).get(
            "avg_ideas_per_user", 0
        )

        comparison["idea_difference"] = {
            "team_avg": team_ideas_per_user,
            "individual_avg": indiv_ideas_per_user,
            "difference": team_ideas_per_user - indiv_ideas_per_user,
            "percentage_difference": (
                (team_ideas_per_user - indiv_ideas_per_user)
                / indiv_ideas_per_user
                * 100
                if indiv_ideas_per_user > 0
                else 0
            ),
        }

        # Compare step metrics
        team_steps_per_user = team_metrics.get("step_metrics", {}).get(
            "avg_steps_per_user", 0
        )
        indiv_steps_per_user = individual_metrics.get("step_metrics", {}).get(
            "avg_steps_per_user", 0
        )

        comparison["step_difference"] = {
            "team_avg": team_steps_per_user,
            "individual_avg": indiv_steps_per_user,
            "difference": team_steps_per_user - indiv_steps_per_user,
            "percentage_difference": (
                (team_steps_per_user - indiv_steps_per_user)
                / indiv_steps_per_user
                * 100
                if indiv_steps_per_user > 0
                else 0
            ),
        }

        # Compare conversion rates
        team_conversion = team_metrics.get("idea_metrics", {}).get(
            "idea_to_step_conversion_rate", 0
        )
        indiv_conversion = individual_metrics.get("idea_metrics", {}).get(
            "idea_to_step_conversion_rate", 0
        )

        comparison["conversion_difference"] = {
            "team_rate": team_conversion,
            "individual_rate": indiv_conversion,
            "difference": team_conversion - indiv_conversion,
            "percentage_difference": (
                (team_conversion - indiv_conversion) / indiv_conversion * 100
                if indiv_conversion > 0
                else 0
            ),
        }

        # Compare engagement levels
        team_high_engagement = (
            team_metrics.get("engagement_distribution", {})
            .get("high", {})
            .get("percentage", 0)
        )
        indiv_high_engagement = (
            individual_metrics.get("engagement_distribution", {})
            .get("high", {})
            .get("percentage", 0)
        )

        comparison["engagement_difference"] = {
            "team_high_engagement": team_high_engagement,
            "individual_high_engagement": indiv_high_engagement,
            "difference": team_high_engagement - indiv_high_engagement,
            "percentage_difference": (
                (team_high_engagement - indiv_high_engagement)
                / indiv_high_engagement
                * 100
                if indiv_high_engagement > 0
                else 0
            ),
        }

        return comparison

    def _analyze_demographics(
        self, team_users: List[Any], individual_users: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze engagement differences across demographic groups.

        Args:
            team_users: Users who are team members
            individual_users: Users who are not team members

        Returns:
            Dict with demographic analysis
        """
        demographics = {
            "user_type": {},
            "department": {},
            "experience": {},
        }

        # Analyze by user type
        user_types = {user_type.value for user_type in UserType}

        for user_type in user_types:
            # Count team members of this type
            team_type_users = [
                user for user in team_users if user.get_user_type().value == user_type
            ]

            # Count individuals of this type
            indiv_type_users = [
                user
                for user in individual_users
                if user.get_user_type().value == user_type
            ]

            # Skip if not enough users
            if len(team_type_users) == 0 or len(indiv_type_users) == 0:
                continue

            # Calculate metrics for each group
            team_metrics = self._calculate_engagement_metrics(
                team_type_users, f"team_{user_type}"
            )
            indiv_metrics = self._calculate_engagement_metrics(
                indiv_type_users, f"individual_{user_type}"
            )

            # Compare metrics
            comparison = self._compare_metrics(team_metrics, indiv_metrics)

            # Add to results
            demographics["user_type"][user_type] = {
                "team_count": len(team_type_users),
                "individual_count": len(indiv_type_users),
                "comparison": comparison,
            }

        # Analyze by department
        # Get departments from users
        all_departments = set()

        for user in team_users + individual_users:
            dept = user.get_department()
            if dept:
                all_departments.add(dept)

        for department in all_departments:
            # Count team members in this department
            team_dept_users = [
                user for user in team_users if user.get_department() == department
            ]

            # Count individuals in this department
            indiv_dept_users = [
                user for user in individual_users if user.get_department() == department
            ]

            # Skip if not enough users
            if len(team_dept_users) < 3 or len(indiv_dept_users) < 3:
                continue

            # Calculate metrics for each group
            team_metrics = self._calculate_engagement_metrics(
                team_dept_users, f"team_{department}"
            )
            indiv_metrics = self._calculate_engagement_metrics(
                indiv_dept_users, f"individual_{department}"
            )

            # Compare metrics
            comparison = self._compare_metrics(team_metrics, indiv_metrics)

            # Add to results
            demographics["department"][department] = {
                "team_count": len(team_dept_users),
                "individual_count": len(indiv_dept_users),
                "comparison": comparison,
            }

        # Analyze by experience level (if available)
        experience_levels = set()

        for user in team_users + individual_users:
            if (
                user.orbit_profile
                and user.orbit_profile.experience
                and user.orbit_profile.experience.strip()
            ):
                experience_levels.add(user.orbit_profile.experience.strip())

        for experience in experience_levels:
            # Count team members with this experience
            team_exp_users = [
                user
                for user in team_users
                if user.orbit_profile
                and user.orbit_profile.experience
                and user.orbit_profile.experience.strip() == experience
            ]

            # Count individuals with this experience
            indiv_exp_users = [
                user
                for user in individual_users
                if user.orbit_profile
                and user.orbit_profile.experience
                and user.orbit_profile.experience.strip() == experience
            ]

            # Skip if not enough users
            if len(team_exp_users) < 3 or len(indiv_exp_users) < 3:
                continue

            # Calculate metrics for each group
            team_metrics = self._calculate_engagement_metrics(
                team_exp_users, f"team_{experience}"
            )
            indiv_metrics = self._calculate_engagement_metrics(
                indiv_exp_users, f"individual_{experience}"
            )

            # Compare metrics
            comparison = self._compare_metrics(team_metrics, indiv_metrics)

            # Add to results
            demographics["experience"][experience] = {
                "team_count": len(team_exp_users),
                "individual_count": len(indiv_exp_users),
                "comparison": comparison,
            }

        return demographics

    def _analyze_single_team_collaboration(
        self, team: Any, include_temporal_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze collaboration patterns within a single team.

        Args:
            team: Team to analyze
            include_temporal_analysis: Whether to include time-based analysis

        Returns:
            Dict with collaboration metrics for the team
        """
        team_metrics = {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "member_count": team.get_member_count(),
            "active_members": 0,
            "shared_idea_count": 0,
            "has_collaboration": False,
            "collaboration_score": 0.0,
            "collaboration_level": "none",
        }

        # Get member emails
        member_emails = team.get_member_emails()

        if not member_emails:
            return team_metrics

        # Track active members
        active_members = []

        # Track ideas and steps by member
        member_ideas = {}
        member_steps = {}

        # Identify potentially shared ideas
        all_ideas = []
        for email in member_emails:
            # Get user's ideas
            user_ideas = self._data_repo.ideas.find_by_owner(email)

            if user_ideas:
                active_members.append(email)
                member_ideas[email] = user_ideas
                all_ideas.extend(user_ideas)

                # Initialize steps list
                member_steps[email] = []

        # Update active member count
        team_metrics["active_members"] = len(active_members)

        if len(active_members) == 0:
            # No active members
            team_metrics["collaboration_level"] = "none"
            return team_metrics
        elif len(active_members) == 1:
            # Only one active member, no collaboration possible
            team_metrics["collaboration_level"] = "single_member"
            return team_metrics

        # Identify shared ideas (where multiple team members contribute steps)
        shared_ideas = []

        for idea in all_ideas:
            if not idea.id:
                continue

            idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

            # Get all steps for this idea
            idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

            # Get unique contributors to this idea
            contributors = set()
            for step in idea_steps:
                if step.owner and step.owner in member_emails:
                    contributors.add(step.owner)

                    # Add to member's steps
                    if step.owner in member_steps:
                        member_steps[step.owner].append(step)

            # Check if multiple team members contributed
            if len(contributors) > 1:
                shared_ideas.append(
                    {
                        "idea_id": idea_id,
                        "contributors": list(contributors),
                        "steps": idea_steps,
                    }
                )

        # Update shared idea count
        team_metrics["shared_idea_count"] = len(shared_ideas)
        team_metrics["has_collaboration"] = len(shared_ideas) > 0

        # Calculate collaboration score
        if len(active_members) > 1:
            # Base score on percentage of ideas that are shared and percentage of active members involved
            idea_sharing_ratio = len(shared_ideas) / len(all_ideas) if all_ideas else 0

            # Get number of members involved in at least one shared idea
            members_with_shared_ideas = set()
            for idea in shared_ideas:
                members_with_shared_ideas.update(idea["contributors"])

            member_involvement_ratio = len(members_with_shared_ideas) / len(
                active_members
            )

            # Combine metrics with weights
            team_metrics["collaboration_score"] = (idea_sharing_ratio * 0.6) + (
                member_involvement_ratio * 0.4
            )

            # Determine collaboration level
            if team_metrics["collaboration_score"] >= 0.6:
                team_metrics["collaboration_level"] = "high"
            elif team_metrics["collaboration_score"] >= 0.3:
                team_metrics["collaboration_level"] = "medium"
            elif team_metrics["collaboration_score"] > 0:
                team_metrics["collaboration_level"] = "low"
            else:
                team_metrics["collaboration_level"] = "none"

        # Add temporal analysis if requested
        if include_temporal_analysis and shared_ideas:
            team_metrics["temporal_patterns"] = self._analyze_team_temporal_patterns(
                shared_ideas
            )

        return team_metrics

    def _analyze_team_temporal_patterns(
        self, shared_ideas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in team collaboration.

        Args:
            shared_ideas: List of ideas with multiple team contributors

        Returns:
            Dict with temporal pattern analysis
        """
        patterns = {
            "synchronous_collaboration": 0,
            "asynchronous_collaboration": 0,
            "avg_collaboration_window_hours": 0,
            "sequential_vs_parallel": {},
        }

        # Define thresholds
        sync_threshold_hours = 1  # Consider collaboration synchronous if within 1 hour

        # Track collaboration windows
        collaboration_windows = []

        # Track sequential vs. parallel work
        sequential_count = 0
        parallel_count = 0

        for idea in shared_ideas:
            # Get steps with timestamps
            steps_with_timestamps = [
                step for step in idea["steps"] if step.get_creation_date()
            ]

            if len(steps_with_timestamps) < 2:
                continue

            # Sort steps by creation date
            sorted_steps = sorted(
                steps_with_timestamps, key=lambda s: s.get_creation_date()
            )

            # Group steps by contributor
            steps_by_contributor = defaultdict(list)
            for step in sorted_steps:
                if step.owner:
                    steps_by_contributor[step.owner].append(step)

            # Skip if only one contributor
            if len(steps_by_contributor) < 2:
                continue

            # Check if collaboration is synchronous or asynchronous
            # Get earliest and latest step for each contributor
            contributor_windows = {}
            for contributor, steps in steps_by_contributor.items():
                if steps:
                    earliest = min(step.get_creation_date() for step in steps)
                    latest = max(step.get_creation_date() for step in steps)
                    contributor_windows[contributor] = (earliest, latest)

            # Check for overlapping time windows
            overlapping = False
            for user1, (start1, end1) in contributor_windows.items():
                for user2, (start2, end2) in contributor_windows.items():
                    if user1 != user2:
                        # Check for overlap
                        if start1 <= end2 and start2 <= end1:
                            overlapping = True
                            break

            # Check time proximity of contributions
            min_gap = float("inf")
            steps_in_sequence = True

            # Check if steps strictly alternate between contributors
            current_contributor = sorted_steps[0].owner
            strictly_alternating = True

            for i in range(1, len(sorted_steps)):
                next_contributor = sorted_steps[i].owner

                # Check gap between steps
                time_gap = (
                    sorted_steps[i].get_creation_date()
                    - sorted_steps[i - 1].get_creation_date()
                ).total_seconds() / 3600  # hours
                min_gap = min(min_gap, time_gap)

                # Check if contributors alternate
                if next_contributor == current_contributor:
                    strictly_alternating = False

                current_contributor = next_contributor

            # Determine if synchronous based on minimum gap
            if min_gap <= sync_threshold_hours:
                patterns["synchronous_collaboration"] += 1
            else:
                patterns["asynchronous_collaboration"] += 1

            # Calculate collaboration window (time from first to last step)
            first_step_time = sorted_steps[0].get_creation_date()
            last_step_time = sorted_steps[-1].get_creation_date()
            window_hours = (last_step_time - first_step_time).total_seconds() / 3600

            collaboration_windows.append(window_hours)

            # Determine if sequential or parallel
            if strictly_alternating or not overlapping:
                sequential_count += 1
            else:
                parallel_count += 1

        # Calculate averages
        if collaboration_windows:
            patterns["avg_collaboration_window_hours"] = sum(
                collaboration_windows
            ) / len(collaboration_windows)

        total_collaborations = sequential_count + parallel_count
        if total_collaborations > 0:
            patterns["sequential_vs_parallel"] = {
                "sequential": {
                    "count": sequential_count,
                    "percentage": sequential_count / total_collaborations,
                },
                "parallel": {
                    "count": parallel_count,
                    "percentage": parallel_count / total_collaborations,
                },
            }

        return patterns

    def _summarize_collaboration_metrics(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize collaboration metrics across teams.

        Args:
            team_metrics: List of metrics for each team

        Returns:
            Dict with summarized collaboration metrics
        """
        summary = {
            "collaboration_levels": {},
            "member_involvement": {},
            "collaborative_activity_patterns": {},
        }

        # Skip if no team metrics
        if not team_metrics:
            return summary

        # Count collaboration levels
        collaboration_levels = {
            "high": 0,
            "medium": 0,
            "low": 0,
            "none": 0,
            "single_member": 0,
        }

        for metrics in team_metrics:
            level = metrics.get("collaboration_level", "none")
            collaboration_levels[level] += 1

        # Calculate percentages
        total_teams = len(team_metrics)
        summary["collaboration_levels"] = {
            level: {
                "count": count,
                "percentage": count / total_teams if total_teams > 0 else 0,
            }
            for level, count in collaboration_levels.items()
        }

        # Analyze member involvement
        member_involvement = {
            "avg_active_ratio": 0,
            "full_team_engagement": 0,
            "partial_engagement": 0,
        }

        for metrics in team_metrics:
            member_count = metrics.get("member_count", 0)
            active_members = metrics.get("active_members", 0)

            if member_count > 0:
                active_ratio = active_members / member_count

                if active_ratio == 1.0:
                    member_involvement["full_team_engagement"] += 1
                else:
                    member_involvement["partial_engagement"] += 1

                # Accumulate for average
                member_involvement["avg_active_ratio"] += active_ratio

        # Calculate average
        if total_teams > 0:
            member_involvement["avg_active_ratio"] /= total_teams

            # Calculate percentages
            member_involvement["full_team_engagement_rate"] = (
                member_involvement["full_team_engagement"] / total_teams
            )
            member_involvement["partial_engagement_rate"] = (
                member_involvement["partial_engagement"] / total_teams
            )

        summary["member_involvement"] = member_involvement

        # Analyze activity patterns from temporal analysis
        teams_with_temporal = [m for m in team_metrics if "temporal_patterns" in m]

        if teams_with_temporal:
            sync_count = sum(
                m["temporal_patterns"].get("synchronous_collaboration", 0)
                for m in teams_with_temporal
            )
            async_count = sum(
                m["temporal_patterns"].get("asynchronous_collaboration", 0)
                for m in teams_with_temporal
            )

            total_patterns = sync_count + async_count

            if total_patterns > 0:
                summary["collaborative_activity_patterns"][
                    "synchronous_vs_asynchronous"
                ] = {
                    "synchronous": {
                        "count": sync_count,
                        "percentage": sync_count / total_patterns,
                    },
                    "asynchronous": {
                        "count": async_count,
                        "percentage": async_count / total_patterns,
                    },
                }

            # Get average collaboration window
            windows = [
                m["temporal_patterns"].get("avg_collaboration_window_hours", 0)
                for m in teams_with_temporal
                if "avg_collaboration_window_hours" in m["temporal_patterns"]
            ]

            if windows:
                summary["collaborative_activity_patterns"][
                    "avg_collaboration_window_hours"
                ] = sum(windows) / len(windows)

            # Summarize sequential vs. parallel patterns
            seq_counts = []
            par_counts = []

            for metrics in teams_with_temporal:
                if (
                    "temporal_patterns" in metrics
                    and "sequential_vs_parallel" in metrics["temporal_patterns"]
                    and "sequential"
                    in metrics["temporal_patterns"]["sequential_vs_parallel"]
                ):

                    seq = metrics["temporal_patterns"]["sequential_vs_parallel"][
                        "sequential"
                    ]
                    par = metrics["temporal_patterns"]["sequential_vs_parallel"][
                        "parallel"
                    ]

                    seq_counts.append(seq["percentage"] if "percentage" in seq else 0)
                    par_counts.append(par["percentage"] if "percentage" in par else 0)

            if seq_counts and par_counts:
                summary["collaborative_activity_patterns"][
                    "avg_sequential_percentage"
                ] = sum(seq_counts) / len(seq_counts)
                summary["collaborative_activity_patterns"][
                    "avg_parallel_percentage"
                ] = sum(par_counts) / len(par_counts)

        return summary

    def _analyze_idea_sharing_patterns(self, teams: List[Any]) -> Dict[str, Any]:
        """
        Analyze patterns in how ideas are shared within teams.

        Args:
            teams: List of teams to analyze

        Returns:
            Dict with idea sharing pattern analysis
        """
        patterns = {
            "ownership_patterns": {},
            "contribution_flow": {},
            "cross_idea_collaboration": {},
        }

        # Skip if no teams
        if not teams:
            return patterns

        # Track ownership patterns
        owner_contributes_most = 0
        equal_contribution = 0
        non_owner_contributes_most = 0

        # Track contribution flow
        creator_first = 0
        parallel_start = 0
        contributor_first = 0

        # Track cross-idea collaboration
        total_teams_with_multiple_ideas = 0
        teams_with_cross_idea_collaboration = 0

        for team in teams:
            # Get member emails
            member_emails = team.get_member_emails()

            if not member_emails:
                continue

            # Get all ideas owned by team members
            all_ideas = []
            for email in member_emails:
                user_ideas = self._data_repo.ideas.find_by_owner(email)
                for idea in user_ideas:
                    all_ideas.append(idea)

            # Skip if no ideas
            if not all_ideas:
                continue

            # Analyze idea ownership and contributions
            ownership_contribution = []

            for idea in all_ideas:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

                # Get steps for this idea
                steps = self._data_repo.steps.find_by_idea_id(idea_id)

                # Filter to steps by team members
                team_steps = [step for step in steps if step.owner in member_emails]

                # Skip if no steps by team members
                if not team_steps:
                    continue

                # Count steps by contributor
                contributor_counts = defaultdict(int)
                for step in team_steps:
                    if step.owner:
                        contributor_counts[step.owner] += 1

                # Skip if only one contributor
                if len(contributor_counts) < 2:
                    continue

                # Get idea owner
                owner = idea.owner

                # Get contributor with most steps
                top_contributor = max(contributor_counts.items(), key=lambda x: x[1])[0]

                # Analyze contribution balance
                max_steps = max(contributor_counts.values())
                min_steps = min(contributor_counts.values())
                step_range = max_steps - min_steps

                if step_range <= 1:
                    equal_contribution += 1
                elif top_contributor == owner:
                    owner_contributes_most += 1
                else:
                    non_owner_contributes_most += 1

                # Analyze contribution flow
                steps_with_dates = [
                    step for step in team_steps if step.get_creation_date()
                ]

                if len(steps_with_dates) >= 2:
                    # Sort by creation date
                    sorted_steps = sorted(
                        steps_with_dates, key=lambda s: s.get_creation_date()
                    )

                    first_contributor = sorted_steps[0].owner
                    second_contributor = sorted_steps[1].owner

                    if first_contributor == owner:
                        creator_first += 1
                    elif first_contributor != owner and second_contributor == owner:
                        contributor_first += 1
                    else:
                        parallel_start += 1

                # Add to ownership contribution list for cross-idea analysis
                ownership_contribution.append(
                    {
                        "idea_id": idea_id,
                        "owner": owner,
                        "contributors": list(contributor_counts.keys()),
                    }
                )

            # Analyze cross-idea collaboration
            if len(ownership_contribution) >= 2:
                total_teams_with_multiple_ideas += 1

                # Check if the same contributors work across multiple ideas
                has_cross_idea = False

                for i in range(len(ownership_contribution)):
                    for j in range(i + 1, len(ownership_contribution)):
                        idea1 = ownership_contribution[i]
                        idea2 = ownership_contribution[j]

                        # Check for common contributors (other than owner)
                        contributors1 = set(idea1["contributors"])
                        contributors2 = set(idea2["contributors"])

                        if idea1["owner"]:
                            contributors1.discard(idea1["owner"])
                        if idea2["owner"]:
                            contributors2.discard(idea2["owner"])

                        if (
                            contributors1
                            and contributors2
                            and contributors1.intersection(contributors2)
                        ):
                            has_cross_idea = True
                            break

                if has_cross_idea:
                    teams_with_cross_idea_collaboration += 1

        # Calculate totals
        total_shared_ideas = (
            owner_contributes_most + equal_contribution + non_owner_contributes_most
        )
        total_flow_patterns = creator_first + parallel_start + contributor_first

        # Calculate pattern percentages
        if total_shared_ideas > 0:
            patterns["ownership_patterns"] = {
                "owner_contributes_most": {
                    "count": owner_contributes_most,
                    "percentage": owner_contributes_most / total_shared_ideas,
                },
                "equal_contribution": {
                    "count": equal_contribution,
                    "percentage": equal_contribution / total_shared_ideas,
                },
                "non_owner_contributes_most": {
                    "count": non_owner_contributes_most,
                    "percentage": non_owner_contributes_most / total_shared_ideas,
                },
            }

        if total_flow_patterns > 0:
            patterns["contribution_flow"] = {
                "creator_first": {
                    "count": creator_first,
                    "percentage": creator_first / total_flow_patterns,
                },
                "parallel_start": {
                    "count": parallel_start,
                    "percentage": parallel_start / total_flow_patterns,
                },
                "contributor_first": {
                    "count": contributor_first,
                    "percentage": contributor_first / total_flow_patterns,
                },
            }

        if total_teams_with_multiple_ideas > 0:
            patterns["cross_idea_collaboration"] = {
                "teams_with_multiple_ideas": total_teams_with_multiple_ideas,
                "teams_with_cross_idea_collaboration": teams_with_cross_idea_collaboration,
                "cross_idea_collaboration_rate": teams_with_cross_idea_collaboration
                / total_teams_with_multiple_ideas,
            }

        return patterns

    def _analyze_team_usage_distribution(self, team: Any) -> Dict[str, Any]:
        """
        Analyze how tool usage is distributed among team members.

        Args:
            team: Team to analyze

        Returns:
            Dict with usage distribution analysis
        """
        result = {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "member_count": team.get_member_count(),
            "active_member_count": 0,
            "active_member_ratio": 0,
            "distribution_type": "inactive",
            "gini_coefficient": 0,
            "coefficient_of_variation": 0,
        }

        # Get member emails
        member_emails = team.get_member_emails()

        if not member_emails:
            return result

        # Initialize activity counts
        activity_counts = {email: 0 for email in member_emails}

        # Count ideas and steps for each member
        for email in member_emails:
            # Count ideas
            ideas = self._data_repo.ideas.find_by_owner(email)
            activity_counts[email] += len(ideas) * 5  # Weight ideas more heavily

            # Count steps
            steps = self._data_repo.steps.find_by_owner(email)
            activity_counts[email] += len(steps)

        # Count active members
        active_members = [
            email for email, count in activity_counts.items() if count > 0
        ]
        result["active_member_count"] = len(active_members)

        if result["member_count"] > 0:
            result["active_member_ratio"] = (
                result["active_member_count"] / result["member_count"]
            )

        # Determine distribution type
        if result["active_member_count"] == 0:
            result["distribution_type"] = "inactive"
        elif result["active_member_count"] == 1:
            result["distribution_type"] = "primary"
        else:
            # Get activity values in sorted order
            activity_values = sorted(activity_counts.values())

            if len(activity_values) >= 2:
                # Calculate Gini coefficient
                gini = self._calculate_gini_coefficient(activity_values)
                result["gini_coefficient"] = gini

                # Calculate coefficient of variation
                if sum(activity_values) > 0:
                    mean = sum(activity_values) / len(activity_values)
                    std_dev = (
                        sum((x - mean) ** 2 for x in activity_values)
                        / len(activity_values)
                    ) ** 0.5
                    cv = std_dev / mean if mean > 0 else 0
                    result["coefficient_of_variation"] = cv

                # Determine distribution type based on Gini coefficient
                if gini < 0.2:
                    result["distribution_type"] = "uniform"
                elif gini > 0.6:
                    result["distribution_type"] = "primary"
                else:
                    result["distribution_type"] = "partial"

        return result

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
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
            return 0

        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate cumulative sum
        cum_values = [sum(sorted_values[: i + 1]) for i in range(n)]
        total = cum_values[-1]

        # Calculate Gini coefficient
        if total == 0:
            return 0

        fair_area = sum(range(1, n + 1)) / n / n
        actual_area = sum((n - i) * sorted_values[i] for i in range(n)) / n / total

        return 2 * (fair_area - actual_area)

    def _analyze_team_roles(
        self, team_distributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in team member roles.

        Args:
            team_distributions: List of team usage distribution analyses

        Returns:
            Dict with team role pattern analysis
        """
        roles = {
            "observed_patterns": {},
            "role_specialization": {},
        }

        # Skip if no team distributions
        if not team_distributions:
            return roles

        # Track primary contributor teams
        primary_teams = [
            t
            for t in team_distributions
            if t.get("distribution_type") == "primary"
            and t.get("active_member_count") > 0
        ]

        # Track partial contribution teams
        partial_teams = [
            t for t in team_distributions if t.get("distribution_type") == "partial"
        ]

        # Track uniform teams
        uniform_teams = [
            t for t in team_distributions if t.get("distribution_type") == "uniform"
        ]

        # Calculate percentages
        total_active_teams = (
            len(primary_teams) + len(partial_teams) + len(uniform_teams)
        )

        if total_active_teams > 0:
            roles["observed_patterns"] = {
                "primary_contributor": {
                    "count": len(primary_teams),
                    "percentage": len(primary_teams) / total_active_teams,
                    "avg_team_size": (
                        sum(t.get("member_count", 0) for t in primary_teams)
                        / len(primary_teams)
                        if primary_teams
                        else 0
                    ),
                },
                "specialized_roles": {
                    "count": len(partial_teams),
                    "percentage": len(partial_teams) / total_active_teams,
                    "avg_team_size": (
                        sum(t.get("member_count", 0) for t in partial_teams)
                        / len(partial_teams)
                        if partial_teams
                        else 0
                    ),
                },
                "balanced_contribution": {
                    "count": len(uniform_teams),
                    "percentage": len(uniform_teams) / total_active_teams,
                    "avg_team_size": (
                        sum(t.get("member_count", 0) for t in uniform_teams)
                        / len(uniform_teams)
                        if uniform_teams
                        else 0
                    ),
                },
            }

        # Calculate role specialization metrics
        if partial_teams:
            avg_gini = sum(t.get("gini_coefficient", 0) for t in partial_teams) / len(
                partial_teams
            )
            avg_cv = sum(
                t.get("coefficient_of_variation", 0) for t in partial_teams
            ) / len(partial_teams)

            roles["role_specialization"] = {
                "avg_gini_coefficient": avg_gini,
                "avg_coefficient_of_variation": avg_cv,
                "specialization_level": (
                    "high" if avg_gini > 0.4 else "medium" if avg_gini > 0.2 else "low"
                ),
            }

        return roles

    def _calculate_team_composition_metrics(self, team: Any) -> Dict[str, Any]:
        """
        Calculate metrics about team composition and tool usage.

        Args:
            team: Team to analyze

        Returns:
            Dict with team composition and usage metrics
        """
        metrics = {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "team_size": team.get_member_count(),
            "member_info": [],
            "diversity_measures": {},
            "engagement_metrics": {},
        }

        # Get member emails
        member_emails = team.get_member_emails()

        if not member_emails:
            return metrics

        # Get user objects for members
        members = []
        for email in member_emails:
            user = self._data_repo.users.find_by_email(email)
            if user:
                members.append(user)

        # Skip if no members found
        if not members:
            return metrics

        # Calculate member info
        metrics["member_info"] = self._calculate_member_info(members)

        # Calculate diversity measures
        metrics["diversity_measures"] = self._calculate_diversity_measures(members)

        # Calculate engagement metrics
        metrics["engagement_metrics"] = self._calculate_team_engagement_metrics(members)

        return metrics

    def _calculate_member_info(self, members: List[Any]) -> List[Dict[str, Any]]:
        """
        Calculate information about team members.

        Args:
            members: List of user objects for team members

        Returns:
            List of member information
        """
        member_info = []

        for user in members:
            if not user.email:
                continue

            # Get user's ideas
            ideas = self._data_repo.ideas.find_by_owner(user.email)

            # Get user's steps
            steps = self._data_repo.steps.find_by_owner(user.email)

            # Calculate engagement level
            engagement_level = user.get_engagement_level().value

            # Get user type
            user_type = (
                user.get_user_type().value if user.get_user_type() else "unknown"
            )

            # Get user department
            department = user.get_department() or "unknown"

            # Get experience level
            experience = (
                user.orbit_profile.experience
                if user.orbit_profile and user.orbit_profile.experience
                else "unknown"
            )

            # Add member info
            member_info.append(
                {
                    "email": user.email,
                    "user_type": user_type,
                    "department": department,
                    "experience": experience,
                    "idea_count": len(ideas),
                    "step_count": len(steps),
                    "engagement_level": engagement_level,
                }
            )

        return member_info

    def _calculate_diversity_measures(self, members: List[Any]) -> Dict[str, Any]:
        """
        Calculate diversity measures for a team.

        Args:
            members: List of user objects for team members

        Returns:
            Dict with diversity measures
        """
        measures = {
            "user_type_diversity": 0,
            "department_diversity": 0,
            "experience_diversity": 0,
            "overall_diversity_score": 0,
        }

        # Skip if too few members
        if len(members) < 2:
            return measures

        # Calculate diversity based on unique values
        user_types = set()
        departments = set()
        experiences = set()

        for user in members:
            if user.get_user_type():
                user_types.add(user.get_user_type().value)

            department = user.get_department()
            if department:
                departments.add(department)

            if user.orbit_profile and user.orbit_profile.experience:
                experiences.add(user.orbit_profile.experience)

        # Calculate diversity scores
        # Normalized to range [0, 1] where 0 is no diversity (all same) and 1 is max diversity
        team_size = len(members)

        if team_size > 1:
            user_type_diversity = (
                (len(user_types) - 1) / (team_size - 1) if user_types else 0
            )
            department_diversity = (
                (len(departments) - 1) / (team_size - 1) if departments else 0
            )
            experience_diversity = (
                (len(experiences) - 1) / (team_size - 1) if experiences else 0
            )

            measures["user_type_diversity"] = user_type_diversity
            measures["department_diversity"] = department_diversity
            measures["experience_diversity"] = experience_diversity

            # Calculate overall diversity score (weighted average)
            measures["overall_diversity_score"] = (
                (user_type_diversity * 0.4)
                + (department_diversity * 0.4)
                + (experience_diversity * 0.2)
            )

        return measures

    def _calculate_team_engagement_metrics(self, members: List[Any]) -> Dict[str, Any]:
        """
        Calculate team engagement metrics.

        Args:
            members: List of user objects for team members

        Returns:
            Dict with engagement metrics
        """
        metrics = {
            "high_engagement_count": 0,
            "medium_engagement_count": 0,
            "low_engagement_count": 0,
            "avg_ideas_per_member": 0,
            "avg_steps_per_member": 0,
            "team_engagement_score": 0,
        }

        # Skip if no members
        if not members:
            return metrics

        # Count engagement levels
        total_ideas = 0
        total_steps = 0

        for user in members:
            if not user.email:
                continue

            # Track engagement level
            engagement_level = user.get_engagement_level()
            if engagement_level == UserEngagementLevel.HIGH:
                metrics["high_engagement_count"] += 1
            elif engagement_level == UserEngagementLevel.MEDIUM:
                metrics["medium_engagement_count"] += 1
            else:
                metrics["low_engagement_count"] += 1

            # Count ideas
            ideas = self._data_repo.ideas.find_by_owner(user.email)
            total_ideas += len(ideas)

            # Count steps
            steps = self._data_repo.steps.find_by_owner(user.email)
            total_steps += len(steps)

        # Calculate averages
        team_size = len(members)
        metrics["avg_ideas_per_member"] = (
            total_ideas / team_size if team_size > 0 else 0
        )
        metrics["avg_steps_per_member"] = (
            total_steps / team_size if team_size > 0 else 0
        )

        # Calculate team engagement score
        if team_size > 0:
            # Weight high engagement more heavily
            engagement_score = (
                (metrics["high_engagement_count"] * 1.0)
                + (metrics["medium_engagement_count"] * 0.5)
                + (metrics["low_engagement_count"] * 0.1)
            ) / team_size

            metrics["team_engagement_score"] = engagement_score

        return metrics

    def _analyze_team_size_correlation(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between team size and tool usage.

        Args:
            team_metrics: List of team composition metrics

        Returns:
            Dict with team size correlation analysis
        """
        correlation = {
            "size_groups": {},
            "size_impact": {},
        }

        # Skip if too few teams
        if len(team_metrics) < 3:
            return correlation

        # Group teams by size
        size_groups = defaultdict(list)

        for metrics in team_metrics:
            team_size = metrics.get("team_size", 0)

            # Skip teams with no size info
            if team_size == 0:
                continue

            # Group into size categories
            if team_size <= 2:
                size_group = "small"
            elif team_size <= 4:
                size_group = "medium"
            else:
                size_group = "large"

            size_groups[size_group].append(metrics)

        # Calculate metrics for each size group
        for size_group, group_metrics in size_groups.items():
            if not group_metrics:
                continue

            # Calculate engagement metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            avg_ideas = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            avg_steps = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add group metrics
            correlation["size_groups"][size_group] = {
                "team_count": len(group_metrics),
                "avg_team_size": sum(m.get("team_size", 0) for m in group_metrics)
                / len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(avg_ideas) / len(avg_ideas) if avg_ideas else 0
                ),
                "avg_steps_per_member": (
                    sum(avg_steps) / len(avg_steps) if avg_steps else 0
                ),
            }

        # Calculate correlations between team size and metrics
        team_sizes = [
            m.get("team_size", 0) for m in team_metrics if m.get("team_size", 0) > 0
        ]

        if not team_sizes:
            return correlation

        # Get engagement scores
        engagement_scores = [
            m.get("engagement_metrics", {}).get("team_engagement_score", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 0
        ]

        # Get idea averages
        idea_avgs = [
            m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 0
        ]

        # Get step averages
        step_avgs = [
            m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 0
        ]

        # Calculate correlations
        engagement_corr = self._calculate_correlation(team_sizes, engagement_scores)
        idea_corr = self._calculate_correlation(team_sizes, idea_avgs)
        step_corr = self._calculate_correlation(team_sizes, step_avgs)

        correlation["size_impact"] = {
            "engagement_correlation": engagement_corr,
            "idea_production_correlation": idea_corr,
            "step_production_correlation": step_corr,
            "overall_size_effect": (
                "positive"
                if (engagement_corr + idea_corr + step_corr) / 3 > 0.1
                else (
                    "negative"
                    if (engagement_corr + idea_corr + step_corr) / 3 < -0.1
                    else "neutral"
                )
            ),
        }

        return correlation

    def _analyze_diversity_correlation(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between team diversity and tool usage.

        Args:
            team_metrics: List of team composition metrics

        Returns:
            Dict with diversity correlation analysis
        """
        correlation = {
            "diversity_groups": {},
            "diversity_impact": {},
        }

        # Skip if too few teams
        if len(team_metrics) < 3:
            return correlation

        # Group teams by diversity level
        diversity_groups = defaultdict(list)

        for metrics in team_metrics:
            diversity_score = metrics.get("diversity_measures", {}).get(
                "overall_diversity_score", 0
            )

            # Skip teams with no diversity info
            if diversity_score == 0 and metrics.get("team_size", 0) <= 1:
                continue

            # Group into diversity categories
            if diversity_score < 0.3:
                diversity_group = "low"
            elif diversity_score < 0.7:
                diversity_group = "medium"
            else:
                diversity_group = "high"

            diversity_groups[diversity_group].append(metrics)

        # Calculate metrics for each diversity group
        for diversity_group, group_metrics in diversity_groups.items():
            if not group_metrics:
                continue

            # Calculate engagement metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            avg_ideas = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            avg_steps = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add group metrics
            correlation["diversity_groups"][diversity_group] = {
                "team_count": len(group_metrics),
                "avg_diversity_score": sum(
                    m.get("diversity_measures", {}).get("overall_diversity_score", 0)
                    for m in group_metrics
                )
                / len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(avg_ideas) / len(avg_ideas) if avg_ideas else 0
                ),
                "avg_steps_per_member": (
                    sum(avg_steps) / len(avg_steps) if avg_steps else 0
                ),
            }

        # Calculate correlations between diversity and metrics
        diversity_scores = [
            m.get("diversity_measures", {}).get("overall_diversity_score", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 1
        ]

        if not diversity_scores:
            return correlation

        # Get engagement scores
        engagement_scores = [
            m.get("engagement_metrics", {}).get("team_engagement_score", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 1
        ]

        # Get idea averages
        idea_avgs = [
            m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 1
        ]

        # Get step averages
        step_avgs = [
            m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
            for m in team_metrics
            if m.get("team_size", 0) > 1
        ]

        # Calculate correlations
        engagement_corr = self._calculate_correlation(
            diversity_scores, engagement_scores
        )
        idea_corr = self._calculate_correlation(diversity_scores, idea_avgs)
        step_corr = self._calculate_correlation(diversity_scores, step_avgs)

        correlation["diversity_impact"] = {
            "engagement_correlation": engagement_corr,
            "idea_production_correlation": idea_corr,
            "step_production_correlation": step_corr,
            "overall_diversity_effect": (
                "positive"
                if (engagement_corr + idea_corr + step_corr) / 3 > 0.1
                else (
                    "negative"
                    if (engagement_corr + idea_corr + step_corr) / 3 < -0.1
                    else "neutral"
                )
            ),
        }

        return correlation

    def _analyze_composition_factors(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze specific team composition factors that influence tool usage.

        Args:
            team_metrics: List of team composition metrics

        Returns:
            Dict with composition factor analysis
        """
        factors = {
            "user_type_impact": {},
            "experience_impact": {},
            "department_mix_impact": {},
        }

        # Skip if too few teams
        if len(team_metrics) < 3:
            return factors

        # Analyze impact of different user types
        user_type_impacts = {}

        for metrics in team_metrics:
            # Skip teams without member info
            if "member_info" not in metrics or not metrics["member_info"]:
                continue

            # Count members by user type
            user_type_counts = defaultdict(int)
            for member in metrics["member_info"]:
                user_type = member.get("user_type", "unknown")
                if user_type and user_type != "unknown":
                    user_type_counts[user_type] += 1

            # Skip if no user type info
            if not user_type_counts:
                continue

            # Calculate team engagement score
            engagement_score = metrics.get("engagement_metrics", {}).get(
                "team_engagement_score", 0
            )

            # Calculate predominant user type
            predominant_type = max(user_type_counts.items(), key=lambda x: x[1])[0]

            # Accumulate engagement scores by predominant type
            if predominant_type not in user_type_impacts:
                user_type_impacts[predominant_type] = {
                    "teams": 0,
                    "total_engagement": 0,
                }

            user_type_impacts[predominant_type]["teams"] += 1
            user_type_impacts[predominant_type]["total_engagement"] += engagement_score

        # Calculate average engagement by user type
        for user_type, impact in user_type_impacts.items():
            if impact["teams"] > 0:
                impact["avg_engagement"] = impact["total_engagement"] / impact["teams"]

        # Sort by average engagement (descending)
        if user_type_impacts:
            sorted_types = sorted(
                user_type_impacts.items(),
                key=lambda x: x[1]["avg_engagement"],
                reverse=True,
            )

            factors["user_type_impact"] = {
                user_type: {
                    "teams": impact["teams"],
                    "avg_engagement": impact["avg_engagement"],
                }
                for user_type, impact in sorted_types
            }

        # Analyze impact of experience levels
        experience_impacts = {}

        for metrics in team_metrics:
            # Skip teams without member info
            if "member_info" not in metrics or not metrics["member_info"]:
                continue

            # Count members by experience
            experience_counts = defaultdict(int)
            for member in metrics["member_info"]:
                experience = member.get("experience", "unknown")
                if experience and experience != "unknown":
                    experience_counts[experience] += 1

            # Skip if no experience info
            if not experience_counts:
                continue

            # Calculate team engagement score
            engagement_score = metrics.get("engagement_metrics", {}).get(
                "team_engagement_score", 0
            )

            # Calculate predominant experience
            predominant_exp = max(experience_counts.items(), key=lambda x: x[1])[0]

            # Accumulate engagement scores by predominant experience
            if predominant_exp not in experience_impacts:
                experience_impacts[predominant_exp] = {
                    "teams": 0,
                    "total_engagement": 0,
                }

            experience_impacts[predominant_exp]["teams"] += 1
            experience_impacts[predominant_exp]["total_engagement"] += engagement_score

        # Calculate average engagement by experience
        for exp, impact in experience_impacts.items():
            if impact["teams"] > 0:
                impact["avg_engagement"] = impact["total_engagement"] / impact["teams"]

        # Sort by average engagement (descending)
        if experience_impacts:
            sorted_exps = sorted(
                experience_impacts.items(),
                key=lambda x: x[1]["avg_engagement"],
                reverse=True,
            )

            factors["experience_impact"] = {
                exp: {
                    "teams": impact["teams"],
                    "avg_engagement": impact["avg_engagement"],
                }
                for exp, impact in sorted_exps
            }

        # Analyze impact of department mix
        department_mix_impacts = {
            "single_department": {
                "teams": 0,
                "total_engagement": 0,
                "avg_engagement": 0,
            },
            "mixed_departments": {
                "teams": 0,
                "total_engagement": 0,
                "avg_engagement": 0,
            },
        }

        for metrics in team_metrics:
            # Skip teams without member info
            if "member_info" not in metrics or not metrics["member_info"]:
                continue

            # Get unique departments
            departments = set()
            for member in metrics["member_info"]:
                department = member.get("department", "unknown")
                if department and department != "unknown":
                    departments.add(department)

            # Skip if no department info
            if not departments:
                continue

            # Calculate team engagement score
            engagement_score = metrics.get("engagement_metrics", {}).get(
                "team_engagement_score", 0
            )

            # Single department or mixed
            if len(departments) == 1:
                department_mix_impacts["single_department"]["teams"] += 1
                department_mix_impacts["single_department"][
                    "total_engagement"
                ] += engagement_score
            else:
                department_mix_impacts["mixed_departments"]["teams"] += 1
                department_mix_impacts["mixed_departments"][
                    "total_engagement"
                ] += engagement_score

        # Calculate average engagement by department mix
        for mix_type, impact in department_mix_impacts.items():
            if impact["teams"] > 0:
                impact["avg_engagement"] = impact["total_engagement"] / impact["teams"]

        factors["department_mix_impact"] = department_mix_impacts

        return factors

    def _breakdown_by_team_size(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Provide detailed breakdown of metrics by team size.

        Args:
            team_metrics: List of team composition metrics

        Returns:
            Dict with detailed breakdown by team size
        """
        breakdown = {}

        # Skip if too few teams
        if len(team_metrics) < 3:
            return breakdown

        # Group teams by exact size
        size_groups = defaultdict(list)

        for metrics in team_metrics:
            team_size = metrics.get("team_size", 0)

            # Skip teams with no size info
            if team_size == 0:
                continue

            size_groups[team_size].append(metrics)

        # Calculate metrics for each size
        for size, group_metrics in size_groups.items():
            if not group_metrics or len(group_metrics) < 2:
                continue  # Need at least 2 teams for meaningful analysis

            # Calculate detailed metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            high_engagement_counts = [
                m.get("engagement_metrics", {}).get("high_engagement_count", 0)
                for m in group_metrics
            ]

            diversity_scores = [
                m.get("diversity_measures", {}).get("overall_diversity_score", 0)
                for m in group_metrics
            ]

            idea_avgs = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            step_avgs = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add detailed metrics
            breakdown[str(size)] = {
                "team_count": len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "high_engagement_ratio": (
                    sum(high_engagement_counts) / (size * len(group_metrics))
                    if size > 0
                    else 0
                ),
                "avg_diversity_score": (
                    sum(diversity_scores) / len(diversity_scores)
                    if diversity_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(idea_avgs) / len(idea_avgs) if idea_avgs else 0
                ),
                "avg_steps_per_member": (
                    sum(step_avgs) / len(step_avgs) if step_avgs else 0
                ),
            }

        return breakdown

    def _breakdown_by_diversity(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Provide detailed breakdown of metrics by diversity level.

        Args:
            team_metrics: List of team composition metrics

        Returns:
            Dict with detailed breakdown by diversity level
        """
        breakdown = {
            "user_type_diversity": {},
            "department_diversity": {},
            "experience_diversity": {},
        }

        # Skip if too few teams
        if len(team_metrics) < 3:
            return breakdown

        # Group teams by user type diversity
        user_type_groups = {
            "low": [],
            "medium": [],
            "high": [],
        }

        # Group teams by department diversity
        department_groups = {
            "low": [],
            "medium": [],
            "high": [],
        }

        # Group teams by experience diversity
        experience_groups = {
            "low": [],
            "medium": [],
            "high": [],
        }

        for metrics in team_metrics:
            diversity = metrics.get("diversity_measures", {})

            # Skip teams with no diversity info
            if not diversity:
                continue

            # Group by user type diversity
            user_type_div = diversity.get("user_type_diversity", 0)
            if user_type_div < 0.3:
                user_type_groups["low"].append(metrics)
            elif user_type_div < 0.7:
                user_type_groups["medium"].append(metrics)
            else:
                user_type_groups["high"].append(metrics)

            # Group by department diversity
            department_div = diversity.get("department_diversity", 0)
            if department_div < 0.3:
                department_groups["low"].append(metrics)
            elif department_div < 0.7:
                department_groups["medium"].append(metrics)
            else:
                department_groups["high"].append(metrics)

            # Group by experience diversity
            experience_div = diversity.get("experience_diversity", 0)
            if experience_div < 0.3:
                experience_groups["low"].append(metrics)
            elif experience_div < 0.7:
                experience_groups["medium"].append(metrics)
            else:
                experience_groups["high"].append(metrics)

        # Calculate metrics for each user type diversity group
        for level, group_metrics in user_type_groups.items():
            if not group_metrics or len(group_metrics) < 2:
                continue  # Need at least 2 teams for meaningful analysis

            # Calculate detailed metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            idea_avgs = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            step_avgs = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add detailed metrics
            breakdown["user_type_diversity"][level] = {
                "team_count": len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(idea_avgs) / len(idea_avgs) if idea_avgs else 0
                ),
                "avg_steps_per_member": (
                    sum(step_avgs) / len(step_avgs) if step_avgs else 0
                ),
            }

        # Calculate metrics for each department diversity group
        for level, group_metrics in department_groups.items():
            if not group_metrics or len(group_metrics) < 2:
                continue  # Need at least 2 teams for meaningful analysis

            # Calculate detailed metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            idea_avgs = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            step_avgs = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add detailed metrics
            breakdown["department_diversity"][level] = {
                "team_count": len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(idea_avgs) / len(idea_avgs) if idea_avgs else 0
                ),
                "avg_steps_per_member": (
                    sum(step_avgs) / len(step_avgs) if step_avgs else 0
                ),
            }

        # Calculate metrics for each experience diversity group
        for level, group_metrics in experience_groups.items():
            if not group_metrics or len(group_metrics) < 2:
                continue  # Need at least 2 teams for meaningful analysis

            # Calculate detailed metrics
            engagement_scores = [
                m.get("engagement_metrics", {}).get("team_engagement_score", 0)
                for m in group_metrics
            ]

            idea_avgs = [
                m.get("engagement_metrics", {}).get("avg_ideas_per_member", 0)
                for m in group_metrics
            ]

            step_avgs = [
                m.get("engagement_metrics", {}).get("avg_steps_per_member", 0)
                for m in group_metrics
            ]

            # Add detailed metrics
            breakdown["experience_diversity"][level] = {
                "team_count": len(group_metrics),
                "avg_engagement_score": (
                    sum(engagement_scores) / len(engagement_scores)
                    if engagement_scores
                    else 0
                ),
                "avg_ideas_per_member": (
                    sum(idea_avgs) / len(idea_avgs) if idea_avgs else 0
                ),
                "avg_steps_per_member": (
                    sum(step_avgs) / len(step_avgs) if step_avgs else 0
                ),
            }

        return breakdown

    def _analyze_team_framework_progression(
        self, teams: List[Any], framework: FrameworkType, framework_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze how teams progress through entrepreneurial frameworks.

        Args:
            teams: List of teams to analyze
            framework: The framework to analyze
            framework_steps: List of steps in the framework

        Returns:
            Dict with team framework progression analysis
        """
        progression = {
            "step_completion_rates": {},
            "progression_patterns": {},
            "collaboration_impact": {},
        }

        # Skip if no teams
        if not teams:
            return progression

        # Track step completion for all team ideas
        step_counts = {step: 0 for step in framework_steps}
        total_ideas = 0

        # Track progression patterns
        linear_progression_count = 0
        non_linear_progression_count = 0

        # Track collaboration impact
        ideas_with_collaboration = []
        ideas_without_collaboration = []

        # Process each team
        for team in teams:
            # Get member emails
            member_emails = team.get_member_emails()

            if not member_emails:
                continue

            # Get all ideas owned by team members
            team_ideas = []
            for email in member_emails:
                ideas = self._data_repo.ideas.find_by_owner(email)
                for idea in ideas:
                    if idea.id:
                        team_ideas.append(idea)

            # Skip if no ideas
            if not team_ideas:
                continue

            # Process each idea
            for idea in team_ideas:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

                # Count completed steps
                completed_steps = []
                for step in framework_steps:
                    if idea.has_step(step):
                        step_counts[step] += 1
                        completed_steps.append(step)

                # Skip if no completed steps
                if not completed_steps:
                    continue

                total_ideas += 1

                # Get steps in order of completion
                steps_in_sequence = []

                # Get all steps for this idea
                idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

                # Filter to framework steps with creation dates
                framework_idea_steps = [
                    step
                    for step in idea_steps
                    if step.framework == framework.value
                    and step.step
                    and step.get_creation_date()
                ]

                # Sort by creation date
                sorted_steps = sorted(
                    framework_idea_steps, key=lambda s: s.get_creation_date()
                )

                # Extract step names
                for step in sorted_steps:
                    if step.step in framework_steps:
                        steps_in_sequence.append(step.step)

                # Determine if progression is linear
                if self._is_linear_progression(steps_in_sequence, framework_steps):
                    linear_progression_count += 1
                else:
                    non_linear_progression_count += 1

                # Check for collaboration
                contributors = set()
                for step in framework_idea_steps:
                    if step.owner and step.owner in member_emails:
                        contributors.add(step.owner)

                # Categorize by collaboration
                if len(contributors) > 1:
                    # Multiple team members contributed
                    ideas_with_collaboration.append(
                        {
                            "idea_id": idea_id,
                            "steps_completed": len(completed_steps),
                            "steps_in_sequence": steps_in_sequence,
                        }
                    )
                else:
                    # Only one team member contributed
                    ideas_without_collaboration.append(
                        {
                            "idea_id": idea_id,
                            "steps_completed": len(completed_steps),
                            "steps_in_sequence": steps_in_sequence,
                        }
                    )

        # Calculate step completion rates
        if total_ideas > 0:
            progression["step_completion_rates"] = {
                self._get_step_display_name(step, framework): {
                    "count": count,
                    "rate": count / total_ideas,
                }
                for step, count in step_counts.items()
            }

        # Calculate progression patterns
        total_progressions = linear_progression_count + non_linear_progression_count
        if total_progressions > 0:
            progression["progression_patterns"] = {
                "linear": {
                    "count": linear_progression_count,
                    "percentage": linear_progression_count / total_progressions,
                },
                "non_linear": {
                    "count": non_linear_progression_count,
                    "percentage": non_linear_progression_count / total_progressions,
                },
            }

        # Calculate collaboration impact
        if ideas_with_collaboration and ideas_without_collaboration:
            avg_steps_with_collab = sum(
                idea["steps_completed"] for idea in ideas_with_collaboration
            ) / len(ideas_with_collaboration)
            avg_steps_without_collab = sum(
                idea["steps_completed"] for idea in ideas_without_collaboration
            ) / len(ideas_without_collaboration)

            progression["collaboration_impact"] = {
                "ideas_with_collaboration": len(ideas_with_collaboration),
                "ideas_without_collaboration": len(ideas_without_collaboration),
                "avg_steps_with_collaboration": avg_steps_with_collab,
                "avg_steps_without_collaboration": avg_steps_without_collab,
                "step_difference": avg_steps_with_collab - avg_steps_without_collab,
                "percentage_difference": (
                    (avg_steps_with_collab - avg_steps_without_collab)
                    / avg_steps_without_collab
                    * 100
                    if avg_steps_without_collab > 0
                    else 0
                ),
            }

        return progression

    def _analyze_individual_framework_progression(
        self, users: List[Any], framework: FrameworkType, framework_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze how individual users progress through entrepreneurial frameworks.

        Args:
            users: List of individual users to analyze
            framework: The framework to analyze
            framework_steps: List of steps in the framework

        Returns:
            Dict with individual framework progression analysis
        """
        progression = {
            "step_completion_rates": {},
            "progression_patterns": {},
            "engagement_correlation": {},
        }

        # Skip if no users
        if not users:
            return progression

        # Track step completion for all individual ideas
        step_counts = {step: 0 for step in framework_steps}
        total_ideas = 0

        # Track progression patterns
        linear_progression_count = 0
        non_linear_progression_count = 0

        # Track engagement correlation
        engagement_data = {
            UserEngagementLevel.HIGH.value: {"ideas": 0, "total_steps": 0},
            UserEngagementLevel.MEDIUM.value: {"ideas": 0, "total_steps": 0},
            UserEngagementLevel.LOW.value: {"ideas": 0, "total_steps": 0},
        }

        # Process each user
        for user in users:
            if not user.email:
                continue

            # Get user's engagement level
            engagement_level = user.get_engagement_level().value

            # Get user's ideas
            user_ideas = self._data_repo.ideas.find_by_owner(user.email)

            # Process each idea
            for idea in user_ideas:
                if not idea.id:
                    continue

                idea_id = idea.id.oid if hasattr(idea.id, "oid") else str(idea.id)

                # Count completed steps
                completed_steps = []
                for step in framework_steps:
                    if idea.has_step(step):
                        step_counts[step] += 1
                        completed_steps.append(step)

                # Skip if no completed steps
                if not completed_steps:
                    continue

                total_ideas += 1

                # Update engagement data
                engagement_data[engagement_level]["ideas"] += 1
                engagement_data[engagement_level]["total_steps"] += len(completed_steps)

                # Get steps in order of completion
                steps_in_sequence = []

                # Get all steps for this idea
                idea_steps = self._data_repo.steps.find_by_idea_id(idea_id)

                # Filter to framework steps with creation dates
                framework_idea_steps = [
                    step
                    for step in idea_steps
                    if step.framework == framework.value
                    and step.step
                    and step.get_creation_date()
                ]

                # Sort by creation date
                sorted_steps = sorted(
                    framework_idea_steps, key=lambda s: s.get_creation_date()
                )

                # Extract step names
                for step in sorted_steps:
                    if step.step in framework_steps:
                        steps_in_sequence.append(step.step)

                # Determine if progression is linear
                if self._is_linear_progression(steps_in_sequence, framework_steps):
                    linear_progression_count += 1
                else:
                    non_linear_progression_count += 1

        # Calculate step completion rates
        if total_ideas > 0:
            progression["step_completion_rates"] = {
                self._get_step_display_name(step, framework): {
                    "count": count,
                    "rate": count / total_ideas,
                }
                for step, count in step_counts.items()
            }

        # Calculate progression patterns
        total_progressions = linear_progression_count + non_linear_progression_count
        if total_progressions > 0:
            progression["progression_patterns"] = {
                "linear": {
                    "count": linear_progression_count,
                    "percentage": linear_progression_count / total_progressions,
                },
                "non_linear": {
                    "count": non_linear_progression_count,
                    "percentage": non_linear_progression_count / total_progressions,
                },
            }

        # Calculate engagement correlation
        for level, data in engagement_data.items():
            if data["ideas"] > 0:
                data["avg_steps_per_idea"] = data["total_steps"] / data["ideas"]
            else:
                data["avg_steps_per_idea"] = 0

        progression["engagement_correlation"] = engagement_data

        return progression

    def _compare_framework_progression(
        self, team_progression: Dict[str, Any], individual_progression: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare framework progression between teams and individuals.

        Args:
            team_progression: Team progression analysis
            individual_progression: Individual progression analysis

        Returns:
            Dict with comparative progression analysis
        """
        comparison = {
            "step_completion_comparison": {},
            "pattern_comparison": {},
            "overall_differences": {},
        }

        # Compare step completion rates
        team_rates = team_progression.get("step_completion_rates", {})
        individual_rates = individual_progression.get("step_completion_rates", {})

        shared_steps = set(team_rates.keys()).intersection(set(individual_rates.keys()))

        for step in shared_steps:
            team_rate = team_rates[step].get("rate", 0)
            indiv_rate = individual_rates[step].get("rate", 0)

            comparison["step_completion_comparison"][step] = {
                "team_rate": team_rate,
                "individual_rate": indiv_rate,
                "difference": team_rate - indiv_rate,
                "percentage_difference": (
                    (team_rate - indiv_rate) / indiv_rate * 100 if indiv_rate > 0 else 0
                ),
            }

        # Compare progression patterns
        team_patterns = team_progression.get("progression_patterns", {})
        indiv_patterns = individual_progression.get("progression_patterns", {})

        if team_patterns and indiv_patterns:
            team_linear = team_patterns.get("linear", {}).get("percentage", 0)
            indiv_linear = indiv_patterns.get("linear", {}).get("percentage", 0)

            comparison["pattern_comparison"] = {
                "team_linear_percentage": team_linear,
                "individual_linear_percentage": indiv_linear,
                "linear_difference": team_linear - indiv_linear,
            }

        # Calculate overall differences
        # Step count comparison
        team_collab_data = team_progression.get("collaboration_impact", {})
        indiv_engagement_data = individual_progression.get("engagement_correlation", {})

        if team_collab_data and "avg_steps_with_collaboration" in team_collab_data:
            # Compare collaborative teams vs. high engagement individuals
            team_collab_avg = team_collab_data.get("avg_steps_with_collaboration", 0)
            indiv_high_avg = indiv_engagement_data.get(
                UserEngagementLevel.HIGH.value, {}
            ).get("avg_steps_per_idea", 0)

            comparison["overall_differences"][
                "collaborative_team_vs_high_individual"
            ] = {
                "team_avg_steps": team_collab_avg,
                "individual_avg_steps": indiv_high_avg,
                "difference": team_collab_avg - indiv_high_avg,
                "percentage_difference": (
                    (team_collab_avg - indiv_high_avg) / indiv_high_avg * 100
                    if indiv_high_avg > 0
                    else 0
                ),
            }

        return comparison

    def _is_linear_progression(
        self, step_sequence: List[str], framework_steps: List[str]
    ) -> bool:
        """
        Determine if a step sequence follows a linear progression.

        Args:
            step_sequence: Sequence of steps in order of completion
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

        # Check if indices are in ascending order
        return all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))

    def _calculate_correlation(
        self, values1: List[float], values2: List[float]
    ) -> float:
        """
        Calculate Pearson correlation coefficient between two lists of values.

        Args:
            values1: First list of values
            values2: Second list of values

        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        if len(values1) != len(values2) or len(values1) < 2:
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
        return cov / (var1 * var2) ** 0.5

    def _analyze_temporal_collaboration(
        self, team_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in team collaboration.

        Args:
            team_metrics: List of team collaboration metrics

        Returns:
            Dict with temporal collaboration patterns
        """
        patterns = {
            "collaboration_timing": {},
            "activity_windows": {},
        }

        # Skip if no team metrics
        teams_with_temporal = [m for m in team_metrics if "temporal_patterns" in m]
        if not teams_with_temporal:
            return patterns

        # Count synchronous vs. asynchronous collaboration
        sync_count = sum(
            m["temporal_patterns"].get("synchronous_collaboration", 0)
            for m in teams_with_temporal
        )
        async_count = sum(
            m["temporal_patterns"].get("asynchronous_collaboration", 0)
            for m in teams_with_temporal
        )

        total_count = sync_count + async_count
        if total_count > 0:
            patterns["collaboration_timing"] = {
                "synchronous": {
                    "count": sync_count,
                    "percentage": sync_count / total_count,
                },
                "asynchronous": {
                    "count": async_count,
                    "percentage": async_count / total_count,
                },
            }

        # Analyze collaboration windows
        windows = []
        for metrics in teams_with_temporal:
            if (
                "temporal_patterns" in metrics
                and "avg_collaboration_window_hours" in metrics["temporal_patterns"]
            ):
                windows.append(
                    metrics["temporal_patterns"]["avg_collaboration_window_hours"]
                )

        if windows:
            # Calculate window statistics
            avg_window = sum(windows) / len(windows)

            # Group windows by duration
            short_windows = sum(1 for w in windows if w < 1)
            medium_windows = sum(1 for w in windows if 1 <= w < 24)
            long_windows = sum(1 for w in windows if w >= 24)

            patterns["activity_windows"] = {
                "avg_window_hours": avg_window,
                "window_distribution": {
                    "short_term": {
                        "count": short_windows,
                        "percentage": short_windows / len(windows),
                    },
                    "medium_term": {
                        "count": medium_windows,
                        "percentage": medium_windows / len(windows),
                    },
                    "long_term": {
                        "count": long_windows,
                        "percentage": long_windows / len(windows),
                    },
                },
            }

        return patterns

    def _get_step_display_name(self, step_name: str, framework: FrameworkType) -> str:
        """
        Get display name for a framework step.

        Args:
            step_name: Step identifier
            framework: Framework type

        Returns:
            str: Human-readable step name
        """
        if framework == FrameworkType.DISCIPLINED_ENTREPRENEURSHIP:
            try:
                step_enum = DisciplinedEntrepreneurshipStep(step_name)
                step_number = step_enum.step_number
                readable_name = step_name.replace("-", " ").title()
                return f"{step_number}. {readable_name}"
            except ValueError:
                return step_name.replace("-", " ").title()
        else:
            return step_name.replace("-", " ").title()
