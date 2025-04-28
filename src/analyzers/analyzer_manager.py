"""
Analyzer Manager for the JetPack/Orbit Tool data analysis system.

This module provides a centralized management system for analyzer components,
handling initialization, lifecycle, and coordination of analysis operations.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

from data.data_repository import DataRepository
from data.models.enums import FrameworkType, Semester, UserEngagementLevel
from analyzers.engagement_analyzer import EngagementAnalyzer
from analyzers.framework_analyzer import FrameworkAnalyzer
from analyzers.learning_analyzer import LearningAnalyzer
from analyzers.team_analyzer import TeamAnalyzer


class AnalyzerManager:
    """
    Manages the lifecycle and coordination of analyzer components.

    This class centralizes the initialization, configuration, and execution
    of various analyzers, providing a unified interface for performing
    different types of analyses on the JetPack/Orbit tool data.
    """

    def __init__(self, data_repository: DataRepository):
        """
        Initialize the analyzer manager.

        Args:
            data_repository: Data repository for accessing all entities
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._data_repo = data_repository

        # Initialize analyzer instances to None
        self._engagement_analyzer = None
        self._framework_analyzer = None
        self._learning_analyzer = None
        self._team_analyzer = None

        # Cache for analysis results
        self._result_cache = {}
        self._cache_enabled = True

    def initialize_analyzers(self) -> bool:
        """
        Initialize all analyzer instances.

        Returns:
            bool: True if all analyzers were initialized successfully
        """
        try:
            self._logger.info("Initializing analyzers...")

            # Create analyzer instances
            self._engagement_analyzer = EngagementAnalyzer(self._data_repo)
            self._framework_analyzer = FrameworkAnalyzer(self._data_repo)
            self._learning_analyzer = LearningAnalyzer(self._data_repo)
            self._team_analyzer = TeamAnalyzer(self._data_repo)

            self._logger.info("All analyzers initialized successfully")
            return True
        except Exception as e:
            self._logger.error(f"Error initializing analyzers: {e}")
            return False

    def get_engagement_analyzer(self) -> Optional[EngagementAnalyzer]:
        """
        Get the engagement analyzer instance.

        Returns:
            EngagementAnalyzer: The engagement analyzer instance or None if not initialized
        """
        if not self._engagement_analyzer:
            self._logger.warning("Engagement analyzer not initialized")
        return self._engagement_analyzer

    def get_framework_analyzer(self) -> Optional[FrameworkAnalyzer]:
        """
        Get the framework analyzer instance.

        Returns:
            FrameworkAnalyzer: The framework analyzer instance or None if not initialized
        """
        if not self._framework_analyzer:
            self._logger.warning("Framework analyzer not initialized")
        return self._framework_analyzer

    def get_learning_analyzer(self) -> Optional[LearningAnalyzer]:
        """
        Get the learning analyzer instance.

        Returns:
            LearningAnalyzer: The learning analyzer instance or None if not initialized
        """
        if not self._learning_analyzer:
            self._logger.warning("Learning analyzer not initialized")
        return self._learning_analyzer

    def get_team_analyzer(self) -> Optional[TeamAnalyzer]:
        """
        Get the team analyzer instance.

        Returns:
            TeamAnalyzer: The team analyzer instance or None if not initialized
        """
        if not self._team_analyzer:
            self._logger.warning("Team analyzer not initialized")
        return self._team_analyzer

    def get_all_analyzers(self) -> Tuple:
        """
        Get all analyzer instances.

        Returns:
            Tuple: (engagement_analyzer, framework_analyzer, learning_analyzer, team_analyzer)
        """
        return (
            self._engagement_analyzer,
            self._framework_analyzer,
            self._learning_analyzer,
            self._team_analyzer,
        )

    def set_course_id(self, course_id: str) -> None:
        """
        Set the course ID for all relevant analyzers.

        Args:
            course_id: Course identifier (e.g., "15.390")
        """
        if self._learning_analyzer:
            self._learning_analyzer.set_course_id(course_id)
            self._logger.debug(f"Course ID set to {course_id} for learning analyzer")

    def enable_cache(self, enabled: bool = True) -> None:
        """
        Enable or disable caching of analysis results.

        Args:
            enabled: Whether caching should be enabled
        """
        self._cache_enabled = enabled
        self._logger.info(
            f"Analysis result caching {'enabled' if enabled else 'disabled'}"
        )

        if not enabled:
            # Clear cache if disabling
            self._result_cache = {}
            self._logger.debug("Analysis result cache cleared")

    def clear_cache(self) -> None:
        """Clear the analysis result cache."""
        self._result_cache = {}
        self._logger.debug("Analysis result cache cleared")

    def _get_cache_key(self, analysis_type: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for an analysis request.

        Args:
            analysis_type: Type of analysis
            params: Analysis parameters

        Returns:
            str: Cache key
        """
        # Convert params to a stable string representation
        param_str = json.dumps(params, sort_keys=True, default=str)
        return f"{analysis_type}:{param_str}"

    def _cache_result(
        self, analysis_type: str, params: Dict[str, Any], result: Any
    ) -> None:
        """
        Cache an analysis result.

        Args:
            analysis_type: Type of analysis
            params: Analysis parameters
            result: Analysis result
        """
        if not self._cache_enabled:
            return

        key = self._get_cache_key(analysis_type, params)
        self._result_cache[key] = result
        self._logger.debug(f"Cached result for {analysis_type}")

    def _get_cached_result(
        self, analysis_type: str, params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Get a cached analysis result if available.

        Args:
            analysis_type: Type of analysis
            params: Analysis parameters

        Returns:
            Any: Cached result or None if not found
        """
        if not self._cache_enabled:
            return None

        key = self._get_cache_key(analysis_type, params)
        result = self._result_cache.get(key)

        if result is not None:
            self._logger.debug(f"Using cached result for {analysis_type}")

        return result

    def analyze_user_engagement(
        self,
        course_id: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[UserEngagementLevel, List[Dict[str, Any]]]:
        """
        Analyze user engagement levels.

        Args:
            course_id: Optional course ID to filter users
            custom_thresholds: Optional custom thresholds for engagement classification

        Returns:
            Dict mapping engagement levels to lists of user data
        """
        # Check if analyzer is initialized
        if not self._engagement_analyzer:
            self._logger.error("Engagement analyzer not initialized")
            return {}

        # Check cache
        params = {"course_id": course_id, "custom_thresholds": custom_thresholds}
        cached_result = self._get_cached_result("user_engagement", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing user engagement...")
            result = self._engagement_analyzer.classify_users_by_engagement(
                course_id=course_id, custom_thresholds=custom_thresholds
            )

            # Cache result
            self._cache_result("user_engagement", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing user engagement: {e}")
            return {}

    def analyze_framework_completion(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze framework step completion metrics.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            include_ideas_without_steps: Whether to include ideas with no steps

        Returns:
            Dict with framework completion metrics
        """
        # Check if analyzer is initialized
        if not self._framework_analyzer:
            self._logger.error("Framework analyzer not initialized")
            return {}

        # Check cache
        params = {
            "framework": framework.value,
            "course_id": course_id,
            "include_ideas_without_steps": include_ideas_without_steps,
        }
        cached_result = self._get_cached_result("framework_completion", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info(f"Analyzing {framework.value} framework completion...")
            result = self._framework_analyzer.get_framework_completion_metrics(
                framework=framework,
                course_id=course_id,
                include_ideas_without_steps=include_ideas_without_steps,
            )

            # Cache result
            self._cache_result("framework_completion", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing framework completion: {e}")
            return {}

    def analyze_tool_impact_on_learning(
        self,
        pre_tool_semester: Union[str, Semester] = Semester.FALL_2023,
        post_tool_semesters: Optional[List[Union[str, Semester]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the impact of the tool on learning outcomes.

        Args:
            pre_tool_semester: Semester before tool introduction
            post_tool_semesters: List of semesters after tool introduction

        Returns:
            Dict with learning outcome comparison results
        """
        # Check if analyzer is initialized
        if not self._learning_analyzer:
            self._logger.error("Learning analyzer not initialized")
            return {}

        # Check cache
        params = {
            "pre_tool_semester": (
                pre_tool_semester.value
                if isinstance(pre_tool_semester, Semester)
                else pre_tool_semester
            ),
            "post_tool_semesters": [
                s.value if isinstance(s, Semester) else s
                for s in (post_tool_semesters or [])
            ],
        }
        cached_result = self._get_cached_result("learning_impact", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing tool impact on learning outcomes...")
            result = self._learning_analyzer.compare_learning_outcomes_by_cohort(
                pre_tool_semester=pre_tool_semester,
                post_tool_semesters=post_tool_semesters,
            )

            # Cache result
            self._cache_result("learning_impact", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing tool impact on learning: {e}")
            return {}

    def analyze_team_collaboration(
        self,
        team_id: Optional[int] = None,
        course_id: Optional[str] = None,
        include_temporal_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze team collaboration patterns.

        Args:
            team_id: Optional specific team ID to analyze
            course_id: Optional course ID to filter teams
            include_temporal_analysis: Whether to include time-based analysis

        Returns:
            Dict with team collaboration pattern analysis
        """
        # Check if analyzer is initialized
        if not self._team_analyzer:
            self._logger.error("Team analyzer not initialized")
            return {}

        # Check cache
        params = {
            "team_id": team_id,
            "course_id": course_id,
            "include_temporal_analysis": include_temporal_analysis,
        }
        cached_result = self._get_cached_result("team_collaboration", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info(
                f"Analyzing team collaboration {'for team ' + str(team_id) if team_id else ''}..."
            )
            result = self._team_analyzer.analyze_team_collaboration_patterns(
                team_id=team_id,
                course_id=course_id,
                include_temporal_analysis=include_temporal_analysis,
            )

            # Cache result
            self._cache_result("team_collaboration", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing team collaboration: {e}")
            return {}

    def analyze_dropout_patterns(
        self, course_id: Optional[str] = None, inactivity_threshold: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze patterns of user dropout from the tool.

        Args:
            course_id: Optional course ID to filter users
            inactivity_threshold: Days of inactivity to consider as dropout

        Returns:
            Dict with dropout analysis results
        """
        # Check if analyzer is initialized
        if not self._engagement_analyzer:
            self._logger.error("Engagement analyzer not initialized")
            return {}

        # Check cache
        params = {"course_id": course_id, "inactivity_threshold": inactivity_threshold}
        cached_result = self._get_cached_result("dropout_patterns", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing user dropout patterns...")
            result = self._engagement_analyzer.analyze_dropout_patterns(
                course_id=course_id, inactivity_threshold=inactivity_threshold
            )

            # Cache result
            self._cache_result("dropout_patterns", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing dropout patterns: {e}")
            return {}

    def analyze_framework_progression(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_steps: int = 3,
        max_patterns: int = 5,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze common progression patterns through framework steps.

        Args:
            framework: The framework to analyze
            min_steps: Minimum number of steps to consider a pattern
            max_patterns: Maximum number of patterns to return
            course_id: Optional course ID to filter users

        Returns:
            Dict with progression pattern analysis
        """
        # Check if analyzer is initialized
        if not self._framework_analyzer:
            self._logger.error("Framework analyzer not initialized")
            return {}

        # Check cache
        params = {
            "framework": framework.value,
            "min_steps": min_steps,
            "max_patterns": max_patterns,
            "course_id": course_id,
        }
        cached_result = self._get_cached_result("framework_progression", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info(f"Analyzing {framework.value} progression patterns...")
            result = self._framework_analyzer.identify_common_progression_patterns(
                framework=framework,
                min_steps=min_steps,
                max_patterns=max_patterns,
                course_id=course_id,
            )

            # Cache result
            self._cache_result("framework_progression", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing framework progression: {e}")
            return {}

    def analyze_tool_version_impact(
        self, course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the impact of different tool versions on learning outcomes.

        Args:
            course_id: Optional course ID to filter data

        Returns:
            Dict with tool version impact analysis
        """
        # Check if analyzer is initialized
        if not self._learning_analyzer:
            self._logger.error("Learning analyzer not initialized")
            return {}

        # Check cache
        params = {"course_id": course_id}
        cached_result = self._get_cached_result("tool_version_impact", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing tool version impact...")
            result = self._learning_analyzer.analyze_tool_version_impact()

            # Cache result
            self._cache_result("tool_version_impact", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing tool version impact: {e}")
            return {}

    def analyze_user_activity(
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
        # Check if analyzer is initialized
        if not self._engagement_analyzer:
            self._logger.error("Engagement analyzer not initialized")
            return {}

        # Check cache
        params = {
            "email": email,
            "include_ideas": include_ideas,
            "include_steps": include_steps,
        }
        cached_result = self._get_cached_result("user_activity", params)
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info(f"Analyzing activity for user {email}...")
            result = self._engagement_analyzer.get_user_activity_timeline(
                email=email, include_ideas=include_ideas, include_steps=include_steps
            )

            # Cache result
            self._cache_result("user_activity", params, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing user activity: {e}")
            return {}

    def analyze_demographic_learning_impact(self) -> Dict[str, Any]:
        """
        Analyze how the tool impacts learning outcomes across different demographics.

        Returns:
            Dict with demographic impact analysis
        """
        # Check if analyzer is initialized
        if not self._learning_analyzer:
            self._logger.error("Learning analyzer not initialized")
            return {}

        # Check cache
        cached_result = self._get_cached_result("demographic_learning", {})
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing demographic learning impact...")
            result = self._learning_analyzer.analyze_demographic_learning_impact()

            # Cache result
            self._cache_result("demographic_learning", {}, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing demographic learning impact: {e}")
            return {}

    def analyze_combined_learning_metrics(self) -> Dict[str, Any]:
        """
        Comprehensive analysis combining multiple learning metrics.

        Returns:
            Dict with combined learning analysis results
        """
        # Check if analyzer is initialized
        if not self._learning_analyzer:
            self._logger.error("Learning analyzer not initialized")
            return {}

        # Check cache
        cached_result = self._get_cached_result("combined_learning", {})
        if cached_result is not None:
            return cached_result

        # Perform analysis
        try:
            self._logger.info("Analyzing combined learning metrics...")
            result = self._learning_analyzer.analyze_combined_learning_metrics()

            # Cache result
            self._cache_result("combined_learning", {}, result)

            return result
        except Exception as e:
            self._logger.error(f"Error analyzing combined learning metrics: {e}")
            return {}

    def run_specific_analysis(
        self, analysis_type: str, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Run a specific analysis and save results to file.

        Args:
            analysis_type: Type of analysis to run
            output_dir: Directory to save results
            **kwargs: Analysis-specific parameters

        Returns:
            Dict with analysis results and output file info
        """
        result = {}
        output_file = None

        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Run specific analysis based on type
            if analysis_type == "user":
                email = kwargs.get("user_email")
                if not email:
                    return {"error": "User email is required for user analysis"}

                result = self.analyze_user_activity(
                    email=email,
                    include_ideas=kwargs.get("include_ideas", True),
                    include_steps=kwargs.get("include_steps", True),
                )

                output_file = (
                    Path(output_dir)
                    / f"user_analysis_{email.split('@')[0]}_{timestamp}.json"
                )

            elif analysis_type == "team":
                team_id = kwargs.get("team_id")
                if team_id is None:
                    return {"error": "Team ID is required for team analysis"}

                result = self.analyze_team_collaboration(
                    team_id=team_id,
                    course_id=kwargs.get("course_id"),
                    include_temporal_analysis=kwargs.get("include_temporal", True),
                )

                output_file = (
                    Path(output_dir) / f"team_analysis_{team_id}_{timestamp}.json"
                )

            elif analysis_type == "semester":
                semester = kwargs.get("semester")
                if not semester:
                    return {"error": "Semester is required for semester analysis"}

                comparison_semester = kwargs.get("comparison_semester")

                # Get semester comparison data from data repository
                result = self._data_repo.get_semester_comparison(
                    semester, comparison_semester
                )

                output_file = (
                    Path(output_dir)
                    / f"semester_comparison_{semester.replace(' ', '_')}_{timestamp}.json"
                )

            elif analysis_type == "framework":
                framework = kwargs.get(
                    "framework", FrameworkType.DISCIPLINED_ENTREPRENEURSHIP
                )

                result = self.analyze_framework_completion(
                    framework=framework,
                    course_id=kwargs.get("course_id"),
                    include_ideas_without_steps=kwargs.get(
                        "include_ideas_without_steps", False
                    ),
                )

                output_file = (
                    Path(output_dir)
                    / f"framework_analysis_{framework.value.replace(' ', '_')}_{timestamp}.json"
                )

            elif analysis_type == "learning":
                result = self.analyze_combined_learning_metrics()

                output_file = Path(output_dir) / f"learning_analysis_{timestamp}.json"

            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

            # Save results to file
            if output_file and result:
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                self._logger.info(f"Analysis results saved to {output_file}")

                return {"result": result, "output_file": str(output_file)}
            else:
                return {"error": "Failed to generate or save analysis results"}

        except Exception as e:
            self._logger.error(f"Error running {analysis_type} analysis: {e}")
            return {"error": str(e)}

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all data in the repository.

        Returns:
            Dict with data summary
        """
        try:
            return self._data_repo.get_data_summary()
        except Exception as e:
            self._logger.error(f"Error getting data summary: {e}")
            return {"error": str(e)}

    def save_data_summary(self, output_dir: str) -> Optional[str]:
        """
        Generate and save a data summary.

        Args:
            output_dir: Output directory

        Returns:
            str: Path to the saved summary file or None if failed
        """
        try:
            # Get data summary
            summary = self._data_repo.get_data_summary()

            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(output_dir) / f"data_summary_{timestamp}.json"

            # Save summary to file
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            self._logger.info(f"Data summary saved to {output_file}")

            return str(output_file)
        except Exception as e:
            self._logger.error(f"Error saving data summary: {e}")
            return None
