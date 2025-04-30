"""
Visualization Manager for Data Analysis System.

This module provides a central manager for all visualization capabilities,
coordinating between different visualizer modules and providing a unified
interface for generating visualizations and reports.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from config.settings import Settings
from src.data.models import FrameworkType

from src.visualizers import (
    EngagementVisualizer,
    FrameworkVisualizer,
    LearningVisualizer,
    TeamVisualizer,
)
from src.analyzers import (
    EngagementAnalyzer,
    FrameworkAnalyzer,
    LearningAnalyzer,
    TeamAnalyzer,
)


class VisualizationManager:
    """
    Central manager for all visualization capabilities.

    This class coordinates between the different visualizer modules,
    providing a unified interface for generating visualizations and reports.
    It manages output directories, visualization themes, and report generation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        output_dir: Optional[str] = None,
        include_timestamps: bool = True,
        default_figsize: Tuple[float, float] = (10, 6),
        theme: str = "default",
        save_formats: List[str] = ["png", "pdf"],
    ):
        """
        Initialize the visualization manager.

        Args:
            settings: Optional application settings
            output_dir: Optional custom output directory
            include_timestamps: Whether to include timestamps in filenames
            default_figsize: Default figure size (width, height) in inches
            theme: Visual theme ('default', 'dark', 'print')
            save_formats: Default formats for saving visualizations
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        # Store configuration
        self._settings = settings or Settings()
        self._include_timestamps = include_timestamps
        self._default_figsize = default_figsize
        self._theme = theme
        self._save_formats = save_formats

        # Set output directory
        if output_dir:
            self._output_dir = Path(output_dir)
        else:
            # Use default project output directory
            self._output_dir = self._settings.OUTPUT_DIR / "visualizations"

        # Create output directory if it doesn't exist
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each visualization type
        self._engagement_dir = self._output_dir / "engagement"
        self._framework_dir = self._output_dir / "framework"
        self._learning_dir = self._output_dir / "learning"
        self._team_dir = self._output_dir / "team"
        self._reports_dir = self._output_dir / "reports"

        for directory in [
            self._engagement_dir,
            self._framework_dir,
            self._learning_dir,
            self._team_dir,
            self._reports_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize analyzer and visualizer instances (placeholders - will be set by set_analyzers)
        self._engagement_analyzer = None
        self._framework_analyzer = None
        self._learning_analyzer = None
        self._team_analyzer = None

        self._engagement_visualizer = None
        self._framework_visualizer = None
        self._learning_visualizer = None
        self._team_visualizer = None

    def set_analyzers(
        self,
        engagement_analyzer: Optional[EngagementAnalyzer] = None,
        framework_analyzer: Optional[FrameworkAnalyzer] = None,
        learning_analyzer: Optional[LearningAnalyzer] = None,
        team_analyzer: Optional[TeamAnalyzer] = None,
    ) -> None:
        """
        Set analyzer instances and initialize corresponding visualizers.

        Args:
            engagement_analyzer: EngagementAnalyzer instance
            framework_analyzer: FrameworkAnalyzer instance
            learning_analyzer: LearningAnalyzer instance
            team_analyzer: TeamAnalyzer instance
        """
        # Store analyzer instances
        self._engagement_analyzer = engagement_analyzer
        self._framework_analyzer = framework_analyzer
        self._learning_analyzer = learning_analyzer
        self._team_analyzer = team_analyzer

        # Initialize visualizers with their corresponding analyzers
        if engagement_analyzer:
            self._engagement_visualizer = EngagementVisualizer(
                engagement_analyzer=engagement_analyzer,
                output_dir=str(self._engagement_dir),
                theme=self._theme,
                default_figsize=self._default_figsize,
                include_timestamps=self._include_timestamps,
            )

        if framework_analyzer:
            self._framework_visualizer = FrameworkVisualizer(
                framework_analyzer=framework_analyzer,
                output_dir=str(self._framework_dir),
                include_timestamps=self._include_timestamps,
                default_figsize=self._default_figsize,
                theme=self._theme,
            )

        if learning_analyzer:
            self._learning_visualizer = LearningVisualizer(
                output_dir=str(self._learning_dir),
            )

        if team_analyzer:
            self._team_visualizer = TeamVisualizer(
                team_analyzer=team_analyzer,
            )

    # Engagement Visualization Methods

    def visualize_engagement_levels(
        self,
        course_id: Optional[str] = None,
        include_demographics: bool = True,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of user engagement levels.

        Args:
            course_id: Optional course ID to filter users
            include_demographics: Whether to include demographic breakdown
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        if not self._engagement_visualizer:
            raise ValueError(
                "EngagementVisualizer not initialized. Call set_analyzers() first."
            )

        return self._engagement_visualizer.create_engagement_level_visualization(
            course_id=course_id,
            include_demographics=include_demographics,
            save_path=save_path or str(self._engagement_dir / "engagement_levels"),
            show_fig=show_fig,
        )

    def visualize_engagement_metrics(
        self,
        metrics: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of engagement metrics.

        Args:
            metrics: Engagement metrics data (if None, fetched from analyzer)
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        if not self._engagement_visualizer:
            raise ValueError(
                "EngagementVisualizer not initialized. Call set_analyzers() first."
            )

        return self._engagement_visualizer.create_engagement_metrics_visualization(
            metrics=metrics,
            save_path=save_path or str(self._engagement_dir / "engagement_metrics"),
            show_fig=show_fig,
        )

    def visualize_dropout_analysis(
        self,
        dropout_data: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of user dropout patterns.

        Args:
            dropout_data: Dropout analysis data (if None, fetched from analyzer)
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        if not self._engagement_visualizer:
            raise ValueError(
                "EngagementVisualizer not initialized. Call set_analyzers() first."
            )

        return self._engagement_visualizer.create_dropout_analysis_visualization(
            dropout_data=dropout_data,
            save_path=save_path or str(self._engagement_dir / "dropout_analysis"),
            show_fig=show_fig,
        )

    def create_engagement_report(
        self,
        course_id: Optional[str] = None,
        include_user_details: bool = False,
        output_dir: Optional[str] = None,
        report_name: str = "Engagement_Analysis_Report",
    ) -> str:
        """
        Create a comprehensive engagement analysis report with multiple visualizations.

        Args:
            course_id: Optional course ID to filter data
            include_user_details: Whether to include detailed user information
            output_dir: Output directory for the report
            report_name: Name of the report directory

        Returns:
            str: Path to the generated report directory
        """
        if not self._engagement_visualizer:
            raise ValueError(
                "EngagementVisualizer not initialized. Call set_analyzers() first."
            )

        return self._engagement_visualizer.create_engagement_report(
            course_id=course_id,
            include_user_details=include_user_details,
            output_dir=output_dir or str(self._reports_dir / "engagement"),
            report_name=report_name,
        )

    # Framework Visualization Methods

    def visualize_framework_completion(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
        save_formats: Optional[List[str]] = None,
        return_figures: bool = False,
    ) -> Optional[Dict[str, plt.Figure]]:
        """
        Create visualizations for framework completion metrics.

        Args:
            framework: The framework to visualize
            course_id: Optional course ID to filter users
            include_ideas_without_steps: Whether to include ideas without steps
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        if not self._framework_visualizer:
            raise ValueError(
                "FrameworkVisualizer not initialized. Call set_analyzers() first."
            )

        return self._framework_visualizer.visualize_framework_completion(
            framework=framework,
            course_id=course_id,
            include_ideas_without_steps=include_ideas_without_steps,
            filename_prefix=None,  # Use default
            save_formats=save_formats or self._save_formats,
            return_figures=return_figures,
        )

    def visualize_progression_patterns(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_steps: int = 3,
        max_patterns: int = 5,
        course_id: Optional[str] = None,
        save_formats: Optional[List[str]] = None,
        return_figures: bool = False,
    ) -> Optional[Dict[str, plt.Figure]]:
        """
        Create visualizations for framework progression patterns.

        Args:
            framework: The framework to visualize
            min_steps: Minimum number of steps to consider a pattern
            max_patterns: Maximum number of patterns to return
            course_id: Optional course ID to filter users
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        if not self._framework_visualizer:
            raise ValueError(
                "FrameworkVisualizer not initialized. Call set_analyzers() first."
            )

        return self._framework_visualizer.visualize_progression_patterns(
            framework=framework,
            min_steps=min_steps,
            max_patterns=max_patterns,
            course_id=course_id,
            filename_prefix=None,  # Use default
            save_formats=save_formats or self._save_formats,
            return_figures=return_figures,
        )

    def create_framework_report(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
        include_category_analysis: bool = True,
        save_formats: Optional[List[str]] = None,
    ) -> str:
        """
        Create a comprehensive report of framework analysis visualizations.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            include_ideas_without_steps: Whether to include ideas without steps
            include_category_analysis: Whether to include analysis by idea category
            save_formats: List of formats to save visualizations in

        Returns:
            str: Path to the generated report directory
        """
        if not self._framework_visualizer:
            raise ValueError(
                "FrameworkVisualizer not initialized. Call set_analyzers() first."
            )

        return self._framework_visualizer.create_framework_report(
            framework=framework,
            course_id=course_id,
            include_ideas_without_steps=include_ideas_without_steps,
            include_category_analysis=include_category_analysis,
            output_dir=str(self._reports_dir / "framework"),
            save_formats=save_formats or self._save_formats,
        )

    # Learning Visualization Methods

    def visualize_course_rating_engagement_correlation(
        self,
        correlation_data: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Visualize correlation between tool usage and course ratings.

        Args:
            correlation_data: Data from LearningAnalyzer or None to fetch from analyzer
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        if not self._learning_visualizer:
            raise ValueError(
                "LearningVisualizer not initialized. Call set_analyzers() first."
            )

        # Get data from analyzer if not provided
        if correlation_data is None and self._learning_analyzer:
            correlation_data = (
                self._learning_analyzer.correlate_tool_usage_with_course_ratings()
            )

        return self._learning_visualizer.visualize_course_rating_engagement_correlation(
            correlation_data=correlation_data,
            save_path=save_path or str(self._learning_dir),
        )

    def visualize_learning_outcomes_by_cohort(
        self,
        outcomes_data: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Visualize comparison of learning outcomes between pre-tool and post-tool cohorts.

        Args:
            outcomes_data: Data from LearningAnalyzer or None to fetch from analyzer
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        if not self._learning_visualizer:
            raise ValueError(
                "LearningVisualizer not initialized. Call set_analyzers() first."
            )

        # Get data from analyzer if not provided
        if outcomes_data is None and self._learning_analyzer:
            outcomes_data = (
                self._learning_analyzer.compare_learning_outcomes_by_cohort()
            )

        return self._learning_visualizer.visualize_learning_outcomes_by_cohort(
            outcomes_data=outcomes_data,
            save_path=save_path or str(self._learning_dir),
        )

    def create_learning_report(self) -> Dict[str, Any]:
        """
        Create a comprehensive visualization report for all learning analyzer metrics.

        Returns:
            Dict[str, Any]: Report metadata and paths
        """
        if not self._learning_visualizer or not self._learning_analyzer:
            raise ValueError(
                "LearningVisualizer and LearningAnalyzer not initialized. Call set_analyzers() first."
            )

        return self._learning_visualizer.create_comprehensive_report(
            learning_analyzer=self._learning_analyzer,
            output_dir=str(self._reports_dir / "learning"),
        )

    # Team Visualization Methods

    def visualize_team_vs_individual_engagement(
        self,
        course_id: Optional[str] = None,
        include_demographic_breakdown: bool = True,
        save_formats: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create visualizations comparing team and individual engagement.

        Args:
            course_id: Optional course ID to filter users
            include_demographic_breakdown: Whether to include demographic analysis
            save_formats: List of formats for saving

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        if not self._team_visualizer:
            raise ValueError(
                "TeamVisualizer not initialized. Call set_analyzers() first."
            )

        output_filename = f"team_vs_individual_engagement"
        if course_id:
            output_filename += f"_{course_id}"

        return self._team_visualizer.visualize_team_vs_individual_engagement(
            course_id=course_id,
            include_demographic_breakdown=include_demographic_breakdown,
            output_filename=output_filename,
            save_formats=save_formats or self._save_formats,
            show_figure=False,  # Don't automatically show in manager context
        )

    def visualize_team_collaboration_patterns(
        self,
        team_id: Optional[int] = None,
        course_id: Optional[str] = None,
        include_temporal_analysis: bool = True,
        save_formats: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create visualizations of team collaboration patterns.

        Args:
            team_id: Optional specific team ID to analyze
            course_id: Optional course ID to filter teams
            include_temporal_analysis: Whether to include time-based analysis
            save_formats: List of formats for saving

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        if not self._team_visualizer:
            raise ValueError(
                "TeamVisualizer not initialized. Call set_analyzers() first."
            )

        output_filename = f"team_collaboration_patterns"
        if team_id:
            output_filename += f"_team_{team_id}"
        elif course_id:
            output_filename += f"_{course_id}"

        return self._team_visualizer.visualize_team_collaboration_patterns(
            team_id=team_id,
            course_id=course_id,
            include_temporal_analysis=include_temporal_analysis,
            output_filename=output_filename,
            save_formats=save_formats or self._save_formats,
            show_figure=False,  # Don't automatically show in manager context
        )

    def create_team_report(
        self,
        course_id: Optional[str] = None,
        team_id: Optional[int] = None,
        include_data_tables: bool = True,
        report_name: str = "comprehensive_team_analysis",
    ) -> Dict[str, str]:
        """
        Create a comprehensive report with all team visualizations.

        Args:
            course_id: Optional course ID to filter data
            team_id: Optional team ID for specific team analysis
            include_data_tables: Whether to include data tables with the visualizations
            report_name: Base name for the report files

        Returns:
            Dict[str, str]: Dictionary of generated file paths
        """
        if not self._team_visualizer:
            raise ValueError(
                "TeamVisualizer not initialized. Call set_analyzers() first."
            )

        return self._team_visualizer.create_comprehensive_team_report(
            course_id=course_id,
            team_id=team_id,
            output_dir=str(self._reports_dir / "team"),
            include_data_tables=include_data_tables,
            report_name=report_name,
        )

    # Combined Visualization Methods

    def create_comprehensive_report(
        self,
        course_id: Optional[str] = None,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        report_name: str = "comprehensive_analysis_report",
        include_user_details: bool = False,
        include_category_analysis: bool = True,
        save_formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Create a comprehensive report with visualizations from all analyzers.

        Args:
            course_id: Optional course ID to filter data
            framework: The framework to analyze
            report_name: Base name for the report
            include_user_details: Whether to include detailed user information
            include_category_analysis: Whether to include analysis by idea category
            save_formats: List of formats to save visualizations in

        Returns:
            Dict[str, str]: Dictionary of report directories and metadata
        """
        # Create report timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base_dir = self._reports_dir / f"{report_name}_{timestamp}"
        report_base_dir.mkdir(parents=True, exist_ok=True)

        # Create report directories for each analyzer
        engagement_dir = report_base_dir / "engagement"
        framework_dir = report_base_dir / "framework"
        learning_dir = report_base_dir / "learning"
        team_dir = report_base_dir / "team"

        for directory in [engagement_dir, framework_dir, learning_dir, team_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create report for each analyzer (if initialized)
        report_paths = {}

        # Create engagement report
        if self._engagement_visualizer:
            try:
                engagement_report = self.create_engagement_report(
                    course_id=course_id,
                    include_user_details=include_user_details,
                    output_dir=str(engagement_dir),
                    report_name="Engagement_Analysis",
                )
                report_paths["engagement"] = engagement_report
            except Exception as e:
                self._logger.error(f"Error creating engagement report: {e}")

        # Create framework report
        if self._framework_visualizer:
            try:
                framework_report = self.create_framework_report(
                    framework=framework,
                    course_id=course_id,
                    include_category_analysis=include_category_analysis,
                    save_formats=save_formats or self._save_formats,
                )
                report_paths["framework"] = framework_report
            except Exception as e:
                self._logger.error(f"Error creating framework report: {e}")

        # Create learning report
        if self._learning_visualizer and self._learning_analyzer:
            try:
                learning_report = self.create_learning_report()
                report_paths["learning"] = learning_report.get("report_dir", "")
            except Exception as e:
                self._logger.error(f"Error creating learning report: {e}")

        # Create team report
        if self._team_visualizer:
            try:
                team_report = self.create_team_report(
                    course_id=course_id,
                    report_name="Team_Analysis",
                )
                report_paths["team"] = team_report
            except Exception as e:
                self._logger.error(f"Error creating team report: {e}")

        # Create comprehensive report summary
        self._create_comprehensive_report_summary(
            report_base_dir, report_paths, course_id, framework
        )

        return {
            "report_directory": str(report_base_dir),
            "sections": report_paths,
        }

    def visualize_combined_metrics(
        self,
        course_id: Optional[str] = None,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        metrics_to_include: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        save_formats: Optional[List[str]] = None,
        show_fig: bool = False,
    ) -> Dict[str, plt.Figure]:
        """
        Create combined visualizations showing metrics from multiple analyzers.

        Args:
            course_id: Optional course ID to filter data
            framework: The framework to analyze
            metrics_to_include: List of metrics to include (defaults to all)
            save_path: Optional path to save visualizations
            save_formats: List of formats to save visualizations in
            show_fig: Whether to display the figures

        Returns:
            Dict[str, Figure]: Dictionary of created figures
        """
        # If metrics not specified, include all
        if metrics_to_include is None:
            metrics_to_include = [
                "engagement_levels",
                "framework_completion",
                "learning_outcomes",
                "team_collaboration",
            ]

        # Create combined visualizations directory if needed
        combined_dir = self._output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        save_dir = save_path or str(combined_dir)
        figures = {}

        # Only create visualizations for which we have analyzers

        # Create engagement visualization if requested and available
        if "engagement_levels" in metrics_to_include and self._engagement_visualizer:
            try:
                fig, _ = self.visualize_engagement_levels(
                    course_id=course_id,
                    include_demographics=True,
                    save_path=f"{save_dir}/combined_engagement_levels",
                    show_fig=show_fig,
                )
                figures["engagement_levels"] = fig
            except Exception as e:
                self._logger.error(
                    f"Error creating engagement levels visualization: {e}"
                )

        # Create framework completion visualization if requested and available
        if "framework_completion" in metrics_to_include and self._framework_visualizer:
            try:
                framework_figs = self.visualize_framework_completion(
                    framework=framework,
                    course_id=course_id,
                    include_ideas_without_steps=False,
                    save_formats=save_formats or self._save_formats,
                    return_figures=True,
                )

                if framework_figs:
                    figures.update(framework_figs)
            except Exception as e:
                self._logger.error(
                    f"Error creating framework completion visualization: {e}"
                )

        # Create learning outcomes visualization if requested and available
        if (
            "learning_outcomes" in metrics_to_include
            and self._learning_visualizer
            and self._learning_analyzer
        ):
            try:
                outcomes_data = (
                    self._learning_analyzer.compare_learning_outcomes_by_cohort()
                )

                result = self.visualize_learning_outcomes_by_cohort(
                    outcomes_data=outcomes_data,
                    save_path=save_dir,
                )

                if "figures" in result:
                    figures.update(result["figures"])
            except Exception as e:
                self._logger.error(
                    f"Error creating learning outcomes visualization: {e}"
                )

        # Create team collaboration visualization if requested and available
        if "team_collaboration" in metrics_to_include and self._team_visualizer:
            try:
                team_fig = self.visualize_team_vs_individual_engagement(
                    course_id=course_id,
                    include_demographic_breakdown=True,
                    save_formats=save_formats or self._save_formats,
                )

                figures["team_engagement"] = team_fig
            except Exception as e:
                self._logger.error(
                    f"Error creating team collaboration visualization: {e}"
                )

        return figures

    def _create_comprehensive_report_summary(
        self,
        report_dir: Path,
        report_paths: Dict[str, str],
        course_id: Optional[str],
        framework: FrameworkType,
    ) -> Path:
        """
        Create a summary file for the comprehensive report.

        Args:
            report_dir: Directory where the report is saved
            report_paths: Dictionary of report section paths
            course_id: Optional course ID used for filtering
            framework: The framework that was analyzed

        Returns:
            Path: Path to the created summary file
        """
        summary_path = report_dir / "report_summary.md"

        # Create markdown content
        md_content = "# Comprehensive Analysis Report\n\n"

        # Add report timestamp
        md_content += (
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        )

        # Add parameters
        md_content += "## Analysis Parameters\n\n"

        if course_id:
            md_content += f"- **Course ID:** {course_id}\n"

        md_content += f"- **Framework:** {framework.value}\n\n"

        # Add table of contents
        md_content += "## Report Sections\n\n"

        for section, path in report_paths.items():
            if path:
                section_display = section.replace("_", " ").title()
                md_content += f"- [**{section_display}**](./{section}): Analysis of {section_display.lower()} metrics\n"

        md_content += "\n## Key Findings\n\n"
        md_content += "This comprehensive report integrates findings from multiple analysis modules:\n\n"

        # Add section overviews
        if "engagement" in report_paths:
            md_content += "### Engagement Analysis\n\n"
            md_content += "- Visualizes user engagement levels and patterns\n"
            md_content += "- Analyzes dropout rates and contributing factors\n"
            md_content += "- Examines temporal patterns in user interactions\n\n"

        if "framework" in report_paths:
            md_content += f"### {framework.value} Framework Analysis\n\n"
            md_content += "- Visualizes framework step completion rates\n"
            md_content += "- Identifies common progression patterns\n"
            md_content += "- Analyzes step dependencies and bottlenecks\n\n"

        if "learning" in report_paths:
            md_content += "### Learning Outcomes Analysis\n\n"
            md_content += "- Correlates tool usage with course ratings\n"
            md_content += "- Compares learning outcomes across cohorts\n"
            md_content += "- Analyzes impact of different tool versions\n\n"

        if "team" in report_paths:
            md_content += "### Team Dynamics Analysis\n\n"
            md_content += "- Compares team vs. individual engagement\n"
            md_content += "- Analyzes collaboration patterns within teams\n"
            md_content += "- Examines the impact of team composition\n\n"

        # Add usage notes
        md_content += "## Usage Notes\n\n"
        md_content += "To explore each analysis section in detail, navigate to the corresponding subdirectory. "
        md_content += "Each section contains its own summary and visualizations relevant to that particular aspect of the analysis.\n\n"

        # Save markdown file
        with open(summary_path, "w") as f:
            f.write(md_content)

        # Try to generate HTML version using markdown
        try:
            import markdown

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Comprehensive Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                    a {{ color: #0366d6; }}
                </style>
            </head>
            <body>
                {markdown.markdown(md_content)}
            </body>
            </html>
            """

            with open(report_dir / "report_summary.html", "w") as f:
                f.write(html_content)

        except ImportError:
            self._logger.warning(
                "Python-Markdown not installed, skipping HTML report generation"
            )

        return summary_path
