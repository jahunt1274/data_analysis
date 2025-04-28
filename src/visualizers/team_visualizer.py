"""
Team visualization module for the data analysis system.

This module provides visualization capabilities for team analysis results,
generating informative charts and diagrams to help understand team dynamics,
collaboration patterns, and framework progression in the JetPack/Orbit tool.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.analyzers.team_analyzer import TeamAnalyzer
from src.data.models.enums import FrameworkType
from src.utils.visualization_creation_utils import (
    create_figure,
    create_subplot_grid,
    configure_axes,
    plot_bar,
    plot_grouped_bars,
    get_color_palette,
    add_reference_line,
    save_figure,
    add_data_table,
)
from src.utils.visualization_data_utils import generate_filename


class TeamVisualizer:
    """
    Visualizer for team dynamics and collaboration patterns.

    This class provides methods for creating visualizations based on
    the analysis results from the TeamAnalyzer class.
    """

    def __init__(self, team_analyzer: TeamAnalyzer):
        """
        Initialize the team visualizer.

        Args:
            team_analyzer: Team analyzer instance to use for data
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._team_analyzer = team_analyzer
        self._output_dir = "team_visualizations"

        # Set default figure parameters
        self._default_figsize = (10, 6)
        self._default_dpi = 100

        # Set default color palette
        self._categorical_palette = "categorical_main"
        self._engagement_palette = "engagement_levels"

    def visualize_team_vs_individual_engagement(
        self,
        course_id: Optional[str] = None,
        include_demographic_breakdown: bool = True,
        figure_size: Tuple[float, float] = None,
        output_filename: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        show_figure: bool = True,
    ) -> Figure:
        """
        Create visualizations comparing team and individual engagement.

        Args:
            course_id: Optional course ID to filter users
            include_demographic_breakdown: Whether to include demographic analysis
            figure_size: Optional custom figure size
            output_filename: Optional filename for saving visualization
            save_formats: List of formats for saving
            show_figure: Whether to show the figure (set to False for headless environments)

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        # Get analysis data from the team analyzer
        engagement_data = self._team_analyzer.compare_team_vs_individual_engagement(
            course_id=course_id,
            include_demographic_breakdown=include_demographic_breakdown,
        )

        # Use default figure size if not provided
        if figure_size is None:
            figure_size = self._default_figsize

        # Create figure with multiple subplots
        if include_demographic_breakdown and "demographic_breakdown" in engagement_data:
            # More complex figure with demographic breakdown
            fig = create_figure(
                width=figure_size[0], height=figure_size[1] * 1.5, dpi=self._default_dpi
            )
            axes = create_subplot_grid(fig, rows=3, cols=2, height_ratios=[1, 1, 1.5])
        else:
            # Simpler figure without demographic breakdown
            fig = create_figure(
                width=figure_size[0], height=figure_size[1], dpi=self._default_dpi
            )
            axes = create_subplot_grid(fig, rows=2, cols=2)

        # Extract metrics for team members and individuals
        team_metrics = engagement_data.get("team_metrics", {})
        individual_metrics = engagement_data.get("individual_metrics", {})
        comparison = engagement_data.get("comparison", {})

        # Plot 1: Idea metrics comparison
        self._create_engagement_metrics_comparison(
            axes[0],
            team_metrics.get("idea_metrics", {}),
            individual_metrics.get("idea_metrics", {}),
            "Idea Metrics Comparison",
        )

        # Plot 2: Step metrics comparison
        self._create_engagement_metrics_comparison(
            axes[1],
            team_metrics.get("step_metrics", {}),
            individual_metrics.get("step_metrics", {}),
            "Step Metrics Comparison",
        )

        # Plot 3: Engagement distribution comparison
        self._create_engagement_distribution_comparison(
            axes[2],
            team_metrics.get("engagement_distribution", {}),
            individual_metrics.get("engagement_distribution", {}),
            "Engagement Level Distribution",
        )

        # Plot 4: Percentage difference in key metrics
        self._create_percentage_difference_chart(
            axes[3], comparison, "Percentage Difference (Team vs Individual)"
        )

        # Add demographic breakdown if requested and available
        if include_demographic_breakdown and "demographic_breakdown" in engagement_data:
            # Plot 5-6: Demographic breakdowns
            demographic_data = engagement_data.get("demographic_breakdown", {})
            self._create_demographic_breakdown_visualization(
                axes[4:], demographic_data, "Engagement by Demographics"
            )

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if output_filename:
            filename = generate_filename(output_filename)
            save_figure(
                fig, filename, directory=self._output_dir, formats=save_formats, dpi=300
            )

        # Show figure if requested
        if show_figure:
            plt.show()

        return fig

    def visualize_team_collaboration_patterns(
        self,
        team_id: Optional[int] = None,
        course_id: Optional[str] = None,
        include_temporal_analysis: bool = True,
        figure_size: Tuple[float, float] = None,
        output_filename: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        show_figure: bool = True,
    ) -> Figure:
        """
        Create visualizations of team collaboration patterns.

        Args:
            team_id: Optional specific team ID to analyze
            course_id: Optional course ID to filter teams
            include_temporal_analysis: Whether to include time-based analysis
            figure_size: Optional custom figure size
            output_filename: Optional filename for saving visualization
            save_formats: List of formats for saving
            show_figure: Whether to show the figure

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        # Get analysis data from the team analyzer
        collaboration_data = self._team_analyzer.analyze_team_collaboration_patterns(
            team_id=team_id,
            course_id=course_id,
            include_temporal_analysis=include_temporal_analysis,
        )

        # Use default figure size if not provided
        if figure_size is None:
            figure_size = self._default_figsize

        # Determine number of subplots based on data and options
        num_plots = 4  # Base number of plots
        if include_temporal_analysis and "temporal_patterns" in collaboration_data:
            num_plots += 2

        # Create figure with appropriate subplots
        fig = create_figure(
            width=figure_size[0],
            height=figure_size[1] * (num_plots / 2),
            dpi=self._default_dpi,
        )

        # Create subplot grid with appropriate layout
        rows = num_plots // 2 + (1 if num_plots % 2 else 0)
        cols = 2
        axes = create_subplot_grid(fig, rows=rows, cols=cols)

        # Extract metrics from data
        overview = collaboration_data.get("overview", {})
        teams_analyzed = collaboration_data.get("teams_analyzed", [])
        collaboration_metrics = collaboration_data.get("collaboration_metrics", {})
        idea_sharing_patterns = collaboration_data.get("idea_sharing_patterns", {})

        # Plot 1: Collaboration level distribution
        if "collaboration_levels" in collaboration_metrics:
            self._create_collaboration_level_chart(
                axes[0],
                collaboration_metrics["collaboration_levels"],
                "Collaboration Level Distribution",
            )

        # Plot 2: Member involvement metrics
        if "member_involvement" in collaboration_metrics:
            self._create_member_involvement_chart(
                axes[1],
                collaboration_metrics["member_involvement"],
                "Team Member Involvement",
            )

        # Plot 3: Collaboration metrics by team
        self._create_team_collaboration_chart(
            axes[2], teams_analyzed, "Collaboration Metrics by Team"
        )

        # Plot 4: Idea sharing patterns
        self._create_idea_sharing_patterns_chart(
            axes[3], idea_sharing_patterns, "Idea Sharing Patterns"
        )

        # Plot 5-6: Temporal patterns if available
        if include_temporal_analysis and "temporal_patterns" in collaboration_data:
            temporal_data = collaboration_data.get("temporal_patterns", {})
            self._create_temporal_collaboration_charts(
                axes[4:6], temporal_data, "Temporal Collaboration Patterns"
            )

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if output_filename:
            filename = generate_filename(output_filename)
            save_figure(
                fig, filename, directory=self._output_dir, formats=save_formats, dpi=300
            )

        # Show figure if requested
        if show_figure:
            plt.show()

        return fig

    def visualize_team_usage_distribution(
        self,
        course_id: Optional[str] = None,
        min_team_size: int = 2,
        figure_size: Tuple[float, float] = None,
        output_filename: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        show_figure: bool = True,
    ) -> Figure:
        """
        Create visualizations of team usage distribution.

        Args:
            course_id: Optional course ID to filter teams
            min_team_size: Minimum number of members for a team to be included
            figure_size: Optional custom figure size
            output_filename: Optional filename for saving visualization
            save_formats: List of formats for saving
            show_figure: Whether to show the figure

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        # Get analysis data from the team analyzer
        distribution_data = self._team_analyzer.get_team_usage_distribution(
            course_id=course_id, min_team_size=min_team_size
        )

        # Use default figure size if not provided
        if figure_size is None:
            figure_size = self._default_figsize

        # Create figure with subplots
        fig = create_figure(
            width=figure_size[0], height=figure_size[1] * 1.5, dpi=self._default_dpi
        )
        axes = create_subplot_grid(fig, rows=3, cols=2)

        # Extract data
        teams_analyzed = distribution_data.get("teams_analyzed", 0)
        distribution_metrics = distribution_data.get("distribution_metrics", {})
        usage_concentration = distribution_data.get("usage_concentration", {})
        role_patterns = distribution_data.get("role_patterns", {})

        # Plot 1: Distribution type distribution
        self._create_distribution_type_chart(
            axes[0], distribution_metrics, "Team Usage Distribution Types"
        )

        # Plot 2: Usage concentration metrics
        self._create_usage_concentration_chart(
            axes[1], usage_concentration, "Team Usage Concentration Metrics"
        )

        # Plot 3: Role patterns analysis
        if "observed_patterns" in role_patterns:
            self._create_role_patterns_chart(
                axes[2], role_patterns["observed_patterns"], "Team Role Patterns"
            )

        # Plot 4: Role specialization analysis
        if "role_specialization" in role_patterns:
            self._create_role_specialization_chart(
                axes[3],
                role_patterns["role_specialization"],
                "Role Specialization Metrics",
            )

        # Plot 5: Gini coefficient distribution
        self._create_gini_coefficient_chart(
            axes[4], distribution_data, "Team Inequality (Gini Coefficient)"
        )

        # Plot 6: Summary info
        self._create_team_distribution_summary(
            axes[5], distribution_data, "Team Usage Distribution Summary"
        )

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if output_filename:
            filename = generate_filename(output_filename)
            save_figure(
                fig, filename, directory=self._output_dir, formats=save_formats, dpi=300
            )

        # Show figure if requested
        if show_figure:
            plt.show()

        return fig

    def visualize_team_composition_correlation(
        self,
        course_id: Optional[str] = None,
        include_detailed_breakdown: bool = True,
        figure_size: Tuple[float, float] = None,
        output_filename: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        show_figure: bool = True,
    ) -> Figure:
        """
        Create visualizations of team composition correlations.

        Args:
            course_id: Optional course ID to filter teams
            include_detailed_breakdown: Whether to include detailed breakdowns
            figure_size: Optional custom figure size
            output_filename: Optional filename for saving visualization
            save_formats: List of formats for saving
            show_figure: Whether to show the figure

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        # Get analysis data from the team analyzer
        composition_data = self._team_analyzer.correlate_team_composition_with_usage(
            course_id=course_id, include_detailed_breakdown=include_detailed_breakdown
        )

        # Use default figure size if not provided
        if figure_size is None:
            figure_size = self._default_figsize

        # Determine number of subplots based on data
        num_plots = 4  # Base number of plots
        if include_detailed_breakdown and "detailed_breakdown" in composition_data:
            num_plots += 2

        # Create figure with appropriate subplots
        fig = create_figure(
            width=figure_size[0],
            height=figure_size[1] * (num_plots / 2),
            dpi=self._default_dpi,
        )

        # Create subplot grid with appropriate layout
        rows = num_plots // 2 + (1 if num_plots % 2 else 0)
        cols = 2
        axes = create_subplot_grid(fig, rows=rows, cols=cols)

        # Extract data
        team_size_correlation = composition_data.get("team_size_correlation", {})
        diversity_correlation = composition_data.get("diversity_correlation", {})
        composition_factors = composition_data.get("composition_factors", {})

        # Plot 1: Team size group metrics
        if "size_groups" in team_size_correlation:
            self._create_size_group_chart(
                axes[0], team_size_correlation["size_groups"], "Metrics by Team Size"
            )

        # Plot 2: Team size correlation metrics
        if "size_impact" in team_size_correlation:
            self._create_size_impact_chart(
                axes[1], team_size_correlation["size_impact"], "Team Size Impact"
            )

        # Plot 3: Diversity group metrics
        if "diversity_groups" in diversity_correlation:
            self._create_diversity_group_chart(
                axes[2],
                diversity_correlation["diversity_groups"],
                "Metrics by Team Diversity",
            )

        # Plot 4: Diversity correlation metrics
        if "diversity_impact" in diversity_correlation:
            self._create_diversity_impact_chart(
                axes[3],
                diversity_correlation["diversity_impact"],
                "Team Diversity Impact",
            )

        # Plots 5-6: Additional breakdown charts if available
        if include_detailed_breakdown and "detailed_breakdown" in composition_data:
            detailed_data = composition_data.get("detailed_breakdown", {})
            self._create_detailed_breakdown_charts(
                axes[4:6], detailed_data, "Detailed Composition Breakdown"
            )

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if output_filename:
            filename = generate_filename(output_filename)
            save_figure(
                fig, filename, directory=self._output_dir, formats=save_formats, dpi=300
            )

        # Show figure if requested
        if show_figure:
            plt.show()

        return fig

    def visualize_team_framework_progression(
        self,
        framework_type: str = "Disciplined Entrepreneurship",
        course_id: Optional[str] = None,
        compare_to_individuals: bool = True,
        figure_size: Tuple[float, float] = None,
        output_filename: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        show_figure: bool = True,
    ) -> Figure:
        """
        Create visualizations of team framework progression.

        Args:
            framework_type: The framework to analyze
            course_id: Optional course ID to filter teams
            compare_to_individuals: Whether to compare with individual users
            figure_size: Optional custom figure size
            output_filename: Optional filename for saving visualization
            save_formats: List of formats for saving
            show_figure: Whether to show the figure

        Returns:
            Figure: Matplotlib figure with visualizations
        """
        # Convert string to enum
        framework_enum = None
        for ft in FrameworkType:
            if ft.value == framework_type:
                framework_enum = ft
                break

        if not framework_enum:
            framework_enum = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP

        # Get analysis data from the team analyzer
        progression_data = self._team_analyzer.analyze_team_framework_progression(
            framework=framework_enum,
            course_id=course_id,
            compare_to_individuals=compare_to_individuals,
        )

        # Use default figure size if not provided
        if figure_size is None:
            figure_size = self._default_figsize

        # Determine number of subplots based on data
        num_plots = 4  # Base number of plots
        has_comparison = compare_to_individuals and "comparison" in progression_data

        if has_comparison:
            num_plots += 2

        # Create figure with appropriate subplots
        fig = create_figure(
            width=figure_size[0],
            height=figure_size[1] * (num_plots / 2),
            dpi=self._default_dpi,
        )

        # Create subplot grid with appropriate layout
        rows = num_plots // 2 + (1 if num_plots % 2 else 0)
        cols = 2
        axes = create_subplot_grid(fig, rows=rows, cols=cols)

        # Extract data
        framework_name = progression_data.get("framework", framework_type)
        team_progression = progression_data.get("team_progression", {})
        individual_progression = progression_data.get("individual_progression", {})
        comparison = progression_data.get("comparison", {})

        # Plot 1: Team step completion rates
        if "step_completion_rates" in team_progression:
            self._create_step_completion_chart(
                axes[0],
                team_progression["step_completion_rates"],
                f"Team Step Completion Rates ({framework_name})",
            )

        # Plot 2: Team progression patterns
        if "progression_patterns" in team_progression:
            self._create_progression_pattern_chart(
                axes[1],
                team_progression["progression_patterns"],
                "Team Progression Patterns",
            )

        # Plot 3: Collaboration impact
        if "collaboration_impact" in team_progression:
            self._create_collaboration_impact_chart(
                axes[2],
                team_progression["collaboration_impact"],
                "Collaboration Impact on Progression",
            )

        # Plot 4: Initial individual vs team metrics if comparison available
        if has_comparison:
            self._create_initial_comparison_chart(
                axes[3], comparison, "Team vs Individual Comparison"
            )

            # Plot 5-6: Detailed comparison charts
            self._create_detailed_comparison_charts(
                axes[4:6], comparison, "Detailed Progression Comparison"
            )
        else:
            # Plot 4: Additional team progression chart
            self._create_summary_progression_chart(
                axes[3], team_progression, "Team Framework Progression Summary"
            )

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if output_filename:
            filename = generate_filename(output_filename)
            save_figure(
                fig, filename, directory=self._output_dir, formats=save_formats, dpi=300
            )

        # Show figure if requested
        if show_figure:
            plt.show()

        return fig

    def create_comprehensive_team_report(
        self,
        course_id: Optional[str] = None,
        team_id: Optional[int] = None,
        output_dir: Optional[str] = None,
        include_data_tables: bool = True,
        report_name: str = "comprehensive_team_analysis",
    ) -> Dict[str, str]:
        """
        Create a comprehensive report with all team visualizations.

        Args:
            course_id: Optional course ID to filter data
            team_id: Optional team ID for specific team analysis
            output_dir: Optional directory for saving visualizations
            include_data_tables: Whether to include data tables with the visualizations
            report_name: Base name for the report files

        Returns:
            Dict[str, str]: Dictionary of generated file paths
        """
        # Set output directory if provided
        if output_dir:
            self._output_dir = output_dir

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d")

        # Dictionary to track generated files
        generated_files = {}

        # 1. Generate team vs individual engagement visualization
        fig_engagement = self.visualize_team_vs_individual_engagement(
            course_id=course_id,
            include_demographic_breakdown=True,
            output_filename=f"{report_name}_engagement_{timestamp}",
            show_figure=False,
        )
        generated_files["engagement"] = (
            f"{self._output_dir}/{report_name}_engagement_{timestamp}.png"
        )

        # 2. Generate team collaboration patterns visualization
        fig_collab = self.visualize_team_collaboration_patterns(
            team_id=team_id,
            course_id=course_id,
            include_temporal_analysis=True,
            output_filename=f"{report_name}_collaboration_{timestamp}",
            show_figure=False,
        )
        generated_files["collaboration"] = (
            f"{self._output_dir}/{report_name}_collaboration_{timestamp}.png"
        )

        # 3. Generate team usage distribution visualization
        fig_usage = self.visualize_team_usage_distribution(
            course_id=course_id,
            min_team_size=2,
            output_filename=f"{report_name}_usage_distribution_{timestamp}",
            show_figure=False,
        )
        generated_files["usage_distribution"] = (
            f"{self._output_dir}/{report_name}_usage_distribution_{timestamp}.png"
        )

        # 4. Generate team composition correlation visualization
        fig_composition = self.visualize_team_composition_correlation(
            course_id=course_id,
            include_detailed_breakdown=True,
            output_filename=f"{report_name}_composition_{timestamp}",
            show_figure=False,
        )
        generated_files["composition"] = (
            f"{self._output_dir}/{report_name}_composition_{timestamp}.png"
        )

        # 5. Generate team framework progression visualization
        fig_progression = self.visualize_team_framework_progression(
            course_id=course_id,
            compare_to_individuals=True,
            output_filename=f"{report_name}_progression_{timestamp}",
            show_figure=False,
        )
        generated_files["progression"] = (
            f"{self._output_dir}/{report_name}_progression_{timestamp}.png"
        )

        # 6. Optionally export data tables for each analysis
        if include_data_tables:
            # Generate data tables for each analysis
            self._export_data_tables(
                course_id=course_id,
                team_id=team_id,
                output_filename=f"{report_name}_data_tables_{timestamp}",
            )
            generated_files["data_tables"] = (
                f"{self._output_dir}/{report_name}_data_tables_{timestamp}.xlsx"
            )

        # Return paths to all generated files
        return generated_files

    # ----------------------
    # Helper methods for visualizations
    # ----------------------

    def _create_engagement_metrics_comparison(
        self,
        ax: Axes,
        team_metrics: Dict[str, float],
        individual_metrics: Dict[str, float],
        title: str,
    ) -> None:
        """
        Create a bar chart comparing team and individual engagement metrics.

        Args:
            ax: Matplotlib axes to plot on
            team_metrics: Dictionary of team metrics
            individual_metrics: Dictionary of individual metrics
            title: Chart title
        """
        # Prepare data for grouped bar chart
        metrics = {}

        for key in set(team_metrics.keys()) | set(individual_metrics.keys()):
            metrics[key] = {
                "Team": team_metrics.get(key, 0),
                "Individual": individual_metrics.get(key, 0),
            }

        # Skip if no data
        if not metrics:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Get colors
        colors = get_color_palette("categorical_main", n_colors=2)

        # Create grouped bar chart
        plot_grouped_bars(
            ax,
            data=metrics,
            orientation="vertical",
            colors=colors,
            add_data_labels=True,
            data_label_format="{:.2f}",
            add_legend=True,
            legend_title="User Type",
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            xticklabel_rotation=45 if len(metrics) > 3 else 0,
            grid=True,
        )

    def _create_engagement_distribution_comparison(
        self,
        ax: Axes,
        team_distribution: Dict[str, Dict[str, float]],
        individual_distribution: Dict[str, Dict[str, float]],
        title: str,
    ) -> None:
        """
        Create a side-by-side bar chart of engagement level distributions.

        Args:
            ax: Matplotlib axes to plot on
            team_distribution: Team engagement distribution
            individual_distribution: Individual engagement distribution
            title: Chart title
        """
        # Prepare data for grouped bar chart
        distribution_data = {}

        # Get all levels
        levels = set()
        if team_distribution:
            levels.update(team_distribution.keys())
        if individual_distribution:
            levels.update(individual_distribution.keys())

        # Format data for grouped bar chart
        for level in levels:
            team_percentage = (
                team_distribution.get(level, {}).get("percentage", 0) * 100
            )
            indiv_percentage = (
                individual_distribution.get(level, {}).get("percentage", 0) * 100
            )

            distribution_data[level] = {
                "Team": team_percentage,
                "Individual": indiv_percentage,
            }

        # Skip if no data
        if not distribution_data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Get colors from engagement palette for levels
        engagement_colors = get_color_palette("engagement_levels")

        # Create grouped bar chart
        plot_grouped_bars(
            ax,
            data=distribution_data,
            orientation="vertical",
            colors=get_color_palette("categorical_main", n_colors=2),
            add_data_labels=True,
            data_label_format="{:.1f}%",
            add_legend=True,
            legend_title="User Type",
        )

        # Configure axes
        configure_axes(ax, title=title, ylabel="Percentage (%)", grid=True)

    def _create_percentage_difference_chart(
        self, ax: Axes, comparison_data: Dict[str, Dict[str, float]], title: str
    ) -> None:
        """
        Create a bar chart showing percentage differences between team and individual metrics.

        Args:
            ax: Matplotlib axes to plot on
            comparison_data: Dictionary with comparison metrics
            title: Chart title
        """
        # Prepare data for bar chart
        diff_data = {}

        # Extract percentage differences
        for category, metrics in comparison_data.items():
            if "percentage_difference" in metrics:
                diff_data[category] = metrics["percentage_difference"]

        # Skip if no data
        if not diff_data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Get colors based on values (positive = green, negative = red)
        colors = []
        for value in diff_data.values():
            if value > 0:
                colors.append("#28A745")  # Green
            else:
                colors.append("#DC3545")  # Red

        # Create bar chart
        plot_bar(
            ax,
            data=diff_data,
            orientation="horizontal",
            color=colors,
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        # Add reference line at zero
        add_reference_line(
            ax,
            value=0,
            orientation="vertical",
            linestyle="-",
            color="#666666",
            add_label=False,
        )

        # Configure axes
        configure_axes(ax, title=title, xlabel="Percentage Difference (%)", grid=True)

    def _create_demographic_breakdown_visualization(
        self, axes: List[Axes], demographic_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create visualizations for demographic breakdown data.

        Args:
            axes: List of Matplotlib axes to plot on
            demographic_data: Demographic breakdown data
            title: Base title for charts
        """
        # Make sure we have enough axes
        if len(axes) < 2:
            return

        # Extract demographic categories
        categories = list(demographic_data.keys())

        # Plot each category
        for i, category in enumerate(categories[:2]):  # Limit to 2 categories
            if i >= len(axes):
                break

            category_data = demographic_data[category]
            ax = axes[i]

            # Prepare data for chart
            chart_data = {}

            for subcategory, data in category_data.items():
                comparison = data.get("comparison", {})
                if "percentage_difference" in comparison.get(
                    "engagement_difference", {}
                ):
                    diff_percentage = comparison["engagement_difference"][
                        "percentage_difference"
                    ]
                    chart_data[subcategory] = diff_percentage

            # Skip if no data
            if not chart_data:
                ax.text(
                    0.5, 0.5, f"No {category} data available", ha="center", va="center"
                )
                continue

            # Get colors based on values
            colors = []
            for value in chart_data.values():
                if value > 0:
                    colors.append("#28A745")  # Green
                else:
                    colors.append("#DC3545")  # Red

            # Create horizontal bar chart
            plot_bar(
                ax,
                data=chart_data,
                orientation="horizontal",
                color=colors,
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            # Add reference line at zero
            add_reference_line(
                ax,
                value=0,
                orientation="vertical",
                linestyle="-",
                color="#666666",
                add_label=False,
            )

            # Configure axes
            configure_axes(
                ax,
                title=f"{title} - {category.replace('_', ' ').title()}",
                xlabel="Engagement Difference (%)",
                grid=True,
            )

    def _create_collaboration_level_chart(
        self, ax: Axes, level_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create a bar chart of collaboration level distribution.

        Args:
            ax: Matplotlib axes to plot on
            level_data: Collaboration level distribution data
            title: Chart title
        """
        # Prepare data for bar chart
        percentages = {}

        for level, data in level_data.items():
            if "percentage" in data:
                percentages[level] = data["percentage"] * 100

        # Skip if no data
        if not percentages:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Create color mapping based on collaboration level
        color_mapping = {
            "high": "#28A745",  # Green
            "medium": "#FFC107",  # Yellow
            "low": "#FD7E14",  # Orange
            "none": "#DC3545",  # Red
            "single_member": "#6C757D",  # Gray
        }

        # Get colors
        colors = [color_mapping.get(level, "#3366CC") for level in percentages.keys()]

        # Create bar chart
        plot_bar(
            ax,
            data=percentages,
            orientation="vertical",
            color=colors,
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Percentage (%)",
            xticklabel_rotation=45 if len(percentages) > 3 else 0,
            grid=True,
        )

    def _create_member_involvement_chart(
        self, ax: Axes, involvement_data: Dict[str, float], title: str
    ) -> None:
        """
        Create a visualization of member involvement metrics.

        Args:
            ax: Matplotlib axes to plot on
            involvement_data: Member involvement metrics
            title: Chart title
        """
        # Extract relevant metrics
        metrics = {}

        # Include relevant metrics for chart
        if "avg_active_ratio" in involvement_data:
            metrics["Average Active Member Ratio"] = (
                involvement_data["avg_active_ratio"] * 100
            )

        if "full_team_engagement_rate" in involvement_data:
            metrics["Full Team Engagement Rate"] = (
                involvement_data["full_team_engagement_rate"] * 100
            )

        if "partial_engagement_rate" in involvement_data:
            metrics["Partial Engagement Rate"] = (
                involvement_data["partial_engagement_rate"] * 100
            )

        # Skip if no data
        if not metrics:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Create bar chart
        plot_bar(
            ax,
            data=metrics,
            orientation="vertical",
            color=get_color_palette("categorical_main", n_colors=len(metrics)),
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Percentage (%)",
            xticklabel_rotation=45 if len(metrics) > 2 else 0,
            grid=True,
        )

    def _create_team_collaboration_chart(
        self, ax: Axes, teams_data: List[Dict[str, Any]], title: str
    ) -> None:
        """
        Create a visualization of collaboration metrics by team.

        Args:
            ax: Matplotlib axes to plot on
            teams_data: List of team data with collaboration metrics
            title: Chart title
        """
        # Prepare data for chart
        team_names = []
        collab_levels = []
        member_counts = []
        active_ratios = []

        for team in teams_data:
            team_name = team.get("team_name", "Unknown")
            if len(team_name) > 15:
                team_name = team_name[:12] + "..."

            collab_level = team.get("collaboration_level", "none")
            member_count = team.get("member_count", 0)

            # Calculate active ratio if available
            active_ratio = (
                team.get("active_members", 0) / member_count if member_count > 0 else 0
            )

            team_names.append(team_name)
            collab_levels.append(collab_level)
            member_counts.append(member_count)
            active_ratios.append(active_ratio)

        # Skip if no data
        if not team_names:
            ax.text(0.5, 0.5, "No team data available", ha="center", va="center")
            return

        # Define color mapping for collaboration levels
        color_mapping = {
            "high": "#28A745",  # Green
            "medium": "#FFC107",  # Yellow
            "low": "#FD7E14",  # Orange
            "none": "#DC3545",  # Red
            "single_member": "#6C757D",  # Gray
        }

        # Create scatter plot of team size vs active ratio, colored by collaboration level
        scatter_colors = [
            color_mapping.get(level, "#3366CC") for level in collab_levels
        ]

        # Create scatter plot
        scatter = ax.scatter(
            member_counts, active_ratios, c=scatter_colors, s=100, alpha=0.8
        )

        # Add team labels
        for i, team_name in enumerate(team_names):
            ax.annotate(
                team_name,
                (member_counts[i], active_ratios[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Create legend for collaboration levels
        legend_elements = []
        for level, color in color_mapping.items():
            if level in collab_levels:
                from matplotlib.lines import Line2D

                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                        label=level.replace("_", " ").title(),
                    )
                )

        if legend_elements:
            ax.legend(handles=legend_elements, title="Collaboration Level")

        # Configure axes
        configure_axes(
            ax,
            title=title,
            xlabel="Team Size (Members)",
            ylabel="Active Member Ratio",
            grid=True,
        )

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

    def _create_idea_sharing_patterns_chart(
        self, ax: Axes, pattern_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create a visualization of idea sharing patterns.

        Args:
            ax: Matplotlib axes to plot on
            pattern_data: Idea sharing pattern data
            title: Chart title
        """
        # Extract ownership and contribution patterns
        ownership_patterns = pattern_data.get("ownership_patterns", {})
        contribution_flow = pattern_data.get("contribution_flow", {})

        # If we don't have both types of data, create a combined chart
        if not ownership_patterns or not contribution_flow:
            combined_data = {}

            # Add available ownership patterns
            for pattern, data in ownership_patterns.items():
                if "percentage" in data:
                    combined_data[f"Ownership: {pattern.replace('_', ' ').title()}"] = (
                        data["percentage"] * 100
                    )

            # Add available contribution flows
            for pattern, data in contribution_flow.items():
                if "percentage" in data:
                    combined_data[f"Flow: {pattern.replace('_', ' ').title()}"] = (
                        data["percentage"] * 100
                    )

            # Skip if no data
            if not combined_data:
                ax.text(0.5, 0.5, "No pattern data available", ha="center", va="center")
                return

            # Create bar chart
            plot_bar(
                ax,
                data=combined_data,
                orientation="vertical",
                color=get_color_palette(
                    "categorical_main", n_colors=len(combined_data)
                ),
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            # Configure axes
            configure_axes(
                ax,
                title=title,
                ylabel="Percentage (%)",
                xticklabel_rotation=45,
                grid=True,
            )

            return

        # If we have both types of data, create a more complex visualization
        # Prepare data for grouped bar chart
        patterns = {}

        # Get all unique pattern names
        owner_patterns = list(ownership_patterns.keys())
        flow_patterns = list(contribution_flow.keys())

        for pattern in owner_patterns:
            display_name = pattern.replace("_", " ").title()
            patterns[display_name] = {
                "Ownership": ownership_patterns[pattern].get("percentage", 0) * 100
            }

        for pattern in flow_patterns:
            display_name = pattern.replace("_", " ").title()
            if display_name in patterns:
                patterns[display_name]["Flow"] = (
                    contribution_flow[pattern].get("percentage", 0) * 100
                )
            else:
                patterns[display_name] = {
                    "Ownership": 0,
                    "Flow": contribution_flow[pattern].get("percentage", 0) * 100,
                }

        # Create grouped bar chart
        plot_grouped_bars(
            ax,
            data=patterns,
            orientation="vertical",
            colors=get_color_palette("categorical_main", n_colors=2),
            add_data_labels=True,
            data_label_format="{:.1f}%",
            add_legend=True,
            legend_title="Pattern Type",
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Percentage (%)",
            xticklabel_rotation=45 if len(patterns) > 2 else 0,
            grid=True,
        )

    def _create_temporal_collaboration_charts(
        self, axes: List[Axes], temporal_data: Dict[str, Any], title: str
    ) -> None:
        """
        Create visualizations for temporal collaboration patterns.

        Args:
            axes: List of Matplotlib axes to plot on
            temporal_data: Temporal collaboration data
            title: Base title for charts
        """
        # Make sure we have enough axes
        if len(axes) < 2:
            return

        # Chart 1: Synchronous vs Asynchronous Collaboration
        ax1 = axes[0]
        sync_data = temporal_data.get("synchronous_vs_asynchronous", {})

        if sync_data:
            # Prepare data for pie chart
            labels = []
            sizes = []

            for collab_type, data in sync_data.items():
                if "percentage" in data:
                    labels.append(collab_type.title())
                    sizes.append(data["percentage"])

            # Skip if no data
            if not labels:
                ax1.text(
                    0.5,
                    0.5,
                    "No synchronous/asynchronous data",
                    ha="center",
                    va="center",
                )
            else:
                # Create pie chart
                ax1.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=get_color_palette("categorical_main", n_colors=len(labels)),
                    startangle=90,
                )
                ax1.axis(
                    "equal"
                )  # Equal aspect ratio ensures that pie is drawn as a circle

                # Set title
                ax1.set_title(f"{title} - Sync vs Async")
        else:
            ax1.text(
                0.5, 0.5, "No synchronous/asynchronous data", ha="center", va="center"
            )

        # Chart 2: Sequential vs Parallel Collaboration
        ax2 = axes[1]
        seq_par_data = temporal_data.get("sequential_vs_parallel", {})

        if seq_par_data:
            # Prepare data for pie chart
            labels = []
            sizes = []

            for collab_type, data in seq_par_data.items():
                if "percentage" in data:
                    labels.append(collab_type.title())
                    sizes.append(data["percentage"])

            # Skip if no data
            if not labels:
                ax2.text(
                    0.5, 0.5, "No sequential/parallel data", ha="center", va="center"
                )
            else:
                # Create pie chart
                ax2.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=get_color_palette("categorical_main", n_colors=len(labels)),
                    startangle=90,
                )
                ax2.axis(
                    "equal"
                )  # Equal aspect ratio ensures that pie is drawn as a circle

                # Set title
                ax2.set_title(f"{title} - Sequential vs Parallel")
        else:
            ax2.text(0.5, 0.5, "No sequential/parallel data", ha="center", va="center")

    def _create_distribution_type_chart(
        self, ax: Axes, distribution_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create a visualization of distribution type frequencies.

        Args:
            ax: Matplotlib axes to plot on
            distribution_data: Distribution type frequency data
            title: Chart title
        """
        # Prepare data for chart
        distribution_types = {}

        for dist_type, data in distribution_data.items():
            if "percentage" in data:
                # Format the type name
                display_name = dist_type.replace("_", " ").title()
                distribution_types[display_name] = data["percentage"] * 100

        # Skip if no data
        if not distribution_types:
            ax.text(
                0.5, 0.5, "No distribution data available", ha="center", va="center"
            )
            return

        # Create color mapping
        color_mapping = {
            "Uniform": "#28A745",  # Green
            "Partial": "#FFC107",  # Yellow
            "Primary": "#FD7E14",  # Orange
            "Inactive": "#DC3545",  # Red
        }

        # Get colors
        colors = [
            color_mapping.get(dist_type, "#3366CC")
            for dist_type in distribution_types.keys()
        ]

        # Create bar chart
        plot_bar(
            ax,
            data=distribution_types,
            orientation="vertical",
            color=colors,
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Percentage (%)",
            xticklabel_rotation=45 if len(distribution_types) > 3 else 0,
            grid=True,
        )

    def _create_usage_concentration_chart(
        self, ax: Axes, concentration_data: Dict[str, float], title: str
    ) -> None:
        """
        Create a bar chart of usage concentration metrics.

        Args:
            ax: Matplotlib axes to plot on
            concentration_data: Usage concentration metrics
            title: Chart title
        """
        # Prepare data for chart
        metrics = {}

        for metric, value in concentration_data.items():
            # Format metric name
            display_name = metric.replace("avg_", "").replace("_", " ").title()

            # Scale based on metric type
            if metric == "avg_gini_coefficient":
                metrics[display_name] = value * 100  # Scale to percentage
            elif metric == "avg_activity_ratio":
                metrics[display_name] = value * 100  # Scale to percentage
            else:
                metrics[display_name] = value

        # Skip if no data
        if not metrics:
            ax.text(
                0.5, 0.5, "No concentration data available", ha="center", va="center"
            )
            return

        # Create bar chart
        plot_bar(
            ax,
            data=metrics,
            orientation="vertical",
            color=get_color_palette("categorical_main", n_colors=len(metrics)),
            add_data_labels=True,
            data_label_format=(
                "{:.1f}"
                if any(name == "Coefficient Of Variation" for name in metrics)
                else "{:.1f}%"
            ),
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Value",
            xticklabel_rotation=45 if len(metrics) > 2 else 0,
            grid=True,
        )

    def _create_role_patterns_chart(
        self, ax: Axes, pattern_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create a visualization of team role patterns.

        Args:
            ax: Matplotlib axes to plot on
            pattern_data: Role pattern data
            title: Chart title
        """
        # Prepare data for chart
        role_patterns = {}
        team_sizes = {}

        for pattern, data in pattern_data.items():
            if "percentage" in data:
                # Format the pattern name
                display_name = pattern.replace("_", " ").title()
                role_patterns[display_name] = data["percentage"] * 100

                # Store average team size if available
                if "avg_team_size" in data:
                    team_sizes[display_name] = data["avg_team_size"]

        # Skip if no data
        if not role_patterns:
            ax.text(
                0.5, 0.5, "No role pattern data available", ha="center", va="center"
            )
            return

        # Create bar chart
        bars = plot_bar(
            ax,
            data=role_patterns,
            orientation="vertical",
            color=get_color_palette("categorical_main", n_colors=len(role_patterns)),
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        # If we have team sizes, add them as text annotations
        if team_sizes:
            for i, (pattern, size) in enumerate(team_sizes.items()):
                ax.text(
                    i,  # Bar position
                    5,  # Fixed position near the bottom
                    f"Avg size: {size:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Percentage (%)",
            xticklabel_rotation=45 if len(role_patterns) > 2 else 0,
            grid=True,
        )

    def _create_role_specialization_chart(
        self, ax: Axes, specialization_data: Dict[str, Any], title: str
    ) -> None:
        """
        Create a visualization of role specialization metrics.

        Args:
            ax: Matplotlib axes to plot on
            specialization_data: Role specialization metrics
            title: Chart title
        """
        # Extract metrics of interest
        metrics = {}

        if "avg_gini_coefficient" in specialization_data:
            metrics["Gini Coefficient"] = (
                specialization_data["avg_gini_coefficient"] * 100
            )

        if "avg_coefficient_of_variation" in specialization_data:
            metrics["Coefficient of Variation"] = specialization_data[
                "avg_coefficient_of_variation"
            ]

        # Create a gauge-like visualization for specialization level
        specialization_level = specialization_data.get("specialization_level", "low")
        level_values = {"low": 0.25, "medium": 0.5, "high": 0.75}
        level_value = level_values.get(specialization_level, 0.5)

        # Skip if no metrics
        if not metrics:
            ax.text(
                0.5, 0.5, "No specialization data available", ha="center", va="center"
            )
            return

        # Create bar chart for metrics
        plot_bar(
            ax,
            data=metrics,
            orientation="vertical",
            color=get_color_palette("categorical_main", n_colors=len(metrics)),
            add_data_labels=True,
            data_label_format=(
                "{:.1f}" if "Coefficient of Variation" in metrics else "{:.1f}%"
            ),
        )

        # Add specialization level as text
        ax.text(
            0.5,
            0.9,
            f"Specialization Level: {specialization_level.title()}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

        # Configure axes
        configure_axes(
            ax,
            title=title,
            ylabel="Value",
            xticklabel_rotation=45 if len(metrics) > 2 else 0,
            grid=True,
        )

    def _create_gini_coefficient_chart(
        self, ax: Axes, distribution_data: Dict[str, Any], title: str
    ) -> None:
        """
        Create a chart showing Gini coefficient distribution across teams.

        Args:
            ax: Matplotlib axes to plot on
            distribution_data: Team distribution data
            title: Chart title
        """
        # Extract team distributions
        team_distributions = []
        for team_id, team_data in enumerate(
            distribution_data.get("teams_analyzed", [])
        ):
            gini = team_data.get("gini_coefficient", -1)
            if gini >= 0:
                team_name = team_data.get("team_name", f"Team {team_id}")
                if len(team_name) > 15:
                    team_name = team_name[:12] + "..."
                team_distributions.append((team_name, gini))

        # Sort by Gini coefficient (descending)
        team_distributions.sort(key=lambda x: x[1], reverse=True)

        # Prepare data for chart
        teams = [team[0] for team in team_distributions]
        gini_values = [team[1] for team in team_distributions]

        # Skip if no data
        if not teams:
            ax.text(
                0.5, 0.5, "No Gini coefficient data available", ha="center", va="center"
            )
            return

        # Create bar chart
        # Color bars based on Gini value (higher = more red)
        colors = []
        for gini in gini_values:
            if gini < 0.2:
                colors.append("#28A745")  # Green - low inequality
            elif gini < 0.4:
                colors.append("#FFC107")  # Yellow - medium inequality
            elif gini < 0.6:
                colors.append("#FD7E14")  # Orange - high inequality
            else:
                colors.append("#DC3545")  # Red - very high inequality

        plot_bar(
            ax,
            data=dict(zip(teams, gini_values)),
            orientation="horizontal",
            color=colors,
            add_data_labels=True,
            data_label_format="{:.2f}",
        )

        # Add reference lines for inequality levels
        add_reference_line(
            ax,
            value=0.2,
            orientation="vertical",
            linestyle="--",
            color="#666666",
            add_label=True,
            label="Low",
        )

        add_reference_line(
            ax,
            value=0.4,
            orientation="vertical",
            linestyle="--",
            color="#666666",
            add_label=True,
            label="Medium",
        )

        add_reference_line(
            ax,
            value=0.6,
            orientation="vertical",
            linestyle="--",
            color="#666666",
            add_label=True,
            label="High",
        )

        # Configure axes
        configure_axes(ax, title=title, xlabel="Gini Coefficient (0-1)", grid=True)

        # Set x-axis limits
        ax.set_xlim(0, 1)

    def _create_team_distribution_summary(
        self, ax: Axes, distribution_data: Dict[str, Any], title: str
    ) -> None:
        """
        Create a summary visualization for team distribution data.

        Args:
            ax: Matplotlib axes to plot on
            distribution_data: Team distribution data
            title: Chart title
        """
        # Extract key metrics
        teams_analyzed = distribution_data.get("teams_analyzed", 0)

        # Create summary table
        data = []

        # Add distribution type counts
        dist_metrics = distribution_data.get("distribution_metrics", {})
        dist_types = ["uniform", "partial", "primary", "inactive"]
        for dist_type in dist_types:
            if dist_type in dist_metrics:
                count = dist_metrics[dist_type].get("count", 0)
                percentage = dist_metrics[dist_type].get("percentage", 0) * 100
                data.append([dist_type.title(), count, f"{percentage:.1f}%"])

        # Add concentration metrics
        concentration = distribution_data.get("usage_concentration", {})
        if concentration:
            data.append(["", "", ""])  # Blank row
            data.append(["Concentration Metrics", "", ""])

            for metric, value in concentration.items():
                display_name = metric.replace("avg_", "").replace("_", " ").title()

                # Format value based on metric type
                if metric == "avg_gini_coefficient":
                    formatted_value = f"{value:.2f} (0-1)"
                elif metric == "avg_activity_ratio":
                    formatted_value = f"{value*100:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"

                data.append([display_name, "", formatted_value])

        # Add role metrics if available
        role_patterns = distribution_data.get("role_patterns", {}).get(
            "observed_patterns", {}
        )
        if role_patterns:
            data.append(["", "", ""])  # Blank row
            data.append(["Role Patterns", "", ""])

            for pattern, pattern_data in role_patterns.items():
                display_name = pattern.replace("_", " ").title()
                count = pattern_data.get("count", 0)
                percentage = pattern_data.get("percentage", 0) * 100
                data.append([display_name, count, f"{percentage:.1f}%"])

        # Skip if no data
        if not data:
            ax.text(0.5, 0.5, "No summary data available", ha="center", va="center")
            return

        # Create table
        add_data_table(
            ax,
            data=data,
            col_labels=["Metric", "Count", "Value"],
            title=title,
            fontsize=9,
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.8],  # Centered in the axes
            header_color="#E6EFF6",
        )

        # Hide axis elements
        ax.axis("off")

    def _create_size_group_chart(
        self, ax: Axes, size_group_data: Dict[str, Dict[str, Any]], title: str
    ) -> None:
        """
        Create a chart showing metrics by team size group.

        Args:
            ax: Matplotlib axes to plot on
            size_group_data: Metrics by team size group
            title: Chart title
        """
        # Prepare data for grouped bar chart
        metrics_by_size = {}

        # Metrics to include
        metric_keys = [
            "avg_engagement_score",
            "avg_ideas_per_member",
            "avg_steps_per_member",
        ]

        # Display names for metrics
        metric_names = {
            "avg_engagement_score": "Engagement Score",
            "avg_ideas_per_member": "Ideas per Member",
            "avg_steps_per_member": "Steps per Member",
        }

        # Extract metrics for each size group
        for size_group, data in size_group_data.items():
            metrics_by_size[size_group] = {}

            for metric in metric_keys:
                if metric in data:
                    metrics_by_size[size_group][metric_names[metric]] = data[metric]

        # Skip if no data
        if not metrics_by_size:
            ax.text(0.5, 0.5, "No size group data available", ha="center", va="center")
            return

        # Create grouped bar chart
        plot_grouped_bars(
            ax,
            data=metrics_by_size,
            orientation="vertical",
            colors=get_color_palette("categorical_main", n_colors=len(metric_names)),
            add_data_labels=True,
            data_label_format="{:.2f}",
            add_legend=True,
            legend_title="Metric",
        )

        # Configure axes
        configure_axes(ax, title=title, ylabel="Value", grid=True)

        # Add team counts if available
        y_pos = 0.02
        for size_group, data in size_group_data.items():
            if "team_count" in data:
                ax.text(
                    size_group,
                    y_pos,
                    f"{data['team_count']} teams",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    transform=ax.get_xaxis_transform(),
                )
