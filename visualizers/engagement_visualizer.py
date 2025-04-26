"""
Engagement visualizer for the data analysis system.

This module provides visualization functionality for engagement analysis data.
It includes methods for creating various visualizations based on engagement metrics,
user demographics, temporal patterns, and comparative analyses.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from ..analyzers.engagement_analyzer import EngagementAnalyzer
from ..utils.visualization_creation_utils import (
    create_figure,
    configure_axes,
    save_figure,
    plot_bar,
    plot_grouped_bars,
    plot_stacked_bars,
    plot_line,
    plot_multi_line,
    plot_heatmap,
    get_color_palette,
    apply_theme,
    add_data_table,
    wrap_labels,
)
from ..utils.visualization_data_utils import (
    export_data_with_visualization,
    create_report_directory,
)


class EngagementVisualizer:
    """
    Visualizer for engagement analysis data.

    This class provides methods for creating visualizations of user engagement
    patterns, metrics, and comparisons. It interfaces with the EngagementAnalyzer
    to access analysis results and generate appropriate visualizations.
    """

    def __init__(
        self,
        engagement_analyzer: EngagementAnalyzer,
        output_dir: Optional[str] = None,
        theme: str = "default",
        default_figsize: Tuple[float, float] = (10, 6),
        include_timestamps: bool = True,
    ):
        """
        Initialize the engagement visualizer.

        Args:
            engagement_analyzer: Engagement analyzer instance to visualize data from
            output_dir: Optional output directory for saving visualizations
            theme: Visualization theme ('default', 'dark', 'print')
            default_figsize: Default figure size (width, height) in inches
            include_timestamps: Whether to include timestamps in saved filenames
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._analyzer = engagement_analyzer
        self._output_dir = output_dir
        self._theme = theme
        self._default_figsize = default_figsize
        self._include_timestamps = include_timestamps

        # Apply theme
        apply_theme(self._theme)

    def create_engagement_level_visualization(
        self,
        course_id: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        include_demographics: bool = True,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of user engagement levels.

        Args:
            course_id: Optional course ID to filter users
            figsize: Optional figure size (width, height) in inches
            include_demographics: Whether to include demographic breakdown
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get engagement data from analyzer
        engagement_data = self._analyzer.classify_users_by_engagement(course_id)

        # Count users at each level
        counts = {
            "High": len(engagement_data.get("HIGH", [])),
            "Medium": len(engagement_data.get("MEDIUM", [])),
            "Low": len(engagement_data.get("LOW", [])),
        }

        # Set up figure using utility function
        if figsize is None:
            figsize = self._default_figsize

        if include_demographics and any(counts.values()):
            # Create a 2x2 figure for engagement levels and demographics
            fig = create_figure(
                width=figsize[0], 
                height=figsize[1] * 1.5, 
                theme=self._theme
            )
            # Create custom grid with different row heights
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
            axs = [
                fig.add_subplot(gs[0, 0]),  # Top left
                fig.add_subplot(gs[0, 1]),  # Top right
                fig.add_subplot(gs[1, 0]),  # Bottom left
                fig.add_subplot(gs[1, 1]),  # Bottom right
            ]
        else:
            # Create a 1x2 figure for just engagement levels
            fig = create_figure(
                width=figsize[0], 
                height=figsize[1], 
                theme=self._theme
            )
            gs = plt.GridSpec(1, 2, figure=fig)
            axs = [
                fig.add_subplot(gs[0, 0]),  # Left
                fig.add_subplot(gs[0, 1]),  # Right
                None, 
                None
            ]

        # Create engagement level bar chart using utility function
        plot_bar(
            axs[0],
            data=counts,
            color=["#28A745", "#FFC107", "#DC3545"],  # Green, Yellow, Red
            add_data_labels=True,
        )
        
        # Configure axes using utility function
        configure_axes(
            axs[0],
            title="User Engagement Levels",
            xlabel="Engagement Level",
            ylabel="Number of Users",
        )

        # Create pie chart of engagement distribution
        if sum(counts.values()) > 0:
            # Note: plot_pie isn't in the utilities, so we use matplotlib directly
            axs[1].pie(
                counts.values(),
                labels=counts.keys(),
                autopct="%1.1f%%",
                colors=["#28A745", "#FFC107", "#DC3545"],
                startangle=90,
                wedgeprops={"edgecolor": "w", "linewidth": 1},
            )
            axs[1].set_title("Engagement Level Distribution")
            axs[1].axis("equal")
        else:
            axs[1].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axs[1].transAxes,
            )

        # If including demographics, add additional plots
        if include_demographics and axs[2] is not None and any(counts.values()):
            # Get additional demographic data
            demographic_data = self._analyzer.get_engagement_by_demographic(
                course_id=course_id, include_types=True, include_departments=True
            )

            # Extract user type data if available
            if "user_type" in demographic_data:
                user_type_data = demographic_data["user_type"]
                user_types = {}
                
                # Extract data in the right format
                for user_type, metrics in user_type_data.items():
                    if metrics["count"] == 0:
                        continue
                    
                    user_types[user_type] = {
                        "High": metrics["engagement_levels"].get("HIGH", 0),
                        "Medium": metrics["engagement_levels"].get("MEDIUM", 0),
                        "Low": metrics["engagement_levels"].get("LOW", 0),
                    }
                
                # Plot stacked bar chart for user types using utility function
                if user_types:
                    plot_stacked_bars(
                        axs[2],
                        data=user_types,
                        orientation="vertical",
                        colors=["#28A745", "#FFC107", "#DC3545"],
                        add_legend=True,
                    )
                    
                    configure_axes(
                        axs[2],
                        title="Engagement by User Type",
                        xlabel="User Type",
                        ylabel="Number of Users",
                        xticklabel_rotation=45,
                    )
                else:
                    axs[2].text(
                        0.5,
                        0.5,
                        "No user type data available",
                        ha="center",
                        va="center",
                        transform=axs[2].transAxes,
                    )

            # Extract department data if available
            if "department" in demographic_data:
                dept_data = demographic_data["department"]
                
                # Find top departments by user count and their engagement
                top_depts = sorted(
                    dept_data.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )[:5]  # Top 5 departments
                
                dept_engagement = {}
                for dept_name, metrics in top_depts:
                    if metrics["count"] == 0:
                        continue
                    
                    dept_engagement[dept_name] = {
                        "High": metrics["engagement_levels"].get("HIGH", 0),
                        "Medium": metrics["engagement_levels"].get("MEDIUM", 0),
                        "Low": metrics["engagement_levels"].get("LOW", 0),
                    }
                
                # Plot stacked bar chart for departments using utility function
                if dept_engagement:
                    plot_stacked_bars(
                        axs[3],
                        data=dept_engagement,
                        orientation="vertical",
                        colors=["#28A745", "#FFC107", "#DC3545"],
                        add_legend=True,
                    )
                    
                    configure_axes(
                        axs[3],
                        title="Engagement by Department (Top 5)",
                        xlabel="Department",
                        ylabel="Number of Users",
                        xticklabel_rotation=45,
                    )
                    
                    # Wrap long department names using utility function
                    wrap_labels(axs[3], which="x", max_length=15)
                else:
                    axs[3].text(
                        0.5,
                        0.5,
                        "No department data available",
                        ha="center",
                        va="center",
                        transform=axs[3].transAxes,
                    )

        # Adjust layout
        plt.tight_layout()

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)
            # Alternative: use more advanced save_visualization utility
            # save_visualization(fig, generate_filename("engagement_levels"), 
            #                   subdirectory="engagement", base_dir=self._output_dir)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_engagement_metrics_visualization(
        self,
        metrics: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of engagement metrics.

        Args:
            metrics: Engagement metrics data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get metrics from analyzer if not provided
        if metrics is None:
            metrics = self._analyzer.get_engagement_metrics_over_time()

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.5)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create a 3x1 grid of subplots
        gs = plt.GridSpec(3, 1, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top
            fig.add_subplot(gs[1, 0]),  # Middle
            fig.add_subplot(gs[2, 0]),  # Bottom
        ]

        # Extract timeline and metrics
        timeline = metrics.get("timeline", [])
        
        # Skip if no data
        if not timeline:
            for ax in axs:
                ax.text(
                    0.5,
                    0.5,
                    "No timeline data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # Prepare data for multi-line visualization
        # Plot new users and active users over time using utility function
        new_users = metrics.get("new_users", [])
        active_users = metrics.get("active_users", [])
        
        user_data = {
            "New Users": {str(i): val for i, val in enumerate(new_users)},
            "Active Users": {str(i): val for i, val in enumerate(active_users)}
        }
        
        # Use plot_multi_line utility function
        plot_multi_line(
            axs[0],
            data=user_data,
            x_values=timeline,
            colors=["#007BFF", "#28A745"],  # Blue, Green
            markers=["o", "s"],
            add_legend=True,
        )
        
        configure_axes(
            axs[0],
            title="User Activity Over Time",
            ylabel="Number of Users",
            xticklabel_rotation=45,
        )
        
        # Plot ideas and steps over time
        new_ideas = metrics.get("new_ideas", [])
        new_steps = metrics.get("new_steps", [])
        
        # The direct use of twinx() doesn't fit with the utility functions,
        # so we'll use a different approach for this subplot
        idea_data = {
            "New Ideas": {str(i): val for i, val in enumerate(new_ideas)}
        }
        
        plot_line(
            axs[1],
            x_data=timeline,
            y_data=new_ideas,
            color="#FFC107",  # Yellow
            marker="o",
            label="New Ideas",
        )
        
        # Add steps line (on second y-axis for better scale management)
        ax_twin = axs[1].twinx()
        plot_line(
            ax_twin,
            x_data=timeline,
            y_data=new_steps,
            color="#DC3545",  # Red
            marker="s",
            label="New Steps",
        )
        
        configure_axes(
            axs[1],
            title="Content Creation Over Time",
            ylabel="Number of Ideas",
            xticklabel_rotation=45,
        )
        ax_twin.set_ylabel("Number of Steps", color="#DC3545")
        ax_twin.tick_params(axis="y", labelcolor="#DC3545")
        
        # Combine legends
        lines_1, labels_1 = axs[1].get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        axs[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        
        # Plot cumulative metrics using utility function
        cumulative_users = metrics.get("cumulative_users", [])
        cumulative_ideas = metrics.get("cumulative_ideas", [])
        cumulative_steps = metrics.get("cumulative_steps", [])
        
        cumulative_data = {
            "Cumulative Users": {str(i): val for i, val in enumerate(cumulative_users)},
            "Cumulative Ideas": {str(i): val for i, val in enumerate(cumulative_ideas)},
            "Cumulative Steps": {str(i): val for i, val in enumerate(cumulative_steps)}
        }
        
        plot_multi_line(
            axs[2],
            data=cumulative_data,
            x_values=timeline,
            colors=["#007BFF", "#FFC107", "#DC3545"],  # Blue, Yellow, Red
            markers=["o", "s", "^"],
            add_legend=True,
        )
        
        configure_axes(
            axs[2],
            title="Cumulative Growth Over Time",
            ylabel="Count",
            xlabel="Time Period",
            xticklabel_rotation=45,
        )

        # Adjust layout
        plt.tight_layout()

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_dropout_analysis_visualization(
        self,
        dropout_data: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of user dropout patterns.

        Args:
            dropout_data: Dropout analysis data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get dropout data from analyzer if not provided
        if dropout_data is None:
            dropout_data = self._analyzer.analyze_dropout_patterns()

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top left
            fig.add_subplot(gs[0, 1]),  # Top right
            fig.add_subplot(gs[1, 0]),  # Bottom left
            fig.add_subplot(gs[1, 1]),  # Bottom right
        ]
        
        # Format for easier access
        axs_flat = axs

        # Check if data is available
        if not dropout_data:
            for ax in axs_flat:
                ax.text(
                    0.5,
                    0.5,
                    "No dropout data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # 1. Dropout Rate Pie Chart (Top Left)
        dropout_rate = dropout_data.get("dropout_rate", 0)
        retention_rate = 1 - dropout_rate
        
        # Pie chart isn't directly available as a utility function
        axs_flat[0].pie(
            [dropout_rate, retention_rate],
            labels=["Dropout", "Retained"],
            autopct="%1.1f%%",
            colors=["#DC3545", "#28A745"],  # Red, Green
            startangle=90,
            wedgeprops={"edgecolor": "w", "linewidth": 1},
        )
        axs_flat[0].set_title(f"Dropout Rate: {dropout_rate:.1%}")
        axs_flat[0].axis("equal")

        # 2. Dropout by Stage Bar Chart (Top Right)
        dropout_by_stage = dropout_data.get("dropout_by_stage", {})
        
        if dropout_by_stage:
            # Format stage names for better readability
            formatted_stages = {
                "pre_idea": "Before Creating Ideas",
                "idea_created_no_steps": "After Idea, No Steps",
                "early_steps": "Early Framework Steps",
                "mid_framework": "Mid Framework",
                "late_framework": "Late Framework",
            }
            
            # Convert data for plotting
            stage_counts = {}
            for stage, count in dropout_by_stage.items():
                stage_name = formatted_stages.get(stage, stage)
                stage_counts[stage_name] = count
            
            # Use utility function to plot bar chart
            plot_bar(
                axs_flat[1],
                data=stage_counts,
                color=get_color_palette("sequential_blue", len(stage_counts)),
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[1],
                title="Dropout by Framework Stage",
                xlabel="Stage",
                ylabel="Number of Users",
                xticklabel_rotation=45,
            )
        else:
            axs_flat[1].text(
                0.5,
                0.5,
                "No stage data available",
                ha="center",
                va="center",
                transform=axs_flat[1].transAxes,
            )

        # 3. Dropout by User Type Bar Chart (Bottom Left)
        dropout_by_user_type = dropout_data.get("dropout_by_user_type", {})
        
        if dropout_by_user_type:
            # Use utility function to plot bar chart
            plot_bar(
                axs_flat[2],
                data=dropout_by_user_type,
                color=get_color_palette("categorical_main", len(dropout_by_user_type)),
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[2],
                title="Dropout by User Type",
                xlabel="User Type",
                ylabel="Number of Users",
                xticklabel_rotation=45,
            )
        else:
            axs_flat[2].text(
                0.5,
                0.5,
                "No user type data available",
                ha="center",
                va="center",
                transform=axs_flat[2].transAxes,
            )

        # 4. Retention Factors Analysis (Bottom Right)
        retention_factors = dropout_data.get("retention_factors", [])
        
        if retention_factors:
            # Look for the team membership factor
            team_data = None
            for factor in retention_factors:
                if factor.get("factor") == "team_membership":
                    team_data = factor.get("data", {})
                    break
            
            if team_data:
                # Prepare data for plotting
                categories = []
                dropout_rates = []
                
                for category, data in team_data.items():
                    if category == "in_team":
                        label = "Team Members"
                    elif category == "not_in_team":
                        label = "Non-Team Members"
                    else:
                        label = category
                    
                    categories.append(label)
                    dropout_rates.append(data.get("dropout_rate", 0) * 100)  # Convert to percentage
                
                # Use utility function to plot bar chart
                plot_bar(
                    axs_flat[3],
                    data={categories[i]: dropout_rates[i] for i in range(len(categories))},
                    color=["#4C72B0", "#DD8452"],  # Blue, Orange
                    add_data_labels=True,
                    data_label_format="{:.1f}%",
                )
                
                configure_axes(
                    axs_flat[3],
                    title="Dropout Rate by Team Membership",
                    xlabel="",
                    ylabel="Dropout Rate (%)",
                )
            else:
                axs_flat[3].text(
                    0.5,
                    0.5,
                    "No retention factor data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[3].transAxes,
                )
        else:
            axs_flat[3].text(
                0.5,
                0.5,
                "No retention factor data available",
                ha="center",
                va="center",
                transform=axs_flat[3].transAxes,
            )

        # Add overall retention metrics as text
        avg_days_active = dropout_data.get("avg_days_active", 0)
        avg_days_to_dropout = dropout_data.get("avg_days_to_dropout", 0)
        
        metrics_text = (
            f"Average Active Days: {avg_days_active:.1f}\n"
            f"Average Days to Dropout: {avg_days_to_dropout:.1f}\n"
            f"Total Dropout Rate: {dropout_rate:.1%}"
        )
        
        fig.text(
            0.5,
            0.01,
            metrics_text,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_demographic_engagement_visualization(
        self,
        demographic_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of engagement patterns by demographic.

        Args:
            demographic_data: Demographic data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get demographic data from analyzer if not provided
        if demographic_data is None:
            demographic_data = self._analyzer.get_engagement_by_demographic()

        # Determine which demographics are available
        available_demographics = []
        for demo_type in ["user_type", "department", "experience"]:
            if demo_type in demographic_data and demographic_data[demo_type]:
                available_demographics.append(demo_type)
        
        # Skip if no data
        if not available_demographics:
            # Create simple figure using utility function
            fig = create_figure(width=self._default_figsize[0], height=self._default_figsize[1])
            ax = fig.add_subplot(111)
            
            ax.text(
                0.5,
                0.5,
                "No demographic data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, ax

        # Set up figure based on available demographics
        if figsize is None:
            figsize = (
                self._default_figsize[0],
                self._default_figsize[1] * (len(available_demographics) * 0.7),
            )

        # Create figure using utility function
        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create grid with 2 columns
        gs = plt.GridSpec(
            len(available_demographics), 
            2, 
            figure=fig,
            width_ratios=[2, 1]
        )
        
        # Create list of axes
        axs = []
        for i in range(len(available_demographics)):
            axs.append([fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])])
        
        # Handle single demographic case
        if len(available_demographics) == 1:
            axs = [axs[0]]

        # Process each demographic type
        for i, demo_type in enumerate(available_demographics):
            demo_data = demographic_data[demo_type]
            
            # Skip if no data for this demographic
            if not demo_data:
                for ax in axs[i]:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {demo_type} data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                continue
            
            # Format titles
            title_map = {
                "user_type": "User Type",
                "department": "Department",
                "experience": "Experience Level",
            }
            
            # 1. Average Metrics by Demographic (Left Column)
            # Prepare data for grouped bar chart
            categories = []
            ideas_per_user = []
            steps_per_user = []
            
            for category, metrics in demo_data.items():
                if metrics["count"] == 0:
                    continue
                
                categories.append(category)
                ideas_per_user.append(metrics.get("avg_ideas_per_user", 0))
                steps_per_user.append(metrics.get("avg_steps_per_user", 0))
            
            # Sort by user count (typically more informative)
            if categories:
                # Prepare data for the grouped bar chart utility
                data = {}
                for i, category in enumerate(categories):
                    data[category] = {
                        "Avg. Ideas per User": ideas_per_user[i],
                        "Avg. Steps per User": steps_per_user[i]
                    }
                
                # Get sorted categories by count
                user_counts = [demo_data[cat]["count"] for cat in categories]
                sorted_indices = np.argsort(user_counts)[::-1]  # Sort descending
                
                # Get top categories if there are many
                max_categories = 8
                if len(categories) > max_categories:
                    sorted_indices = sorted_indices[:max_categories]
                
                # Create filtered data dict with top categories
                sorted_categories = [categories[j] for j in sorted_indices]
                filtered_data = {cat: data[cat] for cat in sorted_categories if cat in data}
                
                # Use plot_grouped_bars utility function
                if filtered_data:
                    plot_grouped_bars(
                        axs[i][0],
                        data=filtered_data,
                        colors=["#4C72B0", "#55A868"],  # Blue, Green
                        add_data_labels=True,
                    )
                    
                    configure_axes(
                        axs[i][0],
                        title=f"Engagement by {title_map.get(demo_type, demo_type)}",
                        xlabel=title_map.get(demo_type, demo_type),
                        ylabel="Average Count",
                        xticklabel_rotation=45,
                    )
                else:
                    axs[i][0].text(
                        0.5,
                        0.5,
                        f"No {demo_type} metric data available",
                        ha="center",
                        va="center",
                        transform=axs[i][0].transAxes,
                    )
            else:
                axs[i][0].text(
                    0.5,
                    0.5,
                    f"No {demo_type} metric data available",
                    ha="center",
                    va="center",
                    transform=axs[i][0].transAxes,
                )
            
            # 2. Engagement Level Distribution (Right Column)
            # Prepare data for stacked percentages
            engagement_data = {}
            
            for category, metrics in demo_data.items():
                if metrics["count"] == 0:
                    continue
                
                high = metrics["engagement_levels"].get("HIGH", 0)
                medium = metrics["engagement_levels"].get("MEDIUM", 0)
                low = metrics["engagement_levels"].get("LOW", 0)
                total = max(1, high + medium + low)  # Avoid division by zero
                
                engagement_data[category] = {
                    "High": (high / total) * 100,
                    "Medium": (medium / total) * 100,
                    "Low": (low / total) * 100,
                }
            
            # Use the same sorted categories for consistency
            if sorted_categories and engagement_data:
                filtered_engagement = {
                    cat: engagement_data.get(cat, {"High": 0, "Medium": 0, "Low": 0})
                    for cat in sorted_categories
                }
                
                # Use plot_stacked_bars utility function
                plot_stacked_bars(
                    axs[i][1],
                    data=filtered_engagement,
                    colors=["#28A745", "#FFC107", "#DC3545"],  # Green, Yellow, Red
                    add_data_labels=True,
                    data_label_format="{:.0f}%",
                )
                
                configure_axes(
                    axs[i][1],
                    title=f"Engagement Levels by {title_map.get(demo_type, demo_type)}",
                    xlabel=title_map.get(demo_type, demo_type),
                    ylabel="Percentage",
                    xticklabel_rotation=45,
                )
                
                # Add legend
                axs[i][1].legend(["High", "Medium", "Low"], loc="upper right")
            else:
                axs[i][1].text(
                    0.5,
                    0.5,
                    f"No {demo_type} engagement data available",
                    ha="center",
                    va="center",
                    transform=axs[i][1].transAxes,
                )

        # Adjust layout
        plt.tight_layout()

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_burst_activity_visualization(
        self,
        burst_data: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of burst activity patterns.

        Args:
            burst_data: Burst activity data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get burst data from analyzer if not provided
        if burst_data is None:
            burst_data = self._analyzer.calculate_user_burst_activity()

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top left
            fig.add_subplot(gs[0, 1]),  # Top right
            fig.add_subplot(gs[1, 0]),  # Bottom left
            fig.add_subplot(gs[1, 1]),  # Bottom right
        ]
        
        # Format for easier access
        axs_flat = axs

        # Check if data is available
        if not burst_data:
            for ax in axs_flat:
                ax.text(
                    0.5,
                    0.5,
                    "No burst activity data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # 1. Burst Hour Distribution (Top Left)
        hour_distribution = burst_data.get("burst_hour_distribution", {})
        
        if hour_distribution:
            # Convert to 24-hour format with proper ordering
            hour_data = {}
            for hour in range(24):
                hour_str = str(hour)
                if hour_str in hour_distribution:
                    hour_data[f"{hour:02d}:00"] = hour_distribution[hour_str]
                else:
                    hour_data[f"{hour:02d}:00"] = 0
            
            # Select a subset of hours for readability (every 3 hours)
            plot_data = {}
            for i, (hour, count) in enumerate(hour_data.items()):
                if i % 3 == 0:  # Every 3 hours
                    plot_data[hour] = count * 100  # Convert to percentage
            
            # Use plot_bar utility function
            plot_bar(
                axs_flat[0],
                data=plot_data,
                color=get_color_palette("sequential_blue", len(plot_data)),
            )
            
            configure_axes(
                axs_flat[0],
                title="Burst Activity by Hour of Day",
                xlabel="Hour",
                ylabel="Frequency (%)",
                xticklabel_rotation=45,
            )
        else:
            axs_flat[0].text(
                0.5,
                0.5,
                "No hour distribution data available",
                ha="center",
                va="center",
                transform=axs_flat[0].transAxes,
            )

        # 2. Burst Day Distribution (Top Right)
        day_distribution = burst_data.get("burst_day_distribution", {})
        
        if day_distribution:
            # Ensure proper day order
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_data = {day: day_distribution.get(day, 0) * 100 for day in days_order}  # Convert to percentage
            
            # Use plot_bar utility function
            plot_bar(
                axs_flat[1],
                data=day_data,
                color=get_color_palette("categorical_main", len(day_data)),
            )
            
            configure_axes(
                axs_flat[1],
                title="Burst Activity by Day of Week",
                xlabel="Day",
                ylabel="Frequency (%)",
            )
            
            # Add weekend vs weekday
            weekend_data = burst_data.get("day_of_week", {}).get("weekend_vs_weekday", {})
            if weekend_data:
                weekend_pct = weekend_data.get("weekend_percentage", 0) * 100
                weekday_pct = 100 - weekend_pct
                
                # Add text annotation
                axs_flat[1].text(
                    0.5,
                    0.95,
                    f"Weekend: {weekend_pct:.1f}%, Weekday: {weekday_pct:.1f}%",
                    ha="center",
                    va="top",
                    transform=axs_flat[1].transAxes,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
                )
        else:
            axs_flat[1].text(
                0.5,
                0.5,
                "No day distribution data available",
                ha="center",
                va="center",
                transform=axs_flat[1].transAxes,
            )

        # 3. Burst Length Distribution (Bottom Left)
        length_distribution = burst_data.get("burst_length_distribution", {})
        
        if length_distribution:
            # Ensure proper order
            length_order = ["<1h", "1-3h", "3-6h", "6-12h", ">12h"]
            length_data = {length: length_distribution.get(length, 0) for length in length_order if length in length_distribution}
            
            # Use plot_bar utility function
            plot_bar(
                axs_flat[2],
                data=length_data,
                color=get_color_palette("sequential_green", len(length_data)),
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[2],
                title="Burst Length Distribution",
                xlabel="Duration",
                ylabel="Number of Bursts",
            )
        else:
            axs_flat[2].text(
                0.5,
                0.5,
                "No length distribution data available",
                ha="center",
                va="center",
                transform=axs_flat[2].transAxes,
            )

        # 4. Common Burst Patterns (Bottom Right)
        burst_patterns = burst_data.get("burst_patterns", [])
        
        if burst_patterns:
            # Prepare data for table
            table_data = []
            for pattern in burst_patterns[:5]:  # Top 5 patterns
                steps = pattern.get("steps", [])
                count = pattern.get("count", 0)
                frequency = pattern.get("frequency", 0)
                
                # Format steps for display
                step_str = " → ".join(steps)
                if len(step_str) > 25:
                    step_str = step_str[:22] + "..."
                
                table_data.append([step_str, count, f"{frequency:.1%}"])
            
            # Create a table using utility function
            if table_data:
                add_data_table(
                    axs_flat[3],
                    data=table_data,
                    col_labels=["Pattern", "Count", "Frequency"],
                    title="Common Burst Patterns",
                    fontsize=9,
                    loc="center",
                )
                axs_flat[3].axis("off")
            else:
                axs_flat[3].text(
                    0.5,
                    0.5,
                    "No common patterns found",
                    ha="center",
                    va="center",
                    transform=axs_flat[3].transAxes,
                )
        else:
            axs_flat[3].text(
                0.5,
                0.5,
                "No burst pattern data available",
                ha="center",
                va="center",
                transform=axs_flat[3].transAxes,
            )

        # Add summary metrics as text
        avg_bursts_per_user = burst_data.get("avg_bursts_per_user", 0)
        avg_activities_per_burst = burst_data.get("avg_activities_per_burst", 0)
        
        metrics_text = (
            f"Average Bursts per User: {avg_bursts_per_user:.1f}\n"
            f"Average Activities per Burst: {avg_activities_per_burst:.1f}"
        )
        
        fig.text(
            0.5,
            0.01,
            metrics_text,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_semester_comparison_visualization(
        self,
        comparison_data: Optional[Dict[str, Any]] = None,
        semesters: Optional[Tuple[str, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization comparing engagement between semesters.

        Args:
            comparison_data: Semester comparison data (if None, fetched from analyzer)
            semesters: Optional tuple of semesters to compare (if None, use latest)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # If data not provided, get it from analyzer using enums
        if comparison_data is None:
            from ..data.models.enums import Semester
            if semesters is None:
                # Default to comparing FALL_2023 and SPRING_2024
                comparison_data = self._analyzer.compare_semester_engagement(
                    semester1=Semester.FALL_2023,
                    semester2=Semester.SPRING_2024,
                )
            else:
                # Convert string semester names to enum values
                semester_map = {
                    "Fall 2023": Semester.FALL_2023,
                    "Spring 2024": Semester.SPRING_2024,
                    "Fall 2024": Semester.FALL_2024,
                    "Spring 2025": Semester.SPRING_2025,
                }
                comparison_data = self._analyzer.compare_semester_engagement(
                    semester1=semester_map.get(semesters[0], Semester.FALL_2023),
                    semester2=semester_map.get(semesters[1], Semester.SPRING_2024),
                )

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top left
            fig.add_subplot(gs[0, 1]),  # Top right
            fig.add_subplot(gs[1, 0]),  # Bottom left
            fig.add_subplot(gs[1, 1]),  # Bottom right
        ]
        
        # Format for easier access
        axs_flat = axs

        # Check if comparison data is available
        engagement_metrics = comparison_data.get("engagement_metrics", {})
        if not engagement_metrics or len(engagement_metrics) < 2:
            for ax in axs_flat:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient semester comparison data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # Get the semester names
        semesters = list(engagement_metrics.keys())

        # 1. User Count Comparison (Top Left)
        user_counts = comparison_data.get("user_counts", {})
        
        if user_counts:
            # Use plot_bar utility function
            plot_bar(
                axs_flat[0],
                data=user_counts,
                color=get_color_palette("categorical_main", len(user_counts)),
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[0],
                title="User Count by Semester",
                xlabel="Semester",
                ylabel="Number of Users",
            )
        else:
            axs_flat[0].text(
                0.5,
                0.5,
                "No user count data available",
                ha="center",
                va="center",
                transform=axs_flat[0].transAxes,
            )

        # 2. Average Ideas and Steps per User (Top Right)
        if engagement_metrics:
            # Prepare data for grouped bar chart
            ideas_per_user = {sem: metrics.get("avg_ideas_per_user", 0) 
                             for sem, metrics in engagement_metrics.items()}
            steps_per_user = {sem: metrics.get("avg_steps_per_user", 0) 
                             for sem, metrics in engagement_metrics.items()}
            
            # Create dictionary format for the plot_grouped_bars utility
            data = {}
            for semester in semesters:
                data[semester] = {
                    "Avg. Ideas per User": ideas_per_user[semester],
                    "Avg. Steps per User": steps_per_user[semester]
                }
            
            # Use plot_grouped_bars utility function
            plot_grouped_bars(
                axs_flat[1],
                data=data,
                colors=["#4C72B0", "#55A868"],  # Blue, Green
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[1],
                title="Engagement per User by Semester",
                xlabel="Semester",
                ylabel="Average Count",
            )
        else:
            axs_flat[1].text(
                0.5,
                0.5,
                "No engagement metrics available",
                ha="center",
                va="center",
                transform=axs_flat[1].transAxes,
            )

        # 3. Engagement Level Distribution (Bottom Left)
        if engagement_metrics:
            # Prepare data for stacked bar chart
            engagement_levels = {}
            
            for semester in semesters:
                if "engagement_levels" in engagement_metrics[semester]:
                    levels = engagement_metrics[semester]["engagement_levels"]
                    engagement_levels[semester] = {
                        "High": levels.get("high", 0),
                        "Medium": levels.get("medium", 0),
                        "Low": levels.get("low", 0),
                    }
            
            if engagement_levels:
                # Use plot_stacked_bars utility function
                plot_stacked_bars(
                    axs_flat[2],
                    data=engagement_levels,
                    colors=["#28A745", "#FFC107", "#DC3545"],  # Green, Yellow, Red
                    add_data_labels=True,
                )
                
                configure_axes(
                    axs_flat[2],
                    title="Engagement Levels by Semester",
                    xlabel="Semester",
                    ylabel="Number of Users",
                )
                
                # Add legend
                axs_flat[2].legend(["High", "Medium", "Low"], loc="upper right")
            else:
                axs_flat[2].text(
                    0.5,
                    0.5,
                    "No engagement level data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[2].transAxes,
                )
        else:
            axs_flat[2].text(
                0.5,
                0.5,
                "No engagement level data available",
                ha="center",
                va="center",
                transform=axs_flat[2].transAxes,
            )

        # 4. Tool Version Impact (Bottom Right)
        tool_version_impact = comparison_data.get("tool_version_impact", {})
        
        if tool_version_impact:
            # Create a table to show the tool versions and key metrics
            table_data = []
            
            for semester in semesters:
                sem_data = tool_version_impact.get(semester, {})
                tool_version = sem_data.get("tool_version", "Unknown")
                
                # Get metrics
                metrics = engagement_metrics.get(semester, {})
                ideas_per_user = metrics.get("avg_ideas_per_user", 0)
                steps_per_user = metrics.get("avg_steps_per_user", 0)
                idea_step_conversion = metrics.get("idea_to_step_conversion", 0)
                
                table_data.append([
                    semester,
                    tool_version,
                    f"{ideas_per_user:.1f}",
                    f"{steps_per_user:.1f}",
                    f"{idea_step_conversion:.0%}"
                ])
            
            # Create the table using utility function
            if table_data:
                add_data_table(
                    axs_flat[3],
                    data=table_data,
                    col_labels=["Semester", "Tool Version", "Avg Ideas", "Avg Steps", "Conversion"],
                    title="Tool Version Impact",
                    fontsize=9,
                    loc="center",
                )
                axs_flat[3].axis("off")
            else:
                axs_flat[3].text(
                    0.5,
                    0.5,
                    "No tool version data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[3].transAxes,
                )
        else:
            axs_flat[3].text(
                0.5,
                0.5,
                "No tool version data available",
                ha="center",
                va="center",
                transform=axs_flat[3].transAxes,
            )

        # Add comparison metrics as text if available
        comparison = comparison_data.get("comparison", {})
        
        if comparison:
            idea_diff = comparison.get("idea_difference", {})
            step_diff = comparison.get("step_difference", {})
            per_user = comparison.get("per_user", {})
            conversion_diff = comparison.get("conversion_difference", 0)
            
            metrics_text = (
                f"Ideas: {idea_diff.get('total', 0):+.0f} ({idea_diff.get('percent', 0):+.1f}%)\n"
                f"Steps: {step_diff.get('total', 0):+.0f} ({step_diff.get('percent', 0):+.1f}%)\n"
                f"Ideas per User: {per_user.get('ideas', 0):+.1f}, Steps per User: {per_user.get('steps', 0):+.1f}"
            )
            
            fig.text(
                0.5,
                0.01,
                f"Semester-to-Semester Changes: {metrics_text}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
            )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_tool_version_impact_visualization(
        self,
        version_data: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of tool version impact on engagement.

        Args:
            version_data: Tool version impact data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get version data from analyzer if not provided
        if version_data is None:
            version_data = self._analyzer.analyze_tool_version_impact()

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top left
            fig.add_subplot(gs[0, 1]),  # Top right
            fig.add_subplot(gs[1, 0]),  # Bottom left
            fig.add_subplot(gs[1, 1]),  # Bottom right
        ]
        
        # Format for easier access
        axs_flat = axs

        # Check if version comparison data is available
        version_comparison = version_data.get("version_comparison", {})
        if not version_comparison or len(version_comparison) < 2:
            for ax in axs_flat:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient version comparison data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # Get the version names
        versions = list(version_comparison.keys())

        # 1. User Count by Version (Top Left)
        user_counts = {ver: data.get("user_count", 0) for ver, data in version_comparison.items()}
        
        if any(user_counts.values()):
            # Use plot_bar utility function
            plot_bar(
                axs_flat[0],
                data=user_counts,
                color=get_color_palette("tool_versions", len(user_counts)),
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[0],
                title="User Count by Tool Version",
                xlabel="Tool Version",
                ylabel="Number of Users",
            )
        else:
            axs_flat[0].text(
                0.5,
                0.5,
                "No user count data available",
                ha="center",
                va="center",
                transform=axs_flat[0].transAxes,
            )

        # 2. Average Ideas and Steps per User (Top Right)
        if version_comparison:
            # Prepare data for grouped bar chart
            ideas_per_user = {ver: data.get("avg_ideas_per_user", 0) 
                             for ver, data in version_comparison.items()}
            steps_per_user = {ver: data.get("avg_steps_per_user", 0) 
                             for ver, data in version_comparison.items()}
            
            # Create dictionary format for the plot_grouped_bars utility
            data = {}
            for version in versions:
                data[version] = {
                    "Avg. Ideas per User": ideas_per_user[version],
                    "Avg. Steps per User": steps_per_user[version]
                }
            
            # Use plot_grouped_bars utility function
            plot_grouped_bars(
                axs_flat[1],
                data=data,
                colors=["#4C72B0", "#55A868"],  # Blue, Green
                add_data_labels=True,
            )
            
            configure_axes(
                axs_flat[1],
                title="Engagement per User by Version",
                xlabel="Tool Version",
                ylabel="Average Count",
            )
        else:
            axs_flat[1].text(
                0.5,
                0.5,
                "No engagement metrics available",
                ha="center",
                va="center",
                transform=axs_flat[1].transAxes,
            )

        # 3. Idea-to-Step Conversion Rate (Bottom Left)
        if version_comparison:
            conversion_rates = {ver: data.get("idea_to_step_conversion", 0) * 100
                               for ver, data in version_comparison.items()}
            
            if any(conversion_rates.values()):
                # Use plot_bar utility function
                plot_bar(
                    axs_flat[2],
                    data=conversion_rates,
                    color=get_color_palette("tool_versions", len(conversion_rates)),
                    add_data_labels=True,
                    data_label_format="{:.1f}%",
                )
                
                configure_axes(
                    axs_flat[2],
                    title="Idea-to-Step Conversion Rate by Version",
                    xlabel="Tool Version",
                    ylabel="Conversion Rate (%)",
                )
            else:
                axs_flat[2].text(
                    0.5,
                    0.5,
                    "No conversion rate data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[2].transAxes,
                )
        else:
            axs_flat[2].text(
                0.5,
                0.5,
                "No conversion rate data available",
                ha="center",
                va="center",
                transform=axs_flat[2].transAxes,
            )

        # 4. Feature Impact Analysis (Bottom Right)
        feature_impact = version_data.get("feature_impact", {})
        cohort_metrics = version_data.get("cohort_metrics", {})
        
        if feature_impact or cohort_metrics:
            # Create a combined table for feature impact and cohort metrics
            table_data = []
            
            # Add version comparison summary
            for version in versions:
                ver_data = version_comparison.get(version, {})
                
                # Get metrics
                user_count = ver_data.get("user_count", 0)
                high_engagement_pct = ver_data.get("high_engagement_percentage", 0) * 100
                
                table_data.append([
                    f"Version {version}",
                    f"{user_count}",
                    f"{high_engagement_pct:.1f}%"
                ])
            
            # Add feature impact data
            if feature_impact:
                for feature, impact in feature_impact.items():
                    avail = impact.get("availability", "Unknown")
                    usage = impact.get("usage_rate", 0) * 100
                    
                    table_data.append([
                        f"Feature: {feature}",
                        f"Available in: {avail}",
                        f"Usage: {usage:.1f}%"
                    ])
            
            # Create the table using utility function
            if table_data:
                add_data_table(
                    axs_flat[3],
                    data=table_data,
                    col_labels=["Item", "Value", "Engagement/Usage"],
                    title="Version & Feature Impact",
                    fontsize=9,
                    loc="center",
                )
                axs_flat[3].axis("off")
            else:
                axs_flat[3].text(
                    0.5,
                    0.5,
                    "No feature impact data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[3].transAxes,
                )
        else:
            axs_flat[3].text(
                0.5,
                0.5,
                "No feature impact data available",
                ha="center",
                va="center",
                transform=axs_flat[3].transAxes,
            )

        # Add version comparison summary as text
        if version_comparison and len(versions) >= 2:
            latest_ver = versions[-1]
            prior_ver = versions[-2]
            
            if latest_ver != "none" and prior_ver != "none":
                latest_data = version_comparison.get(latest_ver, {})
                prior_data = version_comparison.get(prior_ver, {})
                
                # Calculate key improvements
                ideas_improvement = (
                    latest_data.get("avg_ideas_per_user", 0) / 
                    max(0.01, prior_data.get("avg_ideas_per_user", 0)) - 1
                ) * 100
                
                steps_improvement = (
                    latest_data.get("avg_steps_per_user", 0) / 
                    max(0.01, prior_data.get("avg_steps_per_user", 0)) - 1
                ) * 100
                
                conversion_improvement = (
                    latest_data.get("idea_to_step_conversion", 0) / 
                    max(0.01, prior_data.get("idea_to_step_conversion", 0)) - 1
                ) * 100
                
                metrics_text = (
                    f"Version {latest_ver} vs {prior_ver}: "
                    f"Ideas per User: {ideas_improvement:+.1f}%, "
                    f"Steps per User: {steps_improvement:+.1f}%, "
                    f"Conversion Rate: {conversion_improvement:+.1f}%"
                )
                
                fig.text(
                    0.5,
                    0.01,
                    metrics_text,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
                )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_usage_pattern_visualization(
        self,
        usage_data: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of usage patterns.

        Args:
            usage_data: Usage pattern data (if None, fetched from analyzer)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get usage data from analyzer if not provided
        if usage_data is None:
            usage_data = self._analyzer.get_usage_pattern_analysis()

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, 0]),  # Top left
            fig.add_subplot(gs[0, 1]),  # Top right
            fig.add_subplot(gs[1, 0]),  # Bottom left
            fig.add_subplot(gs[1, 1]),  # Bottom right
        ]
        
        # Format for easier access
        axs_flat = axs

        # Check if data is available
        if not usage_data:
            for ax in axs_flat:
                ax.text(
                    0.5,
                    0.5,
                    "No usage pattern data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, axs

        # 1. Time of Day Distribution (Top Left)
        time_of_day = usage_data.get("time_of_day", {})
        
        if time_of_day:
            hour_distribution = time_of_day.get("hour_distribution", {})
            
            if hour_distribution:
                # Convert to 24-hour format with proper ordering
                hour_data = {}
                for hour in range(24):
                    hour_str = str(hour)
                    if hour_str in hour_distribution:
                        hour_data[f"{hour:02d}:00"] = hour_distribution[hour_str] * 100  # Convert to percentage
                    else:
                        hour_data[f"{hour:02d}:00"] = 0
                
                # Select a subset of hours for readability (every 3 hours)
                plot_data = {}
                for i, (hour, count) in enumerate(hour_data.items()):
                    if i % 3 == 0:  # Every 3 hours
                        plot_data[hour] = count
                
                # Use plot_bar utility function
                plot_bar(
                    axs_flat[0],
                    data=plot_data,
                    color=get_color_palette("sequential_blue", len(plot_data)),
                )
                
                configure_axes(
                    axs_flat[0],
                    title="Activity by Hour of Day",
                    xlabel="Hour",
                    ylabel="Percentage (%)",
                    xticklabel_rotation=45,
                )
            else:
                axs_flat[0].text(
                    0.5,
                    0.5,
                    "No hour distribution data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[0].transAxes,
                )
        else:
            axs_flat[0].text(
                0.5,
                0.5,
                "No time of day data available",
                ha="center",
                va="center",
                transform=axs_flat[0].transAxes,
            )

        # 2. Day of Week Distribution (Top Right)
        day_of_week = usage_data.get("day_of_week", {})
        
        if day_of_week:
            day_distribution = day_of_week.get("day_distribution", {})
            
            if day_distribution:
                # Ensure proper day order
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_data = {day: day_distribution.get(day, 0) * 100 for day in days_order}  # Convert to percentage
                
                # Use plot_bar utility function
                plot_bar(
                    axs_flat[1],
                    data=day_data,
                    color=get_color_palette("categorical_main", len(day_data)),
                )
                
                configure_axes(
                    axs_flat[1],
                    title="Activity by Day of Week",
                    xlabel="Day",
                    ylabel="Percentage (%)",
                )
                
                # Add weekend vs weekday
                weekend_data = day_of_week.get("weekend_vs_weekday", {})
                if weekend_data:
                    weekend_pct = weekend_data.get("weekend_percentage", 0) * 100
                    weekday_pct = 100 - weekend_pct
                    
                    # Add text annotation
                    axs_flat[1].text(
                        0.5,
                        0.95,
                        f"Weekend: {weekend_pct:.1f}%, Weekday: {weekday_pct:.1f}%",
                        ha="center",
                        va="top",
                        transform=axs_flat[1].transAxes,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
                    )
            else:
                axs_flat[1].text(
                    0.5,
                    0.5,
                    "No day distribution data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[1].transAxes,
                )
        else:
            axs_flat[1].text(
                0.5,
                0.5,
                "No day of week data available",
                ha="center",
                va="center",
                transform=axs_flat[1].transAxes,
            )

        # 3. Session Length Distribution (Bottom Left)
        session_length = usage_data.get("session_length", {})
        
        if session_length:
            duration_categories = session_length.get("duration_categories", {})
            
            if duration_categories:
                # Ensure proper order
                duration_order = ["under_5min", "5_15min", "15_30min", "30_60min", "1_3hr", "over_3hr"]
                duration_labels = {
                    "under_5min": "< 5 min",
                    "5_15min": "5-15 min",
                    "15_30min": "15-30 min",
                    "30_60min": "30-60 min",
                    "1_3hr": "1-3 hrs",
                    "over_3hr": "> 3 hrs",
                }
                
                duration_data = {
                    duration_labels.get(dur, dur): duration_categories.get(dur, 0)
                    for dur in duration_order if dur in duration_categories
                }
                
                # Use plot_bar utility function
                plot_bar(
                    axs_flat[2],
                    data=duration_data,
                    color=get_color_palette("sequential_green", len(duration_data)),
                    add_data_labels=True,
                )
                
                configure_axes(
                    axs_flat[2],
                    title="Session Length Distribution",
                    xlabel="Duration",
                    ylabel="Number of Sessions",
                )
                
                # Add average duration
                avg_duration = session_length.get("avg_duration_minutes", 0)
                session_count = session_length.get("session_count", 0)
                
                if avg_duration > 0:
                    # Format average duration for display
                    if avg_duration < 60:
                        avg_str = f"{avg_duration:.1f} minutes"
                    else:
                        hours = avg_duration / 60
                        avg_str = f"{hours:.1f} hours"
                    
                    # Add text annotation
                    axs_flat[2].text(
                        0.5,
                        0.95,
                        f"Average: {avg_str}, Total: {session_count} sessions",
                        ha="center",
                        va="top",
                        transform=axs_flat[2].transAxes,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
                    )
            else:
                axs_flat[2].text(
                    0.5,
                    0.5,
                    "No duration category data available",
                    ha="center",
                    va="center",
                    transform=axs_flat[2].transAxes,
                )
        else:
            axs_flat[2].text(
                0.5,
                0.5,
                "No session length data available",
                ha="center",
                va="center",
                transform=axs_flat[2].transAxes,
            )

        # 4. Common Step Sequences (Bottom Right)
        common_sequences = usage_data.get("common_sequences", [])
        
        if common_sequences:
            # Create a table for common sequences
            table_data = []
            
            for i, sequence in enumerate(common_sequences[:5]):  # Top 5 sequences
                steps = sequence.get("steps", [])
                count = sequence.get("count", 0)
                frequency = sequence.get("frequency", 0)
                
                # Format steps for display
                step_str = " → ".join(steps)
                if len(step_str) > 30:
                    step_str = step_str[:27] + "..."
                
                table_data.append([f"{i+1}.", step_str, f"{count}", f"{frequency:.1%}"])
            
            # Create a table using utility function
            if table_data:
                add_data_table(
                    axs_flat[3],
                    data=table_data,
                    col_labels=["#", "Common Step Sequence", "Count", "Frequency"],
                    title="Most Common Step Sequences",
                    fontsize=9,
                    loc="center",
                )
                axs_flat[3].axis("off")
            else:
                axs_flat[3].text(
                    0.5,
                    0.5,
                    "No common sequences found",
                    ha="center",
                    va="center",
                    transform=axs_flat[3].transAxes,
                )
        else:
            axs_flat[3].text(
                0.5,
                0.5,
                "No common sequence data available",
                ha="center",
                va="center",
                transform=axs_flat[3].transAxes,
            )

        # Add usage consistency information as text if available
        usage_consistency = usage_data.get("usage_consistency", {})
        
        if usage_consistency and "avg_activity_rate" in usage_consistency:
            avg_rate = usage_consistency.get("avg_activity_rate", 0)
            user_count = usage_consistency.get("user_count", 0)
            
            user_consistency = usage_consistency.get("user_consistency", {})
            consistent = user_consistency.get("consistent_users", 0)
            sporadic = user_consistency.get("sporadic_users", 0)
            infrequent = user_consistency.get("infrequent_users", 0)
            
            metrics_text = (
                f"Avg. Activity Rate: {avg_rate:.1%}, Users Analyzed: {user_count}\n"
                f"Consistent Users: {consistent}, Sporadic: {sporadic}, Infrequent: {infrequent}"
            )
            
            fig.text(
                0.5,
                0.01,
                metrics_text,
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
            )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, axs

    def create_user_timeline_visualization(
        self,
        timeline_data: Optional[Dict[str, Any]] = None,
        user_email: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create visualization of a user's activity timeline.

        Args:
            timeline_data: User activity timeline data (if None, fetched from analyzer)
            user_email: Email of the user to visualize (required if timeline_data is None)
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get timeline data from analyzer if not provided
        if timeline_data is None:
            if user_email is None:
                raise ValueError("User email must be provided if timeline_data is None")
            
            timeline_data = self._analyzer.get_user_activity_timeline(
                email=user_email,
                include_ideas=True,
                include_steps=True,
            )

        # Set up figure using utility function
        if figsize is None:
            figsize = (self._default_figsize[0], self._default_figsize[1] * 1.2)

        # Create figure using utility function
        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        
        # Create custom grid for timeline and summary plots
        gs = plt.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
        
        # Create the axes
        ax_timeline = fig.add_subplot(gs[0, :])  # Main timeline (entire top row)
        ax_ideas = fig.add_subplot(gs[1, 0])     # Ideas summary (middle left)
        ax_steps = fig.add_subplot(gs[1, 1])     # Steps summary (middle center)
        ax_sessions = fig.add_subplot(gs[1, 2])  # Sessions summary (middle right)
        ax_summary = fig.add_subplot(gs[2, :])   # Engagement summary (bottom row)

        # Access relevant data sections
        user_info = timeline_data.get("user_info", {})
        activity_timeline = timeline_data.get("activity_timeline", [])
        session_data = timeline_data.get("session_data", [])
        engagement_summary = timeline_data.get("engagement_summary", {})

        # Check if data is available
        if not activity_timeline:
            for ax in [ax_timeline, ax_ideas, ax_steps, ax_sessions, ax_summary]:
                ax.text(
                    0.5,
                    0.5,
                    "No timeline data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            
            if save_path:
                save_figure(fig, save_path)
                
            if not show_fig:
                plt.close(fig)
                
            return fig, [ax_timeline, ax_ideas, ax_steps, ax_sessions, ax_summary]

        # 1. Main Activity Timeline (Top)
        
        # Prepare data for timeline
        events_for_timeline = []
        
        for event in activity_timeline:
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("timestamp")
            details = event.get("details", {})
            
            if timestamp:
                # Create event for timeline display
                events_for_timeline.append({
                    "date": timestamp,
                    "category": event_type,
                    "label": self._get_event_label(event_type, details)
                })
        
        # Sort events by date
        events_for_timeline.sort(key=lambda x: x["date"])
        
        # Create timeline visualization - can't directly use plot_timeline here because
        # we need a stacked area chart for activity counts
        if events_for_timeline:
            # Determine date range
            dates = [event["date"] for event in events_for_timeline]
            min_date = min(dates)
            max_date = max(dates)
            
            # Group events by date
            events_by_date = {}
            for event in events_for_timeline:
                date_key = event["date"].strftime("%Y-%m-%d")
                if date_key not in events_by_date:
                    events_by_date[date_key] = []
                events_by_date[date_key].append(event)
            
            # Count events by type for each date
            date_list = []
            idea_counts = []
            step_counts = []
            other_counts = []
            
            for date_key in sorted(events_by_date.keys()):
                date_events = events_by_date[date_key]
                date_list.append(datetime.strptime(date_key, "%Y-%m-%d"))
                
                # Count by type
                idea_count = sum(1 for e in date_events if e["category"] == "idea_creation")
                step_count = sum(1 for e in date_events if e["category"] == "step_creation")
                other_count = sum(1 for e in date_events if e["category"] not in ["idea_creation", "step_creation"])
                
                idea_counts.append(idea_count)
                step_counts.append(step_count)
                other_counts.append(other_count)
            
            # Create stacked area chart
            ax_timeline.fill_between(date_list, 0, idea_counts, label="Ideas", alpha=0.7, color="#FFC107")
            ax_timeline.fill_between(date_list, idea_counts, [idea_counts[i] + step_counts[i] for i in range(len(idea_counts))], 
                                     label="Steps", alpha=0.7, color="#007BFF")
            
            # Add other events if any
            if any(other_counts):
                ax_timeline.fill_between(date_list, 
                                         [idea_counts[i] + step_counts[i] for i in range(len(idea_counts))],
                                         [idea_counts[i] + step_counts[i] + other_counts[i] for i in range(len(idea_counts))],
                                         label="Other", alpha=0.7, color="#28A745")
            
            # Configure timeline
            ax_timeline.set_title(f"Activity Timeline for {user_info.get('name', 'User')}")
            ax_timeline.set_xlabel("Date")
            ax_timeline.set_ylabel("Number of Activities")
            ax_timeline.legend(loc="upper right")
            
            # Format the x-axis dates nicely
            ax_timeline.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax_timeline.get_xticklabels(), rotation=45, ha="right")
            
            # Mark session periods if available
            if session_data:
                for session in session_data:
                    start = session.get("start_time")
                    end = session.get("end_time")
                    
                    if start and end:
                        # Add a highlight for the session duration
                        ax_timeline.axvspan(start, end, alpha=0.2, color="red")
            
            # Add grid
            ax_timeline.grid(True, linestyle="--", alpha=0.7)
        else:
            ax_timeline.text(
                0.5,
                0.5,
                "No timeline events available",
                ha="center",
                va="center",
                transform=ax_timeline.transAxes,
            )

        # 2. Ideas Summary (Middle Left)
        idea_events = [e for e in activity_timeline if e["event_type"] == "idea_creation"]
        
        if idea_events:
            idea_count = len(idea_events)
            
            # Create a simple pie chart - no direct util function for this
            ax_ideas.pie(
                [idea_count, max(0, 1 - idea_count)],  # Dummy slice if only 1
                labels=[f"{idea_count} Ideas", ""] if idea_count > 0 else ["No Ideas", ""],
                autopct="%1.1f%%" if idea_count > 0 else None,
                colors=["#FFC107", "#f8f9fa"],
                startangle=90,
                wedgeprops={"edgecolor": "w", "linewidth": 1},
            )
            ax_ideas.set_title("Ideas Created")
        else:
            ax_ideas.text(
                0.5,
                0.5,
                "No idea data available",
                ha="center",
                va="center",
                transform=ax_ideas.transAxes,
            )

        # 3. Steps Summary (Middle Center)
        step_events = [e for e in activity_timeline if e["event_type"] == "step_creation"]
        
        if step_events:
            step_count = len(step_events)
            
            # Group steps by framework
            steps_by_framework = {}
            
            for event in step_events:
                details = event.get("details", {})
                framework = details.get("framework", "Unknown")
                
                if framework not in steps_by_framework:
                    steps_by_framework[framework] = 0
                
                steps_by_framework[framework] += 1
            
            # Create a pie chart of steps by framework
            if steps_by_framework:
                labels = list(steps_by_framework.keys())
                sizes = list(steps_by_framework.values())
                
                ax_steps.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=get_color_palette("categorical_main", len(labels)),
                    startangle=90,
                    wedgeprops={"edgecolor": "w", "linewidth": 1},
                )
                ax_steps.set_title(f"Steps by Framework ({step_count} Total)")
            else:
                ax_steps.text(
                    0.5,
                    0.5,
                    "No step framework data",
                    ha="center",
                    va="center",
                    transform=ax_steps.transAxes,
                )
        else:
            ax_steps.text(
                0.5,
                0.5,
                "No step data available",
                ha="center",
                va="center",
                transform=ax_steps.transAxes,
            )

        # 4. Sessions Summary (Middle Right)
        if session_data:
            session_count = len(session_data)
            
            # Group sessions by duration
            duration_categories = {
                "under_5min": 0,
                "5_15min": 0,
                "15_30min": 0,
                "30_60min": 0,
                "1_3hr": 0,
                "over_3hr": 0,
            }
            
            for session in session_data:
                duration = session.get("duration_minutes", 0)
                
                if duration < 5:
                    duration_categories["under_5min"] += 1
                elif duration < 15:
                    duration_categories["5_15min"] += 1
                elif duration < 30:
                    duration_categories["15_30min"] += 1
                elif duration < 60:
                    duration_categories["30_60min"] += 1
                elif duration < 180:
                    duration_categories["1_3hr"] += 1
                else:
                    duration_categories["over_3hr"] += 1
            
            # Better labels for display
            duration_labels = {
                "under_5min": "< 5 min",
                "5_15min": "5-15 min",
                "15_30min": "15-30 min",
                "30_60min": "30-60 min",
                "1_3hr": "1-3 hrs",
                "over_3hr": "> 3 hrs",
            }
            
            # Create pie chart for session durations
            labels = []
            sizes = []
            
            for category, count in duration_categories.items():
                if count > 0:
                    labels.append(duration_labels.get(category, category))
                    sizes.append(count)
            
            if sizes:
                ax_sessions.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=get_color_palette("sequential_green", len(labels)),
                    startangle=90,
                    wedgeprops={"edgecolor": "w", "linewidth": 1},
                )
                ax_sessions.set_title(f"Session Durations ({session_count} Total)")
            else:
                ax_sessions.text(
                    0.5,
                    0.5,
                    "No session duration data",
                    ha="center",
                    va="center",
                    transform=ax_sessions.transAxes,
                )
        else:
            ax_sessions.text(
                0.5,
                0.5,
                "No session data available",
                ha="center",
                va="center",
                transform=ax_sessions.transAxes,
            )

        # 5. Engagement Summary (Bottom)
        if engagement_summary:
            # Create a table with key engagement metrics
            table_data = []
            
            # First and last activity
            first_activity = engagement_summary.get("first_activity")
            last_activity = engagement_summary.get("last_activity")
            
            if first_activity and last_activity:
                first_date = first_activity.strftime("%Y-%m-%d")
                last_date = last_activity.strftime("%Y-%m-%d")
                date_range = f"{first_date} to {last_date}"
            else:
                date_range = "Unknown"
            
            # Calculate days since last activity
            days_since_last = None
            if last_activity:
                days_since_last = (datetime.now() - last_activity).days
            
            # Active days
            days_active = engagement_summary.get("days_active", 0)
            
            # Event counts
            total_events = engagement_summary.get("total_events", 0)
            idea_count = engagement_summary.get("idea_count", 0)
            step_count = engagement_summary.get("step_count", 0)
            
            # Format for display
            table_data = [
                ["Active Period", date_range],
                ["Days Active", f"{days_active}"],
                ["Ideas Created", f"{idea_count}"],
                ["Steps Created", f"{step_count}"],
                ["Total Activities", f"{total_events}"],
            ]
            
            if days_since_last is not None:
                table_data.append(["Days Since Last Activity", f"{days_since_last}"])
            
            if "avg_steps_per_day" in engagement_summary:
                avg_steps = engagement_summary["avg_steps_per_day"]
                table_data.append(["Avg. Steps per Active Day", f"{avg_steps:.1f}"])
            
            # Calculate whether user is still active
            is_active = days_since_last is not None and days_since_last < 30
            
            # Add a status indicator
            status = "Active" if is_active else "Inactive"
            status_color = "#28A745" if is_active else "#DC3545"  # Green if active, red if not
            
            table_data.append(["Current Status", status])
            
            # Create the table using utility function
            highlight_row = len(table_data) - 1
            highlight_cells = [(highlight_row, 1, status_color)]
            
            add_data_table(
                ax_summary,
                data=table_data,
                title="Engagement Summary",
                fontsize=10,
                loc="center",
                highlight_cells=highlight_cells,
            )
            
            ax_summary.axis("off")
        else:
            ax_summary.text(
                0.5,
                0.5,
                "No engagement summary data available",
                ha="center",
                va="center",
                transform=ax_summary.transAxes,
            )

        # Add user info in title
        user_name = user_info.get("name", "User")
        user_email = user_info.get("email", "")
        user_type = user_info.get("user_type", "")
        
        title_parts = [part for part in [user_name, user_email, user_type] if part]
        title_text = " - ".join(title_parts)
        
        fig.suptitle(title_text, fontsize=14, fontweight="bold")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title

        # Save if requested using the advanced utility function
        if save_path:
            # Use the more advanced utility to save both the figure and raw data
            if user_email:
                safe_email = user_email.replace("@", "_at_").replace(".", "_")
                filename = f"user_timeline_{safe_email}"
            else:
                filename = "user_timeline"
                
            export_data_with_visualization(
                fig, 
                timeline_data, 
                filename, 
                subdirectory="user_timelines", 
                image_formats=["png", "pdf"],
                data_format="json",
                include_timestamp=self._include_timestamps,
                base_dir=self._output_dir
            )
        
        # Alternative: use simpler save_figure
        # save_figure(fig, save_path)

        # Show or close figure
        if not show_fig:
            plt.close(fig)

        return fig, [ax_timeline, ax_ideas, ax_steps, ax_sessions, ax_summary]

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
        # Create report directory structure using utility function
        report_path, subdirs = create_report_directory(
            report_name=report_name,
            base_dir=output_dir or self._output_dir,
            include_timestamp=self._include_timestamps,
        )
        
        # Store all figures for the report
        figures_path = subdirs["figures"]
        data_path = subdirs["data"]
        
        # 1. Generate Engagement Level Visualization
        self._logger.info("Generating engagement level visualization...")
        fig_levels, _ = self.create_engagement_level_visualization(
            course_id=course_id,
            show_fig=False,
        )
        # Use utility function to save figure
        save_figure(fig_levels, "engagement_levels", directory=str(figures_path))
        plt.close(fig_levels)
        
        # 2. Generate Engagement Metrics Visualization
        self._logger.info("Generating engagement metrics visualization...")
        metrics_data = self._analyzer.get_engagement_metrics_over_time()
        fig_metrics, _ = self.create_engagement_metrics_visualization(
            metrics=metrics_data,
            show_fig=False,
        )
        save_figure(fig_metrics, "engagement_metrics", directory=str(figures_path))
        plt.close(fig_metrics)
        
        # 3. Generate Demographic Analysis
        self._logger.info("Generating demographic analysis visualization...")
        demographic_data = self._analyzer.get_engagement_by_demographic(course_id=course_id)
        fig_demographics, _ = self.create_demographic_engagement_visualization(
            demographic_data=demographic_data,
            show_fig=False,
        )
        save_figure(fig_demographics, "demographic_analysis", directory=str(figures_path))
        plt.close(fig_demographics)
        
        # 4. Generate Dropout Analysis
        self._logger.info("Generating dropout analysis visualization...")
        dropout_data = self._analyzer.analyze_dropout_patterns(course_id=course_id)
        fig_dropout, _ = self.create_dropout_analysis_visualization(
            dropout_data=dropout_data,
            show_fig=False,
        )
        save_figure(fig_dropout, "dropout_analysis", directory=str(figures_path))
        plt.close(fig_dropout)
        
        # 5. Generate Usage Pattern Analysis
        self._logger.info("Generating usage pattern visualization...")
        usage_data = self._analyzer.get_usage_pattern_analysis(course_id=course_id)
        fig_usage, _ = self.create_usage_pattern_visualization(
            usage_data=usage_data,
            show_fig=False,
        )
        save_figure(fig_usage, "usage_patterns", directory=str(figures_path))
        plt.close(fig_usage)
        
        # 6. Generate Semester Comparison
        self._logger.info("Generating semester comparison visualization...")
        from ..data.models.enums import Semester
        comparison_data = self._analyzer.compare_semester_engagement(
            semester1=Semester.FALL_2023,
            semester2=Semester.SPRING_2024,
            course_id=course_id,
        )
        fig_comparison, _ = self.create_semester_comparison_visualization(
            comparison_data=comparison_data,
            show_fig=False,
        )
        save_figure(fig_comparison, "semester_comparison", directory=str(figures_path))
        plt.close(fig_comparison)
        
        # 7. Generate Tool Version Impact
        self._logger.info("Generating tool version impact visualization...")
        version_data = self._analyzer.analyze_tool_version_impact(course_id=course_id)
        fig_versions, _ = self.create_tool_version_impact_visualization(
            version_data=version_data,
            show_fig=False,
        )
        save_figure(fig_versions, "tool_version_impact", directory=str(figures_path))
        plt.close(fig_versions)
        
        # 8. Generate User Timelines (if requested)
        if include_user_details:
            self._logger.info("Generating user timeline visualizations...")
            
            # Get high engagement users
            engagement_levels = self._analyzer.classify_users_by_engagement(course_id=course_id)
            high_engagement_users = engagement_levels.get("HIGH", [])
            
            # Limit to top 5 users to avoid generating too many files
            user_sample = high_engagement_users[:5]
            
            # Create a directory for user timelines
            user_dir = figures_path / "user_timelines"
            user_dir.mkdir(exist_ok=True)
            
            for user_data in user_sample:
                email = user_data.get("email")
                if not email:
                    continue
                
                # Generate timeline visualization
                timeline_data = self._analyzer.get_user_activity_timeline(
                    email=email,
                    include_ideas=True,
                    include_steps=True,
                )
                
                # Skip if insufficient data
                if not timeline_data or "error" in timeline_data:
                    continue
                
                # Use user_timeline visualization function which already uses the advanced export
                self.create_user_timeline_visualization(
                    timeline_data=timeline_data,
                    save_path=str(user_dir),  # This will be handled by the export function
                    show_fig=False,
                )
        
        # 9. Export raw data for reference using utility functions
        self._logger.info("Exporting raw data...")
        
        # Export engagement levels data
        engagement_levels = self._analyzer.classify_users_by_engagement(course_id=course_id)
        with open(data_path / "engagement_levels.json", "w") as f:
            import json
            json.dump(engagement_levels, f, indent=2, default=str)
        
        # Export engagement metrics data
        with open(data_path / "engagement_metrics.json", "w") as f:
            import json
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Create report README with metadata
        self._create_report_readme(report_path, course_id)
        
        self._logger.info(f"Report generated at: {report_path}")
        return str(report_path)

    def _create_report_readme(self, report_path: Path, course_id: Optional[str] = None) -> None:
        """
        Create a README file for the report with metadata.

        Args:
            report_path: Path to the report directory
            course_id: Optional course ID that was used to filter data
        """
        # Create README content with summary information
        from datetime import datetime
        
        readme_content = f"""# Engagement Analysis Report

## Summary
- **Generated On:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Course ID:** {course_id or "All Courses"}

## Contents
This report contains visualizations and data analysis of user engagement with the JetPack/Orbit tool.

### Visualizations
- **engagement_levels.png**: Breakdown of users by engagement level (High, Medium, Low)
- **engagement_metrics.png**: Timeline of engagement metrics over time
- **demographic_analysis.png**: Engagement patterns by user demographics
- **dropout_analysis.png**: Analysis of user dropout patterns
- **usage_patterns.png**: Analysis of tool usage patterns
- **semester_comparison.png**: Comparison of engagement between semesters
- **tool_version_impact.png**: Impact of different tool versions on engagement

### Data
The `/data` directory contains raw data exports used to generate the visualizations.

## Methods
This report was generated using the EngagementVisualizer in conjunction with the EngagementAnalyzer.
The data represents user interactions with the JetPack/Orbit tool, focusing on engagement patterns,
dropout rates, and the impact of different tool versions.
"""
        
        # Write README file
        with open(report_path / "README.md", "w") as f:
            f.write(readme_content)

    def _get_event_label(self, event_type: str, details: Dict[str, Any]) -> str:
        """
        Generate a descriptive label for a timeline event.

        Args:
            event_type: Type of event
            details: Event details dictionary

        Returns:
            str: Descriptive label for the event
        """
        if event_type == "user_creation":
            return "User Created"
        
        elif event_type == "user_login":
            return "User Login"
        
        elif event_type == "idea_creation":
            title = details.get("title", "")
            if title:
                if len(title) > 20:
                    title = title[:17] + "..."
                return f"Idea: {title}"
            else:
                return "New Idea"
        
        elif event_type == "step_creation":
            step = details.get("step", "")
            framework = details.get("framework", "")
            
            if step:
                # Format step name for better readability
                step_display = step.replace("-", " ").title()
                
                if len(step_display) > 20:
                    step_display = step_display[:17] + "..."
                
                if framework:
                    framework_short = framework.split()[0] if " " in framework else framework
                    return f"{framework_short}: {step_display}"
                else:
                    return f"Step: {step_display}"
            else:
                return "New Step"
        
        else:
            return f"Event: {event_type}"

    def create_engagement_heatmap(
        self,
        course_id: Optional[str] = None,
        time_unit: str = "hour",
        metric: str = "all_activity",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
    ) -> Tuple[plt.Figure, Any]:
        """
        Create a heatmap visualization of engagement by time period.

        Args:
            course_id: Optional course ID to filter data
            time_unit: Time unit for analysis ('hour', 'day', 'month')
            metric: Metric to display ('all_activity', 'ideas', 'steps')
            figsize: Optional figure size (width, height) in inches
            save_path: Optional path to save the visualization
            show_fig: Whether to display the figure

        Returns:
            Tuple containing Figure and axes objects
        """
        # Get engagement data from analyzer
        engagement_data = self._analyzer.get_engagement_metrics_over_time(
            interval=time_unit if time_unit != "hour" else "day",  # 'hour' not supported directly
        )
        
        # Get additional data for hourly patterns if needed
        usage_patterns = None
        if time_unit == "hour":
            usage_patterns = self._analyzer.get_usage_pattern_analysis(course_id=course_id)
        
        # Set up figure using utility function
        if figsize is None:
            figsize = self._default_figsize
        
        fig = create_figure(
            width=figsize[0], 
            height=figsize[1], 
            theme=self._theme
        )
        ax = fig.add_subplot(111)
        
        # Prepare heatmap data based on time unit
        if time_unit == "hour" and usage_patterns:
            # For hourly heatmap, use the plot_heatmap utility but need to prepare data
            data, row_labels, col_labels = self._prepare_hourly_heatmap_data(usage_patterns)
            
            if data is not None:
                plot_heatmap(
                    ax,
                    data=data,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    color_map="viridis",
                    add_colorbar=True,
                    add_values=True,
                    value_format="{:.2f}",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data for hourly heatmap",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        
        elif time_unit == "day" and engagement_data:
            # For daily heatmap, use prepare_heatmap_data utility
            data_dict = self._prepare_daily_heatmap_data(engagement_data, metric)
            
            if data_dict:
                # Pass the prepared data to plot_heatmap
                plot_heatmap(
                    ax,
                    data=data_dict["values"],
                    row_labels=data_dict["row_labels"],
                    col_labels=data_dict["col_labels"],
                    color_map="viridis",
                    add_colorbar=True,
                    add_values=True,
                    value_format="{:.0f}",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data for daily heatmap",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        
        elif time_unit == "month" and engagement_data:
            # For monthly heatmap, use prepare_heatmap_data utility
            data_dict = self._prepare_monthly_heatmap_data(engagement_data, metric)
            
            if data_dict:
                plot_heatmap(
                    ax,
                    data=data_dict["values"],
                    row_labels=data_dict["row_labels"],
                    col_labels=data_dict["col_labels"],
                    color_map="viridis",
                    add_colorbar=True,
                    add_values=True,
                    value_format="{:.0f}",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data for monthly heatmap",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        
        else:
            ax.text(
                0.5,
                0.5,
                f"No data available for {time_unit} heatmap",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        
        # Set overall title
        metric_label = {
            "all_activity": "Overall Activity",
            "ideas": "Idea Creation",
            "steps": "Step Completion",
        }.get(metric, "Activity")
        
        period_label = {
            "hour": "Hour of Day",
            "day": "Day of Week",
            "month": "Month of Year",
        }.get(time_unit, time_unit.title())
        
        plt.suptitle(f"{metric_label} by {period_label}", fontsize=14)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save if requested using utility function
        if save_path:
            save_figure(fig, save_path)
        
        # Show or close figure
        if not show_fig:
            plt.close(fig)
        
        return fig, ax
    
    def _prepare_hourly_heatmap_data(self, usage_patterns):
        """
        Prepare data for hourly heatmap visualization using the utility functions.
        
        Returns:
            Tuple of (data, row_labels, col_labels)
        """
        # Extract time of day data
        time_of_day = usage_patterns.get("time_of_day", {})
        
        if not time_of_day:
            return None, None, None
        
        # Process hourly distribution
        hour_distribution = time_of_day.get("hour_distribution", {})
        
        if not hour_distribution:
            return None, None, None
            
        # Extract day of week data
        day_distribution = usage_patterns.get("day_of_week", {}).get("day_distribution", {})
        
        # Create 7x24 matrix (days x hours)
        data = np.zeros((7, 24))
        
        # Fill with data (since we don't have hour x day breakdown)
        # This is a placeholder - in a real implementation, you'd use actual data
        for hour in range(24):
            hour_value = hour_distribution.get(str(hour), 0)
            
            # Scale by day of week pattern if available
            if day_distribution:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                for i, day in enumerate(days):
                    day_value = day_distribution.get(day, 1/7)  # Default to even distribution
                    data[i, hour] = hour_value * day_value * 7  # Scale to maintain total activity
            else:
                # Evenly distribute hour values across days
                for day in range(7):
                    data[day, hour] = hour_value
        
        # Create labels
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        
        return data, day_labels, hour_labels
        
    def _prepare_daily_heatmap_data(self, engagement_data, metric):
        """
        Prepare data for daily heatmap visualization using the utility functions.
        
        Returns:
            Dictionary with keys 'values', 'row_labels', 'col_labels'
        """
        # Extract timeline data
        timeline = engagement_data.get("timeline", [])
        
        if not timeline:
            return None
        
        # Select the appropriate metric data
        if metric == "ideas":
            activity_data = engagement_data.get("new_ideas", [])
        elif metric == "steps":
            activity_data = engagement_data.get("new_steps", [])
        else:  # all_activity
            # Combine ideas and steps
            ideas = engagement_data.get("new_ideas", [])
            steps = engagement_data.get("new_steps", [])
            
            if len(ideas) == len(steps) and len(ideas) > 0:
                activity_data = [ideas[i] + steps[i] for i in range(len(ideas))]
            else:
                activity_data = []
        
        # Skip if no activity data
        if not activity_data or len(activity_data) != len(timeline):
            return None
        
        # Convert timeline strings to dates
        try:
            dates = []
            for date_str in timeline:
                # Handle various date formats
                if "to" in date_str:
                    # Format: "YYYY-MM-DD to YYYY-MM-DD"
                    start_date = date_str.split(" to ")[0]
                    dates.append(datetime.strptime(start_date, "%Y-%m-%d"))
                else:
                    # Try standard format
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except:
            # If date parsing fails, use placeholder dates
            dates = [datetime.now() + timedelta(days=i) for i in range(len(timeline))]
        
        # Create matrix of weeks (rows) by days (columns)
        weeks = {}
        
        for i, date in enumerate(dates):
            week_num = date.isocalendar()[1]  # ISO week number
            day_num = date.weekday()  # 0=Monday, 6=Sunday
            
            if week_num not in weeks:
                weeks[week_num] = [None] * 7
            
            weeks[week_num][day_num] = activity_data[i]
        
        # Convert to numpy array
        week_nums = sorted(weeks.keys())
        data = np.zeros((len(week_nums), 7))
        
        for i, week in enumerate(week_nums):
            for j in range(7):
                if weeks[week][j] is not None:
                    data[i, j] = weeks[week][j]
        
        # Create labels
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        week_labels = [f"Week {w}" for w in week_nums]
        
        return {
            "values": data,
            "row_labels": week_labels,
            "col_labels": day_labels
        }
    
    def _prepare_monthly_heatmap_data(self, engagement_data, metric):
        """
        Prepare data for monthly heatmap visualization using the utility functions.
        
        Returns:
            Dictionary with keys 'values', 'row_labels', 'col_labels'
        """
        # Extract timeline data
        timeline = engagement_data.get("timeline", [])
        
        if not timeline:
            return None
        
        # Select the appropriate metric data
        if metric == "ideas":
            activity_data = engagement_data.get("new_ideas", [])
        elif metric == "steps":
            activity_data = engagement_data.get("new_steps", [])
        else:  # all_activity
            # Combine ideas and steps
            ideas = engagement_data.get("new_ideas", [])
            steps = engagement_data.get("new_steps", [])
            
            if len(ideas) == len(steps) and len(ideas) > 0:
                activity_data = [ideas[i] + steps[i] for i in range(len(ideas))]
            else:
                activity_data = []
        
        # Skip if no activity data
        if not activity_data or len(activity_data) != len(timeline):
            return None
        
        # Convert timeline strings to dates and extract year/month
        try:
            year_months = []
            for date_str in timeline:
                # Handle various date formats
                if "to" in date_str:
                    # Format: "YYYY-MM-DD to YYYY-MM-DD" or just "YYYY-MM"
                    start_date = date_str.split(" to ")[0]
                    if len(start_date) == 7:  # YYYY-MM format
                        dt = datetime.strptime(start_date, "%Y-%m")
                    else:
                        dt = datetime.strptime(start_date, "%Y-%m-%d")
                else:
                    # Try standard formats
                    if len(date_str) == 7:  # YYYY-MM format
                        dt = datetime.strptime(date_str, "%Y-%m")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                
                year_months.append((dt.year, dt.month))
        except:
            # If date parsing fails, use placeholder dates
            current_year = datetime.now().year
            current_month = datetime.now().month
            year_months = []
            for i in range(len(timeline)):
                month = (current_month - i) % 12 or 12  # Ensure 1-12 range
                year = current_year - ((current_month - i) // 12)
                year_months.append((year, month))
            
            # Reverse to get chronological order
            year_months.reverse()
        
        # Get unique years and months
        years = sorted(set(ym[0] for ym in year_months))
        months = list(range(1, 13))  # 1-12
        
        # Create matrix of years (rows) by months (columns)
        data = np.zeros((len(years), 12))
        
        for i, (year, month) in enumerate(year_months):
            year_idx = years.index(year)
            month_idx = month - 1  # 0-based index
            data[year_idx, month_idx] = activity_data[i]
        
        # Create labels
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        year_labels = [str(y) for y in years]
        
        return {
            "values": data,
            "row_labels": year_labels,
            "col_labels": month_labels
        }
    
    def __str__(self):
        """String representation of EngagementVisualizer."""
        return f"EngagementVisualizer(output_dir={self._output_dir}, theme={self._theme})"
    
    def __repr__(self):
        """Detailed string representation of EngagementVisualizer."""
        return f"EngagementVisualizer(output_dir={self._output_dir}, theme={self._theme}, default_figsize={self._default_figsize}, include_timestamps={self._include_timestamps})"