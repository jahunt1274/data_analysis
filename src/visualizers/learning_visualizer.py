"""
Learning outcome visualizer for the data analysis system.

This module provides visualization functionality for the LearningAnalyzer results,
creating insightful visual representations of educational impact, learning outcomes,
and tool effectiveness metrics.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.models.enums import Semester
from src.utils.visualization_creation_utils import (
    create_figure,
    configure_axes,
    plot_bar,
    add_reference_line,
    get_color_palette,
    save_figure,
    create_visualization,
)
from src.utils.visualization_data_utils import (
    prepare_scatter_plot_data,
    prepare_heatmap_data,
    create_report_directory,
    generate_filename,
)


class LearningVisualizer:
    """
    Visualizer for learning outcomes and educational impact analysis.

    This class provides methods for creating visualizations from LearningAnalyzer
    results, including comparisons between cohorts, tool versions, and engagement
    metrics related to learning outcomes.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the learning visualizer.

        Args:
            output_dir: Optional custom output directory for visualizations
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._output_dir = output_dir

        # Define color schemes for consistent visualizations
        self._colors = {
            "engagement_levels": {
                "high": "#28A745",  # Green
                "medium": "#FFC107",  # Yellow/Amber
                "low": "#DC3545",  # Red
            },
            "tool_versions": {
                "none": "#6C757D",  # Gray
                "v1": "#007BFF",  # Blue
                "v2": "#28A745",  # Green
            },
            "semesters": {
                Semester.FALL_2023.value: "#E69F00",
                Semester.SPRING_2024.value: "#56B4E9",
                Semester.FALL_2024.value: "#009E73",
                Semester.SPRING_2025.value: "#F0E442",
            },
            "metrics": {
                "learning_outcome": "#CC79A7",
                "engagement": "#0072B2",
                "completion": "#D55E00",
                "content": "#009E73",
            },
        }

    def visualize_course_rating_engagement_correlation(
        self, correlation_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize correlation between tool usage and course ratings.

        Args:
            correlation_data: Data from LearningAnalyzer.correlate_tool_usage_with_course_ratings()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in correlation_data:
            self._logger.error(
                f"Error in correlation data: {correlation_data['error']}"
            )
            return result

        # Extract semester data
        semester_data = correlation_data.get("semester_data", {})
        if not semester_data:
            self._logger.warning("No semester data available for visualization")
            return result

        # Create dataframe for ratings visualization
        ratings_df = []
        for semester, data in semester_data.items():
            if "error" in data:
                continue

            ratings_df.append(
                {
                    "semester": semester,
                    "overall_score": data["course_rating"]["overall_score"],
                    "high_impact_score": data["course_rating"]["high_impact_overall"],
                    "weighted_engagement": data["engagement"]["weighted_score"],
                    "tool_version": data["tool_version"],
                }
            )

        if not ratings_df:
            self._logger.warning("No valid rating data found for visualization")
            return result

        df = pd.DataFrame(ratings_df)

        # Prepare grouped bar data for ratings
        grouped_data = {
            "values": {
                "Overall Course Rating": dict(zip(df["semester"], df["overall_score"])),
                "High Impact Rating": dict(
                    zip(df["semester"], df["high_impact_score"])
                ),
            }
        }

        # Create visualization for ratings
        fig1, ax1 = create_visualization(
            data=grouped_data,
            viz_type="grouped_bar",
            title="Course Ratings by Semester",
            xlabel="Semester",
            ylabel="Course Rating",
            colors=["#007BFF", "#28A745"],
            add_data_labels=True,
            grid=True,
        )

        # Add engagement line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len(df)),
            df["weighted_engagement"],
            "o-",
            label="Weighted Engagement",
            color="#FFC107",
            linewidth=2,
            markersize=8,
        )
        ax2.set_ylabel("Weighted Engagement")
        ax2.legend(loc="upper right")

        # Add tool version annotations
        for i, (version, eng) in enumerate(
            zip(df["tool_version"], df["weighted_engagement"])
        ):
            ax1.annotate(
                f"v{version}",
                xy=(i, 0.1),
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
            )

        # Add the figure to the result
        result["figures"]["ratings_by_semester"] = fig1

        # Create scatter plot for correlation if data available
        if "correlation" in correlation_data and len(df) > 1:
            # Use utility function to prepare scatter plot data
            scatter_data = prepare_scatter_plot_data(
                df,
                x_field="weighted_engagement",
                y_field="overall_score",
                category_field="tool_version",
                add_trend=True,
            )

            # Create visualization
            fig2, ax = create_visualization(
                data=scatter_data,
                viz_type="scatter",
                title="Correlation: Course Ratings vs. Tool Engagement",
                xlabel="Weighted Engagement Score",
                ylabel="Overall Course Rating",
                add_trend_line=True,
                add_correlation=True,
                grid=True,
                colors=[
                    (
                        self._colors["tool_versions"][f"v{v}"]
                        if v != "none"
                        else self._colors["tool_versions"]["none"]
                    )
                    for v in df["tool_version"]
                ],
            )

            # Add labels for each point
            for i, row in df.iterrows():
                ax.annotate(
                    row["semester"],
                    (row["weighted_engagement"], row["overall_score"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            # Add legend for tool versions
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["none"],
                    label="No Tool",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["v1"],
                    label="Tool v1",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["v2"],
                    label="Tool v2",
                    markersize=10,
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper left")

            # Add the figure to the result
            result["figures"]["correlation_scatter"] = fig2

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_outcome_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_learning_outcomes_by_cohort(
        self, outcomes_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize comparison of learning outcomes between pre-tool and post-tool cohorts.

        Args:
            outcomes_data: Data from LearningAnalyzer.compare_learning_outcomes_by_cohort()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in outcomes_data:
            self._logger.error(f"Error in outcomes data: {outcomes_data['error']}")
            return result

        # Extract pre-tool and post-tool data
        pre_tool = outcomes_data.get("pre_tool", {})
        post_tool = outcomes_data.get("post_tool", {})
        differences = outcomes_data.get("differences", {})

        if not pre_tool or not post_tool:
            self._logger.warning(
                "Insufficient data for cohort comparison visualization"
            )
            return result

        # Prepare data for overall metrics
        metrics = ["overall", "high_impact_overall"]
        labels = ["Overall Score", "High Impact Score"]
        pre_values = [
            pre_tool.get("overall", 0),
            pre_tool.get("high_impact_overall", 0),
        ]
        post_values = [
            post_tool["aggregated"].get("overall", 0),
            post_tool["aggregated"].get("high_impact_overall", 0),
        ]

        # Create data for visualization
        comparison_data = {
            "values": {
                f"Pre-Tool ({pre_tool.get('semester', 'Unknown')})": dict(
                    zip(labels, pre_values)
                ),
                "Post-Tool (Average)": dict(zip(labels, post_values)),
            }
        }

        # Create grouped bar chart using utility function
        fig1, ax = create_visualization(
            data=comparison_data,
            viz_type="grouped_bar",
            title="Learning Outcomes: Pre-Tool vs. Post-Tool Comparison",
            xlabel="Metrics",
            ylabel="Score",
            colors=[
                self._colors["tool_versions"]["none"],
                self._colors["tool_versions"]["v1"],
            ],
            add_data_labels=True,
            grid=True,
        )

        # Add difference annotations
        diff_values = [
            differences.get("overall", 0),
            differences.get("high_impact_overall", 0),
        ]
        x_pos = np.arange(len(metrics))

        for i, diff in enumerate(diff_values):
            color = "#28A745" if diff > 0 else "#DC3545"
            ax.annotate(
                f"{diff:+.2f}",
                xy=(i, max(pre_values[i], post_values[i]) + 0.05),
                ha="center",
                va="bottom",
                fontsize=10,
                color=color,
                weight="bold",
            )

        # Add the figure to the result
        result["figures"]["overall_comparison"] = fig1

        # Create figure for section comparisons if available
        if "sections" in differences:
            # Prepare data for section comparison
            section_data = []
            for section, values in differences["sections"].items():
                section_data.append(
                    {
                        "section": section,
                        "pre_tool": values.get("pre_tool", 0),
                        "post_tool": values.get("post_tool", 0),
                        "difference": values.get("difference", 0),
                        "percent_change": values.get("percent_change", 0),
                    }
                )

            if section_data:
                df = pd.DataFrame(section_data)

                # Sort by absolute difference
                df = df.sort_values(by="difference", key=abs, ascending=False)

                # Prepare data for visualization
                section_comparison = {
                    "values": {
                        f"Pre-Tool ({pre_tool.get('semester', 'Unknown')})": dict(
                            zip(df["section"], df["pre_tool"])
                        ),
                        "Post-Tool (Average)": dict(
                            zip(df["section"], df["post_tool"])
                        ),
                    }
                }

                # Create grouped bar chart using utility function
                fig2, ax = create_visualization(
                    data=section_comparison,
                    viz_type="grouped_bar",
                    title="Section Scores: Pre-Tool vs. Post-Tool Comparison",
                    xlabel="Course Sections",
                    ylabel="Score",
                    colors=[
                        self._colors["tool_versions"]["none"],
                        self._colors["tool_versions"]["v1"],
                    ],
                    add_data_labels=True,
                    grid=True,
                    xticklabel_rotation=45,
                )

                # Add difference annotations
                for i, (section, row) in enumerate(df.iterrows()):
                    color = "#28A745" if row["difference"] > 0 else "#DC3545"
                    ax.annotate(
                        f"{row['difference']:+.2f}",
                        xy=(i, max(row["pre_tool"], row["post_tool"]) + 0.05),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=color,
                        weight="bold",
                    )

                # Add the figure to the result
                result["figures"]["section_comparison"] = fig2

        # Create figure for semester progression if multiple post-tool semesters
        if "by_semester" in post_tool and len(post_tool["by_semester"]) > 1:
            # Prepare data for semester comparison
            semester_data = []

            for sem, data in post_tool["by_semester"].items():
                if "overall" in data and "high_impact_overall" in data:
                    semester_data.append(
                        {
                            "semester": sem,
                            "overall": data["overall"],
                            "high_impact": data["high_impact_overall"],
                            "tool_version": data.get("tool_version", "unknown"),
                        }
                    )

            if semester_data:
                df = pd.DataFrame(semester_data)

                # Add pre-tool data
                pre_df = pd.DataFrame(
                    [
                        {
                            "semester": pre_tool.get("semester", "Unknown"),
                            "overall": pre_tool.get("overall", 0),
                            "high_impact": pre_tool.get("high_impact_overall", 0),
                            "tool_version": "none",
                        }
                    ]
                )

                df = pd.concat([pre_df, df], ignore_index=True)

                # Sort by semester
                df = df.sort_values(by="semester")

                # Prepare data for visualization
                progression_data = {
                    "values": {
                        "Overall Score": dict(zip(df["semester"], df["overall"])),
                        "High Impact Score": dict(
                            zip(df["semester"], df["high_impact"])
                        ),
                    }
                }

                # Create multi-line chart using utility function
                fig3, ax = create_visualization(
                    data=progression_data,
                    viz_type="multi_line",
                    title="Learning Outcomes Progression Across Semesters",
                    xlabel="Semester",
                    ylabel="Score",
                    colors=["#007BFF", "#28A745"],
                    markers=["o", "s"],
                    add_data_labels=True,
                    grid=True,
                )

                # Add tool version markers
                tool_markers = []
                for i, version in enumerate(df["tool_version"]):
                    color = self._colors["tool_versions"].get(version, "#6C757D")
                    ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=color, zorder=0)

                    # Add to tool markers for legend
                    if version not in [m[0] for m in tool_markers]:
                        tool_markers.append((version, color))

                # Add tool version legend
                from matplotlib.patches import Patch

                tool_legend_elements = [
                    Patch(facecolor=color, alpha=0.3, label=f"Tool: {version}")
                    for version, color in tool_markers
                ]

                # Create second legend for tool versions
                second_legend = ax.legend(
                    handles=tool_legend_elements,
                    loc="upper right",
                    title="Tool Version",
                )

                # Add the first legend back
                ax.add_artist(ax.legend(loc="upper left"))

                # Add the figure to the result
                result["figures"]["semester_progression"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_cohort_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_tool_version_impact(
        self, version_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the impact of different tool versions on learning outcomes.

        Args:
            version_data: Data from LearningAnalyzer.analyze_tool_version_impact()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in version_data:
            self._logger.error(f"Error in version impact data: {version_data['error']}")
            return result

        # Extract version data
        v1_data = version_data.get("v1", {})
        v2_data = version_data.get("v2", {})
        differences = version_data.get("differences", {})

        if not v1_data or not v2_data:
            self._logger.warning(
                "Insufficient data for tool version impact visualization"
            )
            return result

        # Prepare data for learning metrics comparison
        metrics = ["overall", "high_impact_overall"]
        labels = ["Overall Score", "High Impact Score"]
        v1_values = [
            v1_data.get("metrics", {}).get("overall", 0),
            v1_data.get("metrics", {}).get("high_impact_overall", 0),
        ]
        v2_values = [
            v2_data.get("metrics", {}).get("overall", 0),
            v2_data.get("metrics", {}).get("high_impact_overall", 0),
        ]

        # Create data for visualization
        metrics_data = {
            "values": {
                "Tool v1": dict(zip(labels, v1_values)),
                "Tool v2": dict(zip(labels, v2_values)),
            }
        }

        # Create grouped bar chart using utility function
        fig1, ax = create_visualization(
            data=metrics_data,
            viz_type="grouped_bar",
            title="Learning Outcomes: Tool v1 vs. Tool v2 Comparison",
            xlabel="Learning Metrics",
            ylabel="Score",
            colors=[
                self._colors["tool_versions"]["v1"],
                self._colors["tool_versions"]["v2"],
            ],
            add_data_labels=True,
            grid=True,
        )

        # Add difference annotations
        diff_values = [
            differences.get("overall", 0),
            differences.get("high_impact_overall", 0),
        ]
        x_pos = np.arange(len(metrics))

        for i, diff in enumerate(diff_values):
            color = "#28A745" if diff > 0 else "#DC3545"
            ax.annotate(
                f"{diff:+.2f}",
                xy=(i, max(v1_values[i], v2_values[i]) + 0.05),
                ha="center",
                va="bottom",
                fontsize=10,
                color=color,
                weight="bold",
            )

        # Add the figure to the result
        result["figures"]["learning_metrics_comparison"] = fig1

        # Create figure for engagement metrics comparison
        if "engagement" in v1_data and "engagement" in v2_data:
            # Extract engagement data
            v1_engagement = v1_data["engagement"]
            v2_engagement = v2_data["engagement"]

            # Prepare engagement metrics data
            eng_metrics = [
                "high_pct",
                "medium_pct",
                "low_pct",
                "weighted_score",
                "avg_content_score",
                "avg_completion_score",
            ]

            eng_labels = [
                "High Engagement %",
                "Medium Engagement %",
                "Low Engagement %",
                "Weighted Engagement",
                "Avg Content Score",
                "Avg Completion Score",
            ]

            # Extract values
            v1_eng_values = []
            v2_eng_values = []

            for metric in eng_metrics:
                if metric in ["high_pct", "medium_pct", "low_pct"]:
                    v1_val = v1_engagement.get("distribution", {}).get(
                        metric.split("_")[0], 0
                    )
                    v2_val = v2_engagement.get("distribution", {}).get(
                        metric.split("_")[0], 0
                    )
                else:
                    v1_val = v1_engagement.get(metric, 0)
                    v2_val = v2_engagement.get(metric, 0)

                v1_eng_values.append(v1_val)
                v2_eng_values.append(v2_val)

            # Create data for visualization
            engagement_data = {
                "values": {
                    "Tool v1": dict(zip(eng_labels, v1_eng_values)),
                    "Tool v2": dict(zip(eng_labels, v2_eng_values)),
                }
            }

            # Create grouped bar chart using utility function
            fig2, ax = create_visualization(
                data=engagement_data,
                viz_type="grouped_bar",
                title="User Engagement: Tool v1 vs. Tool v2 Comparison",
                xlabel="Engagement Metrics",
                ylabel="Value",
                colors=[
                    self._colors["tool_versions"]["v1"],
                    self._colors["tool_versions"]["v2"],
                ],
                add_data_labels=True,
                grid=True,
                xticklabel_rotation=45,
            )

            # Add difference annotations
            diff_eng_values = []
            for metric in eng_metrics:
                diff_eng_values.append(differences.get("engagement", {}).get(metric, 0))

            for i, diff in enumerate(diff_eng_values):
                color = "#28A745" if diff > 0 else "#DC3545"
                ax.annotate(
                    f"{diff:+.2f}",
                    xy=(i, max(v1_eng_values[i], v2_eng_values[i]) + 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    weight="bold",
                )

            # Add the figure to the result
            result["figures"]["engagement_metrics_comparison"] = fig2

        # Create figure for section scores comparison if available
        if "sections" in differences:
            # Get section comparison data
            section_data = []
            for section, values in differences["sections"].items():
                if "v1" in values and "v2" in values:
                    section_data.append(
                        {
                            "section": section,
                            "v1": values.get("v1", 0),
                            "v2": values.get("v2", 0),
                            "difference": values.get("difference", 0),
                            "percent_change": values.get("percent_change", 0),
                        }
                    )

            if section_data:
                df = pd.DataFrame(section_data)

                # Sort by absolute difference
                df = df.sort_values(by="difference", key=abs, ascending=False)

                # Prepare data for visualization
                section_comparison = {
                    "values": {
                        "Tool v1": dict(zip(df["section"], df["v1"])),
                        "Tool v2": dict(zip(df["section"], df["v2"])),
                    }
                }

                # Create grouped bar chart using utility function
                fig3, ax = create_visualization(
                    data=section_comparison,
                    viz_type="grouped_bar",
                    title="Section Scores: Tool v1 vs. Tool v2 Comparison",
                    xlabel="Course Sections",
                    ylabel="Score",
                    colors=[
                        self._colors["tool_versions"]["v1"],
                        self._colors["tool_versions"]["v2"],
                    ],
                    add_data_labels=True,
                    grid=True,
                    xticklabel_rotation=45,
                )

                # Add difference annotations
                for i, (section, row) in enumerate(df.iterrows()):
                    color = "#28A745" if row["difference"] > 0 else "#DC3545"
                    ax.annotate(
                        f"{row['difference']:+.2f}",
                        xy=(i, max(row["v1"], row["v2"]) + 0.05),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=color,
                        weight="bold",
                    )

                # Add the figure to the result
                result["figures"]["section_comparison"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"tool_version_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_framework_engagement(
        self, framework_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize engagement metrics for entrepreneurial frameworks.

        Args:
            framework_data: Data from LearningAnalyzer.calculate_framework_engagement_metrics()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in framework_data:
            self._logger.error(
                f"Error in framework engagement data: {framework_data['error']}"
            )
            return result

        # Extract framework data
        steps_by_framework = framework_data.get("steps_by_framework", {})
        framework_semester_metrics = framework_data.get(
            "framework_semester_metrics", {}
        )
        learning_correlations = framework_data.get("learning_outcome_correlations", {})

        if not steps_by_framework or not framework_semester_metrics:
            self._logger.warning(
                "Insufficient data for framework engagement visualization"
            )
            return result

        # Create figure for steps by framework

        # Prepare data for visualization using utility function
        step_count_data = {"values": steps_by_framework}

        # Create bar chart using utility function
        fig1, ax = create_visualization(
            data=step_count_data,
            viz_type="bar",
            title="Step Counts by Framework",
            xlabel="Framework",
            ylabel="Number of Steps",
            colors=get_color_palette(
                "categorical_main", n_colors=len(steps_by_framework)
            ),
            add_data_labels=True,
            grid=True,
        )

        # Add the figure to the result
        result["figures"]["steps_by_framework"] = fig1

        # Create figure for engagement by semester
        # Prepare data for visualization
        semester_engagement_data = []

        for framework, semester_data in framework_semester_metrics.items():
            for semester, metrics in semester_data.items():
                semester_engagement_data.append(
                    {
                        "semester": semester,
                        "framework": framework,
                        "engagement_rate": metrics.get("engagement_rate", 0),
                        "avg_steps": metrics.get("avg_steps_per_user", 0),
                    }
                )

        if semester_engagement_data:
            df = pd.DataFrame(semester_engagement_data)

            # Prepare data for grouped bar chart
            by_semester_data = {}
            for framework in df["framework"].unique():
                framework_df = df[df["framework"] == framework]
                by_semester_data[framework] = dict(
                    zip(framework_df["semester"], framework_df["engagement_rate"])
                )

            semester_comparison = {"values": by_semester_data}

            # Create grouped bar chart using utility function
            fig2, ax = create_visualization(
                data=semester_comparison,
                viz_type="grouped_bar",
                title="Framework Engagement Rate by Semester",
                xlabel="Semester",
                ylabel="Engagement Rate",
                add_data_labels=True,
                grid=True,
                xticklabel_rotation=45,
            )

            # Add the figure to the result
            result["figures"]["engagement_by_semester"] = fig2

        # Create figure for correlation with learning outcomes
        if learning_correlations:
            # Prepare data for visualization
            correlation_data = {"values": learning_correlations}

            # Create bar chart using utility function
            fig3, ax = create_visualization(
                data=correlation_data,
                viz_type="bar",
                title="Correlation Between Framework Engagement and Learning Outcomes",
                xlabel="Framework",
                ylabel="Correlation Coefficient",
                reference_value=0,
                add_data_labels=True,
                grid=True,
                colors=[
                    (
                        "#28A745"
                        if val >= 0.5
                        else (
                            "#007BFF"
                            if val >= 0.3
                            else (
                                "#FFC107"
                                if val >= 0
                                else "#FD7E14" if val >= -0.3 else "#DC3545"
                            )
                        )
                    )
                    for val in learning_correlations.values()
                ],
            )

            # Add correlation strength legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#28A745", alpha=0.8, label="Strong Positive (≥0.5)"),
                Patch(
                    facecolor="#007BFF", alpha=0.8, label="Moderate Positive (0.3-0.5)"
                ),
                Patch(facecolor="#FFC107", alpha=0.8, label="Weak Positive (0-0.3)"),
                Patch(facecolor="#FD7E14", alpha=0.8, label="Weak Negative (-0.3-0)"),
                Patch(
                    facecolor="#DC3545", alpha=0.8, label="Stronger Negative (<-0.3)"
                ),
            ]

            ax.legend(
                handles=legend_elements, loc="upper right", title="Correlation Strength"
            )

            # Add the figure to the result
            result["figures"]["learning_correlation"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"framework_engagement_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_time_allocation_impact(
        self, time_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize how the tool impacts time allocation for students.

        Args:
            time_data: Data from LearningAnalyzer.analyze_tool_impact_on_time_allocation()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in time_data:
            self._logger.error(f"Error in time allocation data: {time_data['error']}")
            return result

        # Extract time metrics data
        version_metrics = time_data.get("version_metrics", {})
        time_savings = time_data.get("time_savings", {})
        step_consistency = time_data.get("step_consistency", {})

        if not version_metrics:
            self._logger.warning("Insufficient data for time allocation visualization")
            return result

        # Create figure for time metrics comparison using subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Define metrics to visualize
        metrics = [
            "avg_steps_per_idea",
            "avg_total_hours",
            "avg_hours_between_steps",
            "avg_hours_per_step",
        ]

        metric_labels = [
            "Avg Steps per Idea",
            "Avg Total Hours",
            "Avg Hours Between Steps",
            "Avg Hours per Step",
        ]

        # Get values for each tool version
        versions = ["none", "v1", "v2"]
        version_labels = ["No Tool", "Tool v1", "Tool v2"]

        # Plot each metric in a subplot
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Get current axes
            ax = axes[i]

            # Prepare data for this metric
            metric_values = {
                version_label: version_metrics[version].get(metric, 0)
                for version, version_label in zip(versions, version_labels)
                if version in version_metrics
            }

            # Use plot_bar utility function
            plot_bar(
                ax,
                data=metric_values,
                color=[
                    self._colors["tool_versions"][v]
                    for v in versions
                    if v in version_metrics
                ],
                add_data_labels=True,
            )

            # Configure axis
            configure_axes(ax, title=label, ylabel="Value", grid=True)

            # Add percentage difference annotations if available
            if i > 0 and "v2_vs_none" in time_savings:
                metric_key = metric + "_pct" if metric != "avg_steps_per_idea" else None
                if metric_key and metric_key in time_savings["v2_vs_none"]:
                    diff_pct = time_savings["v2_vs_none"][metric_key]
                    color = (
                        "#28A745" if diff_pct < 0 else "#DC3545"
                    )  # Green if time saved
                    ax.annotate(
                        f"{diff_pct:+.1f}%",
                        xy=(0.98, 0.05),
                        xycoords="axes fraction",
                        fontsize=10,
                        color=color,
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )

        # Add overall title
        fig.suptitle("Time Allocation Impact Across Tool Versions", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

        # Add the figure to the result
        result["figures"]["time_metrics_comparison"] = fig

        # Create figure for time savings visualization
        if time_savings and "v2_vs_none" in time_savings:
            # Prepare data for horizontal bar chart
            savings_metrics = [
                ("hours_per_step_pct", "Hours per Step"),
                ("total_hours_pct", "Total Hours"),
            ]

            # Get values with sign inverted (negative % is time saved)
            savings_values = {
                label: -time_savings["v2_vs_none"].get(metric, 0)
                for metric, label in savings_metrics
            }

            # Create horizontal bar chart using utility function
            fig2, ax = create_visualization(
                data={"values": savings_values},
                viz_type="bar",
                title="Time Savings: Tool vs. No Tool",
                xlabel="Time Savings (%)",
                ylabel="",
                orientation="horizontal",
                add_data_labels=True,
                grid=True,
                reference_value=0,
                colors=[
                    "#28A745" if val > 0 else "#DC3545"
                    for val in savings_values.values()
                ],
            )

            # Add savings interpretation
            if any(val > 0 for val in savings_values.values()):
                interpretation = "Tool usage resulted in time savings"
            else:
                interpretation = "Tool usage did not save time"

            ax.annotate(
                interpretation,
                xy=(0.5, 0.01),
                xycoords="figure fraction",
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Add the figure to the result
            result["figures"]["time_savings"] = fig2

        # Create figure for step consistency visualization
        if step_consistency:
            # Prepare data for visualization
            versions = [v for v in ["none", "v1", "v2"] if v in step_consistency]

            # Create bar chart data
            consistency_data = {
                "consistency_score": {
                    v.replace("none", "No Tool")
                    .replace("v", "Tool v"): step_consistency[v]
                    .get("consistency_score", 0)
                    for v in versions
                },
                "avg_completed_steps": {
                    v.replace("none", "No Tool")
                    .replace("v", "Tool v"): step_consistency[v]
                    .get("avg_completed_steps", 0)
                    for v in versions
                },
            }

            # Convert to format for grouped bar chart
            grouped_consistency = {"values": consistency_data}

            # Create grouped bar chart using utility function
            fig3, ax = create_visualization(
                data=grouped_consistency,
                viz_type="grouped_bar",
                title="Step Completion Consistency Across Tool Versions",
                xlabel="Tool Version",
                ylabel="Value",
                add_data_labels=True,
                grid=True,
                colors=["#007BFF", "#28A745"],
            )

            # Add the figure to the result
            result["figures"]["step_consistency"] = fig3

            # Create figure for progression patterns if available
            has_patterns = False
            for v in versions:
                if "progression_patterns" in step_consistency[v]:
                    has_patterns = True
                    break

            if has_patterns:
                # Collect pattern data
                pattern_data = {}
                for v in versions:
                    if "progression_patterns" in step_consistency[v]:
                        version_label = v.replace("none", "No Tool").replace(
                            "v", "Tool v"
                        )
                        pattern_data[version_label] = {
                            pattern.title(): data.get("percentage", 0) * 100
                            for pattern, data in step_consistency[v][
                                "progression_patterns"
                            ].items()
                        }

                # Convert to format for grouped bar chart
                grouped_patterns = {"values": pattern_data}

                # Create grouped bar chart using utility function
                fig4, ax = create_visualization(
                    data=grouped_patterns,
                    viz_type="grouped_bar",
                    title="Framework Progression Patterns Across Tool Versions",
                    xlabel="Progression Pattern",
                    ylabel="Percentage of Users (%)",
                    add_data_labels=True,
                    grid=True,
                )

                # Add the figure to the result
                result["figures"]["progression_patterns"] = fig4

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"time_allocation_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_learning_metrics_summary(
        self, metrics_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a summary visualization of combined learning metrics.

        Args:
            metrics_data: Data from LearningAnalyzer.analyze_combined_learning_metrics()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in metrics_data:
            self._logger.error(
                f"Error in learning metrics data: {metrics_data['error']}"
            )
            return result

        # Extract metrics data
        composite_score = metrics_data.get("composite_effectiveness_score", 0)
        effectiveness = metrics_data.get("effectiveness_interpretation", "")
        key_metrics = metrics_data.get("key_metrics", {})
        recommendations = metrics_data.get("recommendations", [])

        if not key_metrics:
            self._logger.warning("Insufficient data for learning metrics visualization")
            return result

        # Create figure for effectiveness score
        fig1, ax = create_figure(width=10, height=6)

        # Create gauge chart for effectiveness score
        # This is a specialized visualization not in utils, so keep custom implementation
        import matplotlib.patches as mpatches

        # Create background arc
        theta = np.linspace(-np.pi, 0, 100)
        r = 1.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ax.plot(x, y, color="#CCCCCC", linewidth=20, alpha=0.5)

        # Create colored gauge segments
        score_ranges = [
            (-1.0, -0.5, "#DC3545", "Ineffective"),  # Red
            (-0.5, -0.2, "#FD7E14", "Slightly Ineffective"),  # Orange
            (-0.2, 0.2, "#FFC107", "Neutral"),  # Yellow
            (0.2, 0.5, "#17A2B8", "Moderately Effective"),  # Blue
            (0.5, 1.0, "#28A745", "Highly Effective"),  # Green
        ]

        # Draw each segment
        for start, end, color, label in score_ranges:
            segment_theta = np.linspace(np.pi * start, np.pi * end, 50)
            segment_x = r * np.cos(segment_theta)
            segment_y = r * np.sin(segment_theta)

            ax.plot(
                segment_x,
                segment_y,
                color=color,
                linewidth=20,
                alpha=0.7,
                solid_capstyle="butt",
            )

            # Add label
            mid_theta = np.pi * (start + end) / 2
            label_x = 1.2 * r * np.cos(mid_theta)
            label_y = 1.2 * r * np.sin(mid_theta)

            ax.text(
                label_x,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                rotation=90 - mid_theta * 180 / np.pi,
            )

        # Add needle to show score
        score_theta = np.pi * composite_score
        needle_x = [0, r * np.cos(score_theta)]
        needle_y = [0, r * np.sin(score_theta)]

        ax.plot(needle_x, needle_y, "k-", linewidth=2)

        # Add circle at base of needle
        ax.add_patch(mpatches.Circle((0, 0), 0.05, facecolor="black"))

        # Add score and interpretation text
        ax.text(
            0,
            -0.2,
            f"Score: {composite_score:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0,
            -0.35,
            effectiveness,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=next(
                color
                for start, end, color, label in score_ranges
                if start <= composite_score <= end
            ),
        )

        # Configure axes
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        ax.set_title("Tool Effectiveness Composite Score", fontsize=14, pad=20)

        # Add the figure to the result
        result["figures"]["effectiveness_gauge"] = fig1

        # Create figure for key metrics comparison

        # Prepare data for visualization
        metrics_data = {"values": key_metrics}

        # Create bar chart using utility function
        fig2, ax = create_visualization(
            data=metrics_data,
            viz_type="bar",
            title="Key Learning Impact Metrics",
            xlabel="Metric",
            ylabel="Normalized Score (-1 to 1)",
            reference_value=0,
            add_data_labels=True,
            grid=True,
            xticklabel_rotation=0,
            colors=[
                "#28A745" if val >= 0 else "#DC3545" for val in key_metrics.values()
            ],
        )

        # Add custom data labels with original values
        x_pos = np.arange(len(key_metrics))
        values = list(key_metrics.values())

        for i, (metric, val) in enumerate(key_metrics.items()):
            if metric == "time_savings_pct":
                label = f"{val:.1f}%"
            else:
                label = f"{val:.2f}"

            ax.text(
                i,
                val + 0.05 if val >= 0 else val - 0.1,
                label,
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10,
            )

        # Add the figure to the result
        result["figures"]["key_metrics"] = fig2

        # Create figure for recommendations if available
        if recommendations:
            fig3, ax = create_figure(width=14, height=10)

            # Disable axis
            ax.axis("off")

            # Create a formatted text for recommendations
            recommendation_text = (
                "# Tool Effectiveness Recommendations\n\n"
                f"**Overall Assessment: {effectiveness}** (Score: {composite_score:.2f})\n\n"
            )

            for i, rec in enumerate(recommendations[:5]):  # Limit to top 5
                category = rec.get("category", "")
                finding = rec.get("finding", "")
                suggestion = rec.get("suggestion", "")

                recommendation_text += (
                    f"### {i+1}. {category}\n"
                    f"**Finding:** {finding}\n"
                    f"**Recommendation:** {suggestion}\n\n"
                )

            ax.text(
                0.5,
                0.98,
                recommendation_text,
                ha="center",
                va="top",
                fontsize=12,
                transform=ax.transAxes,
                wrap=True,
                family="sans-serif",
            )

            # Apply tight layout
            fig3.tight_layout()

            # Add the figure to the result
            result["figures"]["recommendations"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_summary_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_learning_outcome_metrics(
        self, outcome_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize learning outcome metrics across different cohorts.

        Args:
            outcome_data: Data from LearningAnalyzer.get_learning_outcome_metrics()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in outcome_data:
            self._logger.error(
                f"Error in outcome metrics data: {outcome_data['error']}"
            )
            return result

        # Extract outcome data
        learning_questions = outcome_data.get("learning_outcome_questions", [])
        by_version = outcome_data.get("by_version", {})
        by_semester = outcome_data.get("by_semester", {})
        cohort_aggregates = outcome_data.get("cohort_aggregates", {})
        impact_metrics = outcome_data.get("impact_metrics", {})

        if not by_version or not cohort_aggregates:
            self._logger.warning(
                "Insufficient data for learning outcome metrics visualization"
            )
            return result

        # Create figure for cohort comparison
        # Prepare data for visualization
        cohort_data = {
            "values": {
                cohort: data.get("score", 0)
                for cohort, data in cohort_aggregates.items()
            }
        }

        # Create bar chart using utility function
        fig1, ax = create_visualization(
            data=cohort_data,
            viz_type="bar",
            title="Learning Outcome Scores by Cohort",
            xlabel="Cohort",
            ylabel="Average Score",
            add_data_labels=True,
            grid=True,
            colors=[
                (
                    self._colors["tool_versions"]["none"]
                    if cohort == "pre_tool"
                    else (
                        self._colors["tool_versions"]["v1"]
                        if cohort == "v1"
                        else (
                            self._colors["tool_versions"]["v2"]
                            if cohort == "v2"
                            else "#6C757D"
                        )
                    )
                )
                for cohort in cohort_aggregates
            ],
        )

        # Add question count annotations
        x_pos = np.arange(len(cohort_aggregates))
        for i, (cohort, data) in enumerate(cohort_aggregates.items()):
            question_count = data.get("question_count", 0)
            ax.annotate(
                f"{question_count} questions",
                xy=(i, 0.05),
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

        # Add the figure to the result
        result["figures"]["cohort_comparison"] = fig1

        # Create figure for impact metrics if available
        if impact_metrics:
            # Prepare data for visualization
            comparison_labels = {
                "v1_vs_pre": "Tool v1 vs. Pre-Tool",
                "v2_vs_v1": "Tool v2 vs. Tool v1",
                "v2_vs_pre": "Tool v2 vs. Pre-Tool",
            }

            # Create data structure for comparison visualization
            impact_data = {
                "Absolute Difference": {
                    comparison_labels.get(comp, comp): impact_metrics[comp].get(
                        "absolute_diff", 0
                    )
                    for comp in impact_metrics
                },
                "Percent Change": {
                    comparison_labels.get(comp, comp): impact_metrics[comp].get(
                        "percent_change", 0
                    )
                    for comp in impact_metrics
                },
            }

            # Create grouped bar chart using utility function
            fig2, ax = create_visualization(
                data={"values": impact_data},
                viz_type="grouped_bar",
                title="Learning Outcome Improvements Between Cohorts",
                xlabel="Comparison",
                ylabel="Value",
                add_data_labels=True,
                grid=True,
                colors=["#007BFF", "#28A745"],
            )

            # Create secondary y-axis for percentage
            ax2 = ax.twinx()
            ax2.set_ylabel("Percent Change (%)")

            # Add the figure to the result
            result["figures"]["impact_metrics"] = fig2

        # Create figure for semester comparison if available
        if by_semester:
            # Use heatmap for question scores across semesters
            semesters = list(by_semester.keys())

            # Get all questions from first semester
            first_semester = semesters[0] if semesters else None

            if first_semester and "questions" in by_semester[first_semester]:
                # Identify common questions across semesters
                common_questions = set()

                # Find common questions
                for semester, data in by_semester.items():
                    questions = set(data.get("questions", {}).keys())

                    if not common_questions:
                        common_questions = questions
                    else:
                        common_questions &= questions

                # Select a reasonable number of questions
                if len(common_questions) > 15:
                    # Too many questions, select top learning outcome questions
                    if learning_questions:
                        common_questions = set(learning_questions[:10])
                    else:
                        # Take a subset of questions
                        common_questions = set(list(common_questions)[:10])

                # Create data for visualization
                question_data = []

                for question in common_questions:
                    for semester, data in by_semester.items():
                        if "questions" in data and question in data["questions"]:
                            # Truncate long questions
                            short_q = (
                                question[:37] + "..."
                                if len(question) > 40
                                else question
                            )

                            question_data.append(
                                {
                                    "question": question,
                                    "short_question": short_q,
                                    "semester": semester,
                                    "score": data["questions"][question],
                                }
                            )

                if question_data:
                    df = pd.DataFrame(question_data)

                    # Prepare data for heatmap using utility function
                    heatmap_data = prepare_heatmap_data(
                        df,
                        row_field="short_question",
                        col_field="semester",
                        value_field="score",
                    )

                    # Create heatmap using utility function
                    fig3, ax = create_visualization(
                        data=heatmap_data,
                        viz_type="heatmap",
                        title="Learning Outcome Question Scores by Semester",
                        color_map="YlGnBu",
                        add_values=True,
                        value_format="{:.2f}",
                    )

                    # Add the figure to the result
                    result["figures"]["semester_questions"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_outcome_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_demographic_impact(
        self, demographic_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the learning impact across different student demographics.

        Args:
            demographic_data: Data from LearningAnalyzer.analyze_demographic_learning_impact()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in demographic_data:
            self._logger.error(
                f"Error in demographic impact data: {demographic_data['error']}"
            )
            return result

        # Extract demographic data
        user_type_analysis = demographic_data.get("user_type_analysis", {})
        department_analysis = demographic_data.get("department_analysis", {})
        experience_analysis = demographic_data.get("experience_analysis", {})
        overall_improvement = demographic_data.get("overall_improvement", 0)

        # Create figure for user type analysis
        if "engagement_metrics" in user_type_analysis:
            user_type_metrics = user_type_analysis["engagement_metrics"]

            if user_type_metrics:
                # Prepare data for grouped bar chart
                metrics = {}

                for user_type, data in user_type_metrics.items():
                    if (
                        "weighted_engagement" in data
                        and "avg_content_score" in data
                        and "avg_completion_score" in data
                    ):
                        metrics[user_type] = {
                            "Weighted Engagement": data["weighted_engagement"],
                            "Content Score": data["avg_content_score"],
                            "Completion Score": data["avg_completion_score"],
                        }

                # Create data for visualization
                user_type_data = {"values": {}}

                # Restructure data for grouped bar chart
                for metric in [
                    "Weighted Engagement",
                    "Content Score",
                    "Completion Score",
                ]:
                    user_type_data["values"][metric] = {
                        user_type: metrics[user_type][metric] for user_type in metrics
                    }

                # Create grouped bar chart using utility function
                fig1, ax = create_visualization(
                    data=user_type_data,
                    viz_type="grouped_bar",
                    title="Learning Engagement Metrics by User Type",
                    xlabel="User Type",
                    ylabel="Score",
                    add_data_labels=True,
                    grid=True,
                    colors=["#007BFF", "#28A745", "#FFC107"],
                )

                # Add user count annotations
                for i, user_type in enumerate(metrics.keys()):
                    user_count = user_type_metrics[user_type].get("user_count", 0)
                    ax.annotate(
                        f"n={user_count}",
                        xy=(i, 0.05),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        alpha=0.7,
                    )

                # Add the figure to the result
                result["figures"]["user_type_metrics"] = fig1

        # Create figure for department analysis
        if "engagement_metrics" in department_analysis:
            dept_metrics = department_analysis["engagement_metrics"]

            if dept_metrics:
                # Prepare data for heatmap
                dept_data = []

                for dept, metrics in dept_metrics.items():
                    dept_data.append(
                        {
                            "department": dept,
                            "user_count": metrics.get("user_count", 0),
                            "weighted_engagement": metrics.get(
                                "weighted_engagement", 0
                            ),
                            "avg_content_score": metrics.get("avg_content_score", 0),
                            "avg_completion_score": metrics.get(
                                "avg_completion_score", 0
                            ),
                        }
                    )

                if dept_data:
                    df = pd.DataFrame(dept_data)

                    # Sort by user count
                    df = df.sort_values(by="user_count", ascending=False)

                    # Limit to top 10 departments
                    if len(df) > 10:
                        df = df.head(10)

                    # Prepare data for heatmap
                    heatmap_df = pd.melt(
                        df,
                        id_vars=["department", "user_count"],
                        value_vars=[
                            "weighted_engagement",
                            "avg_content_score",
                            "avg_completion_score",
                        ],
                        var_name="metric",
                        value_name="score",
                    )

                    # Clean up metric names
                    heatmap_df["metric"] = heatmap_df["metric"].map(
                        {
                            "weighted_engagement": "Weighted Engagement",
                            "avg_content_score": "Content Score",
                            "avg_completion_score": "Completion Score",
                        }
                    )

                    # Prepare data for heatmap using utility function
                    heatmap_data = prepare_heatmap_data(
                        heatmap_df,
                        row_field="department",
                        col_field="metric",
                        value_field="score",
                    )

                    # Create heatmap using utility function
                    fig2, ax = create_visualization(
                        data=heatmap_data,
                        viz_type="heatmap",
                        title="Learning Engagement Metrics by Department",
                        color_map="YlGnBu",
                        add_values=True,
                        value_format="{:.2f}",
                    )

                    # Add user count as text
                    for i, dept in enumerate(df["department"]):
                        count = df.loc[df["department"] == dept, "user_count"].iloc[0]
                        ax.text(
                            -0.5,
                            i + 0.5,
                            f"n={count}",
                            va="center",
                            ha="right",
                            fontsize=8,
                        )

                    # Add the figure to the result
                    result["figures"]["department_metrics"] = fig2

        # Create figure for improvement correlation if available
        plot_data = []

        # Process data from all demographic analyses
        # Process user type data
        if "improvement_correlation" in user_type_analysis:
            user_type_corr = user_type_analysis["improvement_correlation"]
            if "by_group" in user_type_corr:
                for group, data in user_type_corr["by_group"].items():
                    plot_data.append(
                        {
                            "group": group,
                            "category": "User Type",
                            "engagement_factor": data.get("engagement_factor", 0),
                            "improvement_efficiency": data.get(
                                "improvement_efficiency", 0
                            ),
                        }
                    )

        # Process department data
        if "improvement_correlation" in department_analysis:
            dept_corr = department_analysis["improvement_correlation"]
            if "by_group" in dept_corr:
                for group, data in dept_corr["by_group"].items():
                    plot_data.append(
                        {
                            "group": group,
                            "category": "Department",
                            "engagement_factor": data.get("engagement_factor", 0),
                            "improvement_efficiency": data.get(
                                "improvement_efficiency", 0
                            ),
                        }
                    )

        # Process experience data
        if "improvement_correlation" in experience_analysis:
            exp_corr = experience_analysis["improvement_correlation"]
            if "by_group" in exp_corr:
                for group, data in exp_corr["by_group"].items():
                    plot_data.append(
                        {
                            "group": group,
                            "category": "Experience",
                            "engagement_factor": data.get("engagement_factor", 0),
                            "improvement_efficiency": data.get(
                                "improvement_efficiency", 0
                            ),
                        }
                    )

        if plot_data:
            # Convert to DataFrame
            df = pd.DataFrame(plot_data)

            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            # Prepare data for scatter plot
            scatter_data = prepare_scatter_plot_data(
                df,
                x_field="engagement_factor",
                y_field="improvement_efficiency",
                category_field="category",
                add_trend=True,
            )

            # Create scatter plot using utility function
            fig3, ax = create_visualization(
                data=scatter_data,
                viz_type="scatter",
                title="Learning Improvement Efficiency vs. Engagement by Demographic Group",
                xlabel="Engagement Factor",
                ylabel="Improvement Efficiency",
                add_trend_line=True,
                grid=True,
                colors={
                    "User Type": "#007BFF",
                    "Department": "#28A745",
                    "Experience": "#FFC107",
                },
            )

            # Add labels for each point
            for i, row in df.iterrows():
                ax.annotate(
                    row["group"],
                    (row["engagement_factor"], row["improvement_efficiency"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            # Add reference lines
            add_reference_line(
                ax,
                1,
                orientation="horizontal",
                color="black",
                linestyle="--",
                alpha=0.3,
            )
            add_reference_line(
                ax, 0, orientation="vertical", color="black", linestyle="--", alpha=0.3
            )

            # Add annotation for interpretation
            ax.annotate(
                "Higher values indicate greater learning improvement\nper unit of engagement",
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Add the figure to the result
            result["figures"]["improvement_efficiency"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"demographic_impact_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_learning_objectives_by_step(
        self, step_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize how different framework steps contribute to learning objectives.

        Args:
            step_data: Data from LearningAnalyzer.analyze_learning_objectives_by_step()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in step_data:
            self._logger.error(
                f"Error in learning objectives by step data: {step_data['error']}"
            )
            return result

        # Extract step data
        step_metrics = step_data.get("step_metrics", {})
        steps_by_impact = step_data.get("steps_by_impact", [])
        impact_groups = step_data.get("impact_groups", {})
        bottlenecks = step_data.get("bottlenecks", {})

        if not step_metrics and not steps_by_impact and not impact_groups:
            self._logger.warning(
                "Insufficient data for learning objectives by step visualization"
            )
            return result

        # Create figure for step impact scores
        if step_metrics:
            # Convert step names to more readable format
            step_labels = {
                step.replace("-", " ").title(): score
                for step, score in step_metrics.items()
            }

            # Sort by absolute impact score
            sorted_steps = dict(
                sorted(step_labels.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            # Create data for visualization
            impact_data = {"values": sorted_steps}

            # Create bar chart using utility function
            fig1, ax = create_visualization(
                data=impact_data,
                viz_type="bar",
                title="Framework Steps Impact on Learning Outcomes",
                xlabel="Framework Step",
                ylabel="Impact Score",
                reference_value=0,
                add_data_labels=True,
                grid=True,
                xticklabel_rotation=45,
                colors=[
                    "#28A745" if score >= 0 else "#DC3545"
                    for score in sorted_steps.values()
                ],
            )

            # Add the figure to the result
            result["figures"]["step_impact_scores"] = fig1

        # Create figure for impact groups
        if impact_groups:
            # Extract impact groups
            high_impact = impact_groups.get("high_impact", [])
            medium_impact = impact_groups.get("medium_impact", [])
            low_impact = impact_groups.get("low_impact", [])

            # Create scatter plot visualization for impact groups
            # This is a specialized visualization that doesn't directly map to utility functions,
            # so we'll use a custom approach

            fig2, ax = create_figure(width=12, height=8)

            # Prepare data
            all_steps = []
            impact_levels = []

            for step in high_impact:
                all_steps.append(step)
                impact_levels.append("High Impact")

            for step in medium_impact:
                all_steps.append(step)
                impact_levels.append("Medium Impact")

            for step in low_impact:
                all_steps.append(step)
                impact_levels.append("Low Impact")

            # Get step numbers if available (from bottlenecks)
            step_numbers = {}

            if "completion_bottlenecks" in bottlenecks:
                for bottleneck in bottlenecks["completion_bottlenecks"]:
                    if "from_step" in bottleneck and "from_step_number" in bottleneck:
                        step_numbers[bottleneck["from_step"]] = bottleneck[
                            "from_step_number"
                        ]
                    if "to_step" in bottleneck and "to_step_number" in bottleneck:
                        step_numbers[bottleneck["to_step"]] = bottleneck[
                            "to_step_number"
                        ]

            # Create sequential numbers for steps without numbers
            for step in all_steps:
                if step not in step_numbers:
                    step_numbers[step] = len(step_numbers) + 1

            # Create x-positions based on step numbers
            x_positions = [step_numbers.get(step, 0) for step in all_steps]

            # Create random y jitter for visual separation
            import random

            y_jitter = [random.uniform(-0.2, 0.2) for _ in range(len(all_steps))]

            # Create colors based on impact level
            impact_colors = {
                "High Impact": "#28A745",
                "Medium Impact": "#FFC107",
                "Low Impact": "#DC3545",
            }

            point_colors = [
                impact_colors.get(level, "#6C757D") for level in impact_levels
            ]

            # Plot points
            sc = ax.scatter(
                x_positions,
                y_jitter,
                c=point_colors,
                s=100,
                alpha=0.8,
                edgecolors="white",
            )

            # Add step labels
            for i, step in enumerate(all_steps):
                ax.annotate(
                    step.replace("-", " ").title(),
                    (x_positions[i], y_jitter[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            # Configure axes
            configure_axes(
                ax,
                title="Framework Steps by Learning Impact",
                xlabel="Step Sequence",
                grid=True,
            )
            ax.set_yticks([])  # Hide y-axis ticks

            # Set integer ticks for x-axis
            max_step_num = max(step_numbers.values()) if step_numbers else 0
            ax.set_xticks(range(1, max_step_num + 1))

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=impact_colors["High Impact"],
                    label="High Impact",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=impact_colors["Medium Impact"],
                    label="Medium Impact",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=impact_colors["Low Impact"],
                    label="Low Impact",
                    markersize=10,
                ),
            ]

            ax.legend(handles=legend_elements, loc="upper right")

            # Apply tight layout
            fig2.tight_layout()

            # Add the figure to the result
            result["figures"]["impact_groups"] = fig2

        # Create figure for bottlenecks
        if bottlenecks and "completion_bottlenecks" in bottlenecks:
            # Extract bottleneck data
            completion_bottlenecks = bottlenecks.get("completion_bottlenecks", [])

            if completion_bottlenecks:
                fig3, ax = create_figure(width=14, height=8)

                # Extract steps and connections
                steps = set()
                connections = []

                for bottleneck in completion_bottlenecks:
                    from_step = bottleneck.get("from_step", "")
                    to_step = bottleneck.get("to_step", "")
                    drop_off = bottleneck.get("drop_off_rate", 0)

                    if from_step and to_step:
                        steps.add(from_step)
                        steps.add(to_step)
                        connections.append((from_step, to_step, drop_off))

                # Create positions for steps
                step_positions = {}

                # Use step numbers if available
                step_numbers = {}
                for bottleneck in completion_bottlenecks:
                    if "from_step" in bottleneck and "from_step_number" in bottleneck:
                        step_numbers[bottleneck["from_step"]] = bottleneck[
                            "from_step_number"
                        ]
                    if "to_step" in bottleneck and "to_step_number" in bottleneck:
                        step_numbers[bottleneck["to_step"]] = bottleneck[
                            "to_step_number"
                        ]

                # Create sequential numbers for steps without numbers
                for step in steps:
                    if step not in step_numbers:
                        step_numbers[step] = len(step_numbers) + 1

                # Set positions based on step numbers (horizontal sequence)
                for step in steps:
                    step_num = step_numbers.get(step, 0)
                    step_positions[step] = (step_num, 0)

                # Normalize positions to 0-1 range
                max_x = (
                    max(pos[0] for pos in step_positions.values())
                    if step_positions
                    else 1
                )

                for step, pos in step_positions.items():
                    step_positions[step] = (pos[0] / max_x, pos[1])

                # Plot steps
                for step, (x, y) in step_positions.items():
                    ax.scatter(
                        x,
                        y,
                        s=200,
                        color="#007BFF",
                        alpha=0.8,
                        edgecolors="white",
                        zorder=2,
                    )

                    # Add step label
                    ax.annotate(
                        step.replace("-", " ").title(),
                        (x, y),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )

                # Plot connections
                for from_step, to_step, drop_off in connections:
                    if from_step in step_positions and to_step in step_positions:
                        from_x, from_y = step_positions[from_step]
                        to_x, to_y = step_positions[to_step]

                        # Calculate arrow properties
                        arrow_width = drop_off * 5  # Scale by drop-off rate
                        arrow_color = (
                            "#DC3545" if drop_off > 0.3 else "#FD7E14"
                        )  # Red for severe, orange for moderate

                        # Draw arrow
                        ax.annotate(
                            "",
                            xy=(to_x, to_y),
                            xytext=(from_x, from_y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color=arrow_color,
                                lw=max(1, arrow_width),
                                alpha=0.8,
                                shrinkA=10,
                                shrinkB=10,
                            ),
                            zorder=1,
                        )

                        # Add drop-off label
                        mid_x = (from_x + to_x) / 2
                        mid_y = (from_y + to_y) / 2 + 0.05

                        ax.text(
                            mid_x,
                            mid_y,
                            f"-{drop_off:.2f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color=arrow_color,
                            fontweight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor="white", alpha=0.8
                            ),
                        )

                # Configure axes
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.3, 0.3)
                ax.set_title("Framework Completion Bottlenecks")
                ax.axis("off")  # Hide axes

                # Add legend
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(
                        facecolor="#DC3545", alpha=0.8, label="Severe Drop-off (>0.3)"
                    ),
                    Patch(
                        facecolor="#FD7E14", alpha=0.8, label="Moderate Drop-off (≤0.3)"
                    ),
                ]

                ax.legend(handles=legend_elements, loc="upper right")

                # Add the figure to the result
                result["figures"]["bottlenecks"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_objectives_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_learning_outcomes_vs_engagement(
        self, engagement_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the relationship between tool engagement and learning outcomes.

        Args:
            engagement_data: Data from LearningAnalyzer.analyze_learning_outcomes_vs_engagement()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in engagement_data:
            self._logger.error(
                f"Error in learning vs engagement data: {engagement_data['error']}"
            )
            return result

        # Extract data
        semester_data = engagement_data.get("semester_data", {})
        correlations = engagement_data.get("correlations", {})

        if not semester_data:
            self._logger.warning(
                "Insufficient data for learning outcomes vs engagement visualization"
            )
            return result

        # Create figure for semester comparison
        # Prepare data for visualization
        semesters = []
        learning_scores = []
        high_engagement = []
        medium_engagement = []
        low_engagement = []
        tool_versions = []

        for semester, data in semester_data.items():
            if "error" in data:
                continue

            semesters.append(semester)
            learning_scores.append(data.get("learning_outcome_score", 0))

            # Extract engagement distribution
            total_users = sum(
                data.get("engagement", {}).get("distribution", {}).values()
            )
            if total_users > 0:
                high_engagement.append(
                    data.get("engagement", {}).get("distribution", {}).get("high", 0)
                    / total_users
                )
                medium_engagement.append(
                    data.get("engagement", {}).get("distribution", {}).get("medium", 0)
                    / total_users
                )
                low_engagement.append(
                    data.get("engagement", {}).get("distribution", {}).get("low", 0)
                    / total_users
                )
            else:
                high_engagement.append(0)
                medium_engagement.append(0)
                low_engagement.append(0)

            tool_versions.append(data.get("tool_version", "none"))

        # Create data for stacked bar chart
        stacked_data = {
            "values": {
                "High Engagement": dict(zip(semesters, high_engagement)),
                "Medium Engagement": dict(zip(semesters, medium_engagement)),
                "Low Engagement": dict(zip(semesters, low_engagement)),
            }
        }

        # Create stacked bar chart using utility function
        fig1, ax = create_visualization(
            data=stacked_data,
            viz_type="stacked_bar",
            title="Engagement Distribution and Learning Outcomes by Semester",
            xlabel="Semester",
            ylabel="User Distribution",
            add_data_labels=True,
            grid=True,
            colors=[
                self._colors["engagement_levels"]["high"],
                self._colors["engagement_levels"]["medium"],
                self._colors["engagement_levels"]["low"],
            ],
        )

        # Add learning outcome scores on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(semesters)),
            learning_scores,
            "o-",
            label="Learning Outcome Score",
            color="#6f42c1",  # Purple
            linewidth=2,
            markersize=8,
        )
        ax2.set_ylabel("Learning Outcome Score")
        ax2.legend(loc="upper right")

        # Add tool version annotations
        for i, version in enumerate(tool_versions):
            ax.annotate(
                f"v{version}" if version != "none" else "No Tool",
                xy=(i, -0.05),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
            )

        # Add the figure to the result
        result["figures"]["semester_comparison"] = fig1

        # Create figure for correlations if available
        if correlations:
            # Prepare data for bar chart
            correlation_data = {"values": correlations}

            # Clean up metric names for display
            cleaned_correlations = {}
            for metric, value in correlations.items():
                if metric == "high_engagement_vs_outcome":
                    cleaned_correlations["High Engagement %"] = value
                elif metric == "avg_ideas_vs_outcome":
                    cleaned_correlations["Avg Ideas per User"] = value
                elif metric == "avg_steps_vs_outcome":
                    cleaned_correlations["Avg Steps per User"] = value
                else:
                    cleaned_correlations[metric.replace("_", " ").title()] = value

            # Create bar chart using utility function
            fig2, ax = create_visualization(
                data={"values": cleaned_correlations},
                viz_type="bar",
                title="Correlation: Engagement Metrics vs. Learning Outcomes",
                xlabel="Engagement Metric",
                ylabel="Correlation with Learning Outcomes",
                reference_value=0,
                add_data_labels=True,
                grid=True,
                colors=[
                    (
                        "#28A745"
                        if val >= 0.5
                        else (
                            "#007BFF"
                            if val >= 0.3
                            else (
                                "#FFC107"
                                if val >= 0
                                else "#FD7E14" if val >= -0.3 else "#DC3545"
                            )
                        )
                    )
                    for val in cleaned_correlations.values()
                ],
            )

            # Add correlation strength legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#28A745", alpha=0.8, label="Strong Positive (≥0.5)"),
                Patch(
                    facecolor="#007BFF", alpha=0.8, label="Moderate Positive (0.3-0.5)"
                ),
                Patch(facecolor="#FFC107", alpha=0.8, label="Weak Positive (0-0.3)"),
                Patch(facecolor="#FD7E14", alpha=0.8, label="Weak Negative (-0.3-0)"),
                Patch(
                    facecolor="#DC3545", alpha=0.8, label="Stronger Negative (<-0.3)"
                ),
            ]

            ax.legend(handles=legend_elements, loc="best", title="Correlation Strength")

            # Add the figure to the result
            result["figures"]["correlations"] = fig2

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"learning_engagement_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def visualize_idea_quality_vs_learning(
        self, quality_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the relationship between idea quality and learning outcomes.

        Args:
            quality_data: Data from LearningAnalyzer.analyze_idea_quality_vs_learning()
            save_path: Optional path to save the visualization

        Returns:
            Dict[str, Any]: Dictionary with figure objects and save paths
        """
        result = {"figures": {}}

        # Skip if there's an error in the data
        if "error" in quality_data:
            self._logger.error(
                f"Error in idea quality vs learning data: {quality_data['error']}"
            )
            return result

        # Extract data
        semester_data = quality_data.get("semester_data", {})
        correlations = quality_data.get("correlations", {})

        if not semester_data:
            self._logger.warning(
                "Insufficient data for idea quality vs learning visualization"
            )
            return result

        # Create figure for semester comparison
        # Prepare data for visualization
        df_data = []

        for semester, data in semester_data.items():
            df_data.append(
                {
                    "semester": semester,
                    "learning_outcome_score": data.get("learning_outcome_score", 0),
                    "avg_quality_score": data.get("idea_metrics", {}).get(
                        "avg_quality_score", 0
                    ),
                    "avg_step_completion_rate": data.get("idea_metrics", {}).get(
                        "avg_step_completion_rate", 0
                    ),
                    "idea_count": data.get("idea_metrics", {}).get("count", 0),
                    "tool_version": data.get("tool_version", "none"),
                }
            )

        if df_data:
            df = pd.DataFrame(df_data)

            # Prepare scatter plot data using utility function
            quality_scatter_data = prepare_scatter_plot_data(
                df,
                x_field="avg_quality_score",
                y_field="learning_outcome_score",
                category_field="tool_version",
                add_trend=True,
            )

            # Create scatter plot using utility function
            fig1, ax = create_visualization(
                data=quality_scatter_data,
                viz_type="scatter",
                title="Relationship Between Idea Quality and Learning Outcomes",
                xlabel="Idea Quality Score",
                ylabel="Learning Outcome Score",
                add_trend_line=True,
                add_correlation=True,
                grid=True,
                colors={
                    "none": self._colors["tool_versions"]["none"],
                    "v1": self._colors["tool_versions"]["v1"],
                    "v2": self._colors["tool_versions"]["v2"],
                },
            )

            # Add semester labels
            for i, row in df.iterrows():
                ax.annotate(
                    row["semester"],
                    (row["avg_quality_score"], row["learning_outcome_score"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            # Add tool version legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["none"],
                    label="No Tool",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["v1"],
                    label="Tool v1",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._colors["tool_versions"]["v2"],
                    label="Tool v2",
                    markersize=10,
                ),
            ]

            ax.legend(handles=legend_elements, loc="best")

            # Add the figure to the result
            result["figures"]["quality_scatter"] = fig1

            # Create scatter plot for completion rate
            completion_scatter_data = prepare_scatter_plot_data(
                df,
                x_field="avg_step_completion_rate",
                y_field="learning_outcome_score",
                category_field="tool_version",
                add_trend=True,
            )

            # Create scatter plot using utility function
            fig2, ax = create_visualization(
                data=completion_scatter_data,
                viz_type="scatter",
                title="Relationship Between Step Completion and Learning Outcomes",
                xlabel="Average Step Completion Rate",
                ylabel="Learning Outcome Score",
                add_trend_line=True,
                add_correlation=True,
                grid=True,
                colors={
                    "none": self._colors["tool_versions"]["none"],
                    "v1": self._colors["tool_versions"]["v1"],
                    "v2": self._colors["tool_versions"]["v2"],
                },
            )

            # Add semester labels
            for i, row in df.iterrows():
                ax.annotate(
                    row["semester"],
                    (row["avg_step_completion_rate"], row["learning_outcome_score"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            # Add tool version legend
            ax.legend(handles=legend_elements, loc="best")

            # Add the figure to the result
            result["figures"]["completion_scatter"] = fig2

        # Create figure for correlation metrics if available
        if correlations:
            # Prepare data for bar chart
            correlation_data = {"values": correlations}

            # Clean up metric names for display
            cleaned_correlations = {}
            for metric, value in correlations.items():
                if metric == "learning_outcome_vs_idea_quality":
                    cleaned_correlations["Idea Quality"] = value
                elif metric == "learning_outcome_vs_step_completion":
                    cleaned_correlations["Step Completion"] = value
                else:
                    cleaned_correlations[metric.replace("_", " ").title()] = value

            # Create bar chart using utility function
            fig3, ax = create_visualization(
                data={"values": cleaned_correlations},
                viz_type="bar",
                title="Correlation: Idea Quality Metrics vs. Learning Outcomes",
                xlabel="Metric",
                ylabel="Correlation with Learning Outcomes",
                reference_value=0,
                add_data_labels=True,
                grid=True,
                colors=[
                    (
                        "#28A745"
                        if val >= 0.5
                        else (
                            "#007BFF"
                            if val >= 0.3
                            else (
                                "#FFC107"
                                if val >= 0
                                else "#FD7E14" if val >= -0.3 else "#DC3545"
                            )
                        )
                    )
                    for val in cleaned_correlations.values()
                ],
            )

            # Add the figure to the result
            result["figures"]["quality_correlations"] = fig3

        # Save figures if requested
        if save_path:
            saved_files = {}

            for name, fig in result["figures"].items():
                filename = generate_filename(f"idea_quality_{name}")
                save_paths = save_figure(fig, filename, directory=save_path)
                saved_files[name] = save_paths

            result["saved_files"] = saved_files

        return result

    def create_comprehensive_report(
        self, learning_analyzer: Any, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive visualization report for all learning analyzer metrics.

        Args:
            learning_analyzer: LearningAnalyzer instance to get data from
            output_dir: Optional custom output directory for report

        Returns:
            Dict[str, Any]: Report metadata and paths
        """
        # Generate report directory
        report_dir, subdirs = create_report_directory(
            "Learning_Outcomes_Report",
            base_dir=output_dir or self._output_dir,
            include_timestamp=True,
        )

        report_result = {
            "report_dir": str(report_dir),
            "subdirs": {name: str(path) for name, path in subdirs.items()},
            "visualizations": {},
        }

        try:
            # Run all analyses and create visualizations

            # 1. Tool usage and course ratings correlation
            correlation_data = (
                learning_analyzer.correlate_tool_usage_with_course_ratings()
            )
            correlation_viz = self.visualize_course_rating_engagement_correlation(
                correlation_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["course_ratings"] = correlation_viz

            # 2. Learning outcomes by cohort
            cohort_data = learning_analyzer.compare_learning_outcomes_by_cohort()
            cohort_viz = self.visualize_learning_outcomes_by_cohort(
                cohort_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["learning_cohorts"] = cohort_viz

            # 3. Tool version impact
            version_data = learning_analyzer.analyze_tool_version_impact()
            version_viz = self.visualize_tool_version_impact(
                version_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["tool_versions"] = version_viz

            # 4. Framework engagement metrics
            framework_data = learning_analyzer.calculate_framework_engagement_metrics()
            framework_viz = self.visualize_framework_engagement(
                framework_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["framework_engagement"] = framework_viz

            # 5. Tool impact on time allocation
            time_data = learning_analyzer.analyze_tool_impact_on_time_allocation()
            time_viz = self.visualize_time_allocation_impact(
                time_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["time_allocation"] = time_viz

            # 6. Combined learning metrics
            metrics_data = learning_analyzer.analyze_combined_learning_metrics()
            metrics_viz = self.visualize_learning_metrics_summary(
                metrics_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["learning_summary"] = metrics_viz

            # 7. Learning outcome metrics
            outcome_metrics = learning_analyzer.get_learning_outcome_metrics()
            outcome_viz = self.visualize_learning_outcome_metrics(
                outcome_metrics, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["outcome_metrics"] = outcome_viz

            # 8. Demographic learning impact
            demographic_data = learning_analyzer.analyze_demographic_learning_impact()
            demographic_viz = self.visualize_demographic_impact(
                demographic_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["demographic_impact"] = demographic_viz

            # 9. Learning objectives by step
            step_data = learning_analyzer.analyze_learning_objectives_by_step()
            step_viz = self.visualize_learning_objectives_by_step(
                step_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["step_impact"] = step_viz

            # 10. Learning outcomes vs engagement
            engagement_data = (
                learning_analyzer.analyze_learning_outcomes_vs_engagement()
            )
            engagement_viz = self.visualize_learning_outcomes_vs_engagement(
                engagement_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["outcomes_vs_engagement"] = engagement_viz

            # 11. Idea quality vs learning
            quality_data = learning_analyzer.analyze_idea_quality_vs_learning()
            quality_viz = self.visualize_idea_quality_vs_learning(
                quality_data, save_path=str(subdirs["figures"])
            )
            report_result["visualizations"]["idea_quality"] = quality_viz

            # Create a README file with report summary
            self._create_report_readme(
                report_dir, report_result, title="Learning Outcomes Analysis Report"
            )

        except Exception as e:
            self._logger.error(f"Error creating comprehensive report: {e}")
            report_result["error"] = str(e)

        return report_result

    def _create_report_readme(
        self,
        report_dir: Path,
        report_data: Dict[str, Any],
        title: str = "Analysis Report",
    ) -> Path:
        """
        Create a README file summarizing the report contents.

        Args:
            report_dir: Directory where the report is saved
            report_data: Report metadata and paths
            title: Report title

        Returns:
            Path: Path to the created README file
        """
        readme_path = report_dir / "README.md"

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Count visualizations
        viz_count = sum(
            len(v.get("figures", {}))
            for v in report_data.get("visualizations", {}).values()
        )

        # Create README content
        content = f"""# {title}

            *Generated on: {timestamp}*

            ## Overview

            This report contains visualizations of learning outcomes and educational impact analysis for the JetPack/Orbit tool. The report includes {viz_count} visualizations across multiple analysis categories.

            ## Contents

            The report is organized into the following directories:

            - `figures/`: Contains all visualization images in PNG, PDF and SVG formats
            - `data/`: Contains raw data used for analysis
            - `tables/`: Contains tabular data extracted from the analysis

            ## Visualization Categories

        """

        # Add visualization categories
        for category, viz_data in report_data.get("visualizations", {}).items():
            figure_count = len(viz_data.get("figures", {}))
            if figure_count > 0:
                category_name = category.replace("_", " ").title()
                content += f"### {category_name}\n\n"
                content += f"Contains {figure_count} visualizations related to {category_name.lower()}.\n\n"

                # List figures in the category
                if "figures" in viz_data:
                    content += "**Visualizations:**\n\n"
                    for fig_name in viz_data["figures"].keys():
                        readable_name = fig_name.replace("_", " ").title()
                        content += f"- {readable_name}\n"
                    content += "\n"

        # Add usage notes
        content += """## Usage Notes

            To view the visualizations, navigate to the `figures/` directory and open the PNG or PDF files. The visualizations are organized by analysis category and provide insights into the effectiveness and impact of the JetPack/Orbit tool on learning outcomes.

            ## Analysis Summary

            The visualizations in this report analyze several key aspects of the tool's educational impact:

            1. **Learning Outcomes by Cohort**: Comparison of learning outcomes between pre-tool and post-tool cohorts.
            2. **Tool Version Impact**: Analysis of differences between tool versions and their impact on learning.
            3. **Framework Engagement**: Analysis of engagement with different entrepreneurial frameworks.
            4. **Time Allocation Impact**: How the tool affects time allocation for entrepreneurship education.
            5. **Demographic Impact**: Analysis of the tool's impact across different student demographics.
            6. **Learning Objectives by Step**: Which framework steps contribute most to learning objectives.
            7. **Outcomes vs. Engagement**: Relationship between tool engagement and learning outcomes.
            8. **Idea Quality vs. Learning**: How idea quality correlates with learning effectiveness.
        """

        # Write content to the README file
        with open(readme_path, "w") as f:
            f.write(content)

        return readme_path
