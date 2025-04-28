"""
Framework visualizer for the data analysis system.

This module provides visualization capabilities for framework analysis results,
generating charts and graphs to represent framework progression, completion rates,
step dependencies, and other key metrics related to entrepreneurial frameworks.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from matplotlib.figure import Figure

from src.data.models.enums import FrameworkType, Semester
from src.analyzers.framework_analyzer import FrameworkAnalyzer
from src.utils.visualization_creation_utils import (
    create_figure,
    configure_axes,
    get_color_palette,
    plot_bar,
    plot_grouped_bars,
    plot_stacked_bars,
    plot_line,
    add_data_table,
    add_reference_line,
    wrap_labels,
    save_figure,
)
from src.utils.visualization_data_utils import (
    generate_filename,
    get_output_path,
)


class FrameworkVisualizer:
    """
    Visualizer for framework analysis results.

    This class generates visualizations for framework analysis results,
    including framework completion rates, progression patterns, step dependencies,
    and other key metrics.
    """

    def __init__(
        self,
        framework_analyzer: FrameworkAnalyzer,
        output_dir: Optional[str] = None,
        include_timestamps: bool = True,
        default_figsize: Tuple[float, float] = (10, 6),
        theme: str = "default",
    ):
        """
        Initialize the framework visualizer.

        Args:
            framework_analyzer: Framework analyzer instance for data access
            output_dir: Optional directory for saving visualizations
            include_timestamps: Whether to include timestamps in filenames
            default_figsize: Default figure size (width, height) in inches
            theme: Visual theme ('default', 'dark', 'print')
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._analyzer = framework_analyzer
        self._include_timestamps = include_timestamps
        self._default_figsize = default_figsize
        self._theme = theme

        # Set output directory
        if output_dir:
            self._output_dir = Path(output_dir)
        else:
            # Use default project output directory
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / "README.md").exists() or (
                    current_dir / ".git"
                ).exists():
                    break
                current_dir = current_dir.parent

            self._output_dir = current_dir / "output" / "visualizations" / "framework"

        # Create output directory if it doesn't exist
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_framework_completion(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for framework completion metrics.

        Args:
            framework: The framework to visualize
            course_id: Optional course ID to filter users
            include_ideas_without_steps: Whether to include ideas without steps
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get framework completion metrics from analyzer
        metrics = self._analyzer.get_framework_completion_metrics(
            framework, course_id, include_ideas_without_steps
        )

        if "error" in metrics:
            self._logger.error(
                f"Error getting framework completion metrics: {metrics['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create step completion rate visualization (bar chart)
        step_completion = {
            step_name: data["rate"]
            for step_name, data in metrics["step_completion_rates"].items()
        }

        # Get step numbers for sorting
        step_numbers = {
            step_name: data["step_number"]
            for step_name, data in metrics["step_completion_rates"].items()
        }

        # Sort steps by step number
        sorted_steps = sorted(step_completion.items(), key=lambda x: step_numbers[x[0]])
        steps_data = {name: rate for name, rate in sorted_steps}

        # Create bar chart
        fig_completion, ax_completion = create_figure(
            *self._default_figsize, theme=self._theme
        )

        # Convert to percentage
        steps_data_pct = {k: v * 100 for k, v in steps_data.items()}

        plot_bar(
            ax_completion,
            data=steps_data_pct,
            orientation="vertical",
            color=get_color_palette("sequential_blue", n_colors=1)[0],
            add_data_labels=True,
            data_label_format="{:.1f}%",
        )

        configure_axes(
            ax_completion,
            title=f"{framework.value} Framework Step Completion Rates",
            xlabel="Framework Steps",
            ylabel="Completion Rate (%)",
            xticklabel_rotation=45,
            ylim=(0, 100),
        )

        # Ensure x-axis labels are readable
        wrap_labels(ax_completion, "x", max_length=20)

        fig_completion.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_step_completion"
        else:
            filename = f"{framework.value.lower()}_step_completion"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_completion,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["step_completion"] = fig_completion

        # 2. Create completion distribution visualization (pie charts)
        fig_dist, axes = plt.subplots(1, 2, figsize=self._default_figsize)

        # User completion distribution
        user_dist = metrics["user_completion_distribution"]

        # Convert keys to percentages of framework
        total_steps = len(metrics["step_completion_rates"])
        user_dist_labels = {
            f"{int(float(k.split('-')[0])/total_steps*100)}-{int(float(k.split('-')[1])/total_steps*100)}%": v
            for k, v in user_dist.items()
        }

        axes[0].pie(
            user_dist_labels.values(),
            labels=user_dist_labels.keys(),
            autopct="%1.1f%%",
            startangle=90,
            colors=get_color_palette(
                "categorical_main", n_colors=len(user_dist_labels)
            ),
        )
        axes[0].set_title("User Completion Distribution")

        # Idea completion distribution
        idea_dist = metrics["idea_completion_distribution"]

        # Convert keys to percentages of framework
        idea_dist_labels = {
            f"{int(float(k.split('-')[0])/total_steps*100)}-{int(float(k.split('-')[1])/total_steps*100)}%": v
            for k, v in idea_dist.items()
        }

        axes[1].pie(
            idea_dist_labels.values(),
            labels=idea_dist_labels.keys(),
            autopct="%1.1f%%",
            startangle=90,
            colors=get_color_palette(
                "categorical_main", n_colors=len(idea_dist_labels)
            ),
        )
        axes[1].set_title("Idea Completion Distribution")

        fig_dist.suptitle(f"{framework.value} Framework Completion Distribution")
        fig_dist.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_completion_distribution"
        else:
            filename = f"{framework.value.lower()}_completion_distribution"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_dist,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["completion_distribution"] = fig_dist

        # 3. Create overall metrics visualization (summary table)
        fig_metrics, ax_metrics = create_figure(8, 4, theme=self._theme)

        # Format metrics for display
        display_metrics = [
            ["Total Users", f"{metrics['overall_metrics']['total_users']}"],
            ["Total Ideas", f"{metrics['overall_metrics']['total_ideas']}"],
            ["Ideas with Steps", f"{metrics['overall_metrics']['ideas_with_steps']}"],
            [
                "Idea to Step Conversion",
                f"{metrics['overall_metrics']['idea_to_step_conversion_rate']:.2f}",
            ],
            [
                "Avg Steps per Idea",
                f"{metrics['overall_metrics']['avg_steps_per_idea']:.2f}",
            ],
            [
                "Avg Completion %",
                f"{metrics['overall_metrics']['avg_completion_percentage']:.2f}%",
            ],
        ]

        # Hide axes
        ax_metrics.axis("off")

        # Add table
        add_data_table(
            ax_metrics,
            data=display_metrics,
            title=f"{framework.value} Framework Metrics Summary",
            fontsize=12,
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.8],
        )

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_metrics_summary"
        else:
            filename = f"{framework.value.lower()}_metrics_summary"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_metrics,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["metrics_summary"] = fig_metrics

        return figures

    def visualize_progression_patterns(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_steps: int = 3,
        max_patterns: int = 5,
        course_id: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for framework progression patterns.

        Args:
            framework: The framework to visualize
            min_steps: Minimum number of steps to consider a pattern
            max_patterns: Maximum number of patterns to return
            course_id: Optional course ID to filter users
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get progression pattern data from analyzer
        pattern_data = self._analyzer.identify_common_progression_patterns(
            framework, min_steps, max_patterns, course_id
        )

        if "error" in pattern_data:
            self._logger.error(
                f"Error getting progression patterns: {pattern_data['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create common patterns visualization
        fig_patterns, ax_patterns = create_figure(
            *self._default_figsize, theme=self._theme
        )

        if pattern_data["common_patterns"]:
            # Extract pattern data
            patterns = {}
            for i, pattern in enumerate(pattern_data["common_patterns"]):
                # Use abbreviated label with first and last step
                step_names = pattern["step_names"]
                if len(step_names) > 2:
                    label = f"{step_names[0]} → ... → {step_names[-1]}"
                else:
                    label = " → ".join(step_names)

                # Add pattern type to label
                label = f"{label} ({pattern['pattern_type']})"

                patterns[label] = pattern["coverage"] * 100  # Convert to percentage

            # Create bar chart of pattern coverage
            plot_bar(
                ax_patterns,
                data=patterns,
                orientation="horizontal",
                color=get_color_palette("sequential_blue", n_colors=len(patterns)),
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_patterns,
                title=f"Common Progression Patterns in {framework.value} Framework",
                xlabel="Coverage (%)",
                ylabel="Pattern",
                xlim=(0, max(patterns.values()) * 1.1),  # Add 10% padding
            )
        else:
            ax_patterns.text(
                0.5,
                0.5,
                "No common patterns found",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_patterns.transAxes,
            )
            ax_patterns.axis("off")

        fig_patterns.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_common_patterns"
        else:
            filename = f"{framework.value.lower()}_common_patterns"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_patterns,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["common_patterns"] = fig_patterns

        # 2. Create progression types visualization (pie chart)
        fig_types, ax_types = create_figure(8, 6, theme=self._theme)

        if pattern_data["progression_types"]:
            # Extract progression type data
            prog_types = {
                prog_type: data["percentage"] * 100  # Convert to percentage
                for prog_type, data in pattern_data["progression_types"].items()
                if data["count"] > 0  # Only include types with data
            }

            # Create pie chart
            wedges, texts, autotexts = ax_types.pie(
                prog_types.values(),
                labels=prog_types.keys(),
                autopct="%1.1f%%",
                startangle=90,
                colors=get_color_palette("categorical_main", n_colors=len(prog_types)),
                explode=[0.05] * len(prog_types),  # Slight explode for all wedges
            )

            # Customize text
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight("bold")

            for text in texts:
                text.set_fontsize(12)

            ax_types.set_title(f"Progression Types in {framework.value} Framework")
        else:
            ax_types.text(
                0.5,
                0.5,
                "No progression type data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_types.transAxes,
            )
            ax_types.axis("off")

        fig_types.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_progression_types"
        else:
            filename = f"{framework.value.lower()}_progression_types"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_types,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["progression_types"] = fig_types

        # 3. Create starting and ending points visualization
        fig_endpoints, (ax_start, ax_end) = plt.subplots(
            1, 2, figsize=self._default_figsize
        )

        if pattern_data["starting_points"] and pattern_data["ending_points"]:
            # Format starting points data
            start_points = {
                f"{data['step_name']}": data["percentage"] * 100
                for step, data in pattern_data["starting_points"].items()
            }

            # Create bar chart for starting points
            plot_bar(
                ax_start,
                data=start_points,
                orientation="horizontal",
                color=get_color_palette("sequential_green", n_colors=1)[0],
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_start,
                title="Common Starting Points",
                xlabel="Percentage",
                ylabel="Step",
                xlim=(0, max(start_points.values()) * 1.1),  # Add 10% padding
            )

            # Format ending points data
            end_points = {
                f"{data['step_name']}": data["percentage"] * 100
                for step, data in pattern_data["ending_points"].items()
            }

            # Create bar chart for ending points
            plot_bar(
                ax_end,
                data=end_points,
                orientation="horizontal",
                color=get_color_palette("sequential_blue", n_colors=1)[0],
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_end,
                title="Common Ending Points",
                xlabel="Percentage",
                ylabel="Step",
                xlim=(0, max(end_points.values()) * 1.1),  # Add 10% padding
            )
        else:
            for ax, title in [(ax_start, "Starting Points"), (ax_end, "Ending Points")]:
                ax.text(
                    0.5,
                    0.5,
                    f"No {title.lower()} data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.axis("off")

        fig_endpoints.suptitle(f"{framework.value} Framework Step Flow", fontsize=14)
        fig_endpoints.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_step_flow"
        else:
            filename = f"{framework.value.lower()}_step_flow"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_endpoints,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["step_flow"] = fig_endpoints

        return figures

    def visualize_step_dependencies(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        min_correlation: float = 0.3,
        course_id: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for step dependencies.

        Args:
            framework: The framework to visualize
            min_correlation: Minimum correlation threshold
            course_id: Optional course ID to filter users
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get step dependencies data from analyzer
        dependencies = self._analyzer.analyze_step_dependencies(
            framework, min_correlation, course_id
        )

        if "error" in dependencies:
            self._logger.error(
                f"Error getting step dependencies: {dependencies['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create prerequisites heatmap
        fig_prereq, ax_prereq = create_figure(12, 10, theme=self._theme)

        if dependencies["prerequisites"]:
            # Prepare data for heatmap
            steps = list(dependencies["prerequisites"].keys())

            # Create matrix for heatmap
            matrix = np.zeros((len(steps), len(steps)))

            for i, step in enumerate(steps):
                prereqs = dependencies["prerequisites"].get(step, {})
                for j, prereq_step in enumerate(steps):
                    if prereq_step in prereqs:
                        matrix[i, j] = prereqs[prereq_step]

            # Create heatmap
            im = ax_prereq.imshow(
                matrix,
                cmap="Blues",
                interpolation="nearest",
                aspect="auto",
                vmin=0,
                vmax=1,
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_prereq)
            cbar.set_label("Prerequisite Probability")

            # Configure axes
            ax_prereq.set_title(f"Step Prerequisites in {framework.value} Framework")
            ax_prereq.set_xlabel("Prerequisite Step")
            ax_prereq.set_ylabel("Dependent Step")

            # Set tick labels
            short_labels = [
                label.split(". ")[0] if ". " in label else label for label in steps
            ]
            ax_prereq.set_xticks(np.arange(len(steps)))
            ax_prereq.set_yticks(np.arange(len(steps)))
            ax_prereq.set_xticklabels(short_labels, rotation=90)
            ax_prereq.set_yticklabels(short_labels)

            # Add grid
            ax_prereq.set_xticks(np.arange(-0.5, len(steps), 1), minor=True)
            ax_prereq.set_yticks(np.arange(-0.5, len(steps), 1), minor=True)
            ax_prereq.grid(which="minor", color="w", linestyle="-", linewidth=2)
            ax_prereq.tick_params(which="minor", bottom=False, left=False)

            # Add text annotations
            for i in range(len(steps)):
                for j in range(len(steps)):
                    if matrix[i, j] >= min_correlation:
                        text_color = "white" if matrix[i, j] > 0.5 else "black"
                        ax_prereq.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )
        else:
            ax_prereq.text(
                0.5,
                0.5,
                "No prerequisite data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_prereq.transAxes,
            )
            ax_prereq.axis("off")

        fig_prereq.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_prerequisites"
        else:
            filename = f"{framework.value.lower()}_prerequisites"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_prereq,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["prerequisites"] = fig_prereq

        # 2. Create network graph of step relationships
        try:
            import networkx as nx

            fig_network, ax_network = create_figure(14, 10, theme=self._theme)

            if dependencies["sequential_patterns"]:
                # Convert step dependencies to a directed graph
                G = nx.DiGraph()

                # Add all nodes first
                all_steps = set(dependencies["sequential_patterns"].keys())
                for steps_dict in dependencies["sequential_patterns"].values():
                    all_steps.update(steps_dict.keys())

                for step in all_steps:
                    G.add_node(step)

                # Add edges with weights based on dependence strength
                for step, dependents in dependencies["sequential_patterns"].items():
                    for dep_step, strength in dependents.items():
                        if strength >= min_correlation:
                            G.add_edge(step, dep_step, weight=strength)

                # Calculate node positions (use spring layout for directed graphs)
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

                # Get node sizes based on degree centrality (how connected they are)
                centrality = nx.degree_centrality(G)
                node_sizes = [2000 * centrality[node] + 500 for node in G.nodes()]

                # Get edge weights for line thickness
                edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

                # Draw the graph
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax_network,
                    node_size=node_sizes,
                    node_color="skyblue",
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=1,
                )

                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax_network,
                    width=edge_weights,
                    alpha=0.6,
                    edge_color="gray",
                    arrowsize=15,
                    connectionstyle="arc3,rad=0.1",
                )

                # Add short labels for readability
                short_labels = {}
                for step in G.nodes():
                    if ". " in step:
                        # Extract just the number part
                        short_labels[step] = step.split(". ")[0]
                    else:
                        # Use the full label for steps without numbers
                        short_labels[step] = step

                nx.draw_networkx_labels(
                    G,
                    pos,
                    ax=ax_network,
                    labels=short_labels,
                    font_size=10,
                    font_weight="bold",
                )

                ax_network.set_title(
                    f"Step Relationships Network in {framework.value} Framework"
                )
                ax_network.axis("off")
            else:
                ax_network.text(
                    0.5,
                    0.5,
                    "No relationship data available",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax_network.transAxes,
                )
                ax_network.axis("off")

            fig_network.tight_layout()

            # Save figure
            if filename_prefix:
                filename = f"{filename_prefix}_relationship_network"
            else:
                filename = f"{framework.value.lower()}_relationship_network"

                if course_id:
                    filename += f"_{course_id}"

            save_path = get_output_path(
                generate_filename(filename, include_timestamp=self._include_timestamps),
                base_dir=str(self._output_dir),
            )

            save_figure(
                fig_network,
                str(save_path),
                formats=save_formats,
                dpi=300,
                tight_layout=True,
            )

            if return_figures:
                figures["relationship_network"] = fig_network

        except ImportError:
            self._logger.warning(
                "NetworkX is required for step relationship network visualization"
            )

        # 3. Create correlation heatmap
        fig_corr, ax_corr = create_figure(12, 10, theme=self._theme)

        if dependencies["step_correlations"]:
            # Process correlation data for visualization
            corr_data = {}

            # Prepare data for heatmap
            for step, correlations in dependencies["step_correlations"].items():
                # Get short display name for the step
                short_name = step.split(". ")[0] if ". " in step else step

                # Process correlations
                for corr_step, corr_value in correlations.items():
                    # Get short display name for correlated step
                    corr_short = (
                        corr_step.split(". ")[0] if ". " in corr_step else corr_step
                    )

                    # Store in correlation matrix
                    if short_name not in corr_data:
                        corr_data[short_name] = {}

                    corr_data[short_name][corr_short] = corr_value

            # Convert to pandas DataFrame for heatmap
            corr_df = pd.DataFrame(corr_data)

            # Fill NaN values with zeros
            corr_df = corr_df.fillna(0)

            # Create heatmap
            sns.heatmap(
                corr_df,
                annot=True,
                cmap="coolwarm",
                center=0.5,
                linewidths=0.5,
                linecolor="white",
                square=True,
                cbar_kws={"label": "Co-occurrence Probability"},
                ax=ax_corr,
                fmt=".2f",
                annot_kws={"size": 8},
            )

            ax_corr.set_title(f"Step Co-occurrences in {framework.value} Framework")
        else:
            ax_corr.text(
                0.5,
                0.5,
                "No correlation data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_corr.transAxes,
            )
            ax_corr.axis("off")

        fig_corr.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_step_correlations"
        else:
            filename = f"{framework.value.lower()}_step_correlations"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_corr,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["step_correlations"] = fig_corr

        return figures

    def visualize_framework_dropout(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        min_idea_count: int = 10,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for framework dropout analysis.

        Args:
            framework: The framework to visualize
            course_id: Optional course ID to filter users
            min_idea_count: Minimum number of ideas needed for analysis
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get dropout analysis data from analyzer
        dropout_data = self._analyzer.get_framework_dropout_analysis(
            framework, course_id, min_idea_count
        )

        if "error" in dropout_data:
            self._logger.error(
                f"Error getting dropout analysis: {dropout_data['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create dropout by step visualization
        fig_dropout, ax_dropout = create_figure(
            *self._default_figsize, theme=self._theme
        )

        if dropout_data["dropout_by_step"]:
            # Extract dropout rates by step
            dropout_rates = {
                step: data["percentage"] * 100  # Convert to percentage
                for step, data in dropout_data["dropout_by_step"].items()
            }

            # Sort by step number if available
            sorted_dropout = {}
            step_numbers = {
                step: data["step_number"]
                for step, data in dropout_data["dropout_by_step"].items()
            }

            for step in sorted(
                dropout_rates.keys(), key=lambda x: step_numbers.get(x, 0)
            ):
                sorted_dropout[step] = dropout_rates[step]

            # Create bar chart
            bars = plot_bar(
                ax_dropout,
                data=sorted_dropout,
                orientation="vertical",
                color=get_color_palette(
                    "sequential_blue", n_colors=len(sorted_dropout)
                ),
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_dropout,
                title=f"Dropout Rates by Step in {framework.value} Framework",
                xlabel="Framework Steps",
                ylabel="Dropout Rate (%)",
                xticklabel_rotation=45,
                ylim=(0, max(sorted_dropout.values()) * 1.1),  # Add 10% padding
            )

            # Ensure x-axis labels are readable
            wrap_labels(ax_dropout, "x", max_length=15)
        else:
            ax_dropout.text(
                0.5,
                0.5,
                "No dropout data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_dropout.transAxes,
            )
            ax_dropout.axis("off")

        fig_dropout.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_dropout_rates"
        else:
            filename = f"{framework.value.lower()}_dropout_rates"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_dropout,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["dropout_rates"] = fig_dropout

        # 2. Create bottleneck visualization
        fig_bottleneck, ax_bottleneck = create_figure(12, 6, theme=self._theme)

        if dropout_data["completion_bottlenecks"]:
            # Extract bottleneck data
            bottlenecks = {}

            for bottleneck in dropout_data["completion_bottlenecks"]:
                label = f"{bottleneck['from_step']} → {bottleneck['to_step']}"
                bottlenecks[label] = (
                    bottleneck["drop_off_rate"] * 100
                )  # Convert to percentage

            # Create bar chart
            bars = plot_bar(
                ax_bottleneck,
                data=bottlenecks,
                orientation="horizontal",
                color=[
                    "#FF9999" if bottleneck["severity"] == "high" else "#FFCC99"
                    for bottleneck in dropout_data["completion_bottlenecks"]
                ],
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_bottleneck,
                title=f"Completion Bottlenecks in {framework.value} Framework",
                xlabel="Drop-off Rate (%)",
                ylabel="Step Transition",
                xlim=(0, max(bottlenecks.values()) * 1.1),  # Add 10% padding
            )
        else:
            ax_bottleneck.text(
                0.5,
                0.5,
                "No bottleneck data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_bottleneck.transAxes,
            )
            ax_bottleneck.axis("off")

        fig_bottleneck.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_bottlenecks"
        else:
            filename = f"{framework.value.lower()}_bottlenecks"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_bottleneck,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["bottlenecks"] = fig_bottleneck

        # 3. Create abandonment factors visualization
        fig_abandon, (ax_complexity, ax_time) = plt.subplots(1, 2, figsize=(14, 7))

        if dropout_data["abandonment_factors"].get("complexity_factors"):
            # Extract complexity factor data
            complexity_data = {}

            for step, data in dropout_data["abandonment_factors"][
                "complexity_factors"
            ].items():
                complexity_data[step] = (
                    data["user_input_rate"] * 100
                )  # Convert to percentage

            # Create bar chart for complexity factors
            plot_bar(
                ax_complexity,
                data=complexity_data,
                orientation="horizontal",
                color=get_color_palette("sequential_blue", n_colors=1)[0],
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_complexity,
                title="User Input Rate at Dropout Steps",
                xlabel="User Input Rate (%)",
                ylabel="Step",
                xlim=(0, 100),
            )

            # Extract time factor data if available
            if dropout_data["abandonment_factors"].get("time_factors"):
                time_data = {}

                for step, data in dropout_data["abandonment_factors"][
                    "time_factors"
                ].items():
                    time_data[step] = data["avg_time_minutes"]

                # Create bar chart for time factors
                plot_bar(
                    ax_time,
                    data=time_data,
                    orientation="horizontal",
                    color=get_color_palette("sequential_green", n_colors=1)[0],
                    add_data_labels=True,
                    data_label_format="{:.1f} min",
                )

                configure_axes(
                    ax_time,
                    title="Average Time Spent on Dropout Steps",
                    xlabel="Average Time (minutes)",
                    ylabel="Step",
                )
            else:
                ax_time.text(
                    0.5,
                    0.5,
                    "No time factor data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax_time.transAxes,
                )
                ax_time.axis("off")
        else:
            for ax, title in [
                (ax_complexity, "Complexity Factors"),
                (ax_time, "Time Factors"),
            ]:
                ax.text(
                    0.5,
                    0.5,
                    f"No {title.lower()} data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.axis("off")

        fig_abandon.suptitle(
            f"Abandonment Factors in {framework.value} Framework", fontsize=14
        )
        fig_abandon.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_abandonment_factors"
        else:
            filename = f"{framework.value.lower()}_abandonment_factors"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_abandon,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["abandonment_factors"] = fig_abandon

        return figures

    def visualize_step_time_intervals(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_version_comparison: bool = True,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for step time interval analysis.

        Args:
            framework: The framework to visualize
            course_id: Optional course ID to filter users
            include_version_comparison: Whether to include tool version comparison
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get step time interval data from analyzer
        interval_data = self._analyzer.calculate_step_time_intervals(
            framework, course_id, include_version_comparison
        )

        if "error" in interval_data:
            self._logger.error(
                f"Error getting step time intervals: {interval_data['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create interval distribution visualization
        fig_dist, ax_dist = create_figure(*self._default_figsize, theme=self._theme)

        if interval_data["interval_distribution"]:
            # Extract distribution data
            distribution = {}

            for category, data in interval_data["interval_distribution"].items():
                # Format category label
                if category == "under_5min":
                    label = "< 5 min"
                elif category == "5_15min":
                    label = "5-15 min"
                elif category == "15_30min":
                    label = "15-30 min"
                elif category == "30_60min":
                    label = "30-60 min"
                elif category == "1_3hr":
                    label = "1-3 hours"
                elif category == "3_24hr":
                    label = "3-24 hours"
                elif category == "over_24hr":
                    label = "> 24 hours"
                else:
                    label = category

                distribution[label] = data["percentage"] * 100  # Convert to percentage

            # Create bar chart
            bars = plot_bar(
                ax_dist,
                data=distribution,
                orientation="vertical",
                color=get_color_palette("sequential_blue", n_colors=len(distribution)),
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_dist,
                title=f"Step Time Interval Distribution in {framework.value} Framework",
                xlabel="Time Interval",
                ylabel="Percentage (%)",
                xticklabel_rotation=45,
                ylim=(0, max(distribution.values()) * 1.1),  # Add 10% padding
            )
        else:
            ax_dist.text(
                0.5,
                0.5,
                "No interval distribution data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_dist.transAxes,
            )
            ax_dist.axis("off")

        fig_dist.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_interval_distribution"
        else:
            filename = f"{framework.value.lower()}_interval_distribution"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_dist,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["interval_distribution"] = fig_dist

        # 2. Create step pair interval visualization
        if interval_data["step_interval_metrics"]:
            # Group step pairs by interval speed
            speed_groups = {
                "very_fast": [],
                "fast": [],
                "moderate": [],
                "slow": [],
                "very_slow": [],
            }

            for step_pair, metrics in interval_data["step_interval_metrics"].items():
                if "speed" in metrics:
                    speed_groups[metrics["speed"]].append(
                        (step_pair, metrics["avg_minutes"])
                    )

            # Create figure with subplots for each speed group
            fig_pairs, axes = plt.subplots(
                len(speed_groups),
                1,
                figsize=(10, 3 * len(speed_groups)),
                sharex=True,
            )

            # Ensure axes is a list even with a single subplot
            if len(speed_groups) == 1:
                axes = [axes]

            speed_titles = {
                "very_fast": "Very Fast Transitions (< 15 min)",
                "fast": "Fast Transitions (15-60 min)",
                "moderate": "Moderate Transitions (1-6 hours)",
                "slow": "Slow Transitions (6-24 hours)",
                "very_slow": "Very Slow Transitions (> 24 hours)",
            }

            # Sort speed groups from fastest to slowest
            speed_order = ["very_fast", "fast", "moderate", "slow", "very_slow"]

            for i, speed in enumerate(speed_order):
                pairs = speed_groups[speed]

                if pairs:
                    # Sort pairs by average time
                    pairs.sort(key=lambda x: x[1])

                    # Create horizontal bar chart
                    pair_data = {pair[0]: pair[1] for pair in pairs}

                    plot_bar(
                        axes[i],
                        data=pair_data,
                        orientation="horizontal",
                        color=get_color_palette("sequential_blue", n_colors=1)[0],
                        add_data_labels=True,
                        data_label_format="{:.1f} min",
                    )

                    configure_axes(
                        axes[i],
                        title=speed_titles[speed],
                        xlabel=(
                            "Average Time (minutes)"
                            if i == len(speed_order) - 1
                            else ""
                        ),
                        ylabel="Step Transition",
                    )
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No {speed} transitions available",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=axes[i].transAxes,
                    )
                    axes[i].axis("off")

            fig_pairs.suptitle(
                f"Step Transition Times in {framework.value} Framework", fontsize=14
            )
            fig_pairs.tight_layout()

            # Save figure
            if filename_prefix:
                filename = f"{filename_prefix}_step_transitions"
            else:
                filename = f"{framework.value.lower()}_step_transitions"

                if course_id:
                    filename += f"_{course_id}"

            save_path = get_output_path(
                generate_filename(filename, include_timestamp=self._include_timestamps),
                base_dir=str(self._output_dir),
            )

            save_figure(
                fig_pairs,
                str(save_path),
                formats=save_formats,
                dpi=300,
                tight_layout=True,
            )

            if return_figures:
                figures["step_transitions"] = fig_pairs

        # 3. Create version comparison visualization if available
        if include_version_comparison and "version_comparison" in interval_data:
            fig_version, (ax_avg, ax_dist) = plt.subplots(1, 2, figsize=(12, 6))

            # Extract version data
            versions = {}
            distributions = {}

            for version, metrics in interval_data.items():
                if version in ["v1", "v2"]:
                    versions[version] = metrics["avg_interval_minutes"]

                    # Get distribution data
                    if "interval_distribution" in metrics:
                        dist_data = {}

                        for category, data in metrics["interval_distribution"].items():
                            # Format category label
                            if category == "under_5min":
                                label = "< 5 min"
                            elif category == "5_15min":
                                label = "5-15 min"
                            elif category == "15_30min":
                                label = "15-30 min"
                            elif category == "30_60min":
                                label = "30-60 min"
                            elif category == "1_3hr":
                                label = "1-3 hours"
                            elif category == "3_24hr":
                                label = "3-24 hours"
                            elif category == "over_24hr":
                                label = "> 24 hours"
                            else:
                                label = category

                            dist_data[label] = (
                                data["percentage"] * 100
                            )  # Convert to percentage

                        distributions[version] = dist_data

            if versions:
                # Create bar chart for average times
                plot_bar(
                    ax_avg,
                    data=versions,
                    orientation="vertical",
                    color=get_color_palette("tool_versions"),
                    add_data_labels=True,
                    data_label_format="{:.1f} min",
                )

                configure_axes(
                    ax_avg,
                    title="Average Step Interval by Tool Version",
                    xlabel="Tool Version",
                    ylabel="Average Time (minutes)",
                    ylim=(0, max(versions.values()) * 1.2),  # Add 20% padding
                )

                # Create grouped bar chart for distributions
                if distributions and len(distributions) > 0:
                    # Get common categories
                    all_categories = set()
                    for dist in distributions.values():
                        all_categories.update(dist.keys())

                    # Prepare data for grouped bars
                    grouped_data = {}
                    for category in all_categories:
                        grouped_data[category] = {
                            version: dist.get(category, 0)
                            for version, dist in distributions.items()
                        }

                    # Create grouped bar chart
                    plot_grouped_bars(
                        ax_dist,
                        data=grouped_data,
                        orientation="vertical",
                        colors=[
                            get_color_palette("tool_versions")[v]
                            for v in distributions.keys()
                        ],
                        add_data_labels=True,
                        data_label_format="{:.1f}%",
                        add_legend=True,
                        legend_title="Tool Version",
                    )

                    configure_axes(
                        ax_dist,
                        title="Interval Distribution by Tool Version",
                        xlabel="Time Interval",
                        ylabel="Percentage (%)",
                        xticklabel_rotation=45,
                    )
                else:
                    ax_dist.text(
                        0.5,
                        0.5,
                        "No distribution data available",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=ax_dist.transAxes,
                    )
                    ax_dist.axis("off")
            else:
                for ax, title in [
                    (ax_avg, "Average Times"),
                    (ax_dist, "Distributions"),
                ]:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {title.lower()} data available",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=ax.transAxes,
                    )
                    ax.axis("off")

            fig_version.suptitle(
                f"Tool Version Comparison for {framework.value} Framework", fontsize=14
            )
            fig_version.tight_layout()

            # Save figure
            if filename_prefix:
                filename = f"{filename_prefix}_version_comparison"
            else:
                filename = f"{framework.value.lower()}_version_comparison"

                if course_id:
                    filename += f"_{course_id}"

            save_path = get_output_path(
                generate_filename(filename, include_timestamp=self._include_timestamps),
                base_dir=str(self._output_dir),
            )

            save_figure(
                fig_version,
                str(save_path),
                formats=save_formats,
                dpi=300,
                tight_layout=True,
            )

            if return_figures:
                figures["version_comparison"] = fig_version

        return figures

    def visualize_cohort_comparison(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        cohorts: Optional[List[Semester]] = None,
        course_id: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for framework progression by cohort.

        Args:
            framework: The framework to visualize
            cohorts: Optional list of semesters to compare
            course_id: Optional course ID to filter users
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get cohort comparison data from analyzer
        cohort_data = self._analyzer.compare_framework_progression_by_cohort(
            framework, cohorts, course_id
        )

        if "error" in cohort_data:
            self._logger.error(
                f"Error getting cohort comparison: {cohort_data['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create cohort metrics visualization
        fig_metrics, ax_metrics = create_figure(
            *self._default_figsize, theme=self._theme
        )

        if cohort_data["cohort_metrics"]:
            # Extract key metrics for each cohort
            cohort_names = list(cohort_data["cohort_metrics"].keys())

            # Sort cohorts chronologically
            cohort_names.sort(
                key=lambda x: (
                    int(x.split()[1]),  # Year
                    0 if x.split()[0] == "Spring" else 1,  # Term (Spring before Fall)
                )
            )

            # Get tool versions for each cohort
            tool_versions = [
                cohort_data["cohort_metrics"][cohort]["tool_version"]
                for cohort in cohort_names
            ]

            # Get completion percentages for each cohort
            completion_pcts = [
                cohort_data["cohort_metrics"][cohort]["completion_percentage"]
                for cohort in cohort_names
            ]

            # Get avg steps per idea for each cohort
            avg_steps = [
                cohort_data["cohort_metrics"][cohort]["avg_steps_per_idea"]
                for cohort in cohort_names
            ]

            # Create grouped bar chart data
            metrics_data = {
                "Completion %": {
                    cohort: pct for cohort, pct in zip(cohort_names, completion_pcts)
                },
                "Avg Steps per Idea": {
                    cohort: steps for cohort, steps in zip(cohort_names, avg_steps)
                },
            }

            # Create figure with subplots
            fig_metrics, (ax_comp, ax_steps) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot completion percentages
            bars1 = plot_bar(
                ax_comp,
                data={
                    cohort: metrics_data["Completion %"][cohort]
                    for cohort in cohort_names
                },
                orientation="vertical",
                color=[
                    get_color_palette("tool_versions")[version]
                    for version in tool_versions
                ],
                add_data_labels=True,
                data_label_format="{:.1f}%",
            )

            configure_axes(
                ax_comp,
                title="Framework Completion Percentage by Cohort",
                xlabel="Cohort",
                ylabel="Completion Percentage (%)",
                xticklabel_rotation=45,
                ylim=(0, max(completion_pcts) * 1.2),  # Add 20% padding
            )

            # Plot avg steps per idea
            bars2 = plot_bar(
                ax_steps,
                data={
                    cohort: metrics_data["Avg Steps per Idea"][cohort]
                    for cohort in cohort_names
                },
                orientation="vertical",
                color=[
                    get_color_palette("tool_versions")[version]
                    for version in tool_versions
                ],
                add_data_labels=True,
                data_label_format="{:.1f}",
            )

            configure_axes(
                ax_steps,
                title="Average Steps per Idea by Cohort",
                xlabel="Cohort",
                ylabel="Average Steps",
                xticklabel_rotation=45,
                ylim=(0, max(avg_steps) * 1.2),  # Add 20% padding
            )

            # Add legend for tool versions
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, color=get_color_palette("tool_versions")[v])
                for v in set(tool_versions)
            ]

            fig_metrics.legend(
                legend_elements,
                list(set(tool_versions)),
                title="Tool Version",
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                ncol=len(set(tool_versions)),
            )
        else:
            ax_metrics.text(
                0.5,
                0.5,
                "No cohort metrics data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_metrics.transAxes,
            )
            ax_metrics.axis("off")

        fig_metrics.suptitle(
            f"Cohort Comparison for {framework.value} Framework", fontsize=14
        )
        fig_metrics.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_cohort_metrics"
        else:
            filename = f"{framework.value.lower()}_cohort_metrics"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_metrics,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["cohort_metrics"] = fig_metrics

        # 2. Create completion trends visualization
        fig_trends, ax_trends = create_figure(*self._default_figsize, theme=self._theme)

        if cohort_data["completion_trends"]:
            # Group trends by step
            step_trends = {}

            for step, trend_data in cohort_data["completion_trends"].items():
                trend_points = trend_data["trend_points"]

                # Extract cohorts and rates
                cohorts = [point["cohort"] for point in trend_points]
                rates = [
                    point["completion_rate"] * 100 for point in trend_points
                ]  # Convert to percentage

                step_trends[step] = {"cohorts": cohorts, "rates": rates}

            # Sort cohorts chronologically
            all_cohorts = set()
            for trend in step_trends.values():
                all_cohorts.update(trend["cohorts"])

            sorted_cohorts = sorted(
                list(all_cohorts),
                key=lambda x: (
                    int(x.split()[1]),  # Year
                    0 if x.split()[0] == "Spring" else 1,  # Term (Spring before Fall)
                ),
            )

            # Select top steps to visualize (to avoid clutter)
            steps_to_visualize = 5

            # Sort steps by overall trend direction and magnitude
            steps_sorted = sorted(
                cohort_data["completion_trends"].items(),
                key=lambda x: abs(x[1]["overall_change"]),
                reverse=True,
            )[:steps_to_visualize]

            # Create line chart
            for step, trend_data in steps_sorted:
                trend_points = trend_data["trend_points"]

                # Extract cohorts and rates
                cohorts = [point["cohort"] for point in trend_points]
                rates = [
                    point["completion_rate"] * 100 for point in trend_points
                ]  # Convert to percentage

                # Plot line
                plot_line(
                    ax_trends,
                    x_data=range(len(cohorts)),
                    y_data=rates,
                    label=step,
                    marker="o",
                    add_data_labels=True,
                    data_label_format="{:.1f}%",
                )

            # Set x-tick labels to cohort names
            ax_trends.set_xticks(range(len(sorted_cohorts)))
            ax_trends.set_xticklabels(sorted_cohorts, rotation=45)

            configure_axes(
                ax_trends,
                title=f"Step Completion Trends in {framework.value} Framework",
                xlabel="Cohort",
                ylabel="Completion Rate (%)",
                ylim=(0, 100),
            )

            # Add legend
            ax_trends.legend(
                title="Framework Steps",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
        else:
            ax_trends.text(
                0.5,
                0.5,
                "No completion trend data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_trends.transAxes,
            )
            ax_trends.axis("off")

        fig_trends.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_completion_trends"
        else:
            filename = f"{framework.value.lower()}_completion_trends"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_trends,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["completion_trends"] = fig_trends

        # 3. Create tool version impact visualization
        if (
            "tool_version_impact" in cohort_data
            and cohort_data["tool_version_impact"]["version_comparison"]
        ):
            fig_impact, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(14, 6))

            # Extract version comparison data
            version_comparison = cohort_data["tool_version_impact"][
                "version_comparison"
            ]

            # Get v1 vs no tool comparison
            v1_comparison = version_comparison.get("v1_vs_no_tool")

            if v1_comparison:
                # Get completion and steps metrics
                completion_diff = v1_comparison["completion_difference"]
                steps_diff = v1_comparison["steps_per_idea_difference"]

                metrics_v1 = {
                    "Completion %": completion_diff,
                    "Steps per Idea": steps_diff,
                }

                # Create bar chart for v1 vs no tool
                bars1 = plot_bar(
                    ax_v1,
                    data=metrics_v1,
                    orientation="vertical",
                    color=[
                        "#007BFF" if val > 0 else "#DC3545"
                        for val in metrics_v1.values()
                    ],
                    add_data_labels=True,
                    data_label_format="{:+.1f}",  # Show sign
                )

                configure_axes(
                    ax_v1,
                    title="V1 vs No Tool",
                    xlabel="Metric",
                    ylabel="Difference",
                )

                # Add zero reference line
                add_reference_line(
                    ax_v1,
                    value=0,
                    orientation="horizontal",
                    color="#666666",
                    linestyle="--",
                )
            else:
                ax_v1.text(
                    0.5,
                    0.5,
                    "No V1 vs No Tool data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax_v1.transAxes,
                )
                ax_v1.axis("off")

            # Get v2 vs v1 comparison
            v2_comparison = version_comparison.get("v2_vs_v1")

            if v2_comparison:
                # Get completion and steps metrics
                completion_diff = v2_comparison["completion_difference"]
                steps_diff = v2_comparison["steps_per_idea_difference"]

                metrics_v2 = {
                    "Completion %": completion_diff,
                    "Steps per Idea": steps_diff,
                }

                # Create bar chart for v2 vs v1
                bars2 = plot_bar(
                    ax_v2,
                    data=metrics_v2,
                    orientation="vertical",
                    color=[
                        "#28A745" if val > 0 else "#DC3545"
                        for val in metrics_v2.values()
                    ],
                    add_data_labels=True,
                    data_label_format="{:+.1f}",  # Show sign
                )

                configure_axes(
                    ax_v2,
                    title="V2 vs V1",
                    xlabel="Metric",
                    ylabel="Difference",
                )

                # Add zero reference line
                add_reference_line(
                    ax_v2,
                    value=0,
                    orientation="horizontal",
                    color="#666666",
                    linestyle="--",
                )
            else:
                ax_v2.text(
                    0.5,
                    0.5,
                    "No V2 vs V1 data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax_v2.transAxes,
                )
                ax_v2.axis("off")

            fig_impact.suptitle(
                f"Tool Version Impact on {framework.value} Framework", fontsize=14
            )
            fig_impact.tight_layout()

            # Save figure
            if filename_prefix:
                filename = f"{filename_prefix}_version_impact"
            else:
                filename = f"{framework.value.lower()}_version_impact"

                if course_id:
                    filename += f"_{course_id}"

            save_path = get_output_path(
                generate_filename(filename, include_timestamp=self._include_timestamps),
                base_dir=str(self._output_dir),
            )

            save_figure(
                fig_impact,
                str(save_path),
                formats=save_formats,
                dpi=300,
                tight_layout=True,
            )

            if return_figures:
                figures["version_impact"] = fig_impact

        return figures

    def visualize_framework_effectiveness(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_category_analysis: bool = True,
        filename_prefix: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
        return_figures: bool = False,
    ) -> Optional[Dict[str, Figure]]:
        """
        Create visualizations for framework effectiveness metrics.

        Args:
            framework: The framework to visualize
            course_id: Optional course ID to filter users
            include_category_analysis: Whether to include analysis by idea category
            filename_prefix: Optional prefix for output filenames
            save_formats: List of formats to save visualizations in
            return_figures: Whether to return the created figures

        Returns:
            Optional[Dict[str, Figure]]: Dictionary of figures if return_figures is True
        """
        # Get framework effectiveness data from analyzer
        effectiveness = self._analyzer.get_framework_effectiveness_metrics(
            framework, course_id, include_category_analysis
        )

        if "error" in effectiveness:
            self._logger.error(
                f"Error getting framework effectiveness: {effectiveness['error']}"
            )
            return None

        # Create a dictionary to store figures if requested
        figures = {} if return_figures else None

        # 1. Create step utility visualization
        fig_utility, ax_utility = create_figure(10, 8, theme=self._theme)

        if effectiveness["step_utility_metrics"]:
            # Extract utility metrics for each step
            utility_data = {}

            for step, metrics in effectiveness["step_utility_metrics"].items():
                # Create a composite utility score (usage rate * user input rate)
                utility_score = metrics["usage_rate"] * metrics["user_input_rate"] * 100
                utility_data[step] = utility_score

            # Sort by utility score
            sorted_utility = {
                k: v
                for k, v in sorted(
                    utility_data.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }

            # Create bar chart
            bars = plot_bar(
                ax_utility,
                data=sorted_utility,
                orientation="horizontal",
                color=get_color_palette(
                    "sequential_blue", n_colors=len(sorted_utility)
                ),
                add_data_labels=True,
                data_label_format="{:.1f}",
            )

            configure_axes(
                ax_utility,
                title=f"Step Utility Scores in {framework.value} Framework",
                xlabel="Utility Score",
                ylabel="Framework Step",
            )
        else:
            ax_utility.text(
                0.5,
                0.5,
                "No step utility data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_utility.transAxes,
            )
            ax_utility.axis("off")

        fig_utility.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_step_utility"
        else:
            filename = f"{framework.value.lower()}_step_utility"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_utility,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["step_utility"] = fig_utility

        # 2. Create user progression metrics visualization
        fig_prog, ax_prog = create_figure(*self._default_figsize, theme=self._theme)

        if effectiveness["user_progression_metrics"]:
            # Extract progression metrics
            metrics = effectiveness["user_progression_metrics"]

            # Display metrics as a table
            data = [
                [
                    "Users with Multiple Ideas",
                    f"{metrics['users_with_multiple_ideas']}",
                ],
                ["Average Step Change", f"{metrics['avg_step_change']:.2f}"],
                [
                    "Improvement Rate",
                    f"{metrics['improvement_rate']:.2f} ({metrics['improvement_rate']*100:.1f}%)",
                ],
            ]

            # Hide axes
            ax_prog.axis("off")

            # Add table
            add_data_table(
                ax_prog,
                data=data,
                title=f"User Progression Metrics in {framework.value} Framework",
                fontsize=12,
                loc="center",
                bbox=[0.1, 0.1, 0.8, 0.8],
            )
        else:
            ax_prog.text(
                0.5,
                0.5,
                "No user progression data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_prog.transAxes,
            )
            ax_prog.axis("off")

        fig_prog.tight_layout()

        # Save figure
        if filename_prefix:
            filename = f"{filename_prefix}_user_progression"
        else:
            filename = f"{framework.value.lower()}_user_progression"

            if course_id:
                filename += f"_{course_id}"

        save_path = get_output_path(
            generate_filename(filename, include_timestamp=self._include_timestamps),
            base_dir=str(self._output_dir),
        )

        save_figure(
            fig_prog,
            str(save_path),
            formats=save_formats,
            dpi=300,
            tight_layout=True,
        )

        if return_figures:
            figures["user_progression"] = fig_prog

        # 3. Create framework impact visualization
        if "framework_impact" in effectiveness:
            # Create correlation visualization
            if "step_score_correlation" in effectiveness["framework_impact"]:
                fig_corr, ax_corr = create_figure(
                    *self._default_figsize, theme=self._theme
                )

                # Extract correlation data
                correlation_data = effectiveness["framework_impact"][
                    "step_score_correlation"
                ]

                if correlation_data:
                    # Create bar chart for correlations
                    correlation_bars = {
                        "Content Score": correlation_data["content_score_correlation"],
                        "Completion Score": correlation_data[
                            "completion_score_correlation"
                        ],
                    }

                    bars = plot_bar(
                        ax_corr,
                        data=correlation_bars,
                        orientation="vertical",
                        color=[
                            "#28A745" if val > 0 else "#DC3545"
                            for val in correlation_bars.values()
                        ],
                        add_data_labels=True,
                        data_label_format="{:.2f}",
                    )

                    configure_axes(
                        ax_corr,
                        title=f"Step Count vs. User Scores Correlation in {framework.value} Framework",
                        xlabel="Score Type",
                        ylabel="Correlation Coefficient",
                        ylim=(-1, 1),
                    )

                    # Add zero reference line
                    add_reference_line(
                        ax_corr,
                        value=0,
                        orientation="horizontal",
                        color="#666666",
                        linestyle="--",
                    )

                    # Add reference lines for correlation strength
                    add_reference_line(
                        ax_corr,
                        value=0.3,
                        orientation="horizontal",
                        color="#28A745",
                        linestyle=":",
                        add_label=True,
                        label="Moderate +",
                    )

                    add_reference_line(
                        ax_corr,
                        value=-0.3,
                        orientation="horizontal",
                        color="#DC3545",
                        linestyle=":",
                        add_label=True,
                        label="Moderate -",
                    )
                else:
                    ax_corr.text(
                        0.5,
                        0.5,
                        "No correlation data available",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax_corr.transAxes,
                    )
                    ax_corr.axis("off")

                fig_corr.tight_layout()

                # Save figure
                if filename_prefix:
                    filename = f"{filename_prefix}_step_score_correlation"
                else:
                    filename = f"{framework.value.lower()}_step_score_correlation"

                    if course_id:
                        filename += f"_{course_id}"

                save_path = get_output_path(
                    generate_filename(
                        filename, include_timestamp=self._include_timestamps
                    ),
                    base_dir=str(self._output_dir),
                )

                save_figure(
                    fig_corr,
                    str(save_path),
                    formats=save_formats,
                    dpi=300,
                    tight_layout=True,
                )

                if return_figures:
                    figures["step_score_correlation"] = fig_corr

            # Create engagement by step count visualization
            if "engagement_by_step_count" in effectiveness["framework_impact"]:
                fig_engage, ax_engage = create_figure(
                    *self._default_figsize, theme=self._theme
                )

                engagement_data = effectiveness["framework_impact"][
                    "engagement_by_step_count"
                ]

                if engagement_data:
                    # Create stacked bar chart for engagement levels
                    engagement_stacked = {}

                    for group, data in engagement_data["engagement_levels"].items():
                        engagement_stacked[group] = {
                            "High": data["high_engagement_percentage"]
                            * 100,  # Convert to percentage
                            "Medium": data["medium_engagement_percentage"] * 100,
                            "Low": data["low_engagement_percentage"] * 100,
                        }

                    # Create stacked bar chart
                    plot_stacked_bars(
                        ax_engage,
                        data=engagement_stacked,
                        orientation="vertical",
                        colors=[
                            get_color_palette("engagement_levels")["high"],
                            get_color_palette("engagement_levels")["medium"],
                            get_color_palette("engagement_levels")["low"],
                        ],
                        add_data_labels=True,
                        data_label_format="{:.1f}%",
                        add_legend=True,
                        legend_title="Engagement Level",
                    )

                    configure_axes(
                        ax_engage,
                        title=f"Engagement Levels by Step Completion in {framework.value} Framework",
                        xlabel="Step Completion Group",
                        ylabel="Percentage (%)",
                        ylim=(0, 100),
                    )
                else:
                    ax_engage.text(
                        0.5,
                        0.5,
                        "No engagement data available",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax_engage.transAxes,
                    )
                    ax_engage.axis("off")

                fig_engage.tight_layout()

                # Save figure
                if filename_prefix:
                    filename = f"{filename_prefix}_engagement_by_completion"
                else:
                    filename = f"{framework.value.lower()}_engagement_by_completion"

                    if course_id:
                        filename += f"_{course_id}"

                save_path = get_output_path(
                    generate_filename(
                        filename, include_timestamp=self._include_timestamps
                    ),
                    base_dir=str(self._output_dir),
                )

                save_figure(
                    fig_engage,
                    str(save_path),
                    formats=save_formats,
                    dpi=300,
                    tight_layout=True,
                )

                if return_figures:
                    figures["engagement_by_completion"] = fig_engage

        # 4. Create category analysis visualization if available
        if include_category_analysis and "category_analysis" in effectiveness:
            category_data = effectiveness["category_analysis"]

            if category_data["categories"]:
                fig_cat, (ax_cat1, ax_cat2) = plt.subplots(1, 2, figsize=(14, 7))

                # Extract category completion data
                cat_completion = {
                    category: data["completion_percentage"]
                    for category, data in category_data["categories"].items()
                }

                # Sort by completion percentage
                sorted_cat_completion = {
                    k: v
                    for k, v in sorted(
                        cat_completion.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                # Create bar chart for category completion
                plot_bar(
                    ax_cat1,
                    data=sorted_cat_completion,
                    orientation="horizontal",
                    color=get_color_palette(
                        "categorical_main", n_colors=len(sorted_cat_completion)
                    ),
                    add_data_labels=True,
                    data_label_format="{:.1f}%",
                )

                configure_axes(
                    ax_cat1,
                    title="Framework Completion by Idea Category",
                    xlabel="Completion Percentage (%)",
                    ylabel="Idea Category",
                )

                # Extract avg steps per idea
                cat_steps = {
                    category: data["avg_steps_per_idea"]
                    for category, data in category_data["categories"].items()
                }

                # Sort by avg steps
                sorted_cat_steps = {
                    k: v
                    for k, v in sorted(
                        cat_steps.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                # Create bar chart for avg steps
                plot_bar(
                    ax_cat2,
                    data=sorted_cat_steps,
                    orientation="horizontal",
                    color=get_color_palette(
                        "sequential_green", n_colors=len(sorted_cat_steps)
                    ),
                    add_data_labels=True,
                    data_label_format="{:.1f}",
                )

                configure_axes(
                    ax_cat2,
                    title="Average Steps per Idea by Category",
                    xlabel="Average Steps",
                    ylabel="Idea Category",
                )

                fig_cat.suptitle(
                    f"Idea Category Analysis for {framework.value} Framework",
                    fontsize=14,
                )
                fig_cat.tight_layout()

                # Save figure
                if filename_prefix:
                    filename = f"{filename_prefix}_category_analysis"
                else:
                    filename = f"{framework.value.lower()}_category_analysis"

                    if course_id:
                        filename += f"_{course_id}"

                save_path = get_output_path(
                    generate_filename(
                        filename, include_timestamp=self._include_timestamps
                    ),
                    base_dir=str(self._output_dir),
                )

                save_figure(
                    fig_cat,
                    str(save_path),
                    formats=save_formats,
                    dpi=300,
                    tight_layout=True,
                )

                if return_figures:
                    figures["category_analysis"] = fig_cat

                # Create step effectiveness by category visualization if available
                if "step_effectiveness_by_category" in category_data:
                    step_effect = category_data["step_effectiveness_by_category"]

                    if step_effect:
                        # Find steps with significant category differences
                        significant_steps = {}

                        for step, categories in step_effect.items():
                            # Calculate range of differences
                            diffs = [
                                data["difference_from_overall"]
                                for data in categories.values()
                            ]
                            diff_range = max(diffs) - min(diffs)

                            if (
                                diff_range >= 0.2
                            ):  # Only show steps with significant differences
                                significant_steps[step] = categories

                        if significant_steps:
                            # Create visualizations for up to 4 steps with the most significant differences
                            steps_to_show = min(4, len(significant_steps))

                            # Sort steps by range of differences
                            sorted_steps = sorted(
                                significant_steps.items(),
                                key=lambda x: max(
                                    abs(data["difference_from_overall"])
                                    for data in x[1].values()
                                ),
                                reverse=True,
                            )[:steps_to_show]

                            # Create figure with subplots
                            fig_steps, axes = plt.subplots(
                                (steps_to_show + 1) // 2,
                                2,
                                figsize=(12, 5 * ((steps_to_show + 1) // 2)),
                            )

                            # Ensure axes is a 2D array
                            if steps_to_show == 1:
                                axes = np.array([[axes]])
                            elif steps_to_show == 2:
                                axes = np.array([axes])

                            # Create bar charts for each step
                            for i, (step, categories) in enumerate(sorted_steps):
                                row, col = i // 2, i % 2

                                # Extract category differences
                                cat_diffs = {
                                    category: data["difference_from_overall"]
                                    * 100  # Convert to percentage
                                    for category, data in categories.items()
                                }

                                # Sort by difference
                                sorted_diffs = {
                                    k: v
                                    for k, v in sorted(
                                        cat_diffs.items(),
                                        key=lambda item: item[1],
                                        reverse=True,
                                    )
                                }

                                # Create bar chart
                                plot_bar(
                                    axes[row, col],
                                    data=sorted_diffs,
                                    orientation="horizontal",
                                    color=[
                                        "#28A745" if val > 0 else "#DC3545"
                                        for val in sorted_diffs.values()
                                    ],
                                    add_data_labels=True,
                                    data_label_format="{:+.1f}%",  # Show sign
                                )

                                configure_axes(
                                    axes[row, col],
                                    title=f"Step: {step}",
                                    xlabel="Difference from Overall (%)",
                                    ylabel="Idea Category",
                                )

                                # Add zero reference line
                                add_reference_line(
                                    axes[row, col],
                                    value=0,
                                    orientation="vertical",
                                    color="#666666",
                                    linestyle="--",
                                )

                            # Hide any unused subplots
                            for i in range(steps_to_show, 4):
                                row, col = i // 2, i % 2
                                if row < len(axes) and col < len(axes[0]):
                                    axes[row, col].axis("off")

                            fig_steps.suptitle(
                                f"Step Effectiveness by Category in {framework.value} Framework",
                                fontsize=14,
                            )
                            fig_steps.tight_layout()

                            # Save figure
                            if filename_prefix:
                                filename = f"{filename_prefix}_step_effectiveness"
                            else:
                                filename = (
                                    f"{framework.value.lower()}_step_effectiveness"
                                )

                                if course_id:
                                    filename += f"_{course_id}"

                            save_path = get_output_path(
                                generate_filename(
                                    filename, include_timestamp=self._include_timestamps
                                ),
                                base_dir=str(self._output_dir),
                            )

                            save_figure(
                                fig_steps,
                                str(save_path),
                                formats=save_formats,
                                dpi=300,
                                tight_layout=True,
                            )

                            if return_figures:
                                figures["step_effectiveness"] = fig_steps

        return figures

    def create_framework_report(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        include_ideas_without_steps: bool = False,
        include_category_analysis: bool = True,
        output_dir: Optional[str] = None,
        save_formats: List[str] = ["png", "pdf"],
    ) -> str:
        """
        Create a comprehensive report of framework analysis visualizations.

        Args:
            framework: The framework to analyze
            course_id: Optional course ID to filter users
            include_ideas_without_steps: Whether to include ideas without steps
            include_category_analysis: Whether to include analysis by idea category
            output_dir: Optional output directory for the report
            save_formats: List of formats to save visualizations in

        Returns:
            str: Path to the generated report directory
        """
        # Set output directory for the report
        if output_dir:
            report_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"{framework.value.lower()}_framework_report"

            if course_id:
                report_name += f"_{course_id}"

            report_name += f"_{timestamp}"

            report_dir = self._output_dir / "reports" / report_name

        # Create report directory
        report_dir.mkdir(parents=True, exist_ok=True)

        # Set filename prefix for visualizations
        prefix = framework.value.lower()

        if course_id:
            prefix += f"_{course_id}"

        # Create subfolders
        figures_dir = report_dir / "figures"
        data_dir = report_dir / "data"

        figures_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        # Dictionary to store report sections
        report_sections = {}

        # 1. Generate completion metrics visualizations
        self._logger.info("Generating framework completion visualizations...")

        completion_figures = self.visualize_framework_completion(
            framework=framework,
            course_id=course_id,
            include_ideas_without_steps=include_ideas_without_steps,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if completion_figures:
            report_sections["completion_metrics"] = completion_figures

        # 2. Generate progression patterns visualizations
        self._logger.info("Generating progression pattern visualizations...")

        progression_figures = self.visualize_progression_patterns(
            framework=framework,
            course_id=course_id,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if progression_figures:
            report_sections["progression_patterns"] = progression_figures

        # 3. Generate step dependencies visualizations
        self._logger.info("Generating step dependencies visualizations...")

        dependencies_figures = self.visualize_step_dependencies(
            framework=framework,
            course_id=course_id,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if dependencies_figures:
            report_sections["step_dependencies"] = dependencies_figures

        # 4. Generate dropout analysis visualizations
        self._logger.info("Generating dropout analysis visualizations...")

        dropout_figures = self.visualize_framework_dropout(
            framework=framework,
            course_id=course_id,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if dropout_figures:
            report_sections["dropout_analysis"] = dropout_figures

        # 5. Generate step time intervals visualizations
        self._logger.info("Generating step time interval visualizations...")

        interval_figures = self.visualize_step_time_intervals(
            framework=framework,
            course_id=course_id,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if interval_figures:
            report_sections["time_intervals"] = interval_figures

        # 6. Generate cohort comparison visualizations
        self._logger.info("Generating cohort comparison visualizations...")

        cohort_figures = self.visualize_cohort_comparison(
            framework=framework,
            course_id=course_id,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if cohort_figures:
            report_sections["cohort_comparison"] = cohort_figures

        # 7. Generate framework effectiveness visualizations
        self._logger.info("Generating framework effectiveness visualizations...")

        effectiveness_figures = self.visualize_framework_effectiveness(
            framework=framework,
            course_id=course_id,
            include_category_analysis=include_category_analysis,
            filename_prefix=prefix,
            save_formats=save_formats,
            return_figures=True,
        )

        if effectiveness_figures:
            report_sections["framework_effectiveness"] = effectiveness_figures

        # Generate report summary in markdown
        self._generate_report_summary(framework, course_id, report_dir, report_sections)

        self._logger.info(f"Framework report generated at: {report_dir}")

        return str(report_dir)

    def _generate_report_summary(
        self,
        framework: FrameworkType,
        course_id: Optional[str],
        report_dir: Path,
        report_sections: Dict[str, Dict[str, Figure]],
        save_formats: List[str],
    ) -> None:
        """
        Generate a markdown summary of the framework analysis report.

        Args:
            framework: The framework analyzed
            course_id: Optional course ID used for filtering
            report_dir: Directory where the report is saved
            report_sections: Dictionary of report sections and figures
            save_formats: List of formats used for saving visualizations
        """
        # Create filename prefix for figure references
        prefix = framework.value.lower()
        if course_id:
            prefix += f"_{course_id}"

        # Create markdown content
        md_content = f"# {framework.value} Framework Analysis Report\n\n"

        # Add report timestamp
        md_content += (
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        )

        # Add course information if applicable
        if course_id:
            md_content += f"**Course ID:** {course_id}\n\n"

        # Add table of contents
        md_content += "## Table of Contents\n\n"

        for i, section in enumerate(report_sections.keys(), 1):
            # Format section name for display
            section_display = section.replace("_", " ").title()
            md_content += (
                f"{i}. [{section_display}](#{section.lower().replace('_', '-')})\n"
            )

        md_content += "\n"

        # Add section content
        for section_name, figures in report_sections.items():
            # Format section name for display
            section_display = section_name.replace("_", " ").title()
            md_content += f"## {section_display}\n\n"

            # Add figure references
            for fig_name, fig in figures.items():
                # Format figure name for display
                fig_display = fig_name.replace("_", " ").title()

                # Get filename for this figure
                fig_filename = f"{prefix}_{fig_name}"

                # Reference both PNG and PDF versions if available
                md_content += f"### {fig_display}\n\n"
                md_content += f"![{fig_display}](figures/{fig_filename}.png)\n\n"

                # Only add PDF version reference if PDFs are included in save formats
                if "pdf" in save_formats:
                    md_content += f"*[PDF Version](figures/{fig_filename}.pdf)*\n\n"
                else:
                    md_content += "\n"

        # Add notes section
        md_content += "## Notes\n\n"
        md_content += (
            "This report was automatically generated using the FrameworkVisualizer.\n"
        )
        md_content += "For questions or additional analysis, please contact the research team.\n\n"

        # Save markdown file
        with open(report_dir / "report_summary.md", "w") as f:
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
                <title>{framework.value} Framework Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; }}
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
