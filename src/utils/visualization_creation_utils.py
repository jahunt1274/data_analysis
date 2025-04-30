"""
Visualization utilities for the data analysis system.

This module provides core visualization functionality for creating consistent
and effective visualizations across the analysis system. It includes functions
for figure management, styling, common chart types, and annotations.

The module is organized in several functional areas:
1. Figure and Axis Management - Creating and configuring matplotlib figures and axes
2. Styling and Theme System - Color management, typography and visual consistency
3. Common Chart Types - Bar, line, scatter, heatmap and other visualization types
4. Annotation Functions - Adding text, statistical, and visual annotations

Dependencies:
- matplotlib
- seaborn
- numpy
- networkx (optional, for network graphs)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

# Set seaborn style
sns.set_style("whitegrid")

# Configure logging
logger = logging.getLogger(__name__)

# Define color palettes
COLOR_PALETTES = {
    # Sequential palettes
    "sequential_blue": [
        "#E6F2FF",
        "#BDDEFF",
        "#94CAFF",
        "#6CB6FF",
        "#44A3FF",
        "#1C8FFF",
        "#007AFF",
        "#0062CC",
        "#004A99",
        "#003266",
    ],
    "sequential_green": [
        "#E6F9F2",
        "#BDECD7",
        "#94E0BC",
        "#6CD4A2",
        "#44C887",
        "#1CBC6D",
        "#00B052",
        "#008D42",
        "#006A31",
        "#004821",
    ],
    # Categorical palettes
    "categorical_main": [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
    ],
    "categorical_steps": [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
    ],
    # Specialized palettes
    "engagement_levels": {
        "high": "#28A745",  # Green
        "medium": "#FFC107",  # Yellow/Amber
        "low": "#DC3545",  # Red
    },
    # Tool versions
    "tool_versions": {
        "none": "#6C757D",  # Gray
        "v1": "#007BFF",  # Blue
        "v2": "#28A745",  # Green
    },
    # Diverging palettes
    "diverging_redblue": [
        "#053061",
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
        "#67001F",
    ],
    # Default palette for when nothing else is specified
    "default": sns.color_palette("tab10", 10).as_hex(),
}

# Theme configurations
THEMES = {
    "default": {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linestyle": "-",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],
    },
    "dark": {
        "figure.facecolor": "#222222",
        "axes.facecolor": "#222222",
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": "#444444",
        "grid.linestyle": "-",
        "text.color": "#EEEEEE",
        "axes.labelcolor": "#EEEEEE",
        "xtick.color": "#EEEEEE",
        "ytick.color": "#EEEEEE",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],
    },
    "print": {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linestyle": ":",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Liberation Serif",
            "Bitstream Vera Serif",
            "serif",
        ],
    },
}


# ----------------------
# Figure Management
# ----------------------


def save_figure(
    fig: Figure,
    filename: str,
    directory: Optional[str] = None,
    formats: List[str] = ["png", "pdf"],
    dpi: int = 300,
    transparent: bool = False,
    tight_layout: bool = True,
    bbox_inches: str = "tight",
) -> List[str]:
    """
    Save a figure to disk in multiple formats.

    Args:
        fig: Matplotlib figure to save
        filename: Base filename (without extension)
        directory: Optional directory path
        formats: List of file formats to save
        dpi: Resolution in dots per inch
        transparent: Whether to save with transparent background
        tight_layout: Whether to apply tight layout before saving
        bbox_inches: Bounding box setting

    Returns:
        List[str]: List of saved file paths
    """
    # Apply tight layout if requested
    if tight_layout:
        fig.tight_layout()

    # Create directory if needed
    if directory:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

    # Save in each format
    saved_files = []

    for fmt in formats:
        # Build full file path
        if directory:
            file_path = str(Path(directory) / f"{filename}.{fmt}")
        else:
            file_path = f"{filename}.{fmt}"

        # Save figure
        fig.savefig(
            file_path,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
        )

        # Add to list of saved files
        saved_files.append(file_path)

        logger.info(f"Saved figure to {file_path}")

    return saved_files


def create_figure(
    width: float = 10.0,
    height: float = 6.0,
    dpi: int = 100,
    size_preset: Optional[str] = None,
    theme: str = "default",
    create_axes: bool = True
) -> Figure:
    """
    Create a matplotlib figure with consistent styling.

    Args:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Figure resolution (dots per inch)
        size_preset: Optional preset size ('standard', 'wide', 'tall', 'presentation')
        theme: Theme name (default, dark, print)
        create_axes: Whether to create and return a default axes object

    Returns:
        Figure: Configured matplotlib figure
    """
    # Apply size presets if specified
    if size_preset:
        if size_preset == "standard":
            width, height = 10, 6
        elif size_preset == "wide":
            width, height = 14, 6
        elif size_preset == "tall":
            width, height = 8, 10
        elif size_preset == "presentation":
            width, height = 16, 9
        else:
            logger.warning(f"Unknown size preset: {size_preset}, using default sizes")

    # Create figure
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # Apply theme
    apply_theme(theme)

    if create_axes:
        ax = fig.add_subplot(111)
        return fig, ax

    return fig


def create_subplot_grid(
    fig: Figure,
    rows: int = 1,
    cols: int = 1,
    width_ratios: Optional[List[float]] = None,
    height_ratios: Optional[List[float]] = None,
    hspace: float = 0.4,
    wspace: float = 0.2,
) -> List[Axes]:
    """
    Create a grid of subplots with specified layout.

    Args:
        fig: Matplotlib figure
        rows: Number of rows
        cols: Number of columns
        width_ratios: Optional list of width ratios for columns
        height_ratios: Optional list of height ratios for rows
        hspace: Horizontal spacing between subplots
        wspace: Vertical spacing between subplots

    Returns:
        List[Axes]: List of axes objects
    """
    # Create GridSpec
    gs = GridSpec(
        rows,
        cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace,
    )

    # Create axes
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))

    return axes


def configure_axes(
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xticks: Optional[List[Any]] = None,
    yticks: Optional[List[Any]] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    xticklabel_rotation: float = 0,
    yticklabel_rotation: float = 0,
    grid: bool = True,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    font_size: Optional[Dict[str, float]] = None,
) -> Axes:
    """
    Configure axis properties consistently.

    Args:
        ax: Matplotlib axes to configure
        title: Optional title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        xlim: Optional x-axis limits (min, max)
        ylim: Optional y-axis limits (min, max)
        xticks: Optional x-axis tick positions
        yticks: Optional y-axis tick positions
        xticklabels: Optional x-axis tick labels
        yticklabels: Optional y-axis tick labels
        xticklabel_rotation: Rotation angle for x-axis tick labels
        yticklabel_rotation: Rotation angle for y-axis tick labels
        grid: Whether to show grid lines
        log_scale_x: Whether to use log scale for x-axis
        log_scale_y: Whether to use log scale for y-axis
        font_size: Optional dictionary of font sizes

    Returns:
        Axes: Configured axes
    """
    # Set title and labels if provided
    if title:
        if font_size and "title" in font_size:
            ax.set_title(title, fontsize=font_size["title"])
        else:
            ax.set_title(title)

    if xlabel:
        if font_size and "xlabel" in font_size:
            ax.set_xlabel(xlabel, fontsize=font_size["xlabel"])
        else:
            ax.set_xlabel(xlabel)

    if ylabel:
        if font_size and "ylabel" in font_size:
            ax.set_ylabel(ylabel, fontsize=font_size["ylabel"])
        else:
            ax.set_ylabel(ylabel)

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # Set axis scales
    if log_scale_x:
        ax.set_xscale("log")

    if log_scale_y:
        ax.set_yscale("log")

    # Set ticks and tick labels if provided
    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        if font_size and "xtick" in font_size:
            ax.set_xticklabels(
                xticklabels, fontsize=font_size["xtick"], rotation=xticklabel_rotation
            )
        else:
            ax.set_xticklabels(xticklabels, rotation=xticklabel_rotation)
    elif xticklabel_rotation != 0:
        # Apply rotation to existing labels
        plt.setp(ax.get_xticklabels(), rotation=xticklabel_rotation)

    if yticklabels is not None:
        if font_size and "ytick" in font_size:
            ax.set_yticklabels(
                yticklabels, fontsize=font_size["ytick"], rotation=yticklabel_rotation
            )
        else:
            ax.set_yticklabels(yticklabels, rotation=yticklabel_rotation)
    elif yticklabel_rotation != 0:
        # Apply rotation to existing labels
        plt.setp(ax.get_yticklabels(), rotation=yticklabel_rotation)

    # Configure grid
    ax.grid(grid, linestyle="--", alpha=0.7, zorder=0)

    return ax


def add_trend_line(
    ax: Axes,
    x_data: List[float],
    y_data: List[float],
    order: int = 1,
    color: str = "#666666",
    linestyle: str = "--",
    linewidth: float = 1.5,
    alpha: float = 0.8,
    extrapolate: float = 0.0,
    add_equation: bool = False,
    add_r_squared: bool = False,
    equation_pos: Tuple[float, float] = (0.05, 0.95),
    equation_fontsize: int = 10,
) -> None:
    """
    Add a trend line to a scatter or line plot.

    Args:
        ax: Matplotlib axes
        x_data: X-axis data
        y_data: Y-axis data
        order: Polynomial order (1 for linear, 2 for quadratic, etc.)
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Line transparency
        extrapolate: Fraction of x-range to extrapolate (0.0 = no extrapolation)
        add_equation: Whether to add the polynomial equation
        add_r_squared: Whether to add the R² value
        equation_pos: Position for equation text (x, y in axes coordinates)
        equation_fontsize: Font size for equation text
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    valid_x = np.array(x_data)[valid_mask]
    valid_y = np.array(y_data)[valid_mask]

    if len(valid_x) < order + 1:
        logger.warning(
            f"Not enough valid data points for polynomial fit of order {order}"
        )
        return

    # Fit polynomial
    coeffs = np.polyfit(valid_x, valid_y, order)
    poly = np.poly1d(coeffs)

    # Create x values for plotting
    x_range = max(valid_x) - min(valid_x)
    x_min = min(valid_x) - extrapolate * x_range
    x_max = max(valid_x) + extrapolate * x_range
    x_plot = np.linspace(x_min, x_max, 100)

    # Plot the trend line
    ax.plot(
        x_plot,
        poly(x_plot),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )

    # Add equation and/or R² if requested
    if add_equation or add_r_squared:
        equation_text = ""

        # Format polynomial equation
        if add_equation:
            if order == 1:
                equation_text = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
            else:
                terms = []
                for i, coeff in enumerate(coeffs):
                    power = order - i
                    if power == 0:
                        terms.append(f"{coeff:.3f}")
                    elif power == 1:
                        terms.append(f"{coeff:.3f}x")
                    else:
                        terms.append(f"{coeff:.3f}x^{power}")
                equation_text = "y = " + " + ".join(terms)

        # Calculate and add R²
        if add_r_squared:
            # Calculate R²
            y_pred = poly(valid_x)
            ss_total = np.sum((valid_y - np.mean(valid_y)) ** 2)
            ss_residual = np.sum((valid_y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            if equation_text:
                equation_text += f"\nR² = {r_squared:.3f}"
            else:
                equation_text = f"R² = {r_squared:.3f}"

        # Add text to plot
        ax.text(
            equation_pos[0],
            equation_pos[1],
            equation_text,
            transform=ax.transAxes,
            fontsize=equation_fontsize,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
        )


def add_reference_line(
    ax: Axes,
    value: float,
    orientation: str = "horizontal",
    color: str = "#666666",
    linestyle: str = "--",
    linewidth: float = 1.5,
    alpha: float = 0.8,
    add_label: bool = False,
    label: Optional[str] = None,
    label_offset: float = 0.01,
) -> None:
    """
    Add a reference line to a plot.

    Args:
        ax: Matplotlib axes
        value: Position for the reference line
        orientation: Line orientation ('horizontal' or 'vertical')
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Line transparency
        add_label: Whether to add a label
        label: Label text (defaults to the value if None)
        label_offset: Offset for label text
    """
    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Create the reference line
    if orientation == "horizontal":
        ax.axhline(
            y=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

        # Add label if requested
        if add_label:
            if label is None:
                label = f"{value:.3g}"

            ax.text(
                x_min + label_offset * (x_max - x_min),
                value,
                label,
                color=color,
                verticalalignment="bottom",
                horizontalalignment="left",
                backgroundcolor="white",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    else:  # vertical
        ax.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

        # Add label if requested
        if add_label:
            if label is None:
                label = f"{value:.3g}"

            ax.text(
                value,
                y_min + label_offset * (y_max - y_min),
                label,
                color=color,
                verticalalignment="bottom",
                horizontalalignment="right",
                rotation=90,
                backgroundcolor="white",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )


def add_data_table(
    ax: Axes,
    data: List[List[Any]],
    col_labels: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    fontsize: int = 10,
    loc: str = "bottom",
    bbox: Optional[List[float]] = None,
    col_widths: Optional[List[float]] = None,
    row_heights: Optional[List[float]] = None,
    highlight_cells: Optional[List[Tuple[int, int, str]]] = None,
    header_color: str = "#E6EFF6",
) -> None:
    """
    Add a data table to a figure.

    Args:
        ax: Matplotlib axes to add table to
        data: 2D list of data values
        col_labels: Optional list of column labels
        row_labels: Optional list of row labels
        title: Optional table title
        fontsize: Font size for table text
        loc: Location ('bottom', 'right', 'top', 'left')
        bbox: Optional bounding box [x, y, width, height]
        col_widths: Optional list of column widths
        row_heights: Optional list of row heights
        highlight_cells: Optional list of (row, col, color) tuples to highlight
        header_color: Background color for header row/column
    """
    # Default bbox if not provided
    if bbox is None:
        if loc == "bottom":
            bbox = [0.0, -0.3, 1.0, 0.2]
        elif loc == "right":
            bbox = [1.05, 0.0, 0.3, 1.0]
        else:
            bbox = [0.0, 0.0, 1.0, 0.2]

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc=loc,
        bbox=bbox,
        colWidths=col_widths,
        rowLoc="center",
        colLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.5)

    # Style header cells
    if col_labels:
        for i, _ in enumerate(col_labels):
            cell = table[0, i]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")

    # Style row label cells
    if row_labels:
        for i, _ in enumerate(row_labels):
            cell = table[i + 1, -1] if col_labels else table[i, -1]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")

    # Apply custom row heights if provided
    if row_heights:
        for i, height in enumerate(row_heights):
            if col_labels:
                i += 1  # Adjust for header row
            for cell_key in table._cells:
                if cell_key[0] == i:
                    table._cells[cell_key].set_height(height)

    # Highlight specific cells if requested
    if highlight_cells:
        for row, col, color in highlight_cells:
            # Adjust for header row if present
            if col_labels:
                row += 1
            if row_labels:
                col += 1

            if (row, col) in table._cells:
                table._cells[(row, col)].set_facecolor(color)

    # Add title if provided
    if title:
        if loc == "bottom":
            ax.text(
                0.5,
                -0.15,
                title,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize + 2,
                fontweight="bold",
            )
        elif loc == "right":
            ax.text(
                1.2,
                1.05,
                title,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize + 2,
                fontweight="bold",
            )


# ----------------------
# Styling and Theme
# ----------------------


def apply_theme(theme_name: str = "default") -> None:
    """
    Apply a predefined theme to matplotlib plots.

    Args:
        theme_name: Name of the theme to apply ('default', 'dark', 'print')
    """
    if theme_name not in THEMES:
        logger.warning(f"Unknown theme: {theme_name}, using default theme")
        theme_name = "default"

    # Get theme settings
    theme = THEMES[theme_name]

    # Apply settings to matplotlib rcParams
    for key, value in theme.items():
        plt.rcParams[key] = value


def get_color_palette(palette_name: str, n_colors: Optional[int] = None) -> List[str]:
    """
    Get a color palette by name.

    Args:
        palette_name: Name of the palette
        n_colors: Optional number of colors to return

    Returns:
        List[str]: List of hex color codes
    """
    # Special handling for engagement levels and tool versions
    if palette_name in ["engagement_levels", "tool_versions"]:
        return COLOR_PALETTES[palette_name]

    # Check if palette exists
    if palette_name not in COLOR_PALETTES:
        logger.warning(f"Unknown palette: {palette_name}, using default palette")
        palette_name = "default"

    # Get palette
    palette = COLOR_PALETTES[palette_name]

    # Return full palette if n_colors not specified
    if n_colors is None:
        return palette

    # Handle case when more colors requested than available
    if n_colors > len(palette):
        # Create a custom palette with the right number of colors
        if "sequential" in palette_name:
            # For sequential palettes, interpolate
            return sns.color_palette(palette, n_colors=n_colors).as_hex()
        else:
            # For categorical palettes, cycle
            return [palette[i % len(palette)] for i in range(n_colors)]

    # Return subset of palette
    return palette[:n_colors]


def create_colormap(palette_name: str, reverse: bool = False) -> Any:
    """
    Create a colormap from a named palette.

    Args:
        palette_name: Name of the palette
        reverse: Whether to reverse the colormap

    Returns:
        Any: Matplotlib colormap
    """
    # Get palette
    palette = get_color_palette(palette_name)

    # Reverse if requested
    if reverse:
        palette = palette[::-1]

    # Create colormap
    return mcolors.ListedColormap(palette)


def style_spines(
    ax: Axes,
    spines_to_show: List[str] = ["bottom", "left"],
    line_width: float = 1.0,
    color: str = "#333333",
) -> None:
    """
    Style axis spines (border lines).

    Args:
        ax: Matplotlib axes
        spines_to_show: List of spines to show ('top', 'right', 'bottom', 'left')
        line_width: Line width for spines
        color: Line color for spines
    """
    # Hide all spines first
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Show and style requested spines
    for spine in spines_to_show:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(line_width)
        ax.spines[spine].set_color(color)


def wrap_labels(ax: Axes, which: str = "x", max_length: int = 20) -> None:
    """
    Wrap long axis labels.

    Args:
        ax: Matplotlib axes
        which: Which axis to wrap labels for ('x', 'y', or 'both')
        max_length: Maximum characters per line
    """
    import textwrap

    def _wrap_text(text):
        if len(text) > max_length:
            return "\n".join(textwrap.wrap(text, width=max_length))
        return text

    if which in ["x", "both"]:
        labels = ax.get_xticklabels()
        ax.set_xticklabels([_wrap_text(label.get_text()) for label in labels])

    if which in ["y", "both"]:
        labels = ax.get_yticklabels()
        ax.set_yticklabels([_wrap_text(label.get_text()) for label in labels])


# ----------------------
# Common Chart Types
# ----------------------


def create_visualization(
    data: Dict[str, Any],
    viz_type: str,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    palette: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Create a visualization based on the specified type.

    This function serves as a high-level interface for creating
    various chart types with consistent styling.

    Args:
        data: Data for the visualization
        viz_type: Type of visualization to create
        fig: Optional existing figure
        ax: Optional existing axes
        title: Optional chart title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        figsize: Figure size if creating new figure
        palette: Optional color palette name
        **kwargs: Additional arguments for specific chart types

    Returns:
        Tuple[Figure, Axes]: Created figure and axes
    """
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Create the visualization based on type
    if viz_type == "bar":
        plot_bar(
            ax,
            data=data["values"],
            orientation=kwargs.get("orientation", "vertical"),
            color=kwargs.get("color", None),
            sort_bars=kwargs.get("sort_bars", False),
            error_data=kwargs.get("error_data", None),
            add_data_labels=kwargs.get("add_data_labels", False),
        )

    elif viz_type == "grouped_bar":
        plot_grouped_bars(
            ax,
            data=data["values"],
            orientation=kwargs.get("orientation", "vertical"),
            colors=kwargs.get("colors", None),
            error_data=kwargs.get("error_data", None),
            add_data_labels=kwargs.get("add_data_labels", False),
            add_legend=kwargs.get("add_legend", True),
            legend_title=kwargs.get("legend_title", None),
        )

    elif viz_type == "stacked_bar":
        plot_stacked_bars(
            ax,
            data=data["values"],
            orientation=kwargs.get("orientation", "vertical"),
            colors=kwargs.get("colors", None),
            add_data_labels=kwargs.get("add_data_labels", False),
            show_total=kwargs.get("show_total", False),
            add_legend=kwargs.get("add_legend", True),
            legend_title=kwargs.get("legend_title", None),
        )

    elif viz_type == "line":
        plot_line(
            ax,
            x_data=data["x"],
            y_data=data["y"],
            label=kwargs.get("label", None),
            color=kwargs.get("color", None),
            marker=kwargs.get("marker", None),
            line_style=kwargs.get("line_style", "-"),
            add_data_labels=kwargs.get("add_data_labels", False),
        )

    elif viz_type == "multi_line":
        plot_multi_line(
            ax,
            data=data["values"],
            x_values=data.get("x_values", None),
            colors=kwargs.get("colors", None),
            markers=kwargs.get("markers", None),
            add_legend=kwargs.get("add_legend", True),
            legend_title=kwargs.get("legend_title", None),
            add_data_labels=kwargs.get("add_data_labels", False),
        )

    elif viz_type == "scatter":
        plot_scatter(
            ax,
            x_data=data["x"],
            y_data=data["y"],
            sizes=kwargs.get("sizes", None),
            colors=kwargs.get("colors", None),
            color_map=kwargs.get("color_map", None),
            alpha=kwargs.get("alpha", 0.7),
            add_labels=kwargs.get("add_labels", False),
            labels=kwargs.get("labels", None),
            add_trend_line=kwargs.get("add_trend_line", False),
            add_correlation=kwargs.get("add_correlation", False),
        )

    elif viz_type == "heatmap":
        plot_heatmap(
            ax,
            data=data["values"],
            row_labels=data.get("row_labels", None),
            col_labels=data.get("col_labels", None),
            color_map=kwargs.get("color_map", "viridis"),
            add_colorbar=kwargs.get("add_colorbar", True),
            add_values=kwargs.get("add_values", True),
            value_format=kwargs.get("value_format", "{:.2f}"),
        )

    elif viz_type == "distribution":
        plot_distribution(
            ax,
            data=data["values"],
            plot_type=kwargs.get("plot_type", "histogram"),
            bins=kwargs.get("bins", 20),
            color=kwargs.get("color", None),
            alpha=kwargs.get("alpha", 0.7),
            add_mean_line=kwargs.get("add_mean_line", True),
            add_median_line=kwargs.get("add_median_line", True),
        )

    elif viz_type == "timeline":
        plot_timeline(
            ax,
            events=data["events"],
            date_column=kwargs.get("date_column", "date"),
            category_column=kwargs.get("category_column", None),
            color_column=kwargs.get("color_column", None),
            label_column=kwargs.get("label_column", None),
            add_labels=kwargs.get("add_labels", False),
            date_format=kwargs.get("date_format", "%Y-%m-%d"),
        )

    elif viz_type == "network":
        create_network_graph(
            ax,
            nodes=data["nodes"],
            edges=data["edges"],
            node_size_column=kwargs.get("node_size_column", None),
            node_color_column=kwargs.get("node_color_column", None),
            node_label_column=kwargs.get("node_label_column", None),
            edge_width_column=kwargs.get("edge_width_column", None),
            with_labels=kwargs.get("with_labels", True),
            layout=kwargs.get("layout", "spring"),
        )

    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    # Configure axes
    configure_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=kwargs.get("xlim", None),
        ylim=kwargs.get("ylim", None),
        xticks=kwargs.get("xticks", None),
        yticks=kwargs.get("yticks", None),
        xticklabels=kwargs.get("xticklabels", None),
        yticklabels=kwargs.get("yticklabels", None),
        xticklabel_rotation=kwargs.get("xticklabel_rotation", 0),
        yticklabel_rotation=kwargs.get("yticklabel_rotation", 0),
        grid=kwargs.get("grid", True),
        log_scale_x=kwargs.get("log_scale_x", False),
        log_scale_y=kwargs.get("log_scale_y", False),
    )

    # Apply custom styling if specified
    if "spines_to_show" in kwargs:
        style_spines(ax, spines_to_show=kwargs["spines_to_show"])

    # Wrap long labels if needed
    if kwargs.get("wrap_labels", False):
        wrap_labels(
            ax,
            which=kwargs.get("wrap_which", "x"),
            max_length=kwargs.get("wrap_max_length", 20),
        )

    # Add trend line if requested - we'll use add_trend_line defined in Annotation Functions
    if kwargs.get("add_trend", False) and viz_type == "scatter":
        add_trend_line(
            ax,
            x_data=data["x"],
            y_data=data["y"],
            order=kwargs.get("trend_order", 1),
            add_equation=kwargs.get("add_equation", False),
            add_r_squared=kwargs.get("add_r_squared", False),
        )

    # Add reference line if requested - using add_reference_line defined in Annotation Functions
    if "reference_value" in kwargs:
        add_reference_line(
            ax,
            value=kwargs["reference_value"],
            orientation=kwargs.get("reference_orientation", "horizontal"),
            add_label=kwargs.get("reference_label", True),
            label=kwargs.get("reference_text", None),
        )

    # Tight layout
    if kwargs.get("tight_layout", True):
        fig.tight_layout()

    return fig, ax


def plot_bar(
    ax: Axes,
    data: Dict[str, Union[float, int]],
    orientation: str = "vertical",
    color: Optional[Union[str, List[str]]] = None,
    sort_bars: bool = False,
    error_data: Optional[Dict[str, Union[float, int]]] = None,
    add_data_labels: bool = False,
    data_label_format: str = "{:.1f}",
    data_label_offset: float = 0.5,
    data_label_fontsize: int = 9,
    bar_width: float = 0.8,
    bar_alpha: float = 0.8,
) -> None:
    """
    Create a bar chart.

    Args:
        ax: Matplotlib axes to plot on
        data: Dictionary mapping categories to values
        orientation: Bar orientation ('vertical' or 'horizontal')
        color: Bar color or list of colors
        sort_bars: Whether to sort bars by value
        error_data: Optional dictionary of error values
        add_data_labels: Whether to add value labels to bars
        data_label_format: Format string for data labels
        data_label_offset: Offset for data labels
        data_label_fontsize: Font size for data labels
        bar_width: Width of bars (0.0-1.0)
        bar_alpha: Opacity of bars (0.0-1.0)
    """
    # Handle empty data
    if not data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Sort data if requested
    if sort_bars:
        sorted_items = sorted(data.items(), key=lambda x: x[1])
        categories = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
    else:
        categories = list(data.keys())
        values = list(data.values())

    # Create x positions
    x_pos = np.arange(len(categories))

    # Set default color if not provided
    if color is None:
        color = get_color_palette("categorical_main", n_colors=1)[0]

    # Create bars
    if orientation == "vertical":
        bars = ax.bar(
            x_pos,
            values,
            width=bar_width,
            alpha=bar_alpha,
            color=color,
            edgecolor="none",
            zorder=3,
        )

        # Add error bars if provided
        if error_data:
            error_values = [error_data.get(cat, 0) for cat in categories]
            ax.errorbar(
                x_pos,
                values,
                yerr=error_values,
                fmt="none",
                ecolor="#333333",
                capsize=3,
                zorder=4,
            )

        # Set x-axis tick labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            categories,
            rotation=45 if max(len(cat) for cat in categories) > 10 else 0,
            ha="right" if max(len(cat) for cat in categories) > 10 else "center",
        )

        # Add data labels if requested
        if add_data_labels:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                label_pos_y = height + data_label_offset
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_pos_y,
                    data_label_format.format(values[i]),
                    ha="center",
                    va="bottom",
                    fontsize=data_label_fontsize,
                )

    else:  # horizontal
        bars = ax.barh(
            x_pos,
            values,
            height=bar_width,
            alpha=bar_alpha,
            color=color,
            edgecolor="none",
            zorder=3,
        )

        # Add error bars if provided
        if error_data:
            error_values = [error_data.get(cat, 0) for cat in categories]
            ax.errorbar(
                values,
                x_pos,
                xerr=error_values,
                fmt="none",
                ecolor="#333333",
                capsize=3,
                zorder=4,
            )

        # Set y-axis tick labels
        ax.set_yticks(x_pos)
        ax.set_yticklabels(categories)

        # Add data labels if requested
        if add_data_labels:
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_pos_x = width + data_label_offset
                ax.text(
                    label_pos_x,
                    bar.get_y() + bar.get_height() / 2,
                    data_label_format.format(values[i]),
                    ha="left",
                    va="center",
                    fontsize=data_label_fontsize,
                )

    # Adjust layout
    if orientation == "vertical":
        ax.set_xlim(-0.5, len(categories) - 0.5)
    else:
        ax.set_ylim(-0.5, len(categories) - 0.5)


def plot_grouped_bars(
    ax: Axes,
    data: Dict[str, Dict[str, Union[float, int]]],
    orientation: str = "vertical",
    colors: Optional[List[str]] = None,
    error_data: Optional[Dict[str, Dict[str, Union[float, int]]]] = None,
    add_data_labels: bool = False,
    data_label_format: str = "{:.1f}",
    bar_width: float = 0.8,
    bar_alpha: float = 0.8,
    add_legend: bool = True,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
) -> None:
    """
    Create a grouped bar chart.

    Args:
        ax: Matplotlib axes to plot on
        data: Nested dictionary mapping categories to group values
        orientation: Bar orientation ('vertical' or 'horizontal')
        colors: List of colors for groups
        error_data: Optional nested dictionary of error values
        add_data_labels: Whether to add value labels to bars
        data_label_format: Format string for data labels
        bar_width: Width of each group (0.0-1.0)
        bar_alpha: Opacity of bars (0.0-1.0)
        add_legend: Whether to add a legend
        legend_title: Optional legend title
        legend_loc: Legend location
    """
    # Handle empty data
    if not data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Get categories and groups
    categories = list(data.keys())
    groups = list(data[categories[0]].keys()) if categories else []

    # Set colors if not provided
    if colors is None:
        colors = get_color_palette("categorical_main", n_colors=len(groups))

    # Number of groups
    n_groups = len(groups)

    # Width of each individual bar
    individual_width = bar_width / n_groups

    # Create x positions for groups
    x_indices = np.arange(len(categories))

    # Plot each group
    bars = []
    for i, group in enumerate(groups):
        # Get values for this group
        values = [data[cat].get(group, 0) for cat in categories]

        # Calculate x positions for this group
        if orientation == "vertical":
            x_pos = x_indices - (bar_width / 2) + (i + 0.5) * individual_width

            # Create bars
            group_bars = ax.bar(
                x_pos,
                values,
                width=individual_width,
                alpha=bar_alpha,
                color=colors[i % len(colors)],
                edgecolor="none",
                zorder=3,
                label=group,
            )

            # Add error bars if provided
            if error_data:
                error_values = [
                    error_data.get(cat, {}).get(group, 0) for cat in categories
                ]
                ax.errorbar(
                    x_pos,
                    values,
                    yerr=error_values,
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    zorder=4,
                )

            # Add data labels if requested
            if add_data_labels:
                for j, bar in enumerate(group_bars):
                    height = bar.get_height()
                    if height != 0:  # Only add labels to non-zero bars
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.1,
                            data_label_format.format(height),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=90,
                        )

        else:  # horizontal
            y_pos = x_indices - (bar_width / 2) + (i + 0.5) * individual_width

            # Create bars
            group_bars = ax.barh(
                y_pos,
                values,
                height=individual_width,
                alpha=bar_alpha,
                color=colors[i % len(colors)],
                edgecolor="none",
                zorder=3,
                label=group,
            )

            # Add error bars if provided
            if error_data:
                error_values = [
                    error_data.get(cat, {}).get(group, 0) for cat in categories
                ]
                ax.errorbar(
                    values,
                    y_pos,
                    xerr=error_values,
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    zorder=4,
                )

            # Add data labels if requested
            if add_data_labels:
                for j, bar in enumerate(group_bars):
                    width = bar.get_width()
                    if width != 0:  # Only add labels to non-zero bars
                        ax.text(
                            width + 0.1,
                            bar.get_y() + bar.get_height() / 2,
                            data_label_format.format(width),
                            ha="left",
                            va="center",
                            fontsize=8,
                        )

        bars.append(group_bars)

    # Set axis labels
    if orientation == "vertical":
        ax.set_xticks(x_indices)
        ax.set_xticklabels(
            categories,
            rotation=45 if max(len(cat) for cat in categories) > 10 else 0,
            ha="right" if max(len(cat) for cat in categories) > 10 else "center",
        )
        ax.set_xlim(-0.5, len(categories) - 0.5)
    else:
        ax.set_yticks(x_indices)
        ax.set_yticklabels(categories)
        ax.set_ylim(-0.5, len(categories) - 0.5)

    # Add legend if requested
    if add_legend and groups:
        legend = ax.legend(
            title=legend_title, loc=legend_loc, frameon=True, framealpha=0.8
        )
        if legend_title:
            legend.get_title().set_fontweight("bold")


def plot_stacked_bars(
    ax: Axes,
    data: Dict[str, Dict[str, Union[float, int]]],
    orientation: str = "vertical",
    colors: Optional[List[str]] = None,
    add_data_labels: bool = False,
    data_label_format: str = "{:.1f}",
    show_total: bool = False,
    bar_width: float = 0.8,
    bar_alpha: float = 0.8,
    add_legend: bool = True,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
) -> None:
    """
    Create a stacked bar chart.

    Args:
        ax: Matplotlib axes to plot on
        data: Nested dictionary mapping categories to stack values
        orientation: Bar orientation ('vertical' or 'horizontal')
        colors: List of colors for stacks
        add_data_labels: Whether to add value labels to bars
        data_label_format: Format string for data labels
        show_total: Whether to show total sum above each stack
        bar_width: Width of bars (0.0-1.0)
        bar_alpha: Opacity of bars (0.0-1.0)
        add_legend: Whether to add a legend
        legend_title: Optional legend title
        legend_loc: Legend location
    """
    # Handle empty data
    if not data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Get categories and stacks
    categories = list(data.keys())
    stacks = list(data[categories[0]].keys()) if categories else []

    # Set colors if not provided
    if colors is None:
        colors = get_color_palette("categorical_main", n_colors=len(stacks))

    # Create x positions
    x_indices = np.arange(len(categories))

    # Initialize bottom (for vertical) or left (for horizontal) positions
    bottoms = np.zeros(len(categories))

    # Plot each stack
    for i, stack in enumerate(stacks):
        # Get values for this stack
        values = [data[cat].get(stack, 0) for cat in categories]

        # Create bars
        if orientation == "vertical":
            bars = ax.bar(
                x_indices,
                values,
                width=bar_width,
                alpha=bar_alpha,
                color=colors[i % len(colors)],
                edgecolor="none",
                zorder=3,
                bottom=bottoms,
                label=stack,
            )

            # Add data labels if requested
            if add_data_labels:
                for j, (value, bottom) in enumerate(zip(values, bottoms)):
                    if value != 0:  # Only add labels to non-zero segments
                        height = value / 2  # Position in middle of segment
                        ax.text(
                            x_indices[j],
                            bottom + height,
                            data_label_format.format(value),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value > (max(values) * 0.25) else "black",
                        )

            # Update bottom positions for next stack
            bottoms += values

        else:  # horizontal
            bars = ax.barh(
                x_indices,
                values,
                height=bar_width,
                alpha=bar_alpha,
                color=colors[i % len(colors)],
                edgecolor="none",
                zorder=3,
                left=bottoms,
                label=stack,
            )

            # Add data labels if requested
            if add_data_labels:
                for j, (value, left) in enumerate(zip(values, bottoms)):
                    if value != 0:  # Only add labels to non-zero segments
                        width = value / 2  # Position in middle of segment
                        ax.text(
                            left + width,
                            x_indices[j],
                            data_label_format.format(value),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value > (max(values) * 0.25) else "black",
                        )

            # Update left positions for next stack
            bottoms += values

    # Add total labels if requested
    if show_total:
        if orientation == "vertical":
            for i, total in enumerate(bottoms):
                ax.text(
                    x_indices[i],
                    total + 0.5,
                    data_label_format.format(total),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        else:
            for i, total in enumerate(bottoms):
                ax.text(
                    total + 0.5,
                    x_indices[i],
                    data_label_format.format(total),
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    # Set axis labels
    if orientation == "vertical":
        ax.set_xticks(x_indices)
        ax.set_xticklabels(
            categories,
            rotation=45 if max(len(cat) for cat in categories) > 10 else 0,
            ha="right" if max(len(cat) for cat in categories) > 10 else "center",
        )
        ax.set_xlim(-0.5, len(categories) - 0.5)
    else:
        ax.set_yticks(x_indices)
        ax.set_yticklabels(categories)
        ax.set_ylim(-0.5, len(categories) - 0.5)

    # Add legend if requested
    if add_legend and stacks:
        legend = ax.legend(
            title=legend_title, loc=legend_loc, frameon=True, framealpha=0.8
        )
        if legend_title:
            legend.get_title().set_fontweight("bold")


def plot_line(
    ax: Axes,
    x_data: List[Any],
    y_data: List[float],
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    line_style: str = "-",
    line_width: float = 2.0,
    marker_size: float = 6.0,
    alpha: float = 1.0,
    add_data_labels: bool = False,
    data_label_format: str = "{:.1f}",
    data_label_fontsize: int = 8,
) -> Line2D:
    """
    Plot a single line.

    Args:
        ax: Matplotlib axes to plot on
        x_data: X-axis data
        y_data: Y-axis data
        label: Optional line label for legend
        color: Line color
        marker: Optional marker style
        line_style: Line style
        line_width: Line width
        marker_size: Marker size
        alpha: Line opacity
        add_data_labels: Whether to add value labels
        data_label_format: Format string for data labels
        data_label_fontsize: Font size for data labels

    Returns:
        Line2D: The created line
    """
    # Set default color if not provided
    if color is None:
        color = get_color_palette("categorical_main", n_colors=1)[0]

    # Plot line
    line = ax.plot(
        x_data,
        y_data,
        label=label,
        color=color,
        marker=marker,
        linestyle=line_style,
        linewidth=line_width,
        markersize=marker_size,
        alpha=alpha,
        zorder=3,
    )[0]

    # Add data labels if requested
    if add_data_labels:
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            # Skip first and last points for better readability
            if i > 0 and i < len(x_data) - 1:
                if i % 2 == 0:  # Only label every other point for better readability
                    continue

            ax.annotate(
                data_label_format.format(y),
                (x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=data_label_fontsize,
            )

    return line


def plot_multi_line(
    ax: Axes,
    data: Dict[str, Dict[str, float]],
    x_values: Optional[List[Any]] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    add_legend: bool = True,
    legend_title: Optional[str] = None,
    legend_loc: str = "best",
    add_data_labels: bool = False,
    data_label_format: str = "{:.1f}",
) -> Dict[str, Line2D]:
    """
    Plot multiple lines on the same axes.

    Args:
        ax: Matplotlib axes to plot on
        data: Nested dictionary with series names mapped to x-y value pairs
        x_values: Optional shared x values (if None, extracted from data)
        colors: Optional list of colors for series
        markers: Optional list of markers for series
        add_legend: Whether to add a legend
        legend_title: Optional legend title
        legend_loc: Legend location
        add_data_labels: Whether to add value labels
        data_label_format: Format string for data labels

    Returns:
        Dict[str, Line2D]: Dictionary mapping series names to line objects
    """
    # Handle empty data
    if not data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return {}

    # Get series names
    series_names = list(data.keys())

    # Set colors if not provided
    if colors is None:
        colors = get_color_palette("categorical_main", n_colors=len(series_names))

    # Set default markers if not provided
    if markers is None:
        # Use no markers by default
        markers = [None] * len(series_names)

    # Determine x values if not provided
    if x_values is None:
        # Extract all x values from the data
        all_x = set()
        for series in data.values():
            all_x.update(series.keys())
        x_values = sorted(all_x)

    # Plot each series
    lines = {}
    for i, series_name in enumerate(series_names):
        series_data = data[series_name]

        # Extract y values for this series, using x_values as the keys
        y_values = [series_data.get(x, float("nan")) for x in x_values]

        # Remove NaN values for plotting
        valid_indices = ~np.isnan(y_values)
        valid_x = [x_values[i] for i, valid in enumerate(valid_indices) if valid]
        valid_y = [y for y, valid in zip(y_values, valid_indices) if valid]

        # Skip if no valid data
        if not valid_x:
            continue

        # Plot line
        line = plot_line(
            ax,
            valid_x,
            valid_y,
            label=series_name,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            add_data_labels=add_data_labels,
            data_label_format=data_label_format,
        )

        lines[series_name] = line

    # Add legend if requested
    if add_legend and lines:
        legend = ax.legend(
            title=legend_title, loc=legend_loc, frameon=True, framealpha=0.8
        )
        if legend_title:
            legend.get_title().set_fontweight("bold")

    return lines


def plot_scatter(
    ax: Axes,
    x_data: List[float],
    y_data: List[float],
    sizes: Optional[List[float]] = None,
    colors: Optional[List[str]] = None,
    color_map: Optional[str] = None,
    alpha: float = 0.7,
    add_labels: bool = False,
    labels: Optional[List[str]] = None,
    add_trend_line: bool = False,
    add_correlation: bool = False,
    marker: str = "o",
) -> None:
    """
    Create a scatter plot.

    Args:
        ax: Matplotlib axes to plot on
        x_data: X-axis data
        y_data: Y-axis data
        sizes: Optional point sizes
        colors: Optional point colors or values for colormap
        color_map: Optional matplotlib colormap name
        alpha: Point opacity
        add_labels: Whether to add labels to points
        labels: Optional list of labels for points
        add_trend_line: Whether to add a linear trend line
        add_correlation: Whether to add correlation coefficient annotation
        marker: Marker style
    """
    # Handle empty data
    if not x_data or not y_data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Set default sizes if not provided
    if sizes is None:
        sizes = [40] * len(x_data)

    # Handle colors
    cmap = None
    if color_map is not None and colors is not None:
        # Using a color map with color values
        cmap = plt.get_cmap(color_map)
        scatter = ax.scatter(
            x_data,
            y_data,
            s=sizes,
            c=colors,
            cmap=cmap,
            alpha=alpha,
            marker=marker,
            zorder=3,
        )

        # Add colorbar
        plt.colorbar(scatter, ax=ax)
    else:
        # Using single color or list of specific colors
        scatter = ax.scatter(
            x_data, y_data, s=sizes, c=colors, alpha=alpha, marker=marker, zorder=3
        )

    # Add labels if requested
    if add_labels and labels:
        for i, (x, y, label) in enumerate(zip(x_data, y_data, labels)):
            ax.annotate(
                label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    # Add trend line if requested
    if add_trend_line and len(x_data) > 1:
        # Filter out NaN values
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        valid_x = np.array(x_data)[valid_mask]
        valid_y = np.array(y_data)[valid_mask]

        if len(valid_x) > 1:
            # Calculate linear regression
            z = np.polyfit(valid_x, valid_y, 1)
            p = np.poly1d(z)

            # Add trend line
            x_range = np.linspace(min(valid_x), max(valid_x), 100)
            ax.plot(x_range, p(x_range), "--", color="#666666", zorder=2)

            # Add correlation coefficient if requested
            if add_correlation:
                correlation = np.corrcoef(valid_x, valid_y)[0, 1]
                ax.annotate(
                    f"r = {correlation:.2f}",
                    xy=(0.95, 0.05),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )


def plot_heatmap(
    ax: Axes,
    data: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    color_map: str = "viridis",
    add_colorbar: bool = True,
    add_values: bool = True,
    value_format: str = "{:.2f}",
    value_threshold: Optional[float] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Create a heatmap visualization.

    Args:
        ax: Matplotlib axes to plot on
        data: 2D array of values
        row_labels: Optional labels for rows
        col_labels: Optional labels for columns
        color_map: Matplotlib colormap name
        add_colorbar: Whether to add a colorbar
        add_values: Whether to add text values to cells
        value_format: Format string for cell values
        value_threshold: Optional threshold for value color (white/black)
        title: Optional heatmap title
        vmin: Optional minimum value for color scaling
        vmax: Optional maximum value for color scaling
    """
    # Handle empty data
    if data.size == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Create heatmap
    im = ax.imshow(data, cmap=color_map, aspect="auto", vmin=vmin, vmax=vmax)

    # Add colorbar if requested
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)

    # Add title if provided
    if title:
        ax.set_title(title)

    # Add row and column labels if provided
    if row_labels:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

    if col_labels:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")

    # Add grid lines
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add text values if requested
    if add_values:
        # Auto-compute threshold if not provided
        if value_threshold is None:
            value_threshold = (data.max() + data.min()) / 2

        # Add text annotations to each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                text_color = "white" if value < value_threshold else "black"
                ax.text(
                    j,
                    i,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )


def plot_distribution(
    ax: Axes,
    data: List[float],
    plot_type: str = "histogram",
    bins: Union[int, List[float]] = 20,
    color: Optional[str] = None,
    alpha: float = 0.7,
    add_mean_line: bool = True,
    add_median_line: bool = True,
    kde_bandwidth: Optional[float] = None,
    rug_plot: bool = False,
    cumulative: bool = False,
) -> None:
    """
    Plot a data distribution using various visualization types.

    Args:
        ax: Matplotlib axes to plot on
        data: List of values to visualize
        plot_type: Type of plot ('histogram', 'kde', 'boxplot', 'violin')
        bins: Number of bins or bin edges for histogram
        color: Fill color
        alpha: Fill opacity
        add_mean_line: Whether to add a vertical line at the mean
        add_median_line: Whether to add a vertical line at the median
        kde_bandwidth: Optional KDE bandwidth parameter
        rug_plot: Whether to add a rug plot
        cumulative: Whether to show cumulative distribution
    """
    # Handle empty data
    if not data:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Set default color if not provided
    if color is None:
        color = get_color_palette("categorical_main", n_colors=1)[0]

    # Calculate statistics
    mean_value = np.nanmean(data)
    median_value = np.nanmedian(data)

    # Create distribution plot based on plot type
    if plot_type == "histogram":
        # Create histogram
        n, bins, patches = ax.hist(
            data,
            bins=bins,
            color=color,
            alpha=alpha,
            cumulative=cumulative,
            density=True if cumulative else False,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # Add KDE curve
        if not cumulative:
            x_grid = np.linspace(min(data), max(data), 1000)
            try:
                kde = sns.kdeplot(
                    data=data,
                    ax=ax,
                    color="#333333",
                    linewidth=2,
                    zorder=4,
                    bw_adjust=kde_bandwidth,
                )
            except:
                # Fallback if KDE fails
                pass

    elif plot_type == "kde":
        # Create KDE plot
        try:
            kde = sns.kdeplot(
                data=data,
                ax=ax,
                color=color,
                fill=True,
                alpha=alpha,
                linewidth=2,
                zorder=3,
                cumulative=cumulative,
                bw_adjust=kde_bandwidth,
            )
        except:
            # Fallback to histogram if KDE fails
            ax.hist(
                data,
                bins=bins,
                color=color,
                alpha=alpha,
                cumulative=cumulative,
                density=True if cumulative else False,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    elif plot_type == "boxplot":
        # Create box plot
        box = ax.boxplot(
            data, patch_artist=True, vert=True, widths=0.5, showfliers=True, zorder=3
        )

        # Style box plot
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        for element in ["whiskers", "caps", "medians"]:
            for line in box[element]:
                line.set_color("#333333")
                line.set_linewidth(1.5)

        for flier in box["fliers"]:
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor("#333333")
            flier.set_alpha(0.7)

    elif plot_type == "violin":
        # Create violin plot
        violin = ax.violinplot(
            data, points=100, showextrema=True, showmedians=True, vert=True
        )

        # Style violin plot
        for pc in violin["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            pc.set_edgecolor("#333333")
            pc.set_linewidth(1)

        for element in ["cmins", "cmaxes", "cbars", "cmedians"]:
            if element in violin:
                violin[element].set_color("#333333")
                violin[element].set_linewidth(1.5)

    # Add rug plot if requested
    if rug_plot and plot_type in ["histogram", "kde"]:
        ax.plot(
            data, [0.01] * len(data), "|", color="#333333", markersize=10, alpha=0.7
        )

    # Add mean line if requested
    if add_mean_line and plot_type in ["histogram", "kde"]:
        ax.axvline(
            mean_value,
            color="#E41A1C",
            linestyle="-",
            linewidth=2,
            zorder=5,
            label=f"Mean: {mean_value:.2f}",
        )

    # Add median line if requested
    if add_median_line and plot_type in ["histogram", "kde"]:
        ax.axvline(
            median_value,
            color="#377EB8",
            linestyle="--",
            linewidth=2,
            zorder=5,
            label=f"Median: {median_value:.2f}",
        )

    # Add legend if lines were added
    if (add_mean_line or add_median_line) and plot_type in ["histogram", "kde"]:
        ax.legend(loc="upper right", frameon=True, framealpha=0.8)


def plot_timeline(
    ax: Axes,
    events: List[Dict[str, Any]],
    date_column: str = "date",
    category_column: Optional[str] = None,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    marker_column: Optional[str] = None,
    label_column: Optional[str] = None,
    default_color: str = "#3366CC",
    default_marker: str = "o",
    default_size: float = 10.0,
    add_labels: bool = False,
    add_grid_lines: bool = True,
    date_format: str = "%Y-%m-%d",
) -> None:
    """
    Create a timeline visualization.

    Args:
        ax: Matplotlib axes to plot on
        events: List of event dictionaries
        date_column: Name of column containing dates
        category_column: Optional column for categorizing events
        color_column: Optional column for determining event colors
        size_column: Optional column for determining event sizes
        marker_column: Optional column for determining event markers
        label_column: Optional column for event labels
        default_color: Default color for events
        default_marker: Default marker for events
        default_size: Default size for events
        add_labels: Whether to add labels to events
        add_grid_lines: Whether to add grid lines
        date_format: Date format string for parsing dates
    """
    # Handle empty data
    if not events:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Group events by category if specified
    if category_column:
        categories = {}
        for event in events:
            category = event.get(category_column, "Unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(event)

        # Get category colors
        category_colors = {
            category: color
            for category, color in zip(
                categories.keys(),
                get_color_palette("categorical_main", n_colors=len(categories)),
            )
        }
    else:
        # Treat all events as one category
        categories = {"All": events}
        category_colors = {"All": default_color}

    # Parse dates
    for event in events:
        date_str = event.get(date_column)
        try:
            if isinstance(date_str, datetime):
                event["parsed_date"] = date_str
            else:
                event["parsed_date"] = datetime.strptime(date_str, date_format)
        except (ValueError, TypeError):
            # Skip events with invalid dates
            event["parsed_date"] = None

    # Filter out events with invalid dates
    valid_events = [event for event in events if event.get("parsed_date")]

    # Get date range
    all_dates = [event["parsed_date"] for event in valid_events]
    min_date = min(all_dates) if all_dates else datetime.now()
    max_date = max(all_dates) if all_dates else datetime.now()

    # Add some padding to date range
    date_range = (max_date - min_date).days
    date_padding = max(1, date_range * 0.05)  # At least 1 day padding

    min_date = min_date - timedelta(days=date_padding)
    max_date = max_date + timedelta(days=date_padding)

    # Plot events by category
    y_positions = {}
    y_offset = 0

    for category, category_events in categories.items():
        valid_category_events = [
            event for event in category_events if event.get("parsed_date")
        ]

        if not valid_category_events:
            continue

        color = category_colors.get(category, default_color)
        y_pos = y_offset

        # Plot each event
        for event in valid_category_events:
            # Get event properties
            date = event["parsed_date"]

            # Get marker size
            if size_column and size_column in event:
                size = event[size_column]
            else:
                size = default_size

            # Get marker style
            if marker_column and marker_column in event:
                marker = event[marker_column]
            else:
                marker = default_marker

            # Get color
            if color_column and color_column in event:
                specific_color = event[color_column]
            else:
                specific_color = color

            # Plot point
            ax.scatter(date, y_pos, s=size, c=specific_color, marker=marker, zorder=4)

            # Add label if requested
            if add_labels and label_column and label_column in event:
                label = event[label_column]
                ax.annotate(
                    label,
                    (date, y_pos),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                )

        # Add category label
        ax.text(
            min_date,
            y_pos,
            category,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="right",
        )

        # Store y position
        y_positions[category] = y_pos

        # Increment y-offset for next category
        y_offset += 1

    # Set axis limits
    ax.set_xlim(min_date, max_date)
    ax.set_ylim(-0.5, y_offset - 0.5)

    # Configure x-axis (dates)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Hide y-axis ticks
    ax.set_yticks([])

    # Add grid lines if requested
    if add_grid_lines:
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)


def create_network_graph(
    ax: Axes,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    node_size_column: Optional[str] = None,
    node_color_column: Optional[str] = None,
    node_label_column: Optional[str] = None,
    edge_width_column: Optional[str] = None,
    edge_color_column: Optional[str] = None,
    default_node_size: float = 300.0,
    default_node_color: str = "#3366CC",
    default_edge_width: float = 1.0,
    default_edge_color: str = "#999999",
    min_edge_width: float = 0.5,
    max_edge_width: float = 5.0,
    with_labels: bool = True,
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    layout: str = "spring",
) -> None:
    """
    Create a network graph visualization.

    Args:
        ax: Matplotlib axes to plot on
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        node_size_column: Optional column for determining node sizes
        node_color_column: Optional column for determining node colors
        node_label_column: Optional column for node labels
        edge_width_column: Optional column for determining edge widths
        edge_color_column: Optional column for determining edge colors
        default_node_size: Default node size
        default_node_color: Default node color
        default_edge_width: Default edge width
        default_edge_color: Default edge color
        min_edge_width: Minimum edge width
        max_edge_width: Maximum edge width
        with_labels: Whether to show node labels
        node_size_scale: Scaling factor for node sizes
        edge_width_scale: Scaling factor for edge widths
        layout: Network layout algorithm ('spring', 'circular', 'random', 'shell')
    """
    try:
        import networkx as nx
    except ImportError:
        ax.text(
            0.5,
            0.5,
            "NetworkX library is required for network graphs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Handle empty data
    if not nodes or not edges:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Create graph
    G = nx.Graph()

    # Add nodes
    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue

        node_attrs = {"name": node_id}

        # Add node attributes
        for key, value in node.items():
            if key != "id":
                node_attrs[key] = value

        G.add_node(node_id, **node_attrs)

    # Add edges
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source is None or target is None:
            continue

        edge_attrs = {}

        # Add edge attributes
        for key, value in edge.items():
            if key not in ["source", "target"]:
                edge_attrs[key] = value

        G.add_edge(source, target, **edge_attrs)

    # Determine layout
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Prepare node attributes for drawing
    node_sizes = []
    node_colors = []
    node_labels = {}

    for node_id in G.nodes():
        node_data = G.nodes[node_id]

        # Determine node size
        if node_size_column and node_size_column in node_data:
            size = node_data[node_size_column] * node_size_scale
        else:
            size = default_node_size * node_size_scale

        node_sizes.append(size)

        # Determine node color
        if node_color_column and node_color_column in node_data:
            color = node_data[node_color_column]
        else:
            color = default_node_color

        node_colors.append(color)

        # Determine node label
        if with_labels:
            if node_label_column and node_label_column in node_data:
                label = node_data[node_label_column]
            else:
                label = node_id

            node_labels[node_id] = label

    # Prepare edge attributes for drawing
    edge_widths = []
    edge_colors = []

    for u, v, edge_data in G.edges(data=True):
        # Determine edge width
        if edge_width_column and edge_width_column in edge_data:
            width = edge_data[edge_width_column] * edge_width_scale
            width = max(min_edge_width, min(width, max_edge_width))
        else:
            width = default_edge_width * edge_width_scale

        edge_widths.append(width)

        # Determine edge color
        if edge_color_column and edge_color_column in edge_data:
            color = edge_data[edge_color_column]
        else:
            color = default_edge_color

        edge_colors.append(color)

    # Draw network
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        arrowstyle="-",
        connectionstyle="arc3,rad=0.1",
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9
    )

    if with_labels:
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels=node_labels, font_size=8, font_weight="bold"
        )

    # Set axis settings
    ax.set_axis_off()
