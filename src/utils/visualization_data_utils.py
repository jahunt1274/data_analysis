"""
Visualization utilities for the data analysis system.

This module provides utility functions to prepare data for visualization and manage
the export and saving of visualizations. It complements the core visualization
capabilities in visualization_creation_utils.py by handling data preprocessing
and file management tasks.

The module is organized in several functional areas:
1. Export and Saving - File naming, format support, and resolution management
2. Data Preprocessing - Transform repository data into visualization-ready formats
3. Common Data Transformations - Aggregation, pivoting, normalization, and scaling
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import json
import re

# Configure logging
logger = logging.getLogger(__name__)


# ----------------------
# Export and Saving
# ----------------------


def get_output_path(
    filename: str,
    subdirectory: Optional[str] = None,
    base_dir: Optional[str] = None,
    create_dirs: bool = True,
) -> Path:
    """
    Generate a consistent file path for output files.

    Args:
        filename: Base filename without extension
        subdirectory: Optional subdirectory within the output directory
        base_dir: Optional base directory (defaults to project output directory)
        create_dirs: Whether to create directories if they don't exist

    Returns:
        Path: Complete path object for the output file
    """
    # Default base directory is the project's output directory
    if base_dir is None:
        # Try to find the project root
        current_dir = Path.cwd()

        # Look for typical project root indicators (like .git, README, etc.)
        while current_dir != current_dir.parent:
            if (current_dir / "README.md").exists() or (current_dir / ".git").exists():
                break
            current_dir = current_dir.parent

        base_dir = current_dir / "output"
    else:
        base_dir = Path(base_dir)

    # Construct the full path
    if subdirectory:
        output_dir = base_dir / subdirectory
    else:
        output_dir = base_dir

    # Create directories if requested
    if create_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def generate_filename(
    base_name: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_timestamp: bool = False,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    sanitize: bool = True,
) -> str:
    """
    Generate a consistent filename with optional prefixes, suffixes, and timestamps.

    Args:
        base_name: Core name for the file
        prefix: Optional prefix to add
        suffix: Optional suffix to add
        include_timestamp: Whether to include a timestamp
        timestamp_format: Format string for the timestamp
        sanitize: Whether to sanitize the filename for cross-platform compatibility

    Returns:
        str: Generated filename without extension
    """
    components = []

    # Add prefix if provided
    if prefix:
        components.append(prefix)

    # Add base name
    components.append(base_name)

    # Add suffix if provided
    if suffix:
        components.append(suffix)

    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime(timestamp_format)
        components.append(timestamp)

    # Join with underscores
    filename = "_".join(components)

    # Sanitize if requested
    if sanitize:
        # Replace characters that are problematic in filenames
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        # Ensure it doesn't start with a dot (hidden file in Unix)
        if filename.startswith("."):
            filename = "_" + filename[1:]

    return filename


def save_visualization(
    fig: Any,
    filename: str,
    subdirectory: Optional[str] = None,
    formats: List[str] = ["png", "pdf", "svg"],
    dpi: Dict[str, int] = {"png": 300, "pdf": 600, "svg": 300},
    transparent: bool = False,
    include_timestamp: bool = False,
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save a visualization in multiple formats with consistent naming.

    Args:
        fig: Matplotlib figure to save
        filename: Base filename without extension
        subdirectory: Optional subdirectory within the output directory
        formats: List of formats to save in
        dpi: Dictionary of resolution settings by format
        transparent: Whether to use transparent background
        include_timestamp: Whether to add timestamp to filename
        base_dir: Optional base directory

    Returns:
        Dict[str, str]: Mapping of format to saved file path
    """
    # Generate the final filename
    final_filename = generate_filename(filename, include_timestamp=include_timestamp)

    # Get the output path
    output_path = get_output_path(
        final_filename, subdirectory=subdirectory, base_dir=base_dir
    )

    # Save in each requested format
    saved_files = {}

    for fmt in formats:
        # Get format-specific DPI
        format_dpi = dpi.get(fmt, 300)

        # Create full path with extension
        full_path = output_path.parent / f"{output_path.name}.{fmt}"

        try:
            # Save the figure
            fig.savefig(
                str(full_path),
                format=fmt,
                dpi=format_dpi,
                transparent=transparent,
                bbox_inches="tight",
            )

            # Record successful save
            saved_files[fmt] = str(full_path)
            logger.info(f"Saved visualization as {fmt} to {full_path}")

        except Exception as e:
            logger.error(f"Failed to save visualization as {fmt}: {e}")

    return saved_files


def export_data_with_visualization(
    fig: Any,
    data: Any,
    filename: str,
    subdirectory: Optional[str] = None,
    image_formats: List[str] = ["png"],
    data_format: str = "csv",
    include_timestamp: bool = False,
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Export both visualization and underlying data together.

    Args:
        fig: Matplotlib figure to save
        data: Data to export (DataFrame, dict, or array)
        filename: Base filename without extension
        subdirectory: Optional subdirectory within the output directory
        image_formats: List of image formats to save
        data_format: Format for data export ('csv', 'json', 'excel')
        include_timestamp: Whether to add timestamp to filename
        base_dir: Optional base directory

    Returns:
        Dict[str, str]: Mapping of format to saved file path
    """
    # Generate the filename
    final_filename = generate_filename(filename, include_timestamp=include_timestamp)

    # Save the visualization
    viz_files = save_visualization(
        fig,
        final_filename,
        subdirectory=subdirectory,
        formats=image_formats,
        include_timestamp=False,  # Already included in final_filename
        base_dir=base_dir,
    )

    # Get the output path for data
    output_path = get_output_path(
        final_filename, subdirectory=subdirectory, base_dir=base_dir
    )

    # Export the data
    data_path = None

    try:
        if isinstance(data, pd.DataFrame):
            # Save DataFrame
            if data_format == "csv":
                data_path = f"{output_path}.csv"
                data.to_csv(data_path, index=True)
            elif data_format == "excel":
                data_path = f"{output_path}.xlsx"
                data.to_excel(data_path, index=True)
            elif data_format == "json":
                data_path = f"{output_path}.json"
                data.to_json(data_path, orient="records", date_format="iso")

        elif isinstance(data, dict) or isinstance(data, list):
            # Save dictionary or list
            data_path = f"{output_path}.json"
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif isinstance(data, np.ndarray):
            # Save NumPy array
            data_path = f"{output_path}.npy"
            np.save(data_path, data)

        else:
            # Try to convert to DataFrame first
            try:
                df = pd.DataFrame(data)
                data_path = f"{output_path}.csv"
                df.to_csv(data_path, index=True)
            except:
                logger.warning(f"Could not export data of type {type(data)}")

        if data_path:
            logger.info(f"Exported data to {data_path}")
            viz_files["data"] = data_path

    except Exception as e:
        logger.error(f"Failed to export data: {e}")

    return viz_files


def create_report_directory(
    report_name: str,
    base_dir: Optional[str] = None,
    include_timestamp: bool = True,
) -> Tuple[Path, Dict[str, Path]]:
    """
    Create a directory structure for a report with subdirectories.

    Args:
        report_name: Name of the report
        base_dir: Optional base directory
        include_timestamp: Whether to include timestamp in directory name

    Returns:
        Tuple[Path, Dict[str, Path]]: Main directory path and subdirectory paths
    """
    # Generate the report directory name
    report_dir_name = generate_filename(
        report_name, include_timestamp=include_timestamp
    )

    # Get the base path
    if base_dir is None:
        # Default to output directory
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / "README.md").exists() or (current_dir / ".git").exists():
                break
            current_dir = current_dir.parent

        base_path = current_dir / "output" / "reports"
    else:
        base_path = Path(base_dir)

    # Create the report directory
    report_path = base_path / report_dir_name
    report_path.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    subdirs = {
        "figures": report_path / "figures",
        "data": report_path / "data",
        "tables": report_path / "tables",
    }

    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    return report_path, subdirs


# ----------------------
# Data Preprocessing
# ----------------------


def prepare_bar_chart_data(
    data: Dict[str, Any],
    category_field: Optional[str] = None,
    value_field: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    top_n: Optional[int] = None,
    include_other: bool = False,
) -> pd.DataFrame:
    """
    Transform raw data into a format suitable for bar charts.

    Args:
        data: Dictionary, list, or DataFrame of data
        category_field: Field to use as categories
        value_field: Field to use as values
        sort_by: Field to sort by (defaults to value_field)
        ascending: Whether to sort in ascending order
        top_n: Optional limit to top N categories
        include_other: Whether to aggregate remaining categories into "Other"

    Returns:
        pd.DataFrame: Processed data ready for visualization
    """
    # Convert input to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Use defaults if fields not specified
    if category_field is None and len(df.columns) > 1:
        category_field = df.columns[0]

    if value_field is None and len(df.columns) > 1:
        value_field = df.columns[1]

    # Handle case where no fields are specified
    if category_field is None or value_field is None:
        if len(df.columns) < 2:
            raise ValueError("Data must have at least two columns")
        category_field = df.columns[0]
        value_field = df.columns[1]

    # Set default sort field
    if sort_by is None:
        sort_by = value_field

    # Group by category and sum values
    result = df.groupby(category_field)[value_field].sum().reset_index()

    # Sort the data
    result = result.sort_values(by=sort_by, ascending=ascending)

    # Limit to top N categories if requested
    if top_n is not None and len(result) > top_n:
        if include_other:
            top_categories = result.head(top_n)
            other_sum = result.iloc[top_n:][value_field].sum()

            # Add "Other" category
            other_row = pd.DataFrame(
                {category_field: ["Other"], value_field: [other_sum]}
            )

            result = pd.concat([top_categories, other_row], ignore_index=True)
        else:
            result = result.head(top_n)

    return result


def prepare_time_series_data(
    data: Dict[str, Any],
    date_field: str,
    value_field: str,
    group_field: Optional[str] = None,
    date_format: str = "%Y-%m-%d",
    resample_freq: Optional[str] = None,
    fill_missing: Optional[str] = None,
    rolling_window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process data for time series visualizations.

    Args:
        data: Dictionary, list, or DataFrame of data
        date_field: Field containing date/time information
        value_field: Field containing values to plot
        group_field: Optional field to group by (for multiple series)
        date_format: Format string for parsing dates
        resample_freq: Optional frequency to resample to ('D', 'W', 'M', etc.)
        fill_missing: Method to fill missing values ('ffill', 'bfill', 'zero', None)
        rolling_window: Optional window size for rolling average

    Returns:
        pd.DataFrame: Processed time series data
    """
    # Convert input to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Ensure date field is datetime
    if df[date_field].dtype != "datetime64[ns]":
        if isinstance(df[date_field].iloc[0], str):
            df[date_field] = pd.to_datetime(df[date_field], format=date_format)
        else:
            df[date_field] = pd.to_datetime(df[date_field])

    # Set the date as index
    df = df.set_index(date_field)

    # Group by if specified
    if group_field:
        # Pivot the data to get one column per group
        pivot_df = df.pivot_table(
            index=df.index, columns=group_field, values=value_field, aggfunc="sum"
        )

        # Handle resampling if requested
        if resample_freq:
            pivot_df = pivot_df.resample(resample_freq).sum()

        # Fill missing values if requested
        if fill_missing:
            if fill_missing == "zero":
                pivot_df = pivot_df.fillna(0)
            elif fill_missing == "ffill":
                pivot_df = pivot_df.fillna(method="ffill")
            elif fill_missing == "bfill":
                pivot_df = pivot_df.fillna(method="bfill")

        # Apply rolling average if requested
        if rolling_window:
            pivot_df = pivot_df.rolling(window=rolling_window).mean()

        return pivot_df

    else:
        # Process single series
        series_df = df[[value_field]].copy()

        # Resample if requested
        if resample_freq:
            series_df = series_df.resample(resample_freq).sum()

        # Fill missing values if requested
        if fill_missing:
            if fill_missing == "zero":
                series_df = series_df.fillna(0)
            elif fill_missing == "ffill":
                series_df = series_df.fillna(method="ffill")
            elif fill_missing == "bfill":
                series_df = series_df.fillna(method="bfill")

        # Apply rolling average if requested
        if rolling_window:
            series_df = series_df.rolling(window=rolling_window).mean()

        return series_df


def prepare_scatter_plot_data(
    data: Dict[str, Any],
    x_field: str,
    y_field: str,
    category_field: Optional[str] = None,
    size_field: Optional[str] = None,
    filter_outliers: bool = False,
    percentile_range: Tuple[float, float] = (0.01, 0.99),
    add_trend: bool = False,
) -> Dict[str, Any]:
    """
    Process data for scatter plot visualizations.

    Args:
        data: Dictionary, list, or DataFrame of data
        x_field: Field to use for x-axis
        y_field: Field to use for y-axis
        category_field: Optional field for categorizing points
        size_field: Optional field for sizing points
        filter_outliers: Whether to filter outliers
        percentile_range: Percentile range to keep when filtering outliers
        add_trend: Whether to add trend line data

    Returns:
        Dict[str, Any]: Processed data with various components
    """
    # Convert input to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Filter outliers if requested
    if filter_outliers:
        for field in [x_field, y_field]:
            if pd.api.types.is_numeric_dtype(df[field]):
                lower = df[field].quantile(percentile_range[0])
                upper = df[field].quantile(percentile_range[1])
                df = df[(df[field] >= lower) & (df[field] <= upper)]

    # Prepare result structure
    result = {
        "x": df[x_field].tolist(),
        "y": df[y_field].tolist(),
        "x_label": x_field,
        "y_label": y_field,
    }

    # Add categories if specified
    if category_field and category_field in df.columns:
        result["categories"] = df[category_field].tolist()
        result["category_field"] = category_field

        # Count points per category
        category_counts = df[category_field].value_counts().to_dict()
        result["category_counts"] = category_counts

    # Add sizes if specified
    if size_field and size_field in df.columns:
        # Normalize sizes between min_size and max_size
        min_size = 20
        max_size = 200

        if pd.api.types.is_numeric_dtype(df[size_field]):
            min_val = df[size_field].min()
            max_val = df[size_field].max()

            # Avoid division by zero
            if max_val > min_val:
                normalized_sizes = min_size + (df[size_field] - min_val) * (
                    max_size - min_size
                ) / (max_val - min_val)
            else:
                normalized_sizes = [min_size] * len(df)

            result["sizes"] = normalized_sizes.tolist()
            result["size_field"] = size_field

    # Add trend data if requested
    if (
        add_trend
        and pd.api.types.is_numeric_dtype(df[x_field])
        and pd.api.types.is_numeric_dtype(df[y_field])
    ):
        # Remove NaN values
        valid_mask = ~(np.isnan(df[x_field]) | np.isnan(df[y_field]))
        valid_x = df.loc[valid_mask, x_field]
        valid_y = df.loc[valid_mask, y_field]

        if len(valid_x) > 1:
            # Calculate trend line
            coeffs = np.polyfit(valid_x, valid_y, 1)
            p = np.poly1d(coeffs)

            # Generate trend line points
            x_range = np.linspace(valid_x.min(), valid_x.max(), 100)
            trend_line = pd.DataFrame({"x": x_range, "y": p(x_range)})

            # Add to result
            result["trend"] = {
                "x": trend_line["x"].tolist(),
                "y": trend_line["y"].tolist(),
                "equation": f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}",
                "r2": np.corrcoef(valid_x, valid_y)[0, 1] ** 2,
            }

    return result


def prepare_heatmap_data(
    data: Dict[str, Any],
    row_field: str,
    col_field: str,
    value_field: str,
    aggregation: str = "mean",
    fill_value: float = 0.0,
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process data for heatmap visualizations.

    Args:
        data: Dictionary, list, or DataFrame of data
        row_field: Field to use for heatmap rows
        col_field: Field to use for heatmap columns
        value_field: Field to use for heatmap values
        aggregation: Aggregation method ('mean', 'sum', 'count', etc.)
        fill_value: Value to use for missing combinations
        normalize: Optional normalization method ('row', 'column', 'all')

    Returns:
        Dict[str, Any]: Processed data for heatmap
    """
    # Convert input to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Create pivot table
    if aggregation == "mean":
        pivot_data = df.pivot_table(
            values=value_field,
            index=row_field,
            columns=col_field,
            aggfunc=np.mean,
            fill_value=fill_value,
        )
    elif aggregation == "sum":
        pivot_data = df.pivot_table(
            values=value_field,
            index=row_field,
            columns=col_field,
            aggfunc=np.sum,
            fill_value=fill_value,
        )
    elif aggregation == "count":
        pivot_data = df.pivot_table(
            values=value_field,
            index=row_field,
            columns=col_field,
            aggfunc="count",
            fill_value=fill_value,
        )
    else:
        pivot_data = df.pivot_table(
            values=value_field,
            index=row_field,
            columns=col_field,
            aggfunc=aggregation,
            fill_value=fill_value,
        )

    # Apply normalization if requested
    if normalize == "row":
        # Normalize each row (divide by row sum)
        row_sums = pivot_data.sum(axis=1)
        normalized_data = pivot_data.div(row_sums, axis=0)
        pivot_data = normalized_data

    elif normalize == "column":
        # Normalize each column (divide by column sum)
        col_sums = pivot_data.sum(axis=0)
        normalized_data = pivot_data.div(col_sums, axis=1)
        pivot_data = normalized_data

    elif normalize == "all":
        # Normalize entire matrix (divide by total sum)
        total_sum = pivot_data.values.sum()
        if total_sum != 0:
            pivot_data = pivot_data / total_sum

    # Convert to dictionary with array data and labels
    result = {
        "values": pivot_data.values,
        "row_labels": pivot_data.index.tolist(),
        "col_labels": pivot_data.columns.tolist(),
        "aggregation": aggregation,
    }

    return result


def prepare_distribution_data(
    data: Dict[str, Any],
    value_field: str,
    group_field: Optional[str] = None,
    bins: int = 20,
    range_limit: Optional[Tuple[float, float]] = None,
    filter_outliers: bool = False,
    calculate_statistics: bool = True,
) -> Dict[str, Any]:
    """
    Process data for distribution visualizations.

    Args:
        data: Dictionary, list, or DataFrame of data
        value_field: Field containing values to analyze
        group_field: Optional field to group by
        bins: Number of bins for histogram
        range_limit: Optional (min, max) range to limit values
        filter_outliers: Whether to filter outliers
        calculate_statistics: Whether to calculate distribution statistics

    Returns:
        Dict[str, Any]: Processed distribution data
    """
    # Convert input to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Apply range limit if specified
    if range_limit:
        df = df[
            (df[value_field] >= range_limit[0]) & (df[value_field] <= range_limit[1])
        ]

    # Filter outliers if requested
    if filter_outliers:
        lower = df[value_field].quantile(0.01)
        upper = df[value_field].quantile(0.99)
        df = df[(df[value_field] >= lower) & (df[value_field] <= upper)]

    if group_field and group_field in df.columns:
        # Process grouped data
        groups = df[group_field].unique()
        result = {
            "groups": {},
            "all": {"values": df[value_field].tolist(), "field": value_field},
        }

        # Process each group
        for group in groups:
            group_data = df[df[group_field] == group][value_field]
            group_result = {"values": group_data.tolist(), "count": len(group_data)}

            # Calculate statistics if requested
            if calculate_statistics:
                group_result.update(
                    {
                        "mean": group_data.mean(),
                        "median": group_data.median(),
                        "std": group_data.std(),
                        "min": group_data.min(),
                        "max": group_data.max(),
                        "q1": group_data.quantile(0.25),
                        "q3": group_data.quantile(0.75),
                    }
                )

            result["groups"][str(group)] = group_result

        return result

    else:
        # Process single distribution
        result = {
            "values": df[value_field].tolist(),
            "field": value_field,
            "count": len(df),
        }

        # Calculate histogram bins
        hist, bin_edges = np.histogram(df[value_field], bins=bins)
        result["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": [
                (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
            ],
        }

        # Calculate statistics if requested
        if calculate_statistics:
            result.update(
                {
                    "mean": df[value_field].mean(),
                    "median": df[value_field].median(),
                    "std": df[value_field].std(),
                    "min": df[value_field].min(),
                    "max": df[value_field].max(),
                    "q1": df[value_field].quantile(0.25),
                    "q3": df[value_field].quantile(0.75),
                }
            )

        return result


# ----------------------
# Common Data Transformations
# ----------------------


def normalize_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0, 1),
) -> pd.DataFrame:
    """
    Normalize numeric columns in a DataFrame.

    Args:
        data: DataFrame to normalize
        columns: List of columns to normalize (defaults to all numeric)
        method: Normalization method ('minmax', 'zscore', 'robust')
        feature_range: Range for minmax scaling

    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df = data.copy()

    # Determine columns to normalize
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Apply normalization to each column
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == "minmax":
                # Min-max scaling
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val > min_val:
                    df[col] = feature_range[0] + (df[col] - min_val) * (
                        feature_range[1] - feature_range[0]
                    ) / (max_val - min_val)
                else:
                    df[col] = feature_range[0]

            elif method == "zscore":
                # Z-score normalization
                mean = df[col].mean()
                std = df[col].std()

                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0

            elif method == "robust":
                # Robust scaling based on median and IQR
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = 0

    return df


def aggregate_data(
    data: pd.DataFrame,
    group_by: Union[str, List[str]],
    agg_dict: Optional[Dict[str, Union[str, List[str]]]] = None,
    include_counts: bool = True,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Aggregate data by specified columns.

    Args:
        data: DataFrame to aggregate
        group_by: Column or list of columns to group by
        agg_dict: Dictionary mapping columns to aggregation functions
        include_counts: Whether to include counts in the result
        reset_index: Whether to reset index after aggregation

    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Create default aggregation dictionary if not provided
    if agg_dict is None:
        agg_dict = {}
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Exclude group_by columns from aggregation
        if isinstance(group_by, str):
            group_cols = [group_by]
        else:
            group_cols = list(group_by)

        numeric_agg_cols = [col for col in numeric_cols if col not in group_cols]

        # Add sum and mean for numeric columns
        for col in numeric_agg_cols:
            agg_dict[col] = ["sum", "mean"]

    # Perform aggregation
    grouped = df.groupby(group_by).agg(agg_dict)

    # Add count if requested
    if include_counts:
        # Count the number of rows in each group
        counts = df.groupby(group_by).size()
        grouped["count"] = counts

    # Reset index if requested
    if reset_index:
        grouped = grouped.reset_index()

    return grouped


def pivot_data(
    data: pd.DataFrame,
    index: Union[str, List[str]],
    columns: str,
    values: Optional[Union[str, List[str]]] = None,
    aggfunc: str = "mean",
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Pivot data for visualization.

    Args:
        data: DataFrame to pivot
        index: Column(s) to use as index
        columns: Column to use for pivot columns
        values: Column(s) to use for values
        aggfunc: Aggregation function to use
        fill_value: Value to fill in missing cells

    Returns:
        pd.DataFrame: Pivoted DataFrame
    """
    # Create pivot table
    pivot_df = pd.pivot_table(
        data=data,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value,
    )

    return pivot_df


def resample_time_data(
    data: pd.DataFrame,
    date_column: str,
    freq: str = "D",
    value_columns: Optional[List[str]] = None,
    method: str = "sum",
    fill_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Resample time-series data to a different frequency.

    Args:
        data: DataFrame with time data
        date_column: Column containing dates
        freq: Frequency to resample to ('D', 'W', 'M', etc.)
        value_columns: Columns to include in the result
        method: Aggregation method ('sum', 'mean', 'count', etc.)
        fill_method: Method to fill missing values (None, 'ffill', 'bfill')

    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Ensure date column is datetime
    if df[date_column].dtype != "datetime64[ns]":
        df[date_column] = pd.to_datetime(df[date_column])

    # Set date as index
    df = df.set_index(date_column)

    # Select columns if specified
    if value_columns:
        df = df[value_columns]

    # Resample data
    if method == "sum":
        resampled = df.resample(freq).sum()
    elif method == "mean":
        resampled = df.resample(freq).mean()
    elif method == "count":
        resampled = df.resample(freq).count()
    elif method == "first":
        resampled = df.resample(freq).first()
    elif method == "last":
        resampled = df.resample(freq).last()
    elif method == "max":
        resampled = df.resample(freq).max()
    elif method == "min":
        resampled = df.resample(freq).min()
    else:
        resampled = df.resample(freq).sum()

    # Fill missing values if requested
    if fill_method:
        if fill_method == "ffill":
            resampled = resampled.fillna(method="ffill")
        elif fill_method == "bfill":
            resampled = resampled.fillna(method="bfill")
        elif fill_method == "zero":
            resampled = resampled.fillna(0)

    # Reset index to make date a column again
    resampled = resampled.reset_index()

    return resampled


def filter_and_sort(
    data: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    filter_type: str = "exact",
    sort_by: Optional[Union[str, List[str]]] = None,
    ascending: Union[bool, List[bool]] = True,
    top_n: Optional[int] = None,
    random_sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter and sort data for visualization.

    Args:
        data: DataFrame to process
        filters: Dictionary of column-value pairs to filter by
        filter_type: Type of filtering ('exact', 'contains', 'range')
        sort_by: Column(s) to sort by
        ascending: Whether to sort in ascending order
        top_n: Optional limit to top N rows after sorting
        random_sample: Optional random sample size

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Apply filters if provided
    if filters:
        for col, value in filters.items():
            if col in df.columns:
                if filter_type == "exact":
                    # Exact match
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]

                elif filter_type == "contains":
                    # String contains (case-insensitive)
                    if pd.api.types.is_string_dtype(df[col]):
                        if isinstance(value, list):
                            # Any of the values
                            mask = df[col].str.contains(value[0], case=False, na=False)
                            for v in value[1:]:
                                mask |= df[col].str.contains(v, case=False, na=False)
                            df = df[mask]
                        else:
                            df = df[df[col].str.contains(value, case=False, na=False)]

                elif filter_type == "range":
                    # Value in range
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        df = df[(df[col] >= value[0]) & (df[col] <= value[1])]

    # Sort if requested
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Limit to top N if requested
    if top_n is not None:
        df = df.head(top_n)

    # Take random sample if requested
    if random_sample is not None:
        if random_sample < len(df):
            df = df.sample(random_sample)

    return df


def transform_for_network_graph(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str] = None,
    node_attr_cols: Optional[List[str]] = None,
    edge_attr_cols: Optional[List[str]] = None,
    min_weight: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform relational data into a format suitable for network graphs.

    Args:
        data: DataFrame with relational data
        source_col: Column with source node IDs
        target_col: Column with target node IDs
        weight_col: Optional column with edge weights
        node_attr_cols: Optional columns to include as node attributes
        edge_attr_cols: Optional columns to include as edge attributes
        min_weight: Optional minimum weight threshold

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with nodes and edges lists
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Initialize result structure
    result = {"nodes": [], "edges": []}

    # Extract unique nodes
    all_nodes = pd.concat([df[source_col], df[target_col]]).unique()

    # Create nodes list
    nodes = []
    for node_id in all_nodes:
        node = {"id": node_id}

        # Add node attributes if specified
        if node_attr_cols:
            # Get rows where this node appears as source or target
            source_rows = df[df[source_col] == node_id]
            target_rows = df[df[target_col] == node_id]

            # Use first occurrence for attributes
            for col in node_attr_cols:
                if col in df.columns and col not in [source_col, target_col]:
                    if not source_rows.empty and not pd.isna(source_rows[col].iloc[0]):
                        node[col] = source_rows[col].iloc[0]
                    elif not target_rows.empty and not pd.isna(
                        target_rows[col].iloc[0]
                    ):
                        node[col] = target_rows[col].iloc[0]

        nodes.append(node)

    # Apply weight filtering if specified
    if min_weight is not None and weight_col in df.columns:
        df = df[df[weight_col] >= min_weight]

    # Create edges list
    edges = []
    for _, row in df.iterrows():
        edge = {"source": row[source_col], "target": row[target_col]}

        # Add weight if specified
        if weight_col and weight_col in df.columns:
            edge["weight"] = row[weight_col]

        # Add edge attributes if specified
        if edge_attr_cols:
            for col in edge_attr_cols:
                if col in df.columns and col not in [
                    source_col,
                    target_col,
                    weight_col,
                ]:
                    edge[col] = row[col]

        edges.append(edge)

    result["nodes"] = nodes
    result["edges"] = edges

    return result


def transform_to_sankey_format(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    value_col: Optional[str] = None,
    source_order: Optional[List[str]] = None,
    target_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Transform data into format suitable for Sankey diagrams.

    Args:
        data: DataFrame with flow data
        source_col: Column with source node labels
        target_col: Column with target node labels
        value_col: Column with flow values (defaults to count)
        source_order: Optional custom ordering of source nodes
        target_order: Optional custom ordering of target nodes

    Returns:
        Dict[str, Any]: Data formatted for Sankey diagram
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Aggregate flows if value_col not specified
    if value_col is None:
        # Count occurrences of each source-target pair
        flow_values = (
            df.groupby([source_col, target_col]).size().reset_index(name="value")
        )
    else:
        # Sum values for each source-target pair
        flow_values = (
            df.groupby([source_col, target_col])[value_col].sum().reset_index()
        )
        flow_values = flow_values.rename(columns={value_col: "value"})

    # Get unique node labels
    sources = df[source_col].unique().tolist()
    targets = df[target_col].unique().tolist()
    all_nodes = list(set(sources + targets))

    # Apply custom ordering if specified
    if source_order:
        ordered_sources = [s for s in source_order if s in sources]
        # Add any sources not in the order list
        for s in sources:
            if s not in ordered_sources:
                ordered_sources.append(s)
        sources = ordered_sources

    if target_order:
        ordered_targets = [t for t in target_order if t in targets]
        # Add any targets not in the order list
        for t in targets:
            if t not in ordered_targets:
                ordered_targets.append(t)
        targets = ordered_targets

    # Create node indices dictionary
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    # Create nodes list with labels and groups
    nodes = []
    for node in all_nodes:
        node_data = {
            "id": node_indices[node],
            "name": node,
            "group": 0,  # Default group
        }

        # Set group: 1 for source-only, 2 for target-only, 3 for both
        if node in sources and node in targets:
            node_data["group"] = 3
        elif node in sources:
            node_data["group"] = 1
        elif node in targets:
            node_data["group"] = 2

        nodes.append(node_data)

    # Create links
    links = []
    for _, row in flow_values.iterrows():
        source = row[source_col]
        target = row[target_col]
        value = row["value"]

        # Skip links with zero or NaN values
        if pd.isna(value) or value == 0:
            continue

        link = {
            "source": node_indices[source],
            "target": node_indices[target],
            "value": value,
        }

        links.append(link)

    # Return formatted data
    return {"nodes": nodes, "links": links}


def transform_to_treemap_format(
    data: pd.DataFrame,
    hierarchy: List[str],
    value_col: str,
    color_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transform data into a hierarchical structure for treemap visualizations.

    Args:
        data: DataFrame with hierarchical data
        hierarchy: List of columns defining the hierarchy levels
        value_col: Column containing the values for sizing
        color_col: Optional column for determining node colors

    Returns:
        Dict[str, Any]: Hierarchical data structure for treemap
    """
    # Copy data to avoid modifying the original
    df = data.copy()

    # Function to recursively build the hierarchy
    def build_tree(df, hierarchy, level=0):
        if level >= len(hierarchy) or df.empty:
            return []

        # Current level column
        col = hierarchy[level]

        # Group by the current level
        grouped = df.groupby(col)

        # Build tree nodes
        result = []
        for name, group in grouped:
            # Skip NaN groups
            if pd.isna(name):
                continue

            # Create node
            node = {"name": str(name)}

            # Add value for leaf nodes or aggregated value for branch nodes
            if level == len(hierarchy) - 1:
                # Leaf node
                node["value"] = group[value_col].sum()

                # Add color information if specified
                if color_col and color_col in df.columns:
                    # Use average for numeric, most common for categorical
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        node["color"] = group[color_col].mean()
                    else:
                        node["color"] = group[color_col].mode()[0]
            else:
                # Branch node - calculate own value and add children
                node["value"] = group[value_col].sum()
                children = build_tree(group, hierarchy, level + 1)

                if children:
                    node["children"] = children

                # Add color information if specified
                if color_col and color_col in df.columns:
                    # Use average for numeric, most common for categorical
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        node["color"] = group[color_col].mean()
                    else:
                        if not group[color_col].empty and not all(
                            pd.isna(group[color_col])
                        ):
                            node["color"] = group[color_col].mode()[0]

            result.append(node)

        return result

    # Build the hierarchical tree
    tree = {"name": "root", "children": build_tree(df, hierarchy)}

    return tree
