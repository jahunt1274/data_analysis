"""
FileManager for the data analysis system.

This module provides a centralized file management system for creating,
saving, and organizing files and directories. It ensures consistent file
naming and directory structures across the codebase.
"""

import os
import logging
import json
import pickle
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from uuid import uuid4


class FileManager:
    """
    Centralized file management system for the data analysis project.

    This class handles all file and directory operations, ensuring consistent
    naming, organization, and structure throughout the application.
    """

    def __init__(self, base_dir: Optional[str] = None, use_timestamps: bool = False):
        """
        Initialize the FileManager with a base directory.

        Args:
            base_dir: Base directory for all outputs. If None, uses the project root/output.
            use_timestamps: Whether to use timestamps in filenames (legacy mode).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_timestamps = use_timestamps

        # Determine base directory if not provided
        if base_dir is None:
            # Find project root by looking for certain indicators
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / "README.md").exists() or (
                    current_dir / ".git"
                ).exists():
                    break
                current_dir = current_dir.parent

            # Set base output directory to project_root/output
            self.base_dir = current_dir / "output"
        else:
            self.base_dir = Path(base_dir)

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Define standard subdirectories
        self.structure = {
            "analysis": {
                "engagement": {},
                "framework": {},
                "learning": {},
                "team": {},
                "combined": {},
            },
            "visualizations": {
                "engagement": {},
                "framework": {},
                "learning": {},
                "team": {},
                "combined": {},
            },
            "reports": {},
            "data": {
                "processed": {},
                "exported": {},
            },
            "logs": {},
            "temp": {},
        }

        # Create the directory structure
        self._create_directory_structure()

        # Initialize a session ID for grouping related outputs
        self.session_id = self._generate_session_id()

        self.logger.info(
            f"FileManager initialized with base directory: {self.base_dir}"
        )
        self.logger.info(f"Session ID: {self.session_id}")

    def _create_directory_structure(self) -> None:
        """Create the standard directory structure if it doesn't exist."""

        def create_nested_dirs(parent_path: Path, structure: Dict) -> None:
            for name, substructure in structure.items():
                dir_path = parent_path / name
                dir_path.mkdir(exist_ok=True)

                if substructure:  # If there are subdirectories
                    create_nested_dirs(dir_path, substructure)

        create_nested_dirs(self.base_dir, self.structure)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for grouping files."""
        # Use timestamp and a short UUID for uniqueness but readability
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        short_uuid = str(uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    def get_path(
        self,
        category: str,
        subcategory: Optional[str] = None,
        filename: Optional[str] = None,
        create_dirs: bool = True,
    ) -> Path:
        """
        Get a standardized path within the directory structure.

        Args:
            category: Top-level category (analysis, visualizations, reports, data, logs)
            subcategory: Optional subcategory (engagement, framework, etc.)
            filename: Optional filename to append to the path
            create_dirs: Whether to create directories if they don't exist

        Returns:
            Path: Constructed path
        """
        if category not in self.structure:
            self.logger.warning(f"Unknown category: {category}, using 'temp' instead")
            category = "temp"

        path = self.base_dir / category

        # Add subcategory if provided and valid
        if subcategory:
            if (
                category in ["analysis", "visualizations"]
                and subcategory in self.structure[category]
            ):
                path = path / subcategory
            else:
                self.logger.warning(
                    f"Unknown subcategory: {subcategory} for {category}"
                )

        # Create directories if needed
        if create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        # Add filename if provided
        if filename:
            path = path / filename

        return path

    def generate_filename(
        self,
        base_name: str,
        category: str,
        extension: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        session_specific: bool = True,
        version: Optional[int] = None,
    ) -> str:
        """
        Generate a standardized filename.

        Args:
            base_name: Core name for the file
            category: Category for context-specific naming conventions
            extension: File extension (without the dot)
            prefix: Optional prefix to add
            suffix: Optional suffix to add
            session_specific: Whether to make the file specific to this session
            version: Optional version number to append

        Returns:
            str: Generated filename
        """
        components = []

        # Add prefix if provided
        if prefix:
            components.append(prefix)

        # Add base name (sanitized)
        base_name = self._sanitize_filename(base_name)
        components.append(base_name)

        # Add suffix if provided
        if suffix:
            components.append(suffix)

        # Add session ID if requested (instead of timestamp)
        if session_specific:
            # Use a short version of the session ID for filenames
            short_session = self.session_id.split("_")[0]  # Just the timestamp part
            components.append(short_session)
        elif self.use_timestamps:
            # Legacy mode: use full timestamps in filenames
            components.append(datetime.now().strftime("%Y%m%d%H%M%S"))

        # Add version if provided
        if version is not None:
            components.append(f"v{version}")

        # Join with underscores
        filename = "_".join(components)

        # Add extension if provided
        if extension:
            if not extension.startswith("."):
                extension = f".{extension}"
            filename += extension

        return filename

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to be safe across different operating systems.

        Args:
            filename: Original filename

        Returns:
            str: Sanitized filename
        """
        # Replace problematic characters
        filename = filename.replace(" ", "_")
        for char in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "%"]:
            filename = filename.replace(char, "_")

        # Ensure it doesn't start with a dot (hidden file in Unix)
        if filename.startswith("."):
            filename = "_" + filename[1:]

        return filename

    def save_file(
        self,
        data: Any,
        filename: str,
        category: str,
        subcategory: Optional[str] = None,
        file_format: str = "auto",
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save data to a file with standardized naming and organization.

        Args:
            data: Data to save
            filename: Base filename (without extension)
            category: Category for directory structure
            subcategory: Optional subcategory
            file_format: Format to save in ("json", "csv", "pickle", "figure", "auto")
            overwrite: Whether to overwrite if file exists
            metadata: Optional metadata to include with the file

        Returns:
            Path: Path to the saved file
        """
        # Determine extension based on format
        extension = self._get_extension_for_format(file_format, data)

        # Generate complete filename
        full_filename = self.generate_filename(
            base_name=filename, category=category, extension=extension
        )

        # Get the full path
        file_path = self.get_path(
            category=category,
            subcategory=subcategory,
            filename=full_filename,
            create_dirs=True,
        )

        # Check if file exists
        if file_path.exists() and not overwrite:
            # Find a unique filename by adding a version number
            base_path = file_path.with_suffix("")
            version = 1
            while file_path.exists():
                version_suffix = f"_v{version}"
                file_path = base_path.with_name(
                    f"{base_path.name}{version_suffix}{file_path.suffix}"
                )
                version += 1

        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix(".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        # Save the file based on format
        try:
            self._save_data_to_file(data, file_path, file_format)
            self.logger.info(f"File saved successfully: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving file {file_path}: {e}")
            raise

    def _get_extension_for_format(self, file_format: str, data: Any) -> str:
        """Determine file extension based on format and data type."""
        if file_format == "auto":
            # Auto-detect format based on data type
            if isinstance(data, plt.Figure):
                return ".png"
            elif isinstance(data, (dict, list)):
                return ".json"
            elif hasattr(data, "to_csv"):  # DataFrame-like
                return ".csv"
            else:
                return ".pkl"

        # Map formats to extensions
        format_extensions = {
            "json": ".json",
            "csv": ".csv",
            "pickle": ".pkl",
            "figure": ".png",
            "pdf": ".pdf",
            "svg": ".svg",
            "excel": ".xlsx",
            "txt": ".txt",
            "markdown": ".md",
            "html": ".html",
        }

        return format_extensions.get(file_format, f".{file_format}")

    def _save_data_to_file(self, data: Any, file_path: Path, file_format: str) -> None:
        """Save data to file based on its format."""
        # Handle different data types and formats
        if isinstance(data, plt.Figure):
            # For matplotlib figures
            data.savefig(
                file_path,
                bbox_inches="tight",
                dpi=300,
                format=file_path.suffix[1:],  # Remove the dot
            )

        elif file_format in ["json", "auto"] and isinstance(data, (dict, list)):
            # For JSON-serializable data
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif file_format == "csv" or (
            file_format == "auto" and hasattr(data, "to_csv")
        ):
            # For DataFrame-like objects
            if hasattr(data, "to_csv"):
                data.to_csv(file_path, index=True)
            else:
                # For list of lists or list of dicts
                with open(file_path, "w", newline="") as f:
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        # List of dictionaries
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        # List of lists
                        writer = csv.writer(f)
                        writer.writerows(data)

        elif file_format == "pickle" or file_format == "pkl":
            # For arbitrary Python objects
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

        elif file_format == "txt" or file_format == "text":
            # For plain text
            with open(file_path, "w") as f:
                f.write(str(data))

        else:
            # Default case
            with open(file_path, "wb" if isinstance(data, bytes) else "w") as f:
                f.write(data)

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        category: str = "visualizations",
        subcategory: Optional[str] = None,
        formats: List[str] = ["png", "pdf"],
        dpi: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Save a matplotlib figure in multiple formats.

        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            category: Category for directory structure
            subcategory: Optional subcategory
            formats: List of formats to save in
            dpi: Resolution for raster formats
            metadata: Optional metadata to include

        Returns:
            Dict[str, Path]: Dictionary mapping formats to saved file paths
        """
        saved_files = {}

        for fmt in formats:
            # Save in each format
            file_path = self.save_file(
                data=fig,
                filename=filename,
                category=category,
                subcategory=subcategory,
                file_format=fmt,
                metadata=metadata,
            )

            saved_files[fmt] = file_path

        return saved_files

    def create_report_directory(
        self, report_name: str, category: str = "reports", with_subdirs: bool = True
    ) -> Tuple[Path, Dict[str, Path]]:
        """
        Create a directory structure for a report.

        Args:
            report_name: Name of the report
            category: Category for directory structure
            with_subdirs: Whether to create standard subdirectories

        Returns:
            Tuple[Path, Dict[str, Path]]: Main directory path and subdirectory paths
        """
        # Generate report directory name
        report_dir_name = self.generate_filename(
            base_name=report_name, category=category, extension=None
        )

        # Get the full path
        report_path = self.get_path(
            category=category, filename=report_dir_name, create_dirs=True
        )

        subdirs = {}

        # Create standard subdirectories if requested
        if with_subdirs:
            for subdir_name in ["figures", "data", "tables"]:
                subdir_path = report_path / subdir_name
                subdir_path.mkdir(exist_ok=True)
                subdirs[subdir_name] = subdir_path

        return report_path, subdirs

    def get_unique_filename(
        self,
        base_name: str,
        category: str,
        subcategory: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> str:
        """
        Get a filename that is guaranteed to be unique in the target directory.

        Args:
            base_name: Core name for the file
            category: Category for directory structure
            subcategory: Optional subcategory
            extension: File extension (without the dot)

        Returns:
            str: Unique filename
        """
        # Generate initial filename
        filename = self.generate_filename(
            base_name=base_name, category=category, extension=extension
        )

        # Get the target directory
        target_dir = self.get_path(
            category=category, subcategory=subcategory, create_dirs=True
        )

        # Check if file exists
        file_path = target_dir / filename

        if not file_path.exists():
            return filename

        # Find a unique name by adding a version number
        base_filename = file_path.stem
        version = 1

        while True:
            versioned_name = f"{base_filename}_v{version}"

            if extension:
                versioned_name += f".{extension}"

            test_path = target_dir / versioned_name

            if not test_path.exists():
                return versioned_name

            version += 1

    def ensure_directory_exists(self, directory_path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory_path: Path to the directory

        Returns:
            Path: Path to the directory
        """
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_files(
        self,
        category: str,
        subcategory: Optional[str] = None,
        extension: Optional[str] = None,
        session_only: bool = False,
    ) -> List[Path]:
        """
        List files in a specific directory.

        Args:
            category: Category for directory structure
            subcategory: Optional subcategory
            extension: Optional file extension to filter by
            session_only: Whether to only include files from current session

        Returns:
            List[Path]: List of file paths
        """
        # Get the directory path
        dir_path = self.get_path(
            category=category, subcategory=subcategory, create_dirs=False
        )

        # Check if directory exists
        if not dir_path.exists():
            return []

        # List files
        files = (
            list(dir_path.iterdir())
            if extension is None
            else list(dir_path.glob(f"*.{extension}"))
        )

        # Filter files by session if requested
        if session_only:
            short_session = self.session_id.split("_")[0]
            files = [f for f in files if short_session in f.name]

        return sorted(files)

    def clear_directory(
        self,
        category: str,
        subcategory: Optional[str] = None,
        only_session: bool = True,
    ) -> None:
        """
        Clear files from a directory.

        Args:
            category: Category for directory structure
            subcategory: Optional subcategory
            only_session: Whether to only clear files from current session
        """
        # Get list of files
        files = self.list_files(
            category=category, subcategory=subcategory, session_only=only_session
        )

        # Delete files
        for file_path in files:
            if file_path.is_file():
                file_path.unlink()
                self.logger.debug(f"Deleted file: {file_path}")

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id

    def new_session(self) -> str:
        """
        Start a new session with a new ID.

        Returns:
            str: New session ID
        """
        self.session_id = self._generate_session_id()
        self.logger.info(f"Started new session: {self.session_id}")
        return self.session_id
