"""
Configuration settings for the data analysis system.

This module provides the Settings class that holds all configuration
parameters for the application, including file paths and analysis options.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class Settings:
    """
    Configuration settings for the data analysis system.

    This class provides centralized configuration management for file paths,
    database connections, and analysis parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings with default values or from config file.

        Args:
            config_path: Optional path to configuration file
        """
        # Default base paths
        self.BASE_DIR = Path(__file__).parent.parent  # Project root directory
        self.INPUT_DIR = self.BASE_DIR / "input"
        self.PROCESSED_DATA_DIR = self.INPUT_DIR / "processed"
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.LOG_DIR = self.BASE_DIR / "logs"

        # Ensure directories exist
        self._create_directories()

        # File paths for data sources
        self.USER_DATA_PATH = self.INPUT_DIR / "users.json"
        self.IDEA_DATA_PATH = self.INPUT_DIR / "ideas.json"
        self.STEP_DATA_PATH = self.INPUT_DIR / "steps.json"
        self.TEAM_DATA_PATH = self.INPUT_DIR / "de_teams.json"
        self.COURSE_EVAL_DATA_PATH = self.INPUT_DIR / "course_evaluations.json"
        self.CATEGORIZED_IDEAS_PATH = self.INPUT_DIR / "categorized_ideas.json"

        # Course ID for primary analysis
        self.PRIMARY_COURSE_ID = "15.390"

        # Analysis parameters
        self.ENGAGEMENT_THRESHOLDS = {
            "HIGH": 0.7,  # Threshold for high engagement
            "MEDIUM": 0.3,  # Threshold for medium engagement
            "LOW": 0.0,  # Threshold for low engagement
        }

        # Semester date ranges
        self.SEMESTER_RANGES = {
            "Fall 2023": {"start": "2023-09-01", "end": "2023-12-31"},
            "Spring 2024": {"start": "2024-01-01", "end": "2024-05-31"},
            "Fall 2024": {"start": "2024-09-01", "end": "2024-12-31"},
            "Spring 2025": {"start": "2025-01-01", "end": "2025-05-31"},
        }

        # Session analysis parameters
        self.SESSION_TIMEOUT_MINUTES = 30  # Time before a new session is considered

        # Cohort definitions
        self.COHORT_DEFINITIONS = {
            "engagement": {
                "HIGH": {"min": 0.7, "max": 1.0},
                "MEDIUM": {"min": 0.3, "max": 0.7},
                "LOW": {"min": 0.0, "max": 0.3},
            },
            "idea_count": {
                "HIGH": {"min": 3, "max": float("inf")},
                "MEDIUM": {"min": 2, "max": 2},
                "LOW": {"min": 0, "max": 1},
            },
            "step_count": {
                "HIGH": {"min": 10, "max": float("inf")},
                "MEDIUM": {"min": 5, "max": 9},
                "LOW": {"min": 0, "max": 4},
            },
        }

        # Cache settings
        self.CACHE_ENABLED = True
        self.CACHE_MAX_SIZE = 1000  # Maximum number of items to cache

        # Load additional settings from config file if provided
        if config_path:
            self._load_from_file(config_path)

        # Override with environment variables if set
        self._load_from_env()

    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.INPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.LOG_DIR.mkdir(exist_ok=True, parents=True)

    def _load_from_file(self, config_path: str) -> None:
        """
        Load settings from a configuration file.

        Args:
            config_path: Path to configuration file
        """
        # Check if the file exists
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file not found: {config_path}")
            return

        # Load based on file extension
        if config_file.suffix.lower() == ".json":
            self._load_from_json(config_file)
        elif config_file.suffix.lower() in [".yml", ".yaml"]:
            self._load_from_yaml(config_file)
        else:
            print(f"Warning: Unsupported config file format: {config_file.suffix}")

    def _load_from_json(self, config_file: Path) -> None:
        """
        Load settings from a JSON file.

        Args:
            config_file: Path to JSON config file
        """
        import json

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Update settings with values from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Error loading JSON config: {e}")

    def _load_from_yaml(self, config_file: Path) -> None:
        """
        Load settings from a YAML file.

        Args:
            config_file: Path to YAML config file
        """
        try:
            import yaml

            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Update settings with values from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except ImportError:
            print("PyYAML not installed, skipping YAML config loading")
        except Exception as e:
            print(f"Error loading YAML config: {e}")

    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        # Define mappings from environment variable names to attributes
        env_mappings = {
            "DATA_ANALYSIS_USER_DATA": "USER_DATA_PATH",
            "DATA_ANALYSIS_IDEA_DATA": "IDEA_DATA_PATH",
            "DATA_ANALYSIS_STEP_DATA": "STEP_DATA_PATH",
            "DATA_ANALYSIS_TEAM_DATA": "TEAM_DATA_PATH",
            "DATA_ANALYSIS_COURSE_EVAL_DATA": "COURSE_EVAL_DATA_PATH",
            "DATA_ANALYSIS_CATEGORIZED_IDEAS": "CATEGORIZED_IDEAS_PATH",
            "DATA_ANALYSIS_PRIMARY_COURSE": "PRIMARY_COURSE_ID",
            "DATA_ANALYSIS_CACHE_ENABLED": "CACHE_ENABLED",
            "DATA_ANALYSIS_CACHE_MAX_SIZE": "CACHE_MAX_SIZE",
        }

        # Update settings from environment variables
        for env_name, attr_name in env_mappings.items():
            if env_name in os.environ and hasattr(self, attr_name):
                env_value = os.environ[env_name]
                attr_value = getattr(self, attr_name)

                # Convert type based on current attribute type
                if isinstance(attr_value, bool):
                    env_value = env_value.lower() in ["true", "1", "yes"]
                elif isinstance(attr_value, int):
                    env_value = int(env_value)
                elif isinstance(attr_value, float):
                    env_value = float(env_value)
                elif isinstance(attr_value, Path):
                    env_value = Path(env_value)

                setattr(self, attr_name, env_value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of settings
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result

    def save_to_file(self, file_path: str) -> None:
        """
        Save current settings to a file.

        Args:
            file_path: Path to save settings
        """
        settings_dict = self.to_dict()
        file_path = Path(file_path)

        # Save based on file extension
        if file_path.suffix.lower() == ".json":
            import json

            with open(file_path, "w") as f:
                json.dump(settings_dict, f, indent=4)
        elif file_path.suffix.lower() in [".yml", ".yaml"]:
            try:
                import yaml

                with open(file_path, "w") as f:
                    yaml.dump(settings_dict, f, default_flow_style=False)
            except ImportError:
                print("PyYAML not installed, saving as JSON instead")
                with open(file_path.with_suffix(".json"), "w") as f:
                    import json

                    json.dump(settings_dict, f, indent=4)
        else:
            print(f"Unsupported file format: {file_path.suffix}, saving as JSON")
            with open(file_path.with_suffix(".json"), "w") as f:
                import json

                json.dump(settings_dict, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Any: Setting value or default
        """
        return getattr(self, key, default)


# Default settings instance
default_settings = Settings()
