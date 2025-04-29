#!/usr/bin/env python3
"""
JetPack/Orbit Data Analysis System

This script provides a comprehensive command-line interface for analyzing
and visualizing data from the JetPack/Orbit entrepreneurship education tool.
It serves as the main entry point for the analysis system, coordinating between
data repositories, analyzers, and visualization components.

Author: [Your Name]
Date: April 2025
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

# Import configuration
from config.settings import Settings

# Import coordinating modules
from src.data.data_repository import DataRepository
from src.analyzers.analyzer_manager import AnalyzerManager
from src.visualizers.visualization_manager import VisualizationManager

# Import model enums for CLI arguments
from src.data.models.enums import FrameworkType, Semester, MetricType


class AnalysisApp:
    """
    Main application class for the JetPack/Orbit data analysis system.

    This class coordinates the entire application workflow, including:
    - Parsing command line arguments
    - Setting up logging
    - Initializing data repositories and analyzers
    - Running analyses
    - Generating visualizations and reports
    """

    def __init__(self):
        """Initialize the application."""
        # Setup initial attributes
        self.args = None
        self.settings = None
        self.logger = None
        self.data_repository = None
        self.analyzer_manager = None
        self.visualization_manager = None

        # Define analysis types
        self.analysis_types = {
            "user": self._analyze_user,
            "team": self._analyze_team,
            "semester": self._analyze_semester,
            "framework": self._analyze_framework,
            "data-summary": self._generate_data_summary,
            "engagement": self._analyze_engagement,
            "learning": self._analyze_learning,
        }

        # Define report types
        self.report_types = {
            "comprehensive": self._generate_comprehensive_report,
            "engagement": self._generate_engagement_report,
            "framework": self._generate_framework_report,
            "learning": self._generate_learning_report,
            "team": self._generate_team_report,
            "summary": self._generate_summary_report,
        }

    def run(self):
        """
        Run the application.

        This is the main entry point that orchestrates the entire workflow.

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        try:
            # Parse command line arguments
            self._parse_arguments()

            # Setup logging
            self._setup_logging()

            # Load configuration
            self._load_configuration()

            # Initialize components
            if not self._initialize_components():
                return 1

            # Execute requested analysis or report
            return self._execute_requested_operation()

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Unhandled exception: {e}")
            else:
                print(f"ERROR: {e}")
            return 1

    def _parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="JetPack/Orbit Data Analysis System",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Configuration options
        config_group = parser.add_argument_group("Configuration Options")
        config_group.add_argument(
            "--config", type=str, help="Path to configuration file"
        )
        config_group.add_argument(
            "--data-dir", type=str, help="Directory containing raw data files"
        )
        config_group.add_argument("--log-dir", type=str, help="Directory for log files")
        config_group.add_argument(
            "--output-dir", type=str, help="Output directory for results"
        )

        # Analysis selection options
        analysis_group = parser.add_argument_group("Analysis Selection")

        # Create a mutually exclusive group for analysis types
        analysis_type = analysis_group.add_mutually_exclusive_group()
        analysis_type.add_argument(
            "--user", metavar="EMAIL", help="Analyze specific user by email"
        )
        analysis_type.add_argument(
            "--team", metavar="ID", type=int, help="Analyze specific team by ID"
        )
        analysis_type.add_argument(
            "--data-summary", action="store_true", help="Generate data summary only"
        )
        analysis_type.add_argument(
            "--engagement", action="store_true", help="Run engagement analysis"
        )
        analysis_type.add_argument(
            "--learning", action="store_true", help="Run learning outcomes analysis"
        )

        # Create a mutually exclusive group for report types
        report_type = analysis_group.add_mutually_exclusive_group()
        report_type.add_argument(
            "--report",
            choices=[
                "comprehensive",
                "engagement",
                "framework",
                "learning",
                "team",
                "summary",
            ],
            help="Generate specified report type",
        )

        # Analysis parameters
        params_group = parser.add_argument_group("Analysis Parameters")
        params_group.add_argument(
            "--course", metavar="ID", default="15.390", help="Course ID to filter data"
        )
        params_group.add_argument(
            "--framework",
            default="Disciplined Entrepreneurship",
            choices=[ft.value for ft in FrameworkType],
            help="Framework to analyze",
        )
        params_group.add_argument(
            "--semester",
            choices=[s.value for s in Semester],
            help="Semester to analyze",
        )

        # Report options
        report_group = parser.add_argument_group("Report Options")
        report_group.add_argument(
            "--include-user-details",
            action="store_true",
            help="Include detailed user information in reports",
        )
        report_group.add_argument(
            "--include-data-tables",
            action="store_true",
            help="Include data tables in reports",
        )

        # Visualization options
        viz_group = parser.add_argument_group("Visualization Options")
        viz_group.add_argument(
            "--theme",
            default="default",
            choices=["default", "dark", "print"],
            help="Theme for visualizations",
        )
        viz_group.add_argument(
            "--formats",
            default="png,pdf",
            help="Comma-separated list of output formats (png, pdf, svg, html)",
        )

        # System options
        sys_group = parser.add_argument_group("System Options")
        sys_group.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        sys_group.add_argument(
            "--no-cache",
            action="store_true",
            help="Disable caching of analysis results",
        )

        self.args = parser.parse_args()

    def _setup_logging(self):
        """Configure logging for the application."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO

        # Determine log directory
        log_dir = self.args.log_dir or "./logs"

        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"data_analysis_{timestamp}.log"

        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Configure console handler with a simpler format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Get logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Logging initialized")

    def _load_configuration(self):
        """Load configuration settings."""
        self.logger.info("Loading configuration settings")

        # Initialize settings with optional config file
        self.settings = Settings(config_path=self.args.config)

        # Override settings from command line arguments if provided
        if self.args.data_dir:
            self.logger.info(
                f"Using data directory from arguments: {self.args.data_dir}"
            )
            # Here you might want to update settings if needed

        if self.args.output_dir:
            self.logger.info(
                f"Using output directory from arguments: {self.args.output_dir}"
            )
            # Here you might want to update settings

        self.logger.info("Configuration loaded successfully")

    def _initialize_components(self) -> bool:
        """
        Initialize the core components of the system.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Initialize data repository
            self.logger.info("Initializing data repository")
            self.data_repository = DataRepository(self.settings)

            # Load data
            if self.args.data_dir:
                result = self.data_repository.load_data_from_directory(
                    self.args.data_dir
                )
                for filename, count in result.items():
                    self.logger.info(f"Loaded {count} records from {filename}")
            else:
                self.logger.info("Connecting to default data sources")
                self.data_repository.connect()

            # Verify data loaded successfully by getting a summary
            data_summary = self.data_repository.get_data_summary()
            if "error" in data_summary:
                self.logger.error(f"Error in data repository: {data_summary['error']}")
                return False

            # Log data summary
            user_count = data_summary.get("users", {}).get("count", 0)
            idea_count = data_summary.get("ideas", {}).get("count", 0)
            step_count = data_summary.get("steps", {}).get("count", 0)
            team_count = data_summary.get("teams", {}).get("count", 0)

            self.logger.info(
                f"Data loaded: {user_count} users, {idea_count} ideas, "
                f"{step_count} steps, {team_count} teams"
            )

            # Initialize analyzer manager
            self.logger.info("Initializing analyzer manager")
            self.analyzer_manager = AnalyzerManager(self.data_repository)

            # Set cache settings
            if self.args.no_cache:
                self.analyzer_manager.enable_cache(False)
                self.logger.info("Analysis result caching disabled")

            # Initialize analyzers
            if not self.analyzer_manager.initialize_analyzers():
                self.logger.error("Failed to initialize analyzers")
                return False

            # Initialize visualization manager
            self.logger.info(
                f"Initializing visualization manager with theme: {self.args.theme}"
            )
            output_dir = self.args.output_dir or self.settings.OUTPUT_DIR

            self.visualization_manager = VisualizationManager(
                settings=self.settings,
                output_dir=output_dir,
                include_timestamps=True,
                theme=self.args.theme,
                save_formats=self.args.formats.split(","),
            )

            # Set analyzers in visualization manager
            (
                engagement_analyzer,
                framework_analyzer,
                learning_analyzer,
                team_analyzer,
            ) = self.analyzer_manager.get_all_analyzers()

            self.visualization_manager.set_analyzers(
                engagement_analyzer=engagement_analyzer,
                framework_analyzer=framework_analyzer,
                learning_analyzer=learning_analyzer,
                team_analyzer=team_analyzer,
            )

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error initializing components: {e}")
            return False

    def _execute_requested_operation(self) -> int:
        """
        Execute the requested analysis or report generation.

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        # Determine the requested operation type
        operation_type, operation_params = self._get_requested_operation()

        if not operation_type:
            self.logger.error("No analysis or report type specified")
            return 1

        self.logger.info(f"Executing {operation_type} operation")

        try:
            # Execute the requested operation
            result = operation_params["func"](**operation_params.get("params", {}))

            # Check result
            if not result:
                self.logger.error(f"{operation_type} operation failed")
                return 1

            self.logger.info(f"{operation_type} operation completed successfully")
            return 0

        except Exception as e:
            self.logger.exception(f"Error executing {operation_type} operation: {e}")
            return 1

    def _get_requested_operation(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Determine the requested operation from command line arguments.

        Returns:
            Tuple[Optional[str], Dict[str, Any]]: Operation type and parameters
        """
        # Check for analysis operations
        if self.args.data_summary:
            return "data-summary", {
                "func": self._generate_data_summary,
                "params": {"output_dir": self.args.output_dir or "./output"},
            }

        if self.args.user:
            return "user-analysis", {
                "func": self._analyze_user,
                "params": {"email": self.args.user},
            }

        if self.args.team is not None:
            return "team-analysis", {
                "func": self._analyze_team,
                "params": {"team_id": self.args.team},
            }

        if self.args.engagement:
            return "engagement-analysis", {
                "func": self._analyze_engagement,
                "params": {"course_id": self.args.course},
            }

        if self.args.learning:
            return "learning-analysis", {"func": self._analyze_learning, "params": {}}

        # Check for report operations
        if self.args.report:
            report_func = self.report_types.get(self.args.report)
            if not report_func:
                return None, {}

            return f"{self.args.report}-report", {
                "func": report_func,
                "params": {
                    "course_id": self.args.course,
                    "framework": self._get_framework_type(),
                    "include_user_details": self.args.include_user_details,
                    "include_data_tables": self.args.include_data_tables,
                },
            }

        # Check if semester analysis is requested
        if self.args.semester:
            return "semester-analysis", {
                "func": self._analyze_semester,
                "params": {"semester": self.args.semester},
            }

        # If a framework other than the default is specified, run framework analysis
        if self.args.framework != "Disciplined Entrepreneurship":
            return "framework-analysis", {
                "func": self._analyze_framework,
                "params": {
                    "framework": self._get_framework_type(),
                    "course_id": self.args.course,
                },
            }

        # If no specific operation is requested, default to comprehensive report
        return "comprehensive-report", {
            "func": self._generate_comprehensive_report,
            "params": {
                "course_id": self.args.course,
                "framework": self._get_framework_type(),
                "include_user_details": self.args.include_user_details,
                "include_data_tables": self.args.include_data_tables,
            },
        }

    def _get_framework_type(self) -> FrameworkType:
        """
        Get the FrameworkType enum value from the framework argument.

        Returns:
            FrameworkType: The framework type enum value
        """
        return next(
            (ft for ft in FrameworkType if ft.value == self.args.framework),
            FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        )

    # === Analysis Methods ===

    def _analyze_user(self, email: str) -> bool:
        """
        Analyze a specific user.

        Args:
            email: User's email address

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info(f"Analyzing user: {email}")

        try:
            # Get output directory
            output_dir = self.args.output_dir or "./output/users"

            # Run user analysis
            result = self.analyzer_manager.run_specific_analysis(
                analysis_type="user",
                output_dir=output_dir,
                user_email=email,
                include_ideas=True,
                include_steps=True,
            )

            if "error" in result:
                self.logger.error(f"User analysis error: {result['error']}")
                return False

            if "output_file" in result:
                self.logger.info(
                    f"User analysis results saved to: {result['output_file']}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing user {email}: {e}")
            return False

    def _analyze_team(self, team_id: int) -> bool:
        """
        Analyze a specific team.

        Args:
            team_id: Team ID

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info(f"Analyzing team: {team_id}")

        try:
            # Get output directory
            output_dir = self.args.output_dir or "./output/teams"

            # Run team analysis
            result = self.analyzer_manager.run_specific_analysis(
                analysis_type="team",
                output_dir=output_dir,
                team_id=team_id,
                course_id=self.args.course,
                include_temporal=True,
            )

            if "error" in result:
                self.logger.error(f"Team analysis error: {result['error']}")
                return False

            if "output_file" in result:
                self.logger.info(
                    f"Team analysis results saved to: {result['output_file']}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing team {team_id}: {e}")
            return False

    def _analyze_semester(self, semester: str) -> bool:
        """
        Analyze a specific semester.

        Args:
            semester: Semester name

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info(f"Analyzing semester: {semester}")

        try:
            # Get output directory
            output_dir = self.args.output_dir or "./output/semesters"

            # Find a different semester to compare with
            semester_order = ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]
            current_index = semester_order.index(semester)

            # Try to get the previous semester, or the next one if current is the first
            comparison_semester = (
                semester_order[current_index - 1]
                if current_index > 0
                else semester_order[current_index + 1]
            )

            # Run semester analysis
            result = self.analyzer_manager.run_specific_analysis(
                analysis_type="semester",
                output_dir=output_dir,
                semester=semester,
                comparison_semester=comparison_semester,
            )

            if "error" in result:
                self.logger.error(f"Semester analysis error: {result['error']}")
                return False

            if "output_file" in result:
                self.logger.info(
                    f"Semester comparison results saved to: {result['output_file']}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing semester {semester}: {e}")
            return False

    def _analyze_framework(
        self, framework: FrameworkType, course_id: Optional[str] = None
    ) -> bool:
        """
        Analyze a specific framework.

        Args:
            framework: Framework type
            course_id: Optional course ID to filter data

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info(f"Analyzing framework: {framework.value}")

        try:
            # Get output directory
            output_dir = self.args.output_dir or "./output/frameworks"

            # Run framework analysis
            result = self.analyzer_manager.run_specific_analysis(
                analysis_type="framework",
                output_dir=output_dir,
                framework=framework,
                course_id=course_id,
                include_ideas_without_steps=False,
            )

            if "error" in result:
                self.logger.error(f"Framework analysis error: {result['error']}")
                return False

            if "output_file" in result:
                self.logger.info(
                    f"Framework analysis results saved to: {result['output_file']}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing framework {framework.value}: {e}")
            return False

    def _analyze_engagement(self, course_id: Optional[str] = None) -> bool:
        """
        Analyze user engagement.

        Args:
            course_id: Optional course ID to filter data

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info(f"Analyzing user engagement for course: {course_id}")

        try:
            # Get engagement levels
            engagement_data = self.analyzer_manager.analyze_user_engagement(
                course_id=course_id
            )

            # Analyze dropout patterns
            dropout_data = self.analyzer_manager.analyze_dropout_patterns(
                course_id=course_id
            )

            # Create visualizations
            self.visualization_manager.visualize_engagement_levels(
                course_id=course_id, include_demographics=True
            )

            self.visualization_manager.visualize_dropout_analysis(
                dropout_data=dropout_data
            )

            # Get output directory
            output_dir = self.args.output_dir or "./output/engagement"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save analysis results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(output_dir) / f"engagement_analysis_{timestamp}.json"

            # Combine results
            combined_results = {
                "engagement_levels": engagement_data,
                "dropout_analysis": dropout_data,
            }

            with open(output_file, "w") as f:
                json.dump(combined_results, f, indent=2, default=str)

            self.logger.info(f"Engagement analysis results saved to: {output_file}")

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing engagement: {e}")
            return False

    def _analyze_learning(self) -> bool:
        """
        Analyze learning outcomes.

        Returns:
            bool: True if analysis was successful
        """
        self.logger.info("Analyzing learning outcomes")

        try:
            # Get output directory
            output_dir = self.args.output_dir or "./output/learning"

            # Run learning analysis
            result = self.analyzer_manager.run_specific_analysis(
                analysis_type="learning", output_dir=output_dir
            )

            if "error" in result:
                self.logger.error(f"Learning analysis error: {result['error']}")
                return False

            if "output_file" in result:
                self.logger.info(
                    f"Learning analysis results saved to: {result['output_file']}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error analyzing learning outcomes: {e}")
            return False

    def _generate_data_summary(self, output_dir: str) -> bool:
        """
        Generate a data summary.

        Args:
            output_dir: Output directory

        Returns:
            bool: True if summary was generated successfully
        """
        self.logger.info("Generating data summary")

        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Get data summary
            summary_path = self.analyzer_manager.save_data_summary(output_dir)

            if not summary_path:
                self.logger.error("Failed to generate data summary")
                return False

            self.logger.info(f"Data summary saved to: {summary_path}")
            return True

        except Exception as e:
            self.logger.exception(f"Error generating data summary: {e}")
            return False

    # === Report Generation Methods ===

    def _generate_comprehensive_report(
        self,
        course_id: Optional[str] = None,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        include_user_details: bool = False,
        include_data_tables: bool = False,
    ) -> bool:
        """
        Generate a comprehensive report with all analyses.

        Args:
            course_id: Optional course ID to filter data
            framework: Framework type to analyze
            include_user_details: Whether to include user details
            include_data_tables: Whether to include data tables

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating comprehensive report for course: {course_id}")

        try:
            result = self.visualization_manager.create_comprehensive_report(
                course_id=course_id,
                framework=framework,
                report_name="comprehensive_analysis_report",
                include_user_details=include_user_details,
                include_category_analysis=True,
                save_formats=self.args.formats.split(","),
            )

            if not result or "report_directory" not in result:
                self.logger.error("Failed to generate comprehensive report")
                return False

            self.logger.info(
                f"Comprehensive report generated at: {result['report_directory']}"
            )
            return True

        except Exception as e:
            self.logger.exception(f"Error generating comprehensive report: {e}")
            return False

    def _generate_engagement_report(
        self,
        course_id: Optional[str] = None,
        include_user_details: bool = False,
        **kwargs,
    ) -> bool:
        """
        Generate an engagement analysis report.

        Args:
            course_id: Optional course ID to filter data
            include_user_details: Whether to include user details

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating engagement report for course: {course_id}")

        try:
            report_path = self.visualization_manager.create_engagement_report(
                course_id=course_id, include_user_details=include_user_details
            )

            if not report_path:
                self.logger.error("Failed to generate engagement report")
                return False

            self.logger.info(f"Engagement report generated at: {report_path}")
            return True

        except Exception as e:
            self.logger.exception(f"Error generating engagement report: {e}")
            return False

    def _generate_framework_report(
        self,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        course_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Generate a framework analysis report.

        Args:
            framework: Framework type to analyze
            course_id: Optional course ID to filter data

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating framework report for: {framework.value}")

        try:
            report_path = self.visualization_manager.create_framework_report(
                framework=framework,
                course_id=course_id,
                include_ideas_without_steps=False,
                include_category_analysis=True,
                save_formats=self.args.formats.split(","),
            )

            if not report_path:
                self.logger.error("Failed to generate framework report")
                return False

            self.logger.info(f"Framework report generated at: {report_path}")
            return True

        except Exception as e:
            self.logger.exception(f"Error generating framework report: {e}")
            return False

    def _generate_learning_report(self, **kwargs) -> bool:
        """
        Generate a learning outcomes report.

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info("Generating learning outcomes report")

        try:
            result = self.visualization_manager.create_learning_report()

            if not result or "report_dir" not in result:
                self.logger.error("Failed to generate learning report")
                return False

            self.logger.info(f"Learning report generated at: {result['report_dir']}")
            return True

        except Exception as e:
            self.logger.exception(f"Error generating learning report: {e}")
            return False

    def _generate_team_report(
        self,
        course_id: Optional[str] = None,
        team_id: Optional[int] = None,
        include_data_tables: bool = False,
        **kwargs,
    ) -> bool:
        """
        Generate a team analysis report.

        Args:
            course_id: Optional course ID to filter data
            team_id: Optional team ID for specific team analysis
            include_data_tables: Whether to include data tables

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating team report for course: {course_id}")

        try:
            result = self.visualization_manager.create_team_report(
                course_id=course_id,
                team_id=team_id,
                include_data_tables=include_data_tables,
                report_name="team_analysis",
            )

            if not result:
                self.logger.error("Failed to generate team report")
                return False

            # Team report returns paths to generated files, not a single directory
            self.logger.info(f"Team report files generated: {', '.join(result.keys())}")
            return True

        except Exception as e:
            self.logger.exception(f"Error generating team report: {e}")
            return False

    def _generate_summary_report(
        self,
        course_id: Optional[str] = None,
        framework: FrameworkType = FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
        **kwargs,
    ) -> bool:
        """
        Generate a summary report with key visualizations.

        Args:
            course_id: Optional course ID to filter data
            framework: Framework type to analyze

        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating summary report for course: {course_id}")

        try:
            # Generate a brief summary with key visualizations
            figures = self.visualization_manager.visualize_combined_metrics(
                course_id=course_id,
                framework=framework,
                save_formats=self.args.formats.split(","),
            )

            if not figures:
                self.logger.error(
                    "Failed to generate visualizations for summary report"
                )
                return False

            # Determine output directory
            output_dir = str(Path(self.visualization_manager._output_dir) / "combined")
            self.logger.info(f"Summary visualizations saved to: {output_dir}")

            return True

        except Exception as e:
            self.logger.exception(f"Error generating summary report: {e}")
            return False


def main():
    """Main function to run the analysis application."""
    app = AnalysisApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
