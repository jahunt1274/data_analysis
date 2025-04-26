"""
Main script for running data analysis and visualization.

This script provides a comprehensive interface for loading data,
performing analysis, and generating visualizations and reports
for the JetPack/Orbit tool usage data.
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime
import json

from config.settings import Settings
from data.models.enums import FrameworkType, Semester, MetricType

# Import analyzers
from analyzers.engagement_analyzer import EngagementAnalyzer
from analyzers.framework_analyzer import FrameworkAnalyzer
from analyzers.learning_analyzer import LearningAnalyzer
from analyzers.team_analyzer import TeamAnalyzer

# Import data repository
from data.data_repository import DataRepository

# Import visualization manager
from visualizers.visualization_manager import VisualizationManager


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        log_dir: Optional directory for log files
        log_level: Logging level

    Returns:
        Logger instance
    """
    # Create log directory if specified and doesn't exist
    if log_dir:
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

        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        # Simple console logging if no log directory specified
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Data Analysis and Visualization System for JetPack/Orbit Tool"
    )

    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--data-dir", type=str, help="Directory containing raw data files"
    )
    parser.add_argument("--log-dir", type=str, help="Directory for log files")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for visualizations"
    )

    # Analysis options
    parser.add_argument(
        "--course-id", type=str, default="15.390", help="Course ID to filter data"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="Disciplined Entrepreneurship",
        choices=[ft.value for ft in FrameworkType],
        help="Framework to analyze",
    )
    parser.add_argument(
        "--semester",
        type=str,
        default=None,
        choices=[s.value for s in Semester],
        help="Semester to analyze",
    )

    # Report options
    parser.add_argument(
        "--report-type",
        type=str,
        default="comprehensive",
        choices=[
            "comprehensive",
            "engagement",
            "framework",
            "learning",
            "team",
            "summary",
        ],
        help="Type of report to generate",
    )
    parser.add_argument(
        "--include-user-details",
        action="store_true",
        help="Include detailed user information in reports",
    )
    parser.add_argument(
        "--include-data-tables",
        action="store_true",
        help="Include data tables in reports",
    )

    # Specific analysis options
    parser.add_argument(
        "--team-id",
        type=int,
        default=None,
        help="Specific team ID to analyze",
    )
    parser.add_argument(
        "--user-email",
        type=str,
        default=None,
        help="Specific user email to analyze",
    )

    # Visualization options
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        choices=["default", "dark", "print"],
        help="Theme for visualizations",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated list of output formats (png, pdf, svg, html)",
    )

    # Action options
    parser.add_argument(
        "--data-summary",
        action="store_true",
        help="Generate data summary only",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def load_data(data_repo, data_dir=None, logger=None):
    """
    Load data into the data repository.

    Args:
        data_repo: Data repository instance
        data_dir: Directory containing raw data files
        logger: Logger instance

    Returns:
        bool: True if data was loaded successfully
    """
    if logger:
        logger.info("Loading data into repository...")

    try:
        # Connect to data sources (this will load from default paths if data_dir is None)
        if data_dir:
            # Load from specified directory
            result = data_repo.load_data_from_directory(data_dir)
            if logger:
                for file_name, count in result.items():
                    logger.info(f"Loaded {count} records from {file_name}")
        else:
            # Connect using default paths in settings
            data_repo.connect()

        # Get data summary to verify loading
        summary = data_repo.get_data_summary()

        if logger:
            user_count = summary.get("users", {}).get("count", 0)
            idea_count = summary.get("ideas", {}).get("count", 0)
            step_count = summary.get("steps", {}).get("count", 0)
            team_count = summary.get("teams", {}).get("count", 0)

            logger.info(
                f"Data loaded successfully: {user_count} users, {idea_count} ideas, "
                + f"{step_count} steps, {team_count} teams"
            )

        return True
    except Exception as e:
        if logger:
            logger.error(f"Error loading data: {e}")
        return False


def initialize_analyzers(data_repo, logger=None):
    """
    Initialize analyzer instances.

    Args:
        data_repo: Data repository instance
        logger: Logger instance

    Returns:
        tuple: Analyzer instances (engagement, framework, learning, team)
    """
    if logger:
        logger.info("Initializing analyzers...")

    try:
        engagement_analyzer = EngagementAnalyzer(data_repo)
        framework_analyzer = FrameworkAnalyzer(data_repo)
        learning_analyzer = LearningAnalyzer(data_repo)
        team_analyzer = TeamAnalyzer(data_repo)

        if logger:
            logger.info("Analyzers initialized successfully")

        return (
            engagement_analyzer,
            framework_analyzer,
            learning_analyzer,
            team_analyzer,
        )
    except Exception as e:
        if logger:
            logger.error(f"Error initializing analyzers: {e}")
        return (None, None, None, None)


def save_data_summary(data_repo, output_dir, logger=None):
    """
    Generate and save a data summary.

    Args:
        data_repo: Data repository instance
        output_dir: Output directory
        logger: Logger instance

    Returns:
        str: Path to the saved summary file
    """
    if logger:
        logger.info("Generating data summary...")

    try:
        # Get data summary
        summary = data_repo.get_data_summary()

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"data_summary_{timestamp}.json"

        # Save summary to file
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if logger:
            logger.info(f"Data summary saved to {output_file}")

        return str(output_file)
    except Exception as e:
        if logger:
            logger.error(f"Error saving data summary: {e}")
        return None


def generate_reports(viz_manager, args, logger=None):
    """
    Generate reports based on command line arguments.

    Args:
        viz_manager: Visualization manager instance
        args: Command line arguments
        logger: Logger instance

    Returns:
        str: Path to the generated report
    """
    if logger:
        logger.info(f"Generating {args.report_type} report...")

    # Get framework type from argument
    framework = next(
        (ft for ft in FrameworkType if ft.value == args.framework),
        FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
    )

    # Parse formats
    save_formats = args.formats.split(",")

    # Generate report based on selected type
    report_path = None

    try:
        if args.report_type == "comprehensive":
            result = viz_manager.create_comprehensive_report(
                course_id=args.course_id,
                framework=framework,
                include_user_details=args.include_user_details,
                include_category_analysis=True,
                save_formats=save_formats,
            )
            report_path = result.get("report_directory")

        elif args.report_type == "engagement":
            report_path = viz_manager.create_engagement_report(
                course_id=args.course_id,
                include_user_details=args.include_user_details,
            )

        elif args.report_type == "framework":
            report_path = viz_manager.create_framework_report(
                framework=framework,
                course_id=args.course_id,
                include_category_analysis=True,
                save_formats=save_formats,
            )

        elif args.report_type == "learning":
            result = viz_manager.create_learning_report()
            report_path = result.get("report_dir", "")

        elif args.report_type == "team":
            result = viz_manager.create_team_report(
                course_id=args.course_id,
                team_id=args.team_id,
                include_data_tables=args.include_data_tables,
            )
            # Team report returns paths to generated files, not a single directory
            if logger:
                logger.info(f"Team report files generated: {', '.join(result.keys())}")
            report_path = list(result.values())[0] if result else None

        elif args.report_type == "summary":
            # Generate a brief summary with key visualizations
            figures = viz_manager.visualize_combined_metrics(
                course_id=args.course_id,
                framework=framework,
                save_formats=save_formats,
            )

            # The combined metrics function returns figure objects
            report_path = str(Path(viz_manager._output_dir) / "combined")

        if logger and report_path:
            logger.info(f"Report generated at: {report_path}")

        return report_path

    except Exception as e:
        if logger:
            logger.error(f"Error generating report: {e}")
        return None


def run_specific_analysis(data_repo, viz_manager, args, logger=None):
    """
    Run specific analysis based on command line arguments.

    Args:
        data_repo: Data repository instance
        viz_manager: Visualization manager instance
        args: Command line arguments
        logger: Logger instance

    Returns:
        bool: True if analysis was run successfully
    """
    try:
        # User-specific analysis
        if args.user_email:
            if logger:
                logger.info(f"Analyzing user: {args.user_email}")

            user_data = data_repo.get_user_ideas_steps(args.user_email)

            # Save user data to file
            output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                output_dir
                / f"user_analysis_{args.user_email.split('@')[0]}_{timestamp}.json"
            )

            with open(output_file, "w") as f:
                json.dump(user_data, f, indent=2, default=str)

            if logger:
                logger.info(f"User analysis saved to {output_file}")

            return True

        # Team-specific analysis
        elif args.team_id is not None:
            if logger:
                logger.info(f"Analyzing team: {args.team_id}")

            # Generate team visualization
            viz_manager.visualize_team_collaboration_patterns(
                team_id=args.team_id,
                include_temporal_analysis=True,
                save_formats=args.formats.split(","),
            )

            # Get and save team engagement data
            team_data = data_repo.get_team_engagement(args.team_id)

            output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"team_analysis_{args.team_id}_{timestamp}.json"

            with open(output_file, "w") as f:
                json.dump(team_data, f, indent=2, default=str)

            if logger:
                logger.info(f"Team analysis saved to {output_file}")

            return True

        # Semester-specific analysis
        elif args.semester:
            if logger:
                logger.info(f"Analyzing semester: {args.semester}")

            # Find a different semester to compare with
            semester_order = ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]
            current_index = semester_order.index(args.semester)

            # Try to get the previous semester, or the next one if current is the first
            comparison_semester = (
                semester_order[current_index - 1]
                if current_index > 0
                else semester_order[current_index + 1]
            )

            # Generate semester comparison
            comparison_data = data_repo.get_semester_comparison(
                args.semester, comparison_semester
            )

            # Save comparison data
            output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                output_dir
                / f"semester_comparison_{args.semester.replace(' ', '_')}_{timestamp}.json"
            )

            with open(output_file, "w") as f:
                json.dump(comparison_data, f, indent=2, default=str)

            if logger:
                logger.info(f"Semester comparison saved to {output_file}")

            return True

        return False
    except Exception as e:
        if logger:
            logger.error(f"Error running specific analysis: {e}")
        return False


def main():
    """Main function to run the data analysis and visualization system."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = args.log_dir or "./logs"
    logger = setup_logging(log_dir, log_level)
    logger.info("Starting data analysis and visualization system")

    # Load settings
    settings = Settings(config_path=args.config)
    logger.info(
        f"Loaded settings from {args.config if args.config else 'default configuration'}"
    )

    # Set output directory
    output_dir = args.output_dir or settings.OUTPUT_DIR
    logger.info(f"Output directory set to: {output_dir}")

    # Initialize data repository
    data_repo = DataRepository(settings)
    logger.info("Initialized data repository")

    # Load data
    if not load_data(data_repo, args.data_dir, logger):
        logger.error("Failed to load data. Exiting.")
        return 1

    # Generate data summary if requested
    if args.data_summary:
        summary_path = save_data_summary(data_repo, output_dir, logger)
        if summary_path:
            logger.info(f"Data summary generated at: {summary_path}")
            return 0
        else:
            logger.error("Failed to generate data summary.")
            return 1

    # Initialize analyzers
    engagement_analyzer, framework_analyzer, learning_analyzer, team_analyzer = (
        initialize_analyzers(data_repo, logger)
    )

    if not all(
        [engagement_analyzer, framework_analyzer, learning_analyzer, team_analyzer]
    ):
        logger.error("Failed to initialize analyzers. Exiting.")
        return 1

    # Initialize visualization manager
    logger.info(f"Initializing visualization manager with theme: {args.theme}")
    viz_manager = VisualizationManager(
        settings=settings,
        output_dir=output_dir,
        include_timestamps=True,
        theme=args.theme,
        save_formats=args.formats.split(","),
    )

    # Set analyzers in visualization manager
    viz_manager.set_analyzers(
        engagement_analyzer=engagement_analyzer,
        framework_analyzer=framework_analyzer,
        learning_analyzer=learning_analyzer,
        team_analyzer=team_analyzer,
    )
    logger.info("Visualization manager initialized")

    # Run specific analysis if requested
    specific_analysis_run = run_specific_analysis(data_repo, viz_manager, args, logger)

    # Generate reports if no specific analysis was run or if a report type was specified
    if not specific_analysis_run or args.report_type:
        report_path = generate_reports(viz_manager, args, logger)

        if not report_path:
            logger.error("Failed to generate report.")
            return 1

    logger.info("Data analysis and visualization complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
