"""
Main script for running data analysis and visualization.

This script demonstrates how to use the VisualizationManager class
to generate visualizations and reports from the data analysis system.
"""

import os
import logging
from pathlib import Path
import argparse
from datetime import datetime

from config.settings import Settings
from data.models.enums import FrameworkType, Semester

# Import analyzers
from analyzers.engagement_analyzer import EngagementAnalyzer
from analyzers.framework_analyzer import FrameworkAnalyzer
from analyzers.learning_analyzer import LearningAnalyzer
from analyzers.team_analyzer import TeamAnalyzer

# Import data repository
from data.data_repository import DataRepository

# Import visualization manager
from visualizers.visualization_manager import VisualizationManager


def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Analysis and Visualization System"
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")

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
        "--output-dir", type=str, help="Output directory for visualizations"
    )

    parser.add_argument(
        "--report-type",
        type=str,
        default="comprehensive",
        choices=["comprehensive", "engagement", "framework", "learning", "team"],
        help="Type of report to generate",
    )

    parser.add_argument(
        "--include-user-details",
        action="store_true",
        help="Include detailed user information in reports",
    )

    return parser.parse_args()


def main():
    """Main function to run the data analysis and visualization system."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging()
    logger.info("Starting data analysis and visualization system")

    # Load settings
    settings = Settings(config_path=args.config)
    logger.info(
        f"Loaded settings from {args.config if args.config else 'default configuration'}"
    )

    # Set output directory
    output_dir = args.output_dir or settings.OUTPUT_DIR

    # Initialize data repository
    data_repo = DataRepository(settings)
    logger.info("Initialized data repository")

    # Initialize analyzers
    engagement_analyzer = EngagementAnalyzer(data_repo)
    framework_analyzer = FrameworkAnalyzer(data_repo)
    learning_analyzer = LearningAnalyzer(data_repo)
    team_analyzer = TeamAnalyzer(data_repo)
    logger.info("Initialized analyzers")

    # Initialize visualization manager
    viz_manager = VisualizationManager(
        settings=settings,
        output_dir=output_dir,
        include_timestamps=True,
        theme="default",
    )

    # Set analyzers in visualization manager
    viz_manager.set_analyzers(
        engagement_analyzer=engagement_analyzer,
        framework_analyzer=framework_analyzer,
        learning_analyzer=learning_analyzer,
        team_analyzer=team_analyzer,
    )
    logger.info("Initialized visualization manager")

    # Get framework type from argument
    framework = next(
        (ft for ft in FrameworkType if ft.value == args.framework),
        FrameworkType.DISCIPLINED_ENTREPRENEURSHIP,
    )

    # Generate report based on selected type
    report_path = None

    if args.report_type == "comprehensive":
        logger.info("Generating comprehensive report")
        result = viz_manager.create_comprehensive_report(
            course_id=args.course_id,
            framework=framework,
            include_user_details=args.include_user_details,
        )
        report_path = result["report_directory"]

    elif args.report_type == "engagement":
        logger.info("Generating engagement report")
        report_path = viz_manager.create_engagement_report(
            course_id=args.course_id,
            include_user_details=args.include_user_details,
        )

    elif args.report_type == "framework":
        logger.info("Generating framework report")
        report_path = viz_manager.create_framework_report(
            framework=framework,
            course_id=args.course_id,
        )

    elif args.report_type == "learning":
        logger.info("Generating learning report")
        result = viz_manager.create_learning_report()
        report_path = result.get("report_dir", "")

    elif args.report_type == "team":
        logger.info("Generating team report")
        result = viz_manager.create_team_report(
            course_id=args.course_id,
        )
        # Team report returns paths to generated files, not a single directory
        logger.info(f"Team report files generated: {', '.join(result.keys())}")

    # Print report location if available
    if report_path:
        logger.info(f"Report generated at: {report_path}")

    logger.info("Data analysis and visualization complete")


if __name__ == "__main__":
    main()
