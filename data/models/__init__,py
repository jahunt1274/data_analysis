"""
Models package for the data analysis system.

This package contains all the data models used for representing and
analyzing the JetPack/Orbit tool usage data.
"""

# Re-export enums
from .enums import (
    FrameworkType,
    ToolVersion,
    Semester,
    UserEngagementLevel,
    UserType,
    DisciplinedEntrepreneurshipStep,
    StartupTacticsStep,
    StepPrefix,
    IdeaCategory,
    MetricType,
)

# Re-export base models
from .base_model import ObjectId, DateField

# Re-export entity models
from .user_model import User, UserScores, OrbitProfile, StudentAffiliation
from .idea_model import Idea, FrameworkProgress
from .step_model import Step
from .team_model import Team, TeamMember
from .course_model import (
    CourseEvaluation,
    EvaluationMetric,
    EvaluationQuestion,
    SemesterInfo,
    Scale,
    ScaleValue,
)

# Define all models for easy access
__all__ = [
    # Enums
    "FrameworkType",
    "ToolVersion",
    "Semester",
    "UserEngagementLevel",
    "UserType",
    "DisciplinedEntrepreneurshipStep",
    "StartupTacticsStep",
    "StepPrefix",
    "IdeaCategory",
    "MetricType",
    # Base models
    "ObjectId",
    "DateField",
    # User models
    "User",
    "UserScores",
    "OrbitProfile",
    "StudentAffiliation",
    # Idea models
    "Idea",
    "FrameworkProgress",
    # Step models
    "Step",
    # Team models
    "Team",
    "TeamMember",
    # Course models
    "CourseEvaluation",
    "EvaluationMetric",
    "EvaluationQuestion",
    "SemesterInfo",
    "Scale",
    "ScaleValue",
]
