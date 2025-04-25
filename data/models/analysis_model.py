"""
Analysis result models for the JetPack/Orbit analysis system.
Contains models for representing analysis outputs, including
metrics, distributions, and statistical results used in reporting.
"""
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel

from .enums import Semester, EngagementLevel


class CategoryDistribution(BaseModel):
    """Distribution of ideas by category."""
    category: str
    count: int
    percentage: float


class CohortMetrics(BaseModel):
    """Metrics for a specific cohort."""
    semester: Semester
    tool_version: Optional[str] = None
    user_count: int
    idea_count: int
    step_count: int
    avg_ideas_per_user: float
    avg_steps_per_idea: float
    avg_steps_per_user: float
    category_distribution: List[CategoryDistribution]
    framework_distribution: Dict[str, float]
    
    # Team metrics
    team_count: Optional[int] = 0
    avg_ideas_per_team: Optional[float] = 0
    avg_steps_per_team: Optional[float] = 0
    
    # Engagement levels
    high_engagement_count: Optional[int] = 0
    medium_engagement_count: Optional[int] = 0
    low_engagement_count: Optional[int] = 0
    inactive_count: Optional[int] = 0


class EngagementMetrics(BaseModel):
    """Metrics for user engagement."""
    user_email: str
    user_name: str
    cohort: Semester
    department: Optional[str] = None
    idea_count: int
    step_count: int
    avg_steps_per_idea: float
    frameworks_used: List[str]
    primary_framework: Optional[str] = None
    last_activity_date: Optional[datetime] = None
    first_activity_date: Optional[datetime] = None
    activity_span_days: Optional[int] = 0
    engagement_level: EngagementLevel


class StepProgressionMetrics(BaseModel):
    """Metrics for progression through framework steps."""
    step_name: str
    completion_count: int
    completion_percentage: float
    avg_content_length: float
    avg_time_to_complete: Optional[float] = None
    common_next_step: Optional[str] = None
    common_prev_step: Optional[str] = None
    dropout_rate: Optional[float] = None


class TeamAnalysisResult(BaseModel):
    """Results from team analysis."""
    team_name: str
    team_size: int
    member_emails: List[str]
    cohort: Semester
    
    # Engagement metrics
    total_ideas: int
    total_steps: int
    avg_ideas_per_member: float
    avg_steps_per_idea: float
    
    # Collaboration patterns
    single_owner_percentage: float  # % of ideas owned by one team member
    has_tool_champion: bool  # One member responsible for most tool interaction
    champion_email: Optional[str] = None
    collaboration_score: float  # Measure of distributed work (0=single user, 1=perfectly distributed)
    
    # Progression metrics
    framework_coverage: Dict[str, float]  # Coverage of each framework
    most_completed_steps: List[str]
    avg_step_completion_time: Optional[float] = None