"""
Enumerations for the data analysis system.

This module defines all the enumerations used throughout the system,
providing type safety and documentation for categorical data.
Compatible with Pydantic models.
"""

from enum import Enum
from typing import Dict, List, Optional, ClassVar, Type

from src.utils.safe_ops import safe_lower


class FrameworkType(str, Enum):
    """Types of entrepreneurial frameworks supported by the JetPack/Orbit tool."""

    DISCIPLINED_ENTREPRENEURSHIP = "Disciplined Entrepreneurship"
    STARTUP_TACTICS = "Startup Tactics"
    MY_JOURNEY = "My Journey"
    PRODUCT_MANAGEMENT = "Product Management"
    DISCIPLINED_ENTREPRENEURSHIP_FORMATTED = "disciplined-entrepreneurship"
    STARTUP_TACTICS_FORMATTED = "startup-tactics"
    MY_JOURNEY_FORMATTED = "my-journey"
    PRODUCT_MANAGEMENT_FORMATTED = "product-management"

    @classmethod
    def get_all_values(cls) -> List[str]:
        """Returns a list of all framework values"""
        return [e.value for e in cls]

    # FUTURE: Add validation for case variations and extra whitespace
    # in framework names if the data source becomes less standardized


class ToolVersion(str, Enum):
    """Versions of the JetPack/Orbit tool used for each cohort."""

    NONE = "none"  # Fall 2023: No tool
    V1 = "v1"  # Spring 2024: JetPack v1
    V2 = "v2"  # Fall 2024 & Spring 2025: JetPack v2

    # FUTURE: Add validation for different representations of versions
    # (e.g., "V1" vs "v1" vs "version 1") if needed


class Semester(str, Enum):
    """Academic semesters for cohort analysis with tool version mapping."""

    FALL_2023 = "Fall 2023"  # Control group (no tool)
    SPRING_2024 = "Spring 2024"  # JetPack v1
    FALL_2024 = "Fall 2024"  # JetPack v2
    SPRING_2025 = "Spring 2025"  # JetPack v2 (upcoming)

    # Class variable mapping semesters to tool versions
    tool_version_map: ClassVar[Dict[str, ToolVersion]] = {
        "Fall 2023": ToolVersion.NONE,
        "Spring 2024": ToolVersion.V1,
        "Fall 2024": ToolVersion.V2,
        "Spring 2025": ToolVersion.V2,
    }

    @classmethod
    def get_tool_version(cls, semester: str) -> ToolVersion:
        """Get the tool version used in a given semester"""
        if isinstance(semester, cls):
            return cls.tool_version_map[semester.value]
        return cls.tool_version_map.get(semester, ToolVersion.NONE)

    # FUTURE: Add validation for alternative semester representations
    # (e.g., "F23" vs "Fall 2023") if the data source varies


class UserEngagementLevel(str, Enum):
    """User engagement levels for segmentation analysis."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    # FUTURE: Add validation for case variations if needed


class UserType(str, Enum):
    """Types of users based on academic or professional affiliation."""

    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    MBA = "MBA"
    PHD = "PhD"
    FACULTY = "faculty"
    STAFF = "staff"
    ALUMNI = "alumni"
    OTHER = "other"

    @classmethod
    def from_affiliation(cls, affiliation: Dict) -> "UserType":
        """Extract user type from student_affiliation or institution.affiliation"""
        if not affiliation:
            return cls.OTHER

        # Check type field
        if "type" in affiliation:
            type_str = affiliation["type"]
            # Try direct mapping
            for member in cls:
                if member.value == type_str:
                    return member

        # Check student_type field
        if "student_type" in affiliation:
            student_type = affiliation["student_type"]
            if "undergraduate" in safe_lower(student_type):
                return cls.UNDERGRADUATE
            elif "graduate" in safe_lower(student_type):
                return cls.GRADUATE
            elif "mba" in safe_lower(student_type):
                return cls.MBA
            elif "phd" in safe_lower(student_type):
                return cls.PHD

        # Check title field for non-students
        if "title" in affiliation:
            title = affiliation["title"]
            if title:
                if "faculty" in safe_lower(title) or "professor" in safe_lower(title):
                    return cls.FACULTY
                elif "staff" in safe_lower(title):
                    return cls.STAFF

        return cls.OTHER

    # FUTURE: Add more robust validation for different representations
    # of user types across various data sources


class DisciplinedEntrepreneurshipStep(str, Enum):
    """
    Steps in the Disciplined Entrepreneurship framework (24 steps).
    """

    MARKET_SEGMENTATION = "market-segmentation"
    BEACHHEAD_MARKET = "beachhead-market"
    END_USER_PROFILE = "end-user-profile"
    BEACHHEAD_TAM_SIZE = "beachhead-tam-size"
    PERSONA = "persona"
    LIFE_CYCLE_USE_CASE = "life-cycle-use-case"
    HIGH_LEVEL_SPECS = "high-level-specs"
    QUANTIFY_VALUE_PROPOSITION = "quantify-value-proposition"
    NEXT_10_CUSTOMERS = "next-10-customers"
    DEFINE_CORE = "define-core"
    CHART_COMPETITIVE_POSITION = "chart-competitive-position"
    DETERMINE_DMU = "determine-dmu"
    MAP_SALES_PROCESS = "map-sales-process"
    FOLLOW_ON_TAM = "follow-on-tam"
    DESIGN_BUSINESS_MODEL = "design-business-model"
    PRICING_FRAMEWORK = "pricing-framework"
    LTV = "ltv"
    MAP_CUSTOMER_ACQUISITION_PROCESS = "map-customer-acquisition-process"
    COCA = "coca"
    IDENTIFY_KEY_ASSUMPTIONS = "identify-key-assumptions"
    TEST_KEY_ASSUMPTIONS = "test-key-assumptions"
    DEFINE_MVBP = "define-mvbp"
    SHOW_DOGS_WILL_EAT_DOG_FOOD = "show-dogs-will-eat-dog-food"
    DEVELOP_PRODUCT_PLAN = "develop-product-plan"

    # Define the step order outside the enum initialization
    @classmethod
    def get_step_order(cls) -> List[str]:
        """Get the list of steps in order"""
        return [
            "market-segmentation",
            "beachhead-market",
            "end-user-profile",
            "beachhead-tam-size",
            "persona",
            "life-cycle-use-case",
            "high-level-specs",
            "quantify-value-proposition",
            "next-10-customers",
            "define-core",
            "chart-competitive-position",
            "determine-dmu",
            "map-sales-process",
            "follow-on-tam",
            "design-business-model",
            "pricing-framework",
            "ltv",
            "map-customer-acquisition-process",
            "coca",
            "identify-key-assumptions",
            "test-key-assumptions",
            "define-mvbp",
            "show-dogs-will-eat-dog-food",
            "develop-product-plan",
        ]

    @classmethod
    def get_all_step_values(cls) -> List[str]:
        """Returns a list of all step values in order"""
        # Simply return the result of get_step_order - no need for copy()
        # since get_step_order creates a new list each time
        return cls.get_step_order()

    @classmethod
    def get_step_number(cls, step) -> int:
        """Get the step number (1-24) for a given step value"""
        if step is None:
            return 0

        if isinstance(step, cls):
            step_value = step.value
        else:
            step_value = step

        try:
            return cls.get_step_order().index(step_value) + 1
        except (ValueError, TypeError):
            return 0

    @classmethod
    def get_by_step_number(
        cls, step_number: int
    ) -> Optional["DisciplinedEntrepreneurshipStep"]:
        """Get the step enum by its number (1-24)"""
        if (
            not isinstance(step_number, int)
            or step_number < 1
            or step_number > len(cls.get_step_order())
        ):
            return None

        try:
            step_value = cls.get_step_order()[step_number - 1]
            for member in cls:
                if member.value == step_value:
                    return member
        except (IndexError, TypeError):
            pass

        return None

    @property
    def step_number(self) -> int:
        """Get the step number for this step"""
        return self.__class__.get_step_number(self)

    @property
    def next_step(self) -> Optional["DisciplinedEntrepreneurshipStep"]:
        """Get the next step in the sequence, or None if this is the last step"""
        num = self.step_number
        if num < len(self.__class__.get_step_order()):
            return self.__class__.get_by_step_number(num + 1)
        return None

    @property
    def previous_step(self) -> Optional["DisciplinedEntrepreneurshipStep"]:
        """Get the previous step in the sequence, or None if this is the first step"""
        num = self.step_number
        if num > 1:
            return self.__class__.get_by_step_number(num - 1)
        return None


class StartupTacticsStep(str, Enum):
    """
    Steps in the Startup Tactics framework.

    These correspond to fields in the idea document and the 'step' field
    in the steps collection when framework is "Startup Tactics".
    """

    GOALS = "goals"
    SYSTEMS = "systems"
    MARKET_RESEARCH = "market-research"
    ASSETS = "assets"
    MARKETING = "marketing"
    SALES = "sales"
    PRODUCT_ROADMAP = "product-roadmap"
    DESIGN = "design"
    USER_TESTING = "user-testing"
    ENGINEERING = "engineering"
    LEGAL = "legal"
    FINANCE = "finance"
    PITCH_DECK = "pitch-deck"
    FUNDRAISING = "fundraising"
    HIRING = "hiring"

    # FUTURE: Add mapping for common name variations (with spaces, without hyphens, etc.)
    # if we need to support non-standardized inputs

    @classmethod
    def get_all_step_values(cls) -> List[str]:
        """Returns a list of all step values"""
        return [e.value for e in cls]


class StepPrefix(str, Enum):
    """
    Prefixes used in the idea document for different versions of steps.
    """

    NONE = ""  # Standard step (e.g., "market-segmentation")
    AI = "ai-"  # AI-generated content (e.g., "ai-market-segmentation")
    SELECTED = (
        "selected-"  # User-selected content (e.g., "selected-market-segmentation")
    )

    @classmethod
    def extract_base_step(cls, field_name: str) -> str:
        """
        Extract the base step name from a field with a prefix

        Example:
            extract_base_step("ai-market-segmentation") -> "market-segmentation"
        """
        for prefix in [cls.AI.value, cls.SELECTED.value]:
            if field_name.startswith(prefix):
                return field_name[len(prefix) :]
        return field_name

    @classmethod
    def get_prefix(cls, field_name: str) -> "StepPrefix":
        """
        Get the prefix from a field name

        Example:
            get_prefix("ai-market-segmentation") -> StepPrefix.AI
        """
        if field_name.startswith(cls.AI.value):
            return cls.AI
        elif field_name.startswith(cls.SELECTED.value):
            return cls.SELECTED
        return cls.NONE


class IdeaCategory(str, Enum):
    """
    Categories for classifying ideas based on the categorized ideas data.
    """

    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    FINTECH = "Financial Technology"
    SUSTAINABILITY = "Sustainability"
    ENTERPRISE_SOFTWARE = "Enterprise Software"
    CONSUMER_APPS = "Consumer Applications"
    ECOMMERCE = "E-Commerce"
    SOCIAL_IMPACT = "Social Impact"
    AI_ML = "Artificial Intelligence / Machine Learning"
    HARDWARE = "Hardware / IoT"
    OTHER = "Other"

    # FUTURE: Add category aliases mapping for fuzzy matching of categories
    # if we need to categorize ideas with non-standard descriptions

    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Returns a list of all category values"""
        return [e.value for e in cls]


class MetricType(str, Enum):
    """
    Types of metrics used in the analysis.
    """

    ENGAGEMENT = "engagement"  # User engagement metrics
    COMPLETION = "completion"  # Framework completion metrics
    CONTENT = "content"  # Content quality metrics
    TIME = "time"  # Time-based metrics (duration, intervals)
