"""
Base model classes for MongoDB document representation.
Contains utility models for handling MongoDB's special field types
including ObjectId and DateField with proper field aliasing to handle
the "$" character in field names.
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class ObjectId(BaseModel):
    """MongoDB ObjectId representation."""

    oid: str = Field(alias="$oid")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "$oid": "507f1f77bcf86cd799439011"
            }  # Custom JSON schema to handle the $oid field
        },
    )


class DateField(BaseModel):
    """MongoDB date field representation."""

    date: str = Field(alias="$date")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {"$date": "2023-01-01T00:00:00Z"}},
    )

    def to_datetime(self) -> datetime:
        """Convert to datetime object."""
        if isinstance(self.date, datetime):
            return self.date
        return datetime.fromisoformat(self.date.replace("Z", "+00:00"))
