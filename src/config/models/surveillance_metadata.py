from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SurveillanceMetadata(BaseModel):
    camera_type: Optional[str] = Field(None, description="e.g., 'dome', 'fixed', etc.")
    mount_type: Optional[str] = Field(None, description="e.g., 'wall', 'pole', etc.")
    zone: Optional[str] = Field(None, description="e.g., 'town', 'building', etc.")
    operator: Optional[str] = Field(None, description="Entity operating the camera")
    manufacturer: Optional[str] = Field(None, description="Manufacturer of the camera")
    public: Optional[bool] = Field(
        None, description="True if public surveillance, False if private"
    )
    surveillance_type: Optional[str] = Field(
        None, description="e.g., 'camera', 'guard', etc."
    )
    start_date: Optional[str] = Field(
        None, description="ISO date string (e.g., '2024-12-01')"
    )
    sensitive: bool = Field(
        False, description="Flagged as sensitive (e.g., police, government)"
    )

    sensitive_reason: Optional[str] = Field(
        None, max_length=60, description="The LLM justification for the sensitive tag"
    )

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.endswith("?"):
            return value  # Allow fuzzy date like "2024?"
        # Ensure it’s a valid partial or full ISO date
        import re

        if re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", value):
            return value
        raise ValueError(
            "start_date must be in format YYYY, YYYY-MM, YYYY-MM-DD or YYYY?"
        )

    @classmethod
    def from_raw(cls, element: dict, enriched_fields: dict) -> "SurveillanceMetadata":
        """
        Combines the raw OSM element and LLM-enriched metadata into a validated object.
        """
        return cls(
            camera_type=enriched_fields.get("camera_type"),
            mount_type=enriched_fields.get("mount_type"),
            zone=enriched_fields.get("zone"),
            operator=enriched_fields.get("operator"),
            manufacturer=enriched_fields.get("manufacturer"),
            public=enriched_fields.get("public"),
            surveillance_type=enriched_fields.get("surveillance_type"),
            start_date=enriched_fields.get("start_date"),
            sensitive=enriched_fields.get("sensitive", False),
            sensitive_reason=enriched_fields.get("sensitive_reason"),
        )
