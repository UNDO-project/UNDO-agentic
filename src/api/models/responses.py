"""
Pydantic response models for API endpoints.

This module defines the response schemas for all API endpoints, providing
automatic validation, serialization, and documentation.
"""

from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class TaskStatus(str, Enum):
    """Task execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResponse(BaseModel):
    """
    Response model for task creation.

    :param task_id: Unique identifier for the created task
    :param status: Current status of the task
    :param message: Human-readable message about the task
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "task_id": "abc123",
                    "status": "pending",
                    "message": "Pipeline started for Berlin",
                }
            ]
        }
    )

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    message: str = Field(..., description="Human-readable status message")


class TaskStatusResponse(BaseModel):
    """
    Response model for task status queries.

    :param id: Task identifier
    :param type: Type of task (scrape, analyze, route, pipeline)
    :param status: Current execution status
    :param progress: Progress percentage (0-100)
    :param result: Task result data (only present when completed)
    :param error: Error message (only present when failed)
    :param created_at: ISO timestamp when task was created
    :param started_at: ISO timestamp when task started execution
    :param completed_at: ISO timestamp when task completed
    :param metadata: Additional task metadata
    """

    id: str = Field(..., description="Task identifier")
    type: str = Field(..., description="Task type (scrape, analyze, route, pipeline)")
    status: TaskStatus = Field(..., description="Current execution status")
    progress: int = Field(..., description="Progress percentage (0-100)")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task result data (present when completed)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (present when failed)",
    )
    created_at: str = Field(..., description="ISO timestamp of task creation")
    started_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when task started",
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when task completed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata",
    )


class RouteMetricsResponse(BaseModel):
    """
    Response model for route metrics.

    :param length_m: Total route length in meters
    :param exposure_score: Cameras per kilometer along the route
    :param camera_count: Number of cameras near the route
    :param baseline_length_m: Baseline shortest path length in meters
    :param baseline_exposure_score: Baseline path exposure score
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "length_m": 1523.4,
                    "exposure_score": 2.3,
                    "camera_count": 12,
                    "baseline_length_m": 1450.0,
                    "baseline_exposure_score": 4.1,
                }
            ]
        }
    )

    length_m: float = Field(..., description="Total route length in meters")
    exposure_score: float = Field(
        ...,
        description="Exposure score (cameras per kilometer)",
    )
    camera_count: int = Field(..., description="Number of cameras near route")
    baseline_length_m: float = Field(
        ...,
        description="Baseline shortest path length in meters",
    )
    baseline_exposure_score: float = Field(
        ...,
        description="Baseline path exposure score",
    )


class RouteResponse(BaseModel):
    """
    Response model for route computation.

    :param route_id: Unique identifier for the route (cache key)
    :param city: City name where route was computed
    :param metrics: Route performance metrics
    :param geojson_path: Path to route GeoJSON file
    :param map_path: Path to interactive HTML map
    :param from_cache: Whether route was served from cache
    """

    route_id: str = Field(..., description="Unique route identifier")
    city: str = Field(..., description="City name")
    metrics: RouteMetricsResponse = Field(..., description="Route metrics")
    geojson_path: str = Field(..., description="Path to route GeoJSON file")
    map_path: str = Field(..., description="Path to interactive HTML map")
    from_cache: bool = Field(
        ...,
        description="Whether route was served from cache",
    )


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    :param status: Health status (healthy, degraded, unhealthy)
    :param timestamp: ISO timestamp of health check
    :param service: Service name
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2025-12-05T10:30:00Z",
                    "service": "Agentic Surveillance Research API",
                }
            ]
        }
    )

    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO timestamp")
    service: str = Field(..., description="Service name")


class VersionResponse(BaseModel):
    """
    Response model for version endpoint.

    :param version: Application version
    :param api_version: API version
    :param description: Service description
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "version": "2.4.0",
                    "api_version": "v1",
                    "description": "Agentic Surveillance Research API",
                }
            ]
        }
    )

    version: str = Field(..., description="Application version")
    api_version: str = Field(..., description="API version")
    description: str = Field(..., description="Service description")
