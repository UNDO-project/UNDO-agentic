"""
Health check and system status endpoints.

This module provides basic health monitoring and version information endpoints.
"""

from fastapi import APIRouter
from datetime import datetime

from src.api.models.responses import HealthResponse, VersionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API service.

    :return: Health status with timestamp
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="agentic-surveillance-research-api",
    )


@router.get("/version", response_model=VersionResponse)
async def version_info():
    """
    API version information endpoint.

    Returns version information about the API and application.

    :return: Version details
    """
    return VersionResponse(
        version="2.3.0",
        api_version="v1",
        description="Agentic Surveillance Research API",
    )
