"""
Pydantic request models for API endpoints.

This module defines the request schemas for all API endpoints, providing
automatic validation, serialization, and documentation.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from src.config.pipeline_config import AnalysisScenario


class OutputOverrides(BaseModel):
    """
    Per-toggle overrides layered on top of the chosen ``AnalysisScenario``.

    Every field is optional. Only the fields the caller sets are merged
    into the preset's ``PipelineConfig``; unset fields keep the preset's
    default. Pass an empty object to keep the preset baseline exactly.
    """

    generate_geojson: Optional[bool] = Field(
        default=None, description="Write the enriched-camera GeoJSON"
    )
    compute_stats: Optional[bool] = Field(
        default=None, description="Compute the summary statistics dict"
    )
    generate_chart: Optional[bool] = Field(
        default=None, description="Render the public/private privacy pie chart"
    )
    generate_heatmap: Optional[bool] = Field(
        default=None, description="Render the folium heatmap (HTML)"
    )
    generate_hotspots: Optional[bool] = Field(
        default=None, description="Compute DBSCAN hotspots GeoJSON"
    )
    plot_zone_sensitivity: Optional[bool] = Field(
        default=None, description="Render the zone-sensitivity stacked bar chart"
    )
    plot_sensitivity_reasons: Optional[bool] = Field(
        default=None, description="Render the sensitivity-reasons bar chart"
    )
    plot_hotspots: Optional[bool] = Field(
        default=None, description="Render the hotspots scatter plot (PNG)"
    )


class ScrapeRequest(BaseModel):
    """
    Request model for scraping surveillance data from OpenStreetMap.

    :param city: City name to scrape (e.g., "Berlin", "Athens")
    :param country: Optional ISO country code for disambiguation (e.g., "DE", "GR")
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"city": "Berlin", "country": "DE"},
                {"city": "Athens", "country": "GR"},
                {"city": "Lund", "country": "SE"},
            ]
        }
    )

    city: str = Field(..., description="City name to scrape")
    country: Optional[str] = Field(
        default=None,
        description="ISO country code (2 letters) for disambiguation",
    )


class AnalyzeRequest(BaseModel):
    """
    Request model for analyzing scraped surveillance data.

    :param data_path: Path to the scraped data file (JSON format)
    :param scenario: Analysis scenario preset defining which visualizations to generate
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"data_path": "overpass_data/Berlin.json", "scenario": "full"},
                {"data_path": "overpass_data/Athens.json", "scenario": "basic"},
            ]
        }
    )

    data_path: str = Field(
        ...,
        description="Path to scraped data file",
    )
    scenario: AnalysisScenario = Field(
        default=AnalysisScenario.BASIC,
        description="Analysis scenario preset (basic, full)",
    )


class RouteComputeRequest(BaseModel):
    """
    Request model for computing low-surveillance walking route.

    :param city: City name for routing
    :param country: Optional ISO country code
    :param start_lat: Starting point latitude (-90 to 90)
    :param start_lon: Starting point longitude (-180 to 180)
    :param end_lat: Ending point latitude (-90 to 90)
    :param end_lon: Ending point longitude (-180 to 180)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "city": "Berlin",
                    "country": "DE",
                    "start_lat": 52.52,
                    "start_lon": 13.40,
                    "end_lat": 52.50,
                    "end_lon": 13.42,
                },
                {
                    "city": "Athens",
                    "country": "GR",
                    "start_lat": 37.9838,
                    "start_lon": 23.7275,
                    "end_lat": 37.9755,
                    "end_lon": 23.7348,
                },
            ]
        }
    )

    city: str = Field(..., description="City name for routing")
    country: Optional[str] = Field(
        default=None,
        description="ISO country code for disambiguation",
    )
    start_lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Starting point latitude"
    )
    start_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Starting point longitude"
    )
    end_lat: float = Field(..., ge=-90.0, le=90.0, description="Ending point latitude")
    end_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Ending point longitude"
    )


class PipelineRequest(BaseModel):
    """
    Request model for running the complete surveillance analysis pipeline.

    Orchestrates scraping, analysis, and optionally routing in a single request.

    :param city: City name to analyze
    :param country: Optional ISO country code
    :param scenario: Analysis scenario preset
    :param overrides: Optional per-toggle overrides layered over the preset
    :param routing_config: Optional routing configuration (enables routing if provided)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"city": "Berlin", "country": "DE", "scenario": "basic"},
                {
                    "city": "Berlin",
                    "country": "DE",
                    "scenario": "basic",
                    "overrides": {"generate_heatmap": True},
                },
                {
                    "city": "Lund",
                    "country": "SE",
                    "scenario": "full",
                    "routing_config": {
                        "city": "Lund",
                        "country": "SE",
                        "start_lat": 55.7047,
                        "start_lon": 13.1910,
                        "end_lat": 55.7058,
                        "end_lon": 13.1932,
                    },
                },
            ]
        }
    )

    city: str = Field(..., description="City name to analyze")
    country: Optional[str] = Field(
        default=None,
        description="ISO country code for disambiguation",
    )
    scenario: AnalysisScenario = Field(
        default=AnalysisScenario.BASIC,
        description="Analysis scenario preset (basic, full)",
    )
    overrides: Optional[OutputOverrides] = Field(
        default=None,
        description=(
            "Per-toggle overrides layered on top of the scenario preset. "
            "Only fields explicitly set are applied; unset fields inherit "
            "the preset's value."
        ),
    )
    routing_config: Optional[RouteComputeRequest] = Field(
        default=None,
        description="Optional routing configuration (enables routing when provided)",
    )
    force_refresh: bool = Field(
        default=False,
        description=(
            "Bypass the scrape cache and refetch from Overpass even if a "
            "valid cache entry exists for this city. Use to capture cameras "
            "newly tagged in OSM before the configured TTL expires."
        ),
    )
