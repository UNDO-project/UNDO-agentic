from typing import Optional

from pydantic import BaseModel, Field

# ============================================================================
# Input Schemas for Tools
# ============================================================================


class LoadOverpassInput(BaseModel):
    """Input schema for load_overpass_elements tool."""

    path: str = Field(
        description="File path to the Overpass JSON dump to load. Can be absolute or relative."
    )


class SaveEnrichedInput(BaseModel):
    """Input schema for save_enriched_elements tool."""

    elements: str = Field(
        description="JSON string of enriched surveillance elements to save"
    )
    path: str = Field(
        description="File path of the source file. Output will be saved alongside it with '_enriched' suffix."
    )


class SaveOverpassInput(BaseModel):
    """Input schema for save_overpass_dump tool."""

    data: str = Field(description="JSON string of Overpass API response data to save")
    city: str = Field(description="Name of the city for naming the output file")
    dest: str = Field(
        description="Destination directory or file path where data will be saved"
    )


class ToGeoJSONInput(BaseModel):
    """Input schema for to_geojson tool."""

    enriched_file: str = Field(description="Path to enriched JSON file to convert")
    output_file: Optional[str] = Field(
        default=None, description="Optional output path for GeoJSON file"
    )


class ToHeatmapInput(BaseModel):
    """Input schema for to_heatmap tool."""

    geojson_path: str = Field(
        description="Path to GeoJSON file containing surveillance points"
    )
    output_html: str = Field(description="Output path for HTML heatmap file")


class ComputeStatisticsInput(BaseModel):
    """Input schema for compute_statistics tool."""

    elements: str = Field(
        description="JSON string of enriched surveillance elements to analyze"
    )


class PrivatePublicPieInput(BaseModel):
    """Input schema for private_public_pie tool."""

    stats: str = Field(description="JSON string of statistics from compute_statistics")
    output_dir: str = Field(description="Directory path to save the pie chart")


class PlotZoneSensitivityInput(BaseModel):
    """Input schema for plot_zone_sensitivity tool."""

    stats: str = Field(description="JSON string of statistics from compute_statistics")
    output_dir: str = Field(description="Directory path to save the chart")
    top_n: int = Field(default=10, description="Number of top zones to display")


class PlotSensitivityReasonsInput(BaseModel):
    """Input schema for plot_sensitivity_reasons tool."""

    enriched_file: str = Field(description="Path to enriched JSON file")
    output_file: str = Field(description="Output path for the reasons chart")
    top_n: int = Field(default=5, description="Number of top reasons to display")


class PlotHotspotsInput(BaseModel):
    """Input schema for plot_hotspots tool."""

    hotspots_file: str = Field(description="Path to hotspots GeoJSON file")
    output_file: str = Field(description="Output path for the hotspots visualization")
