from typing import Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AnalysisScenario(str, Enum):
    """
    Output presets for the analyzer's visualization stage.

    ``BASIC`` is the minimal-output baseline (enriched data + stats).
    ``FULL`` flips every output toggle on. Power users override
    individual flags via ``OutputOverrides`` on the request layer
    (or the CLI's ``--heatmap``/``--charts``/``--hotspots`` flags)
    rather than picking from a long list of named bundles.
    """

    BASIC = "basic"  # Enriched data + GeoJSON + stats; no charts or maps
    FULL = "full"  # All visualizations and statistics


class PipelineConfig(BaseModel):
    """Configuration for the surveillance data analysis pipeline."""

    # General settings
    scenario: AnalysisScenario = Field(
        default=AnalysisScenario.BASIC, description="Predefined analysis scenario"
    )

    # Scraper settings
    scrape_enabled: bool = Field(default=True, description="Enable data scraping step")
    country_code: Optional[str] = Field(
        default=None, description="ISO country code for disambiguation"
    )
    force_refresh: bool = Field(
        default=False,
        description=(
            "Bypass the scrape cache and refetch from Overpass even if a "
            "valid cache entry exists. Use to capture newly-tagged cameras "
            "before the configured TTL expires."
        ),
    )

    # Analyzer settings
    analyze_enabled: bool = Field(default=True, description="Enable data analysis step")

    # Routing settings
    routing_enabled: bool = Field(
        default=False, description="Enable low-surveillance routing step"
    )
    start_lat: Optional[float] = Field(default=None, description="Starting latitude")
    start_lon: Optional[float] = Field(default=None, description="Starting longitude")
    end_lat: Optional[float] = Field(default=None, description="Ending latitude")
    end_lon: Optional[float] = Field(default=None, description="Ending longitude")

    # Visualization flags
    generate_geojson: bool = Field(default=True, description="Generate GeoJSON output")
    generate_heatmap: bool = Field(
        default=False, description="Generate heatmap visualization"
    )
    generate_hotspots: bool = Field(
        default=False, description="Generate hotspot clusters"
    )
    compute_stats: bool = Field(default=True, description="Compute statistics")
    generate_chart: bool = Field(default=False, description="Generate pie chart")
    plot_zone_sensitivity: bool = Field(
        default=False, description="Plot zone sensitivity chart"
    )
    plot_sensitivity_reasons: bool = Field(
        default=False, description="Plot sensitivity reasons chart"
    )
    plot_hotspots: bool = Field(
        default=False, description="Plot hotspots visualization"
    )
    plot_operator_distribution: bool = Field(
        default=False,
        description="Plot top-N operator distribution as a horizontal bar chart",
    )
    plot_manufacturer_distribution: bool = Field(
        default=False,
        description="Plot top-N manufacturer distribution as a horizontal bar chart",
    )
    plot_install_timeline: bool = Field(
        default=False,
        description="Plot a bar chart of camera count by install year (start_date)",
    )
    generate_report: bool = Field(
        default=False,
        description=(
            "Generate an LLM-written markdown report (<city>_report.md) "
            "summarizing the city's surveillance posture. Depends on "
            "compute_stats=True; failures land in visualization_errors "
            "and never abort the run."
        ),
    )

    # Pipeline behavior
    stop_on_error: bool = Field(
        default=True,
        description="Stop pipeline on first error (vs. continue with partial results)",
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")

    # Output settings
    output_dir: str = Field(
        default="overpass_data", description="Base directory for outputs"
    )

    @model_validator(mode="after")
    def validate_routing_coordinates(self) -> "PipelineConfig":
        """
        Validate that routing coordinates are provided when routing is enabled.

        :return: The validated PipelineConfig instance
        :raises ValueError: If routing is enabled but coordinates are missing
        """
        if self.routing_enabled:
            if any(
                coord is None
                for coord in [
                    self.start_lat,
                    self.start_lon,
                    self.end_lat,
                    self.end_lon,
                ]
            ):
                raise ValueError(
                    "Routing enabled but missing required coordinates. "
                    "Provide start_lat, start_lon, end_lat, and end_lon."
                )
        return self

    @classmethod
    def from_scenario(cls, scenario: AnalysisScenario) -> "PipelineConfig":
        """
        Create configuration from a predefined output preset.

        Two presets are supported: ``BASIC`` (geojson + stats) and ``FULL``
        (every output toggle on). Callers needing a custom toggle bundle
        should start from a preset and override individual flags via
        ``OutputOverrides`` (API) or the CLI toggle flags.

        :param scenario: Analysis scenario to use
        :return: Configured PipelineConfig instance
        """
        if scenario == AnalysisScenario.FULL:
            return cls(
                scenario=scenario,
                generate_geojson=True,
                generate_heatmap=True,
                generate_hotspots=True,
                compute_stats=True,
                generate_chart=True,
                plot_zone_sensitivity=True,
                plot_sensitivity_reasons=True,
                plot_hotspots=True,
                plot_operator_distribution=True,
                plot_manufacturer_distribution=True,
                plot_install_timeline=True,
                generate_report=True,
            )

        # BASIC baseline: enriched data + stats, no charts or maps.
        return cls(
            scenario=scenario,
            generate_geojson=True,
            compute_stats=True,
        )

    def to_analyzer_options(self) -> Dict[str, Any]:
        """
        Convert config to analyzer options dictionary.

        :return: Dictionary of analyzer options
        """
        return {
            "generate_geojson": self.generate_geojson,
            "generate_heatmap": self.generate_heatmap,
            "generate_hotspots": self.generate_hotspots,
            "compute_stats": self.compute_stats,
            "generate_chart": self.generate_chart,
            "plot_zone_sensitivity": self.plot_zone_sensitivity,
            "plot_sensitivity_reasons": self.plot_sensitivity_reasons,
            "plot_hotspots": self.plot_hotspots,
            "plot_operator_distribution": self.plot_operator_distribution,
            "plot_manufacturer_distribution": self.plot_manufacturer_distribution,
            "plot_install_timeline": self.plot_install_timeline,
            "generate_report": self.generate_report,
        }
