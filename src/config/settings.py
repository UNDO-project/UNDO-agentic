from pathlib import Path
from typing import Optional, Union, Dict, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """
    Base settings for Ollama interaction
    """

    base_url: str = Field(
        default="http://localhost:11434/api/generate", description="The Ollama base url"
    )
    timeout_seconds: float = Field(
        default=30.0, description="Timeout for calling Ollama"
    )
    stream: bool = Field(default=False, description="Flag to denote chunked streaming.")
    model: str = Field(default="llama3:latest", description="Ollama model name.")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OLLAMA_", extra="allow"
    )


class DatabaseSettings(BaseSettings):
    """
    Configuration for the SQLModel-based SQLite database.
    """

    url: str = Field(default=None, description="The url of the database")
    echo: bool = Field(default=False, description="SQLAlchemy echo for debugging SQL")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SQLITE_DB_", extra="allow"
    )


class LoggingSettings(BaseSettings):
    """
    Configuration for Loguru logging sinks.
    """

    level: str = Field(default="DEBUG", description="The log level")
    console: bool = Field(default=True, description="Show logs in console")
    enable_file: bool = Field(
        default=False, description="Flag to denote persistence of logs"
    )
    filepath: Optional[Path] = Field(
        default="logs/agents.log", description="Optional file path for logs"
    )
    rotation: str = Field(default="10 MB", description="Roll log after this size")
    retention: str = Field(
        default="7 days", description="Keep logs for this amount of time"
    )
    compression: str = Field(default="zip", description="Compress old logs")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOG_", extra="allow")


class OverpassSettings(BaseSettings):
    """
    Configuration for Overpass pipeline.
    """

    endpoint: str = Field(
        default="https://overpass-api.de/api/interpreter",
        description="The Overpass API endpoint",
    )
    headers: Dict[str, Any] = Field(
        default={"User-Agent": "ACS-Agent/0.1 (contact@email)"},
        description="The headers used for making request to Overpass",
    )
    dir: Union[Path, str] = Field(
        default=Path("overpass_data"),
        description="The Path to the Overpass data directory",
    )
    query_timeout: int = Field(
        default=25, description="The timeout for the Overpass query"
    )
    timeout: int = Field(
        default=60, description="The timeout for the request made to Overpass API"
    )
    retry_http: set[int] = Field(
        default={429, 500, 502, 503, 504},
        description="The HTTP statuses to retry after hitting",
    )
    max_attempts: int = Field(default=4, description="The maximum number of retries")
    base_delay: float = Field(
        default=2.0, description="The number of delay between retries in seconds"
    )
    cache_ttl_hours: float = Field(
        default=24.0,
        description=(
            "How long a scrape cache entry stays valid before the next scan "
            "must re-fetch from Overpass. Override via OVERPASS_CACHE_TTL_HOURS."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OVERPASS_", extra="allow"
    )


class HeatmapSettings(BaseSettings):
    """
    Configuration for the heatmaps
    """

    radius: int = Field(default=15, description="The radius of points")
    blur: int = Field(default=10, description="The blur of points")


class HotspotSettings(BaseSettings):
    """
    Thresholds for the four-layer hotspot pipeline.

    Defaults are tuned for European/North American city-scale runs
    (hundreds to low thousands of cameras). Override per-deployment
    via env vars prefixed with ``HOTSPOT_``.
    """

    # Planar KDE (HSR#2)
    kde_bandwidth: Union[str, float] = Field(
        default="silverman",
        description=(
            "KDE bandwidth selector. Either a positive float in metres "
            "or a rule-of-thumb name accepted by KDEpy "
            "(``'silverman'``, ``'scott'``, ``'ISJ'``)."
        ),
    )
    kde_grid_resolution_m: int = Field(
        default=50,
        description=(
            "Spacing in metres between KDE grid samples. Smaller values "
            "give crisper contours but quadratically more cells."
        ),
    )

    # Getis-Ord Gi* hex grid (HSR#3)
    h3_resolution: int = Field(
        default=9,
        description=(
            "H3 hex resolution for the Gi* statistical layer. Resolution "
            "9 ≈ 0.1 km² hexes (metro-wide view); 10 ≈ 0.015 km² "
            "(neighbourhood view). Increase for denser cities."
        ),
    )
    gi_star_p_threshold: float = Field(
        default=0.05,
        description=(
            "FDR-adjusted p-value above which a hex is classified "
            "``not_significant`` rather than hot/cold."
        ),
    )

    # HDBSCAN polygons (HSR#1)
    hdbscan_min_cluster_size: int = Field(
        default=5,
        description=(
            "Minimum number of cameras for a cluster to be reported. "
            "Smaller values fragment noisy areas; larger values miss "
            "small but real clusters."
        ),
    )
    hdbscan_min_samples: int = Field(
        default=3,
        description=(
            "HDBSCAN's ``min_samples`` parameter — the conservativeness "
            "of the clusterer. Larger values keep more points as noise."
        ),
    )

    @field_validator("h3_resolution")
    @classmethod
    def _validate_h3_resolution(cls, value: int) -> int:
        if not 0 <= value <= 15:
            raise ValueError("h3_resolution must be between 0 and 15")
        return value

    @field_validator("gi_star_p_threshold")
    @classmethod
    def _validate_p_threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("gi_star_p_threshold must be in (0, 1)")
        return value

    @field_validator("hdbscan_min_cluster_size", "hdbscan_min_samples")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value < 2:
            raise ValueError("must be at least 2")
        return value

    @field_validator("kde_grid_resolution_m")
    @classmethod
    def _validate_grid_resolution(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("kde_grid_resolution_m must be positive")
        return value

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="HOTSPOT_", extra="allow"
    )


class DistrictSettings(BaseSettings):
    """
    Configuration for the administrative-district aggregation layer.

    ``admin_level`` semantics vary by country, so the default is a
    starting point rather than a universal truth — override per city via
    the CLI ``--district-admin-level`` flag or ``DISTRICT_DEFAULT_ADMIN_LEVEL``.
    """

    default_admin_level: int = Field(
        default=9,
        description=(
            "OSM admin_level fetched for districts when the caller does "
            "not specify one. 9 is a common city-district level; adjust "
            "per country (e.g. 7/8 for municipalities)."
        ),
    )

    @field_validator("default_admin_level")
    @classmethod
    def _validate_admin_level(cls, value: int) -> int:
        if not 1 <= value <= 12:
            raise ValueError("default_admin_level must be between 1 and 12")
        return value

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DISTRICT_", extra="allow"
    )


class RouteSettings(BaseSettings):
    """
    Configuration for the low-surveillance routing logic.

    The settings control how far from the path cameras are considered and how
    much longer than the baseline shortest path a route is allowed to be.
    """

    buffer_radius_m: float = Field(
        default=50.0,
        description="Radius in metres around the path within which cameras are counted.",
    )
    stretch_factor: float = Field(
        default=1.4,
        description=(
            "Maximum allowed ratio between the chosen route length and the "
            "baseline shortest-path length."
        ),
    )
    max_candidates: int = Field(
        default=5,
        description="Maximum number of candidate routes to score per request.",
    )
    network_type: str = Field(
        default="walk",
        description="OSM network type to request from the routing backend (e.g. 'walk').",
    )
    snap_distance_threshold_m: float = Field(
        default=500.0,
        description=(
            "Maximum distance in metres for snapping coordinates to the network. "
            "Coordinates farther than this from any road will raise an error."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ROUTE_", extra="allow"
    )


class LangChainSettings(BaseSettings):
    """
    Configuration for LangChain compatible settings
    """

    # # LangSmith Tracing Configuration
    # tracing_enabled: bool = Field(
    #     default=False, description="Enable LangSmith tracing for observability"
    # )
    # api_key: Optional[str] = Field(
    #     default=None, description="LangSmith API key for tracing"
    # )
    # endpoint: str = Field(
    #     default="https://api.smith.langchain.com",
    #     description="LangSmith API endpoint",
    # )
    # project_name: str = Field(
    #     default="agentic-counter-surveillance",
    #     description="LangSmith project name for organizing traces",
    # )

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    ollama_model: str = Field(
        default="llama3:latest",
        description="Ollama model name for LangChain integration",
    )
    ollama_timeout: float = Field(
        default=30.0, description="Timeout for Ollama requests in seconds"
    )
    ollama_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)",
    )

    # Agent configuration
    agent_max_iterations: int = Field(
        default=10, description="Maximum iterations for agent execution loops"
    )
    agent_max_execution_time: float = Field(
        default=120.0, description="Maximum execution time for agent in seconds"
    )
    agent_verbose: bool = Field(
        default=True, description="Enable verbose logging for agent operations"
    )

    # Memory configuration
    memory_enabled: bool = Field(
        default=True, description="Enable conversation memory for agents"
    )
    memory_max_tokens: int = Field(
        default=2000, description="Maximum tokens to keep in conversation memory"
    )

    # Tool configuration
    tool_timeout: float = Field(
        default=60.0, description="Timeout for individual tool executions in seconds"
    )

    # Analyzer batching
    batch_size: int = Field(
        default=8,
        description=(
            "Number of surveillance elements per LLM batch call. Override "
            "via LANGCHAIN_BATCH_SIZE. Larger values trade memory pressure "
            "on the Ollama side for fewer round-trips."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LANGCHAIN_", extra="allow"
    )

    @field_validator("ollama_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return value

    @field_validator("agent_max_iterations")
    @classmethod
    def validate_max_iterations(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Maximum iterations must be positive")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if not 1 <= value <= 64:
            raise ValueError("batch_size must be between 1 and 64")
        return value
