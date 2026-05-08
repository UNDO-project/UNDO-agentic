from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CameraFilter(BaseModel):
    """Filter applied to camera features before routing exposure scoring.

    The enriched GeoJSON already carries ``operator``, ``surveillance_type``,
    and ``sensitive`` per feature; this model lets a route request narrow the
    cameras the routing pipeline considers without touching the analyzer.

    Defaults are an inert filter (no fields set ⇒ ``matches`` returns ``True``
    for every camera), so omitting the field on a ``RouteRequest`` preserves
    pre-Issue-#6 behaviour byte-for-byte.

    :param sensitive_only: When True, drop cameras whose ``sensitive``
                           property is not truthy.
    :param operators: When set, keep only cameras whose ``operator`` is in
                      the list. Empty list means "keep none" — pass ``None``
                      (the default) to disable the operator constraint.
    :param surveillance_types: Same semantics as ``operators`` but for the
                               ``surveillance_type`` property.
    """

    sensitive_only: bool = Field(
        default=False,
        description="Score only cameras flagged as sensitive by the analyzer.",
    )
    operators: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of operator names. Cameras whose operator "
            "is not in the list are dropped before scoring."
        ),
    )
    surveillance_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of surveillance_type values (e.g. 'camera', "
            "'guard'). Cameras with a different type are dropped."
        ),
    )

    def is_inert(self) -> bool:
        """Return ``True`` when this filter would let every camera through.

        Used as a fast-path in tools/agents so we can skip per-feature
        property reads when no filter was actually requested.
        """
        return (
            not self.sensitive_only
            and self.operators is None
            and self.surveillance_types is None
        )

    def matches(self, props: Dict[str, Any]) -> bool:
        """Apply the filter to a single feature's properties dict.

        :param props: The ``properties`` block of a GeoJSON feature.
        :return: ``True`` when the camera passes every active constraint.
        """
        if self.sensitive_only and not props.get("sensitive"):
            return False
        if self.operators is not None and props.get("operator") not in set(
            self.operators
        ):
            return False
        if self.surveillance_types is not None and props.get(
            "surveillance_type"
        ) not in set(self.surveillance_types):
            return False
        return True


class RouteRequest(BaseModel):
    """Input parameters for computing a low-surveillance route.

    The request captures the city context and the start / end locations
    expressed as latitude and longitude.

    :param city: Name of the city for which the route is computed.
    :param country: Optional ISO country code used for disambiguation.
    :param start_lat: Latitude of the starting point.
    :param start_lon: Longitude of the starting point.
    :param end_lat: Latitude of the ending point.
    :param end_lon: Longitude of the ending point.
    :param data_path: Optional override to the input data file if it differs
                      from the standard pipeline outputs.
    :param mode: Optional logical travel mode (for now primarily "walk").
    """

    city: str = Field(..., description="City name used for routing context.")
    country: Optional[str] = Field(
        default=None,
        description="Optional ISO country code for city disambiguation.",
    )
    start_lat: float = Field(..., description="Latitude of the starting point.")
    start_lon: float = Field(..., description="Longitude of the starting point.")
    end_lat: float = Field(..., description="Latitude of the ending point.")
    end_lon: float = Field(..., description="Longitude of the ending point.")
    data_path: Optional[Path] = Field(
        default=None,
        description=(
            "Optional path to an existing data file when not using the "
            "default pipeline location."
        ),
    )
    mode: Optional[str] = Field(
        default="walk", description="Logical travel mode, e.g. 'walk' or 'bike'."
    )
    camera_filter: Optional[CameraFilter] = Field(
        default=None,
        description=(
            "Optional camera filter. When provided, only matching cameras "
            "are loaded for scoring; the default (None) preserves prior "
            "behaviour and considers every camera in the dataset."
        ),
    )


class RouteMetrics(BaseModel):
    """Summary metrics describing a computed route.

    These metrics are used both for scoring candidate routes and for reporting
    the characteristics of the final chosen route.

    :param length_m: Total length of the route in metres.
    :param exposure_score: Aggregate exposure score based on cameras near
                           the path.
    :param camera_count_near_route: Total number of cameras within the
                                    configured buffer radius of the route.
    :param baseline_length_m: Length in metres of the baseline shortest route
                              used as a reference.
    :param baseline_exposure_score: Exposure score of the baseline route.
    """

    length_m: float = Field(..., description="Total route length in metres.")
    exposure_score: float = Field(
        ..., description="Aggregate exposure score along the route."
    )
    camera_count_near_route: int = Field(
        ..., description="Number of cameras falling within the route buffer."
    )
    camera_count_total: Optional[int] = Field(
        default=None,
        description=(
            "Total cameras considered for scoring after any camera filter "
            "was applied. ``None`` for legacy results that pre-date the "
            "filter; otherwise the denominator for the 'X of Y cameras "
            "considered' UI affordance."
        ),
    )
    baseline_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Length of the baseline shortest path used for comparison, "
            "expressed in metres."
        ),
    )
    baseline_exposure_score: Optional[float] = Field(
        default=None,
        description="Exposure score of the baseline shortest path, if computed.",
    )


class RouteResult(BaseModel):
    """Output artefacts and metrics for a low-surveillance route.

    :param city: City name for which the route was computed.
    :param route_geojson_path: Filesystem path to the GeoJSON representation
                               of the route.
    :param route_map_path: Filesystem path to the HTML map visualising the
                           route and nearby cameras.
    :param metrics: Computed :class:`RouteMetrics` instance.
    :param from_cache: Flag indicating whether the result was served entirely
                       from cache.
    """

    route_id: str = Field(..., description="Unique route identifier.")
    city: str = Field(..., description="City name used for routing context.")
    route_geojson_path: Path = Field(
        ..., description="Path to the GeoJSON file describing the route."
    )
    route_map_path: Path = Field(
        ..., description="Path to the HTML map with the rendered route."
    )
    metrics: RouteMetrics = Field(..., description="Summary statistics.")
    from_cache: bool = Field(
        default=False,
        description="Whether the route was served from a cached computation.",
    )
