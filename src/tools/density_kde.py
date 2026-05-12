"""
Planar kernel-density estimation for the surveillance heatmap layer.

Replaces the previous "feed raw camera points to folium.HeatMap" path. Folium's HeatMap plugin interpolates with its own undocumented
radius/blur, which is fine for a sketch but indefensible as a published
density layer. By computing the density on a metric grid via FFT-based
KDE first and feeding the *weighted* grid into folium, the visual is
derived from a principled surface — and the same surface yields contour
polygons (``<city>_density.geojson``) that researchers can cite.

All density work happens in the local UTM zone (via
:func:`src.tools.geo_projection.project_to_utm`) so bandwidths are in
metres rather than degrees of latitude.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import HeatMap
from KDEpy import FFTKDE

from src.config.logger import logger
from src.config.settings import HeatmapSettings, HotspotSettings
from src.tools.geo_projection import project_to_utm, unproject_from_utm

# Percentile bands the density is contoured at. Nested polygons: p95 ⊂
# p90 ⊂ p75 ⊂ p50. The dashboard renders them as four filled layers
# with decreasing opacity so the eye lands on the densest cores first.
DEFAULT_CONTOUR_PERCENTILES: Tuple[int, ...] = (50, 75, 90, 95)


@dataclass
class KDEResult:
    """Outcome of a single :func:`compute_kde` call."""

    # (M, 3) array of (lat, lon, weight) triples for folium HeatMap.
    # Sub-sampled from the full grid to drop near-zero cells the visual
    # would never show anyway.
    weighted_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    # Per-percentile list of polygon rings, each ring a list of
    # ``(lat, lon)`` pairs (closed: first == last).
    contours_latlon: Dict[int, List[List[Tuple[float, float]]]] = field(
        default_factory=dict
    )

    # The density value at each percentile — useful for legend labels.
    thresholds: Dict[int, float] = field(default_factory=dict)

    # Centroid of the input points in lat/lon, for centring the folium map.
    centroid_latlon: Tuple[float, float] = (0.0, 0.0)

    epsg: str = ""

    def is_empty(self) -> bool:
        return self.weighted_points.shape[0] == 0


def _read_point_features(geojson_path: Union[str, Path]) -> List[Tuple[float, float]]:
    """Extract ``[(lat, lon), ...]`` from a Point FeatureCollection."""
    raw = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    points: List[Tuple[float, float]] = []
    for feat in raw.get("features", []):
        geom = feat.get("geometry") or {}
        if (geom.get("type") or "").lower() != "point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        points.append((float(lat), float(lon)))
    return points


def _resolve_bandwidth(bw: Union[str, float], coords_utm: np.ndarray) -> float:
    """
    Translate a bandwidth selector into a metric scalar.

    KDEpy's ``silverman`` / ``scott`` / ``ISJ`` rules are implemented
    only for 1-D data, so for our 2-D camera grid we compute the
    multivariate analogue ourselves: Silverman's rule of thumb in
    ``d`` dimensions is ``σ * n^(-1/(d+4))``. We use the
    arithmetic mean of the per-axis standard deviations as ``σ`` and
    ``d = 2``, which gives an isotropic bandwidth that's a sensible
    default for city-scale camera distributions.
    """
    if isinstance(bw, (int, float)):
        return float(bw)

    name = str(bw).lower()
    n = max(coords_utm.shape[0], 1)
    sigma = float((coords_utm.std(axis=0)).mean())
    if sigma <= 0:
        # Degenerate input (all points identical). Fall back to a
        # 50 m bandwidth so the FFT still has something to convolve.
        return 50.0

    if name in ("silverman", "scott"):
        # Both rules collapse to ``σ * n^(-1/6)`` in 2-D up to a
        # constant of order unity; use the same expression for both.
        return sigma * (n ** (-1.0 / 6.0))

    raise ValueError(
        f"Unknown KDE bandwidth selector: {bw!r}. Use a numeric value, "
        f"'silverman', or 'scott'."
    )


def _grid_dimension_for_extent(extent_m: float, resolution_m: int) -> int:
    """
    Pick a power-of-two grid dimension so FFTKDE stays efficient while
    still hitting roughly ``resolution_m`` spacing in the data area.

    Capped at 1024 (≈ 8 MB density grid) so a sprawling rural input
    can't accidentally allocate gigabytes.
    """
    raw = max(64, int(np.ceil(extent_m / max(resolution_m, 1))))
    log2 = int(np.ceil(np.log2(raw)))
    return min(1024, 2**log2)


def _polygon_rings_at_level(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    density_2d: np.ndarray,
    level: float,
    epsg: str,
) -> List[List[Tuple[float, float]]]:
    """
    Extract closed contour rings at ``level`` and project to WGS84.

    Uses matplotlib's contouring (Agg backend, figure discarded) because
    its output is stable across matplotlib versions via ``allsegs``. Open
    contours that hit the grid edge are closed by repeating the first
    vertex; the resulting polygon may have a straight clipped edge,
    which is the right behaviour for a visual layer.
    """
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(grid_x, grid_y, density_2d, levels=[level])
        rings: List[List[Tuple[float, float]]] = []
        # ``allsegs[0]`` is the list of segments at our single level.
        segments = cs.allsegs[0] if cs.allsegs else []
        for seg in segments:
            if len(seg) < 3:
                continue
            seg_latlon = unproject_from_utm(np.asarray(seg), epsg)
            ring = [(float(lat), float(lon)) for lat, lon in seg_latlon]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings.append(ring)
    finally:
        plt.close(fig)
    return rings


def compute_kde(
    geojson_path: Union[str, Path],
    settings: HotspotSettings,
    percentiles: Tuple[int, ...] = DEFAULT_CONTOUR_PERCENTILES,
) -> KDEResult:
    """
    Run planar KDE on the points in ``geojson_path``.

    :param geojson_path: Point FeatureCollection of cameras.
    :param settings: :class:`HotspotSettings` providing ``kde_bandwidth``
        and ``kde_grid_resolution_m``.
    :param percentiles: Which percentiles of the density grid to contour.
        Defaults to ``(50, 75, 90, 95)`` — the standard nested-isopleth
        set used in published surveillance density maps.
    :return: :class:`KDEResult`. Empty when fewer than three points are
        supplied (KDE on one or two points is not meaningful and the
        contour step would silently produce nothing).
    """
    points = _read_point_features(geojson_path)
    if len(points) < 3:
        return KDEResult()

    coords_utm, epsg = project_to_utm(points)
    centroid_lat = float(np.mean([p[0] for p in points]))
    centroid_lon = float(np.mean([p[1] for p in points]))

    extent_m = float(max(np.ptp(coords_utm[:, 0]), np.ptp(coords_utm[:, 1]), 1.0))
    n_grid = _grid_dimension_for_extent(extent_m, settings.kde_grid_resolution_m)

    try:
        bandwidth = _resolve_bandwidth(settings.kde_bandwidth, coords_utm)
        kde = FFTKDE(kernel="gaussian", bw=bandwidth).fit(coords_utm)
        grid, density_flat = kde.evaluate(n_grid)
    except (ValueError, RuntimeError) as e:
        # FFTKDE can fail on collinear or extremely concentrated input
        # ("data falls outside grid"). Log and degrade gracefully — the
        # caller still gets a structurally valid empty result.
        logger.warning(f"FFTKDE failed ({e}); skipping density surface")
        return KDEResult(centroid_latlon=(centroid_lat, centroid_lon), epsg=epsg)

    grid = np.asarray(grid)
    density_flat = np.asarray(density_flat)

    # Recover the axis vectors from the (N², 2) grid. KDEpy varies x
    # fastest, so unique() reproduces the two axes in ascending order.
    x_axis = np.unique(grid[:, 0])
    y_axis = np.unique(grid[:, 1])
    # Reshape (N², ) → (Ny, Nx). KDEpy's flat order has y outer, x inner
    # — matches a meshgrid built with default indexing="xy".
    density_2d = density_flat.reshape(len(y_axis), len(x_axis))

    # Percentile thresholds on the grid density values. Above-zero floor
    # so a sea of zeros (sparse outskirts) doesn't drag p50 to 0 and
    # produce a contour that swallows the entire map.
    nonzero = density_flat[density_flat > 0]
    if nonzero.size == 0:
        return KDEResult(centroid_latlon=(centroid_lat, centroid_lon), epsg=epsg)

    thresholds = {int(p): float(np.percentile(nonzero, p)) for p in percentiles}

    contours_latlon: Dict[int, List[List[Tuple[float, float]]]] = {}
    for p in percentiles:
        contours_latlon[int(p)] = _polygon_rings_at_level(
            x_axis, y_axis, density_2d, thresholds[int(p)], epsg
        )

    # For folium: weight each *camera point* by the KDE density at its
    # location, rather than dumping the entire grid in. Feeding ~10⁴
    # grid cells to ``L.heatLayer`` at radius 15 px makes every cell's
    # blob overlap its neighbours and saturates the whole urban
    # footprint to solid red — the visual stops being informative.
    # Using the camera points keeps the heatmap as sparse and crisp
    # as the legacy raw-points version while still letting the KDE
    # surface drive the colour ramp (clustered cameras get high
    # weights; isolated ones get low weights).
    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else 1.0
    x_idx = np.clip(
        ((coords_utm[:, 0] - x_axis[0]) / dx).astype(int),
        0,
        len(x_axis) - 1,
    )
    y_idx = np.clip(
        ((coords_utm[:, 1] - y_axis[0]) / dy).astype(int),
        0,
        len(y_axis) - 1,
    )
    density_at_cameras = density_2d[y_idx, x_idx]
    max_density = float(density_at_cameras.max())
    if max_density <= 0:
        weighted_points = np.empty((0, 3))
    else:
        weights = density_at_cameras / max_density
        lats = np.array([p[0] for p in points], dtype=float)
        lons = np.array([p[1] for p in points], dtype=float)
        weighted_points = np.column_stack([lats, lons, weights])

    return KDEResult(
        weighted_points=weighted_points,
        contours_latlon=contours_latlon,
        thresholds=thresholds,
        centroid_latlon=(centroid_lat, centroid_lon),
        epsg=epsg,
    )


def write_density_geojson(result: KDEResult, output_file: Union[str, Path]) -> Path:
    """
    Write ``<city>_density.geojson`` — a Polygon FeatureCollection of
    KDE contours with ``percentile`` and ``density`` properties.

    Polygons at higher percentiles nest inside lower-percentile ones;
    we emit them in ascending percentile order so a renderer that draws
    in document order ends up with the densest cores on top.
    """
    features = []
    for percentile in sorted(result.contours_latlon):
        density = result.thresholds.get(percentile, 0.0)
        for ring in result.contours_latlon[percentile]:
            ring_lonlat = [(lon, lat) for lat, lon in ring]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring_lonlat],
                    },
                    "properties": {
                        "percentile": int(percentile),
                        "density": float(density),
                    },
                }
            )

    out = {"type": "FeatureCollection", "features": features}
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def write_kde_heatmap_html(
    result: KDEResult,
    output_html: Union[str, Path],
    heatmap_settings: HeatmapSettings = HeatmapSettings(),
) -> Path:
    """
    Render a folium HeatMap from the KDE-weighted grid.

    The ``radius`` / ``blur`` knobs from :class:`HeatmapSettings` still
    apply — the upgrade replaces *what* is fed to the plugin (KDE-grid
    weighted points instead of raw cameras), not the plugin itself.
    Existing user overrides in ``.env`` keep working.
    """
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if result.is_empty():
        # Render an empty map so the consumer's iframe still resolves
        # rather than 404'ing — the chain treats this as "no heatmap
        # available" via the visualization-error path.
        m = folium.Map(location=result.centroid_latlon, zoom_start=13)
        m.save(str(out_path))
        return out_path

    m = folium.Map(location=result.centroid_latlon, zoom_start=13)
    # folium expects [lat, lon, weight] triples.
    HeatMap(
        result.weighted_points.tolist(),
        radius=heatmap_settings.radius,
        blur=heatmap_settings.blur,
    ).add_to(m)
    m.save(str(out_path))
    return out_path
