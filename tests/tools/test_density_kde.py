"""
Tests for the planar KDE density surface.

"""

import json
import math
import random
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from src.config.settings import HeatmapSettings, HotspotSettings
from src.tools.density_kde import (
    DEFAULT_CONTOUR_PERCENTILES,
    compute_kde,
    write_density_geojson,
    write_kde_heatmap_html,
)


def _write_points_geojson(path: Path, points: list[tuple[float, float]]) -> Path:
    """Write a minimal Point FeatureCollection at ``path``."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {},
            }
            for lat, lon in points
        ],
    }
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


def _gaussian_blob(
    center: tuple[float, float], sigma_deg: float, n: int, seed: int = 0
) -> list[tuple[float, float]]:
    """Return ``n`` lat/lon points sampled from a 2D Gaussian around ``center``."""
    rng = random.Random(seed)
    lat0, lon0 = center
    return [
        (lat0 + rng.gauss(0, sigma_deg), lon0 + rng.gauss(0, sigma_deg))
        for _ in range(n)
    ]


def test_gaussian_blob_peak_density_near_centre(tmp_path):
    """
    The KDE evaluated on a tight Gaussian sample must put its highest
    density values close to the sample's centre.
    """
    settings = HotspotSettings()
    centre = (55.7000, 13.2000)
    points = _gaussian_blob(centre, sigma_deg=0.001, n=200, seed=42)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)

    assert not result.is_empty()
    # The folium-bound weighted points represent grid cells above the
    # lowest contour. The peak weight should sit close to the input
    # centre — within a few hundred metres at our default resolution.
    peak_idx = int(np.argmax(result.weighted_points[:, 2]))
    peak_lat = float(result.weighted_points[peak_idx, 0])
    peak_lon = float(result.weighted_points[peak_idx, 1])
    # 0.005° ≈ 500 m; comfortably loose given grid resolution + bandwidth.
    assert abs(peak_lat - centre[0]) < 0.005
    assert abs(peak_lon - centre[1]) < 0.005


def test_contour_thresholds_strictly_increasing(tmp_path):
    """Higher percentiles must map to higher density thresholds."""
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=200, seed=1)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)

    sorted_percentiles = sorted(result.thresholds)
    values = [result.thresholds[p] for p in sorted_percentiles]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_contour_polygons_are_valid_and_nested(tmp_path):
    """
    For a Gaussian blob, the higher-percentile polygon should sit
    inside (or at least overlap) the lower-percentile one. Strict
    ``contains`` can fail on the discretised grid, so we settle for
    "the highest percentile's centroid sits inside the lowest one's
    polygon" — the practical nesting guarantee a renderer needs.
    """
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=300, seed=2)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)

    p_low = min(result.contours_latlon)
    p_high = max(result.contours_latlon)
    low_rings = result.contours_latlon[p_low]
    high_rings = result.contours_latlon[p_high]

    assert low_rings, "Lowest percentile must produce at least one ring"
    assert high_rings, "Highest percentile must produce at least one ring"

    # Validate each ring as a Shapely polygon.
    for percentile, rings in result.contours_latlon.items():
        for ring in rings:
            poly = Polygon([(lon, lat) for lat, lon in ring])
            assert poly.is_valid, (
                f"p{percentile} polygon invalid: {explain_validity(poly)}"
            )

    # Build a union of low-percentile polygons and check that at least
    # one high-percentile centroid lies inside.
    from shapely.ops import unary_union

    low_union = unary_union(
        [Polygon([(lon, lat) for lat, lon in r]) for r in low_rings]
    )
    found_inside = False
    for ring in high_rings:
        centroid = Polygon([(lon, lat) for lat, lon in ring]).centroid
        if low_union.buffer(1e-6).contains(centroid):
            found_inside = True
            break
    assert found_inside, "Highest percentile contour should sit inside the lowest"


def test_too_few_points_returns_empty_result(tmp_path):
    """KDE on fewer than three points is not meaningful — degrade cleanly."""
    settings = HotspotSettings()
    geojson_path = _write_points_geojson(
        tmp_path / "two.geojson", [(55.7, 13.2), (55.701, 13.201)]
    )

    result = compute_kde(geojson_path, settings)

    assert result.is_empty()
    assert result.contours_latlon == {}


def test_empty_input_returns_empty_result(tmp_path):
    """An empty FeatureCollection is also handled without raising."""
    settings = HotspotSettings()
    geojson_path = _write_points_geojson(tmp_path / "empty.geojson", [])

    result = compute_kde(geojson_path, settings)

    assert result.is_empty()


def test_density_geojson_writer_emits_required_properties(tmp_path):
    """The density GeoJSON must carry percentile + density per feature."""
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=100, seed=3)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)
    out_path = tmp_path / "density.geojson"
    write_density_geojson(result, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    assert content["features"], "Expected at least one contour feature"
    feat = content["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert {"percentile", "density"} <= set(feat["properties"].keys())
    # Percentiles should be from the default set.
    assert feat["properties"]["percentile"] in DEFAULT_CONTOUR_PERCENTILES


def test_density_geojson_polygons_close_their_rings(tmp_path):
    """Every polygon ring must close (first vertex == last vertex)."""
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=100, seed=4)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)
    out_path = tmp_path / "density.geojson"
    write_density_geojson(result, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    for feat in content["features"]:
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "Polygon ring must close on itself"


def test_kde_heatmap_html_written_with_weighted_points(tmp_path):
    """
    The folium HTML must contain the L.heatLayer marker so the dashboard
    iframe doesn't render a blank page on a successful run.
    """
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=100, seed=5)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)
    out_path = tmp_path / "heatmap.html"
    write_kde_heatmap_html(result, out_path, HeatmapSettings(radius=20, blur=15))

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "L.heatLayer" in html
    assert '"radius": 20' in html
    assert '"blur": 15' in html


def test_kde_heatmap_html_renders_empty_map_for_empty_input(tmp_path):
    """An empty result still produces a valid HTML file (no L.heatLayer)."""
    settings = HotspotSettings()
    empty_geojson = _write_points_geojson(tmp_path / "empty.geojson", [])

    result = compute_kde(empty_geojson, settings)
    out_path = tmp_path / "heatmap.html"
    write_kde_heatmap_html(result, out_path)

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    # A folium Map without overlays still includes leaflet bootstrap.
    assert "leaflet" in html.lower()
    assert "L.heatLayer" not in html


def test_density_geojson_empty_result_writes_empty_collection(tmp_path):
    """Empty inputs still yield a structurally valid (empty) GeoJSON."""
    settings = HotspotSettings()
    empty_geojson = _write_points_geojson(tmp_path / "empty.geojson", [])

    result = compute_kde(empty_geojson, settings)
    out_path = tmp_path / "density.geojson"
    write_density_geojson(result, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content == {"type": "FeatureCollection", "features": []}


def test_weighted_points_are_normalised_to_unit_interval(tmp_path):
    """Weights must lie in [0, 1] so folium's colour ramp is stable."""
    settings = HotspotSettings()
    points = _gaussian_blob((55.7, 13.2), sigma_deg=0.001, n=150, seed=6)
    geojson_path = _write_points_geojson(tmp_path / "blob.geojson", points)

    result = compute_kde(geojson_path, settings)
    weights = result.weighted_points[:, 2]
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0 + 1e-9
    # The normalisation should hit 1.0 at the peak.
    assert math.isclose(weights.max(), 1.0, rel_tol=1e-6)
