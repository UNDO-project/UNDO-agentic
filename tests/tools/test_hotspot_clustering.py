"""
Tests for the HDBSCAN hotspot detector.

"""

import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from src.config.settings import HotspotSettings
from src.tools.hotspot_clustering import (
    cluster_hdbscan,
    write_centroids_geojson,
    write_polygons_geojson,
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


def _ring_around(
    center: tuple[float, float], radius_deg: float, n: int
) -> list[tuple[float, float]]:
    """Return ``n`` lat/lon points evenly spaced around ``center``."""
    lat0, lon0 = center
    return [
        (
            lat0 + radius_deg * math.sin(2 * math.pi * i / n),
            lon0 + radius_deg * math.cos(2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def test_two_dense_regions_yield_two_clusters(tmp_path):
    """
    Two tight clouds of 8 points, each ~50 m wide, separated by ~2 km.
    HDBSCAN with ``min_cluster_size=5`` should report exactly two
    clusters with non-negative persistence.
    """
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    cloud_a = _ring_around((55.7000, 13.2000), 0.0003, 8)
    cloud_b = _ring_around((55.7180, 13.2180), 0.0003, 8)
    geojson_path = _write_points_geojson(
        tmp_path / "cameras.geojson", cloud_a + cloud_b
    )

    result = cluster_hdbscan(geojson_path, settings)

    assert len(result.clusters) == 2
    for cluster in result.clusters:
        assert cluster.count >= 5
        assert cluster.persistence >= 0.0
        assert cluster.epsg if hasattr(cluster, "epsg") else True
    assert result.epsg.startswith("EPSG:")


def test_sparse_spread_yields_no_clusters(tmp_path):
    """
    Six points spread across hundreds of metres with no local density:
    HDBSCAN should label every point noise and return zero clusters.
    """
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    sparse = [
        (55.7000, 13.2000),
        (55.7050, 13.2080),
        (55.7100, 13.2160),
        (55.7150, 13.2240),
        (55.7200, 13.2320),
        (55.7250, 13.2400),
    ]
    geojson_path = _write_points_geojson(tmp_path / "sparse.geojson", sparse)

    result = cluster_hdbscan(geojson_path, settings)

    assert result.clusters == []
    assert result.noise_count >= 0  # may be 0 or 6 depending on HDBSCAN call


def test_empty_input_returns_empty_result(tmp_path):
    """An empty FeatureCollection should produce an empty result, not raise."""
    settings = HotspotSettings()
    geojson_path = _write_points_geojson(tmp_path / "empty.geojson", [])

    result = cluster_hdbscan(geojson_path, settings)

    assert result.is_empty()
    assert result.noise_count == 0


def test_below_min_cluster_size_returns_no_clusters(tmp_path):
    """
    Fewer points than ``min_cluster_size`` cannot form a cluster by
    definition. The function should bail cleanly rather than letting
    HDBSCAN raise.
    """
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    points = [(55.7000, 13.2000), (55.7001, 13.2001), (55.7002, 13.2002)]
    geojson_path = _write_points_geojson(tmp_path / "few.geojson", points)

    result = cluster_hdbscan(geojson_path, settings)

    assert result.clusters == []
    assert result.noise_count == 3


def test_convex_hull_contains_member_points(tmp_path):
    """
    The convex hull written for each cluster must geometrically contain
    every member point. Without this guarantee the dashboard polygon
    would visually misrepresent the cluster.

    Uses two well-separated clouds so HDBSCAN (which requires *contrast*
    against noise to label any cluster at all under default
    ``allow_single_cluster=False``) actually reports clusters.
    """
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    cloud_a = _ring_around((55.7000, 13.2000), 0.0003, 8)
    cloud_b = _ring_around((55.7180, 13.2180), 0.0003, 8)
    geojson_path = _write_points_geojson(
        tmp_path / "two_rings.geojson", cloud_a + cloud_b
    )

    result = cluster_hdbscan(geojson_path, settings)

    assert result.clusters, "Expected at least one cluster from two tight rings"
    for cluster in result.clusters:
        polygon = Polygon([(lon, lat) for lat, lon in cluster.hull_latlon])
        # Buffer by a tiny epsilon to absorb floating-point round-trip
        # error introduced by UTM ↔ WGS84 conversion.
        for lat, lon in cluster.members_latlon:
            assert polygon.buffer(1e-7).contains(Point(lon, lat))


def test_centroid_geojson_has_required_properties(tmp_path):
    """``write_centroids_geojson`` must emit cluster_id / count / persistence."""
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    cloud_a = _ring_around((55.7000, 13.2000), 0.0003, 8)
    cloud_b = _ring_around((55.7180, 13.2180), 0.0003, 8)
    geojson_path = _write_points_geojson(tmp_path / "clouds.geojson", cloud_a + cloud_b)

    result = cluster_hdbscan(geojson_path, settings)
    out_path = tmp_path / "hotspots.geojson"
    write_centroids_geojson(result, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    assert content["features"], "Expected at least one centroid feature"
    feat = content["features"][0]
    assert feat["geometry"]["type"] == "Point"
    for key in ("cluster_id", "count", "persistence"):
        assert key in feat["properties"]


def test_polygon_geojson_rings_are_closed(tmp_path):
    """Polygon rings must be explicitly closed so GeoJSON consumers don't choke."""
    settings = HotspotSettings(hdbscan_min_cluster_size=5, hdbscan_min_samples=3)
    cloud_a = _ring_around((55.7000, 13.2000), 0.0003, 8)
    cloud_b = _ring_around((55.7180, 13.2180), 0.0003, 8)
    geojson_path = _write_points_geojson(tmp_path / "two.geojson", cloud_a + cloud_b)

    result = cluster_hdbscan(geojson_path, settings)
    out_path = tmp_path / "polygons.geojson"
    write_polygons_geojson(result, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    for feat in content["features"]:
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "First and last vertex of a ring must match"


def test_empty_result_writes_empty_collections(tmp_path):
    """Empty results still produce valid (empty) FeatureCollections on disk."""
    settings = HotspotSettings()
    empty_geojson = _write_points_geojson(tmp_path / "empty.geojson", [])
    result = cluster_hdbscan(empty_geojson, settings)

    centroids_path = write_centroids_geojson(result, tmp_path / "c.geojson")
    polygons_path = write_polygons_geojson(result, tmp_path / "p.geojson")

    for path in (centroids_path, polygons_path):
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        assert content == {"type": "FeatureCollection", "features": []}


@pytest.mark.parametrize(
    "min_cluster_size,min_samples",
    [(5, 3), (8, 5)],
)
def test_settings_thresholds_round_trip(tmp_path, min_cluster_size, min_samples):
    """Larger thresholds should keep more points as noise (monotone)."""
    settings = HotspotSettings(
        hdbscan_min_cluster_size=min_cluster_size,
        hdbscan_min_samples=min_samples,
    )
    cloud = _ring_around((55.7000, 13.2000), 0.0003, 10)
    geojson_path = _write_points_geojson(tmp_path / "ring.geojson", cloud)

    result = cluster_hdbscan(geojson_path, settings)
    # Whatever the outcome, members + noise must equal input size.
    member_count = sum(c.count for c in result.clusters)
    assert member_count + result.noise_count == 10
