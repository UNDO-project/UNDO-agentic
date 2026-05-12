"""
Tests for the cameras-per-road-km headline metric.

The road-km figure is the load-bearing number for the report; the tests
pin its arithmetic on a synthetic graph (no OSMnx call) and verify the
graph cache is honoured when present, so the second pipeline pass for
the same city doesn't trigger a network fetch.
"""

import json
from pathlib import Path
from unittest.mock import patch
from typing import List, Tuple

import networkx as nx
import pytest

from src.config.settings import RouteSettings
from src.tools.density_metrics import (
    DensityMetrics,
    _hull_area_km2,
    _street_length_km,
    compute_road_km_density,
    write_density_metrics_json,
)


def _write_points_geojson(path: Path, points: List[Tuple[float, float]]) -> Path:
    """Minimal Point FeatureCollection — same helper shape the other tests use."""
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


def _synthetic_pedestrian_graph() -> nx.MultiDiGraph:
    """
    Build a MultiDiGraph whose undirected street length totals exactly
    10,000 m — 10 km. Ten 1-km edges arranged in a line. Coordinates
    sit near Lund so the UTM projection picks a meaningful zone.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    base_lat, base_lon = 55.70, 13.20
    for i in range(11):
        # ~1 km between consecutive longitudes at 55.7°N is ~0.0157°,
        # but we set edge["length"] explicitly so the geometry only
        # needs to give the convex hull something to chew on.
        g.add_node(i, x=base_lon + i * 0.016, y=base_lat)
    for i in range(10):
        g.add_edge(i, i + 1, length=1000.0)
    return g


# -- pure-function tests --


def test_street_length_undirected_counts_each_segment_once():
    """A MultiDiGraph with both forward + back edges still totals 10 km."""
    g = _synthetic_pedestrian_graph()
    # Add reverse edges so the directed total is 20 km — the
    # undirected sum must stay at 10 km.
    for u, v in [(0, 1), (1, 2)]:
        g.add_edge(v, u, length=1000.0)
    assert _street_length_km(g) == pytest.approx(10.0)


def test_street_length_empty_graph_returns_zero():
    """An empty graph yields 0 km, not a crash."""
    assert _street_length_km(nx.MultiDiGraph()) == 0.0


def test_hull_area_returns_positive_for_spread_nodes():
    """Spread node positions yield a non-zero convex-hull area."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    # ~1 km square near Lund
    g.add_node(0, x=13.20, y=55.70)
    g.add_node(1, x=13.21, y=55.70)
    g.add_node(2, x=13.21, y=55.71)
    g.add_node(3, x=13.20, y=55.71)
    area = _hull_area_km2(g)
    # 1km x 1km ≈ 1 km², allow loose tolerance for UTM distortion +
    # the fact that 0.01° of longitude at 55.7°N is ~0.63 km.
    assert 0.4 < area < 1.5


def test_hull_area_too_few_nodes_returns_zero():
    """Fewer than three nodes can't form a hull."""
    g = nx.MultiDiGraph()
    g.add_node(0, x=13.2, y=55.7)
    g.add_node(1, x=13.21, y=55.7)
    assert _hull_area_km2(g) == 0.0


# -- compute_road_km_density --


@patch("src.tools.density_metrics.build_pedestrian_graph")
def test_compute_density_uses_cached_graph_no_osm_fetch(mock_build, tmp_path):
    """
    Five cameras + a 10 km synthetic graph must give 0.5 cameras/road-km.

    The mock stands in for ``build_pedestrian_graph``, which itself
    short-circuits to the cache when the file exists. Patching here
    keeps the assertion focused on the arithmetic rather than the
    cache plumbing (which is already covered in test_routing_tools).
    """
    mock_build.return_value = _synthetic_pedestrian_graph()

    points = [(55.700 + 1e-4 * i, 13.200 + 1e-4 * i) for i in range(5)]
    geojson = _write_points_geojson(tmp_path / "cameras.geojson", points)

    metrics = compute_road_km_density(
        geojson,
        city="Lund",
        country="SE",
        route_settings=RouteSettings(),
        cache_dir=tmp_path / ".graph_cache",
    )

    assert metrics.total_cameras == 5
    assert metrics.total_road_km == pytest.approx(10.0)
    assert metrics.cameras_per_road_km == pytest.approx(0.5)
    # Nodes share a latitude, so the hull degenerates to a thin sliver —
    # the projection's float fuzz gives a small but non-zero area. The
    # substantive arithmetic (road-km) is what the metric pins on.
    assert metrics.area_km2 >= 0.0
    assert metrics.cameras_per_km2 >= 0.0
    assert metrics.provenance["city"] == "Lund"
    assert metrics.provenance["country"] == "SE"
    assert metrics.provenance["network_type"] == "walk"
    assert metrics.provenance["area_source"] == "convex_hull_utm"
    # No fresh OSM call — the mock replaces the network entry point.
    mock_build.assert_called_once()


@patch("src.tools.density_metrics.build_pedestrian_graph")
def test_compute_density_zero_road_km_avoids_divide_by_zero(mock_build, tmp_path):
    """An empty graph must yield 0.0 cameras/road-km, not ZeroDivisionError."""
    mock_build.return_value = nx.MultiDiGraph()

    points = [(55.7, 13.2)]
    geojson = _write_points_geojson(tmp_path / "cameras.geojson", points)

    metrics = compute_road_km_density(
        geojson,
        city="Lund",
        country="SE",
        route_settings=RouteSettings(),
        cache_dir=tmp_path / ".graph_cache",
    )

    assert metrics.cameras_per_road_km == 0.0
    assert metrics.cameras_per_km2 == 0.0


@patch("src.tools.density_metrics.build_pedestrian_graph")
def test_compute_density_reuses_existing_graphml_no_download(
    mock_build, tmp_path, monkeypatch
):
    """
    When a cached ``.graphml`` already lives at the expected path, the
    routing-tools builder loads it instead of downloading. We assert
    that by patching ``ox.graph_from_place`` and confirming it is
    never invoked.
    """
    # Defer the patch chain — we need the *real* build_pedestrian_graph
    # to exercise its cache hit, with osmnx's download path forbidden.
    mock_build.side_effect = None
    from src.tools import routing_tools

    cache_dir = tmp_path / ".graph_cache"
    cache_dir.mkdir()

    import hashlib

    cache_key = hashlib.sha256(b"Lund_SE_walk").hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.graphml"
    cache_file.touch()

    fake_graph = _synthetic_pedestrian_graph()
    # Bypass our outer mock for this case — call the real builder.
    mock_build.side_effect = (
        lambda *_args, **_kwargs: routing_tools.build_pedestrian_graph(  # noqa: E501
            *_args, **_kwargs
        )
    )
    monkeypatch.setattr(routing_tools.ox, "load_graphml", lambda _p: fake_graph)
    monkeypatch.setattr(
        routing_tools.ox,
        "graph_from_place",
        lambda *_a, **_kw: pytest.fail(
            "graph_from_place should not be called when cache is warm"
        ),
    )

    points = [(55.7, 13.2)] * 3
    geojson = _write_points_geojson(tmp_path / "cameras.geojson", points)

    metrics = compute_road_km_density(
        geojson,
        city="Lund",
        country="SE",
        route_settings=RouteSettings(),
        cache_dir=cache_dir,
    )

    assert metrics.total_road_km == pytest.approx(10.0)


# -- writer --


def test_write_density_metrics_json_round_trips(tmp_path):
    """The JSON written matches the dataclass field-for-field."""
    metrics = DensityMetrics(
        total_cameras=42,
        total_road_km=83.5,
        cameras_per_road_km=0.5031,
        area_km2=12.4,
        cameras_per_km2=3.39,
        provenance={
            "city": "Lund",
            "country": "SE",
            "network_type": "walk",
            "graph_hash": "deadbeef00000000",
            "area_source": "convex_hull_utm",
        },
    )
    out_path = tmp_path / "lund_density_metrics.json"
    write_density_metrics_json(metrics, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total_cameras"] == 42
    assert payload["cameras_per_road_km"] == 0.5031
    assert payload["provenance"]["graph_hash"] == "deadbeef00000000"
    assert set(payload.keys()) == {
        "total_cameras",
        "total_road_km",
        "cameras_per_road_km",
        "area_km2",
        "cameras_per_km2",
        "provenance",
    }
