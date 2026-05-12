"""
Tests for the Getis-Ord Gi* hex grid.

The asserts are written against *behaviour* rather than exact numbers
where possible — z-scores depend on float-precision details of the
weights matrix, but the qualitative outcome (a tight cluster
classifies ``hot_*``; an empty input returns no cells; FDR adjustment
shrinks the rejection set) is stable across reasonable settings.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.config.settings import HotspotSettings
from src.tools.spatial_stats import (
    _bh_adjust,
    _classify,
    compute_gi_star,
    write_gi_star_geojson,
)


def _write_points_geojson(path: Path, points: List[Tuple[float, float]]) -> Path:
    """Emit a minimal Point FeatureCollection."""
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


def _make_dense_plus_scatter(
    centre: Tuple[float, float],
    n_dense: int = 200,
    n_scatter: int = 200,
    seed: int = 0,
) -> List[Tuple[float, float]]:
    """
    A tight Gaussian blob + a uniform scatter across a wider footprint.

    The scatter contributes the "expected" baseline that turns the
    blob into a statistically detectable hot-spot rather than just a
    region with cameras.
    """
    rng = random.Random(seed)
    lat0, lon0 = centre
    points: List[Tuple[float, float]] = []
    for _ in range(n_dense):
        points.append((lat0 + rng.gauss(0, 0.0003), lon0 + rng.gauss(0, 0.0003)))
    for _ in range(n_scatter):
        points.append(
            (
                lat0 + rng.uniform(-0.02, 0.02),
                lon0 + rng.uniform(-0.02, 0.02),
            )
        )
    return points


# -- Compute --


def test_dense_cluster_yields_hot_classification(tmp_path):
    """
    A dense Gaussian blob embedded in a city-wide scatter must produce
    at least one ``hot_99`` or ``hot_95`` hex with FDR-adjusted p
    below the configured threshold.
    """
    settings = HotspotSettings()
    points = _make_dense_plus_scatter((55.7, 13.2))
    geojson_path = _write_points_geojson(tmp_path / "points.geojson", points)

    grid = compute_gi_star(geojson_path, settings)

    assert not grid.is_empty()
    hot_cells = [c for c in grid.cells if c.category in {"hot_95", "hot_99"}]
    assert hot_cells, "Expected at least one hot hex from a dense cluster"
    # The hottest hex should sit close to the cluster centre.
    peak = max(grid.cells, key=lambda c: c.gi_star_z)
    assert peak.category in {"hot_95", "hot_99"}
    assert peak.p_fdr < settings.gi_star_p_threshold
    assert peak.gi_star_z > 1.96  # > 95th percentile of the standard normal
    # And lie within ~1 km of the planted centre.
    assert abs(peak.centroid_latlon[0] - 55.7) < 0.01
    assert abs(peak.centroid_latlon[1] - 13.2) < 0.01


def test_empty_input_returns_empty_grid(tmp_path):
    """An empty FeatureCollection produces no cells, no exception."""
    settings = HotspotSettings()
    geojson_path = _write_points_geojson(tmp_path / "empty.geojson", [])

    grid = compute_gi_star(geojson_path, settings)

    assert grid.is_empty()
    assert grid.cells == []


def test_single_hex_falls_back_to_not_significant(tmp_path):
    """
    All points landing in a single hex leaves Gi* with no neighbours —
    we still emit the hex but classify it ``not_significant`` rather
    than crashing or omitting it.
    """
    settings = HotspotSettings()
    # Five points within ~10m — guaranteed to share an H3 res-9 hex.
    points = [(55.70000 + i * 1e-6, 13.20000 + i * 1e-6) for i in range(5)]
    geojson_path = _write_points_geojson(tmp_path / "single.geojson", points)

    grid = compute_gi_star(geojson_path, settings)

    assert len(grid.cells) == 1
    assert grid.cells[0].category == "not_significant"
    assert grid.cells[0].count == 5


# -- BH adjustment --


def test_bh_adjust_preserves_ordering():
    """BH-adjusted q-values share the order of the raw p-values."""
    raw = np.array([0.001, 0.04, 0.20, 0.50, 0.90])
    adj = _bh_adjust(raw)
    assert np.all(np.argsort(adj) == np.argsort(raw))
    # And BH adjustment never *reduces* the smallest p (q ≥ p for every i).
    assert np.all(adj >= raw - 1e-12)


def test_bh_adjust_changes_classification_at_boundary():
    """
    With many tests, BH-adjusted thresholds are stricter than raw p.
    A hex at ``p = 0.04`` (significant raw) often becomes
    ``not_significant`` after FDR — exactly the safety the layer
    advertises to journalists who otherwise over-claim hot-spots.
    """
    # 100 raw p-values: one weak signal at 0.04, rest uniform >= 0.5.
    raw = np.concatenate([[0.04], np.linspace(0.5, 1.0, 99)])
    adj = _bh_adjust(raw)
    # Raw 0.04 < 0.05 ⇒ would be hot under naive thresholding.
    assert raw[0] < 0.05
    # But BH adjustment: q = 0.04 * 100 / 1 = 4.0 → clipped to 1.0.
    # So the same hex is no longer significant.
    assert adj[0] > 0.05


# -- Classifier --


def test_classify_handles_signs_and_thresholds():
    """Sign of z splits hot vs cold; magnitude of p_fdr splits 95 vs 99."""
    assert _classify(3.5, 0.001, 0.05) == "hot_99"
    assert _classify(2.1, 0.04, 0.05) == "hot_95"
    assert _classify(-3.5, 0.001, 0.05) == "cold_99"
    assert _classify(-2.1, 0.04, 0.05) == "cold_95"
    assert _classify(0.5, 0.5, 0.05) == "not_significant"
    # Non-finite z degrades cleanly.
    assert _classify(float("nan"), 0.001, 0.05) == "not_significant"


# -- Writer --


def test_write_geojson_emits_required_properties(tmp_path):
    """
    The Polygon FeatureCollection must carry the four properties the
    frontend layer-toggle relies on, plus a closed ring per polygon.
    """
    settings = HotspotSettings()
    points = _make_dense_plus_scatter((55.7, 13.2))
    geojson_path = _write_points_geojson(tmp_path / "points.geojson", points)
    grid = compute_gi_star(geojson_path, settings)

    out_path = tmp_path / "gi_star.geojson"
    write_gi_star_geojson(grid, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    assert content["features"], "Expected at least one hex feature"
    feat = content["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    required = {"count", "gi_star_z", "p_fdr", "category", "h3_index"}
    assert required <= set(feat["properties"].keys())
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "Polygon ring must close"


def test_write_geojson_empty_input(tmp_path):
    """An empty grid still serialises to a valid (empty) FeatureCollection."""
    settings = HotspotSettings()
    geojson_path = _write_points_geojson(tmp_path / "empty.geojson", [])
    grid = compute_gi_star(geojson_path, settings)
    out_path = tmp_path / "gi_star.geojson"
    write_gi_star_geojson(grid, out_path)

    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content == {"type": "FeatureCollection", "features": []}
