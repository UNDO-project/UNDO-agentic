"""
Tests for administrative-district aggregation.

The spatial join, per-class tallies, citywide summary, and the
``unassigned`` (outside-all-districts) bucket are exercised against
synthetic polygons and points — no OSM/network access. A single
network-touching boundary fetch is marked ``slow`` and excluded from the
default run.
"""

import csv
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src.tools.district_aggregation import (
    aggregate_cameras_by_district,
    fetch_admin_boundaries,
    write_districts_csv,
    write_districts_geojson,
)


def _boundaries():
    """Two adjacent squares near Malmö (WGS84)."""
    d1 = Polygon([(13.0, 55.6), (13.1, 55.6), (13.1, 55.7), (13.0, 55.7)])
    d2 = Polygon([(13.1, 55.6), (13.2, 55.6), (13.2, 55.7), (13.1, 55.7)])
    return gpd.GeoDataFrame(
        {"name": ["A", "B"], "admin_level": [9, 9], "geometry": [d1, d2]},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _write_cameras(path: Path, rows):
    """rows: list of (lon, lat, operator)."""
    feats = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"operator": op},
        }
        for lon, lat, op in rows
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    return path


# Operator → class mapping the LLM would have produced (police vs other).
_CLASSES = {
    "Polismyndigheten": "police",
    "Politi": "police",
    "ACME": "other_identified",
}


def test_aggregate_counts_classes_per_district(tmp_path):
    cams = _write_cameras(
        tmp_path / "city_enriched.geojson",
        [
            (13.05, 55.65, "Polismyndigheten"),  # A: police
            (13.06, 55.66, "Politi"),  # A: police
            (13.04, 55.64, None),  # A: untagged
            (13.15, 55.65, "ACME"),  # B: other
        ],
    )
    districts, summary = aggregate_cameras_by_district(cams, _boundaries(), _CLASSES)

    a = districts[districts.name == "A"].iloc[0]
    b = districts[districts.name == "B"].iloc[0]
    assert int(a.police_count) == 2
    assert int(a.untagged_count) == 1
    assert int(a.total_cameras) == 3
    assert round(float(a.untagged_share), 4) == round(1 / 3, 4)
    assert int(b.other_identified_count) == 1
    assert int(b.total_cameras) == 1

    assert summary["police_count"] == 2
    assert summary["unassigned"] == 0
    assert summary["total_cameras"] == 4


def test_aggregate_counts_unassigned_outside_all_districts(tmp_path):
    cams = _write_cameras(
        tmp_path / "city_enriched.geojson",
        [
            (13.05, 55.65, "Polismyndigheten"),  # inside A
            (20.0, 50.0, "Polismyndigheten"),  # far outside both
        ],
    )
    districts, summary = aggregate_cameras_by_district(cams, _boundaries(), _CLASSES)
    assert summary["unassigned"] == 1
    assert summary["cameras_in_districts"] == 1
    # Citywide total counts the unassigned camera too.
    assert summary["total_cameras"] == 2
    assert int(districts["total_cameras"].sum()) == 1


def test_aggregate_empty_cameras(tmp_path):
    cams = _write_cameras(tmp_path / "city_enriched.geojson", [])
    districts, summary = aggregate_cameras_by_district(cams, _boundaries(), {})
    assert summary["total_cameras"] == 0
    assert summary["untagged_share"] == 0.0
    assert int(districts["total_cameras"].sum()) == 0


def test_writers_emit_geojson_and_csv(tmp_path):
    cams = _write_cameras(
        tmp_path / "city_enriched.geojson",
        [(13.05, 55.65, "Polismyndigheten"), (13.15, 55.65, "ACME")],
    )
    districts, summary = aggregate_cameras_by_district(cams, _boundaries(), _CLASSES)

    gout = write_districts_geojson(
        districts, summary, tmp_path / "city_districts.geojson"
    )
    fc = json.loads(Path(gout).read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert fc["summary"]["police_count"] == 1
    props = fc["features"][0]["properties"]
    for key in (
        "name",
        "admin_level",
        "total_cameras",
        "police_count",
        "other_identified_count",
        "untagged_count",
        "untagged_share",
    ):
        assert key in props

    cout = write_districts_csv(districts, summary, tmp_path / "city_districts.csv")
    rows = list(csv.DictReader(Path(cout).read_text().splitlines()))
    assert len(rows) == 3  # two districts + citywide totals row
    totals = [r for r in rows if r["name"] == "__ALL__"]
    assert len(totals) == 1
    assert int(totals[0]["police_count"]) == 1


def test_fetch_admin_boundaries_filters_to_requested_level(monkeypatch):
    """
    ``features_from_place`` returns every ``boundary=administrative`` relation
    intersecting the geocoded area — county, municipality, districts and
    sub-districts all at once (real Malmö admin_level=9 query yields levels
    4/7/8/9/10). ``fetch_admin_boundaries`` must keep only the requested
    level, or the aggregation double-counts across the nested hierarchy.
    """
    import osmnx as ox

    county = Polygon([(12.0, 55.0), (14.0, 55.0), (14.0, 56.0), (12.0, 56.0)])
    kommun = Polygon([(13.0, 55.5), (13.3, 55.5), (13.3, 55.8), (13.0, 55.8)])
    d1 = Polygon([(13.0, 55.6), (13.1, 55.6), (13.1, 55.7), (13.0, 55.7)])
    d2 = Polygon([(13.1, 55.6), (13.2, 55.6), (13.2, 55.7), (13.1, 55.7)])
    sub = Polygon([(13.05, 55.65), (13.07, 55.65), (13.07, 55.67), (13.05, 55.67)])
    mixed = gpd.GeoDataFrame(
        {
            "name": ["Skåne län", "Malmö kommun", "Norr", "Söder", "Östervärn"],
            "admin_level": ["4", "7", "9", "9", "10"],
            "geometry": [county, kommun, d1, d2, sub],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    monkeypatch.setattr(ox, "features_from_place", lambda *a, **k: mixed)

    gdf = fetch_admin_boundaries("Malmö", admin_level=9, country="Sweden")
    assert set(gdf["name"]) == {"Norr", "Söder"}  # only the level-9 pair
    assert "Skåne län" not in set(gdf["name"])  # county dropped
    assert "Malmö kommun" not in set(gdf["name"])  # municipality dropped


def test_fetch_admin_boundaries_empty_when_level_absent(monkeypatch):
    """A requested level with no matching boundary yields an empty frame."""
    import osmnx as ox

    only7 = gpd.GeoDataFrame(
        {
            "name": ["Malmö kommun"],
            "admin_level": ["7"],
            "geometry": [Polygon([(13.0, 55.5), (13.3, 55.5), (13.3, 55.8)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    monkeypatch.setattr(ox, "features_from_place", lambda *a, **k: only7)

    gdf = fetch_admin_boundaries("Malmö", admin_level=9, country="Sweden")
    assert gdf.empty


@pytest.mark.slow
def test_fetch_admin_boundaries_network():
    """Live OSM fetch — excluded from the default ``-m 'not slow'`` run."""
    gdf = fetch_admin_boundaries("Malmö", admin_level=9, country="Sweden")
    # We don't assert an exact count (OSM changes); just that we got polygons
    # in WGS84 with the expected schema when the fetch succeeds.
    if not gdf.empty:
        assert set(["name", "admin_level", "geometry"]).issubset(gdf.columns)
        assert str(gdf.crs).upper().endswith("4326")
