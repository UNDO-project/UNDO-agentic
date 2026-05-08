import json
from pathlib import Path

from src.tools.io_tools import (
    cache_hit,
    load_overpass_elements,
    save_enriched_elements,
    to_geojson,
    visualization_cache_key,
    write_sidecar,
)

ELEMENTS = [
    {"id": 1, "lat": 10.0, "lon": 20.0, "tags": {"foo": "bar"}, "analysis": {"a": 1}},
    {"id": 2},  # should be skipped by to_geojson (no lat/lon)
    {"id": 3, "lat": 0.0, "lon": 0.0, "tags": {}, "analysis": {}},
]


def test_load_overpass_elements(tmp_path):
    # prepare a dump with a few elements
    dump = {"elements": [{"id": "x"}, {"id": "y"}]}
    dump_path = tmp_path / "dump.json"
    dump_path.write_text(json.dumps(dump), encoding="utf-8")

    loaded = load_overpass_elements(dump_path)
    assert isinstance(loaded, list)
    assert loaded == dump["elements"]


def test_save_enriched_elements(tmp_path):
    # create a fake source file so save_enriched_elements can build its name
    source = tmp_path / "city.json"
    source.write_text("{}", encoding="utf-8")

    enriched = [{"id": 42}, {"id": 43}]
    out = save_enriched_elements(enriched, source)
    out_path = Path(out)

    # file should exist and contain exactly {"elements": enriched}
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed == {"elements": enriched}


def test_to_geojson_without_writing(tmp_path):
    # write an enriched JSON file with ELEMENTS
    enriched = {"elements": ELEMENTS}
    enriched_path = tmp_path / "in.json"
    enriched_path.write_text(json.dumps(enriched), encoding="utf-8")

    geo = to_geojson(enriched_path)
    # should be a FeatureCollection
    assert geo["type"] == "FeatureCollection"
    # only two elements have both lat and lon
    assert len(geo["features"]) == 2

    # check geometry & properties merging
    feat = geo["features"][0]
    assert feat["geometry"] == {"type": "Point", "coordinates": [20.0, 10.0]}
    # props should include both tags and analysis
    assert feat["properties"]["foo"] == "bar"
    assert feat["properties"]["a"] == 1


def test_to_geojson_with_writing(tmp_path):
    enriched = {"elements": ELEMENTS}
    enriched_path = tmp_path / "data.json"
    enriched_path.write_text(json.dumps(enriched), encoding="utf-8")
    out_geojson = tmp_path / "out.geojson"

    geo = to_geojson(enriched_path, out_geojson)
    # file was written
    assert out_geojson.exists()
    # contents match returned dict
    written = json.loads(out_geojson.read_text(encoding="utf-8"))
    assert written == geo
    assert written["type"] == "FeatureCollection"


# -- Per-artifact visualisation cache (Architecture Proposal #5) --


def test_visualization_cache_key_is_deterministic():
    """Same inputs → same key, regardless of dict ordering."""
    k1 = visualization_cache_key("abc123", "heatmap", {"top_n": 10, "x": 1})
    k2 = visualization_cache_key("abc123", "heatmap", {"x": 1, "top_n": 10})
    assert k1 == k2


def test_visualization_cache_key_changes_with_raw_hash():
    k1 = visualization_cache_key("abc", "heatmap", {})
    k2 = visualization_cache_key("xyz", "heatmap", {})
    assert k1 != k2


def test_visualization_cache_key_changes_with_vis_name():
    k1 = visualization_cache_key("abc", "heatmap", {})
    k2 = visualization_cache_key("abc", "hotspots", {})
    assert k1 != k2


def test_visualization_cache_key_changes_with_options():
    """Acceptance criterion: changing top_n must invalidate the cache."""
    k1 = visualization_cache_key("abc", "zone_sensitivity", {"top_n": 10})
    k2 = visualization_cache_key("abc", "zone_sensitivity", {"top_n": 5})
    assert k1 != k2


def test_cache_hit_false_when_artifact_missing(tmp_path):
    artifact = tmp_path / "heatmap.html"  # never created
    write_sidecar(artifact, "key123")  # sidecar without artifact
    assert cache_hit(artifact, "key123") is False


def test_cache_hit_false_when_sidecar_missing(tmp_path):
    artifact = tmp_path / "heatmap.html"
    artifact.write_text("<html/>", encoding="utf-8")
    assert cache_hit(artifact, "key123") is False


def test_cache_hit_false_on_key_mismatch(tmp_path):
    artifact = tmp_path / "heatmap.html"
    artifact.write_text("<html/>", encoding="utf-8")
    write_sidecar(artifact, "old-key")
    assert cache_hit(artifact, "new-key") is False


def test_cache_hit_true_on_artifact_plus_matching_sidecar(tmp_path):
    artifact = tmp_path / "heatmap.html"
    artifact.write_text("<html/>", encoding="utf-8")
    write_sidecar(artifact, "key123")
    assert cache_hit(artifact, "key123") is True


def test_cache_hit_false_on_corrupt_sidecar(tmp_path):
    """A malformed sidecar JSON is treated as a miss, not an error."""
    artifact = tmp_path / "heatmap.html"
    artifact.write_text("<html/>", encoding="utf-8")
    sidecar = artifact.with_name(artifact.name + ".cache.json")
    sidecar.write_text("not-json{", encoding="utf-8")
    assert cache_hit(artifact, "key123") is False


def test_write_sidecar_records_key_and_timestamp(tmp_path):
    artifact = tmp_path / "heatmap.html"
    artifact.write_text("<html/>", encoding="utf-8")
    write_sidecar(artifact, "key123")

    sidecar = artifact.with_name(artifact.name + ".cache.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["key"] == "key123"
    assert "ts" in payload and isinstance(payload["ts"], str)
