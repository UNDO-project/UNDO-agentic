"""
Tests for file-serving endpoints.

These tests pin the ``/api/v1/outputs/*`` route surface against the
standardized ``<city>_<artifact>.<ext>`` filename convention adopted.
The fixture city is lowercase to match the production
filename casing produced by ``SurveillanceDataCollector.scrape``
(which lowercases user input when building paths).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

TEST_CITY = "testcity"
TEST_ROUTE_ID = "test123"

# Smallest legal PNG: 8-byte signature + IHDR + IDAT + IEND. Used by tests
# that assert ``image/png`` content-type so FileResponse doesn't object to
# a malformed payload.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452000000010000000108060000001f15c4890000000d4944415478"
    "9c63000100000005000100"
    "5e3aff7c0000000049454e44ae426082"
)


@pytest.fixture
def mock_output_files(tmp_path, monkeypatch):
    """
    Create mock output files matching the analyzer's actual filenames
    and per-city directory layout (``overpass_data/<city>/``).

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :return: Path to mock per-city output directory
    """
    base_dir = tmp_path / "overpass_data"
    base_dir.mkdir()
    output_dir = base_dir / TEST_CITY
    output_dir.mkdir()

    # Enriched + raw GeoJSON
    (output_dir / f"{TEST_CITY}_enriched.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )
    (output_dir / f"{TEST_CITY}.json").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )

    # Maps (heatmap HTML, hotspots PNG)
    (output_dir / f"{TEST_CITY}_heatmap.html").write_text("<html>Heatmap</html>")
    (output_dir / f"{TEST_CITY}_hotspots.png").write_bytes(PNG_BYTES)

    # Charts: privacy + sensitivity reasons
    (output_dir / f"{TEST_CITY}_privacy.png").write_bytes(PNG_BYTES)
    (output_dir / f"{TEST_CITY}_sensitivity_reasons.png").write_bytes(PNG_BYTES)

    # Per-route files live under routes/. Mirror the production layout.
    routes_dir = output_dir / "routes"
    routes_dir.mkdir()
    (routes_dir / f"route_{TEST_ROUTE_ID}.html").write_text("<html>Route</html>")
    (routes_dir / f"route_{TEST_ROUTE_ID}.geojson").write_text(
        json.dumps({"type": "Feature", "geometry": {}, "properties": {}})
    )

    # LLM-generated city report
    (output_dir / f"{TEST_CITY}_report.md").write_text(
        "## Overview\nstubbed report.\n", encoding="utf-8"
    )

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", base_dir)

    return output_dir


def test_get_city_geojson_enriched(mock_output_files):
    """Test retrieving enriched GeoJSON for a city."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/geojson")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"

    data = response.json()
    assert data["type"] == "FeatureCollection"


def test_get_city_geojson_raw(mock_output_files):
    """Test retrieving raw scraped GeoJSON for a city."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/geojson?enriched=false")

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"


def test_get_city_geojson_not_found():
    """Test retrieving GeoJSON for non-existent city returns 404."""
    response = client.get("/api/v1/outputs/NonExistentCity/geojson")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_city_map_heatmap(mock_output_files):
    """Heatmap endpoint serves ``<city>_heatmap.html`` as text/html."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/map?map_type=heatmap")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert b"Heatmap" in response.content


def test_get_city_map_hotspots(mock_output_files):
    """Hotspots endpoint serves ``<city>_hotspots.png`` as image/png."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/map?map_type=hotspots")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_get_city_map_invalid_type(mock_output_files):
    """Test requesting invalid map type returns 400."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/map?map_type=invalid")

    assert response.status_code == 400
    assert "invalid map_type" in response.json()["detail"].lower()


def test_get_city_route_map(mock_output_files):
    """Per-route HTML lookup by route_id."""
    response = client.get(
        f"/api/v1/outputs/{TEST_CITY}/route/{TEST_ROUTE_ID}?filetype=map"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert b"Route" in response.content


def test_get_city_route_geojson(mock_output_files):
    """Per-route GeoJSON lookup by route_id."""
    response = client.get(
        f"/api/v1/outputs/{TEST_CITY}/route/{TEST_ROUTE_ID}?filetype=geojson"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"

    data = response.json()
    assert data["type"] == "Feature"


def test_get_city_route_invalid_filetype(mock_output_files):
    """Test requesting invalid route filetype returns 400."""
    response = client.get(
        f"/api/v1/outputs/{TEST_CITY}/route/{TEST_ROUTE_ID}?filetype=invalid"
    )

    assert response.status_code == 400
    assert "invalid filetype" in response.json()["detail"].lower()


def test_get_city_charts_privacy(mock_output_files):
    """Privacy chart serves ``<city>_privacy.png``."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/charts?chart=privacy")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_get_city_charts_sensitivity(mock_output_files):
    """Sensitivity chart serves the per-city sensitivity PNG."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/charts?chart=sensitivity")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_get_city_charts_invalid_type(mock_output_files):
    """Test requesting invalid chart type returns 400."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/charts?chart=invalid")

    assert response.status_code == 400
    assert "invalid chart type" in response.json()["detail"].lower()


def test_get_city_report(mock_output_files):
    """Report endpoint serves ``<city>_report.md`` as text/markdown."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "## Overview" in response.text


def test_get_city_report_not_found():
    """A city with no generated report returns 404."""
    response = client.get("/api/v1/outputs/NonExistentCity/report")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_city_files(mock_output_files):
    """Test listing all files for a city."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/list")

    assert response.status_code == 200
    data = response.json()

    assert data["city"] == TEST_CITY
    assert data["file_count"] > 0
    assert isinstance(data["files"], list)

    for file_info in data["files"]:
        assert "name" in file_info
        assert "path" in file_info
        assert "size_bytes" in file_info
        assert "modified" in file_info
        assert "type" in file_info


def test_list_city_files_excludes_cache_sidecars(tmp_path, monkeypatch):
    """
    ``*.cache.json`` sidecars written by the per-artifact
    visualisation cache must never appear in the user-facing list,
    even though they live in the same per-city directory.
    """
    base_dir = tmp_path / "overpass_data"
    output_dir = base_dir / TEST_CITY
    output_dir.mkdir(parents=True)
    (output_dir / f"{TEST_CITY}_heatmap.html").write_text("<html/>")
    (output_dir / f"{TEST_CITY}_heatmap.html.cache.json").write_text(
        '{"key": "deadbeef", "ts": "2026-05-08T00:00:00Z"}'
    )
    (output_dir / f"{TEST_CITY}_report.md").write_text("## Overview\n")
    (output_dir / f"{TEST_CITY}_report.md.cache.json").write_text(
        '{"key": "cafe", "ts": "2026-05-08T00:00:00Z"}'
    )

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", base_dir)

    response = client.get(f"/api/v1/outputs/{TEST_CITY}/list")
    assert response.status_code == 200
    names = {f["name"] for f in response.json()["files"]}

    assert f"{TEST_CITY}_heatmap.html" in names
    assert f"{TEST_CITY}_report.md" in names
    assert not any(name.endswith(".cache.json") for name in names), (
        "Sidecar files leaked into the user-facing outputs list"
    )


@pytest.fixture
def mock_hotspot_artifacts(tmp_path, monkeypatch):
    """
    Per-city directory pre-populated with the five hotspot-redesign
    artifacts: centroids, polygons, KDE contours, Gi* hexes,
    and the density-metrics JSON. Used to pin the named GET routes
    introduced in HSR#5 plus the ``/list`` integration.
    """
    base_dir = tmp_path / "overpass_data"
    output_dir = base_dir / TEST_CITY
    output_dir.mkdir(parents=True)

    empty_fc = json.dumps({"type": "FeatureCollection", "features": []})
    (output_dir / f"{TEST_CITY}_density.geojson").write_text(empty_fc)
    (output_dir / f"{TEST_CITY}_gi_star.geojson").write_text(empty_fc)
    (output_dir / f"{TEST_CITY}_hotspots.geojson").write_text(empty_fc)
    (output_dir / f"{TEST_CITY}_hotspot_polygons.geojson").write_text(empty_fc)
    (output_dir / f"{TEST_CITY}_districts.geojson").write_text(empty_fc)
    (output_dir / f"{TEST_CITY}_districts.csv").write_text(
        "name,admin_level,total_cameras,police_count,"
        "other_identified_count,untagged_count,untagged_share\n"
        "__ALL__,,0,0,0,0,0.0\n"
    )
    (output_dir / f"{TEST_CITY}_density_metrics.json").write_text(
        json.dumps(
            {
                "total_cameras": 42,
                "total_road_km": 83.5,
                "cameras_per_road_km": 0.5031,
                "area_km2": 12.4,
                "cameras_per_km2": 3.39,
                "provenance": {
                    "city": TEST_CITY,
                    "country": "SE",
                    "network_type": "walk",
                    "graph_hash": "deadbeef00000000",
                    "area_source": "convex_hull_utm",
                },
            }
        )
    )

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", base_dir)

    return output_dir


# HSR#5: parametric coverage of the five named hotspot artifact routes.
# One declarative table = one row per artifact, so adding a future layer
# is a single new entry rather than another copy-pasted test body.
HOTSPOT_ROUTE_CASES = [
    ("density.geojson", "_density.geojson", "application/geo+json"),
    ("density_metrics.json", "_density_metrics.json", "application/json"),
    ("gi_star.geojson", "_gi_star.geojson", "application/geo+json"),
    ("hotspots.geojson", "_hotspots.geojson", "application/geo+json"),
    (
        "hotspot_polygons.geojson",
        "_hotspot_polygons.geojson",
        "application/geo+json",
    ),
    ("districts.geojson", "_districts.geojson", "application/geo+json"),
    ("districts.csv", "_districts.csv", "text/csv; charset=utf-8"),
]


@pytest.mark.parametrize(
    "route_leaf,_file_suffix,content_type",
    HOTSPOT_ROUTE_CASES,
)
def test_hotspot_artifact_endpoint_serves_file(
    mock_hotspot_artifacts, route_leaf, _file_suffix, content_type
):
    """Each named hotspot artifact endpoint returns the file with the right MIME."""
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/{route_leaf}")
    assert response.status_code == 200
    assert response.headers["content-type"] == content_type


@pytest.mark.parametrize(
    "route_leaf,_file_suffix,_content_type",
    HOTSPOT_ROUTE_CASES,
)
def test_hotspot_artifact_endpoint_404_when_missing(
    tmp_path, monkeypatch, route_leaf, _file_suffix, _content_type
):
    """
    Missing artifact → 404 with a clear ``not found`` detail. Per-city
    directory exists but contains no files; this is the typical
    "ran BASIC, requested FULL artifact" path.
    """
    base_dir = tmp_path / "overpass_data"
    (base_dir / TEST_CITY).mkdir(parents=True)

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", base_dir)

    response = client.get(f"/api/v1/outputs/{TEST_CITY}/{route_leaf}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_density_metrics_endpoint_returns_headline_number(mock_hotspot_artifacts):
    """
    The density-metrics route serves valid JSON whose body carries the
    headline ``cameras_per_road_km`` number — the field the frontend
    pulls for its city-card tile.
    """
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/density_metrics.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cameras_per_road_km"] == pytest.approx(0.5031)
    assert payload["provenance"]["area_source"] == "convex_hull_utm"


def test_list_city_files_includes_all_hotspot_artifacts(mock_hotspot_artifacts):
    """
    ``/list`` is the entrypoint the dashboard uses to discover which
    layers are present for a given city. All five hotspot-redesign
    artifacts must appear by their exact on-disk filename so the
    client-side ``findFile`` substring lookup matches.
    """
    response = client.get(f"/api/v1/outputs/{TEST_CITY}/list")
    assert response.status_code == 200
    names = {f["name"] for f in response.json()["files"]}

    expected = {
        f"{TEST_CITY}_density.geojson",
        f"{TEST_CITY}_density_metrics.json",
        f"{TEST_CITY}_gi_star.geojson",
        f"{TEST_CITY}_hotspots.geojson",
        f"{TEST_CITY}_hotspot_polygons.geojson",
        f"{TEST_CITY}_districts.geojson",
        f"{TEST_CITY}_districts.csv",
    }
    assert expected <= names


def test_list_city_files_includes_distribution_charts(tmp_path, monkeypatch):
    """
    Every distribution chart adopts the standardized
    ``<city>_<artifact>.png`` convention and must appear in ``/list``
    so the dashboard's ``findFile`` lookup can discover them.
    """
    base_dir = tmp_path / "overpass_data"
    output_dir = base_dir / TEST_CITY
    output_dir.mkdir(parents=True)
    (output_dir / f"{TEST_CITY}_operator_distribution.png").write_bytes(PNG_BYTES)
    (output_dir / f"{TEST_CITY}_manufacturer_distribution.png").write_bytes(PNG_BYTES)
    (output_dir / f"{TEST_CITY}_install_timeline.png").write_bytes(PNG_BYTES)

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", base_dir)

    response = client.get(f"/api/v1/outputs/{TEST_CITY}/list")
    assert response.status_code == 200
    names = {f["name"] for f in response.json()["files"]}

    assert f"{TEST_CITY}_operator_distribution.png" in names
    assert f"{TEST_CITY}_manufacturer_distribution.png" in names
    assert f"{TEST_CITY}_install_timeline.png" in names


def test_list_city_files_no_files():
    """Listing a city with no outputs returns an empty list (HTTP 200)."""
    response = client.get("/api/v1/outputs/NonExistentCity/list")

    assert response.status_code == 200
    data = response.json()

    assert data["city"] == "NonExistentCity"
    assert data["file_count"] == 0
    assert data["files"] == []


def test_get_file_by_name(mock_output_files):
    """Test retrieving a file by its name (city is a query param)."""
    response = client.get(
        f"/api/v1/outputs/file/{TEST_CITY}_enriched.geojson",
        params={"city": TEST_CITY},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"


def test_get_file_by_name_with_dots():
    """Test that filenames with .. are rejected by validation logic."""
    filename_with_dots = "..test.json"
    assert ".." in filename_with_dots


def test_get_file_by_name_not_found(mock_output_files):
    """Test that non-existent files return 404 (with the city query param set)."""
    response = client.get(
        "/api/v1/outputs/file/nonexistent.json",
        params={"city": TEST_CITY},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_mime_type_detection():
    """Test MIME type detection for various file extensions."""
    from src.api.routes.outputs import get_mime_type

    assert get_mime_type(Path("file.json")) == "application/json"
    assert get_mime_type(Path("file.geojson")) == "application/geo+json"
    assert get_mime_type(Path("file.html")) == "text/html"
    assert get_mime_type(Path("file.png")) == "image/png"
    assert get_mime_type(Path("file.jpg")) == "image/jpeg"
    assert get_mime_type(Path("file.csv")) == "text/csv"
    assert get_mime_type(Path("file.unknown")) == "application/octet-stream"


def test_validate_path_security(tmp_path, monkeypatch):
    """Test that validate_path prevents directory traversal."""
    from src.api.routes import outputs
    from fastapi import HTTPException

    output_dir = tmp_path / "overpass_data"
    output_dir.mkdir()

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", output_dir)

    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret")

    with pytest.raises(HTTPException) as exc_info:
        outputs.validate_path(outside_file)

    assert exc_info.value.status_code == 400
    assert "directory traversal" in exc_info.value.detail.lower()


def test_validate_path_directory(tmp_path, monkeypatch):
    """Test that validate_path rejects directories."""
    from src.api.routes import outputs
    from fastapi import HTTPException

    output_dir = tmp_path / "overpass_data"
    output_dir.mkdir()

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", output_dir)

    with pytest.raises(HTTPException) as exc_info:
        outputs.validate_path(output_dir)

    assert exc_info.value.status_code == 400
    assert "not a file" in exc_info.value.detail.lower()
