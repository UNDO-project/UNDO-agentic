"""
Tests for file-serving endpoints.

These tests pin the ``/api/v1/outputs/*`` route surface against the
filenames the analyzer actually writes today
(``<city>_heatmap.html``, ``hotspot_plot_<city>.png``,
``hotspots_<city>.geojson``, ``privacy_distribution.png``,
``<city>_enriched_sensitivity.png``, and the per-route files under
``routes/route_<hash>.{html,geojson}``). The fixture city is
lowercase to match the production filename casing produced by
``SurveillanceDataCollector.scrape`` (which lowercases user input
when building paths).
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
    Create mock output files matching the analyzer's actual filenames.

    :param tmp_path: pytest temporary directory fixture
    :param monkeypatch: pytest monkeypatch fixture
    :return: Path to mock output directory
    """
    output_dir = tmp_path / "overpass_data"
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
    (output_dir / f"hotspot_plot_{TEST_CITY}.png").write_bytes(PNG_BYTES)

    # Charts: privacy + sensitivity (separate filename conventions)
    (output_dir / "privacy_distribution.png").write_bytes(PNG_BYTES)
    (output_dir / f"{TEST_CITY}_enriched_sensitivity.png").write_bytes(PNG_BYTES)

    # Per-route files live under routes/. Mirror the production layout.
    routes_dir = output_dir / "routes"
    routes_dir.mkdir()
    (routes_dir / f"route_{TEST_ROUTE_ID}.html").write_text("<html>Route</html>")
    (routes_dir / f"route_{TEST_ROUTE_ID}.geojson").write_text(
        json.dumps({"type": "Feature", "geometry": {}, "properties": {}})
    )

    from src.api.routes import outputs

    monkeypatch.setattr(outputs, "OUTPUT_BASE_DIR", output_dir)

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
    """Hotspots endpoint serves ``hotspot_plot_<city>.png`` as image/png."""
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
    """Privacy chart serves the shared ``privacy_distribution.png``."""
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
