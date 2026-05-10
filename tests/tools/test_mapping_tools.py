import json

import pytest

from src.config.settings import HeatmapSettings
from src.tools.mapping_tools import to_heatmap


def test_to_heatmap_creates_html(geojson_file, tmp_path):
    """Test that to_heatmap creates an HTML file with expected content"""
    output_path = tmp_path / "test_heatmap.html"
    result = to_heatmap(geojson_file, output_path)

    assert result == output_path
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "L.heatLayer" in content
    assert "leaflet_heat.min.js" in content


def test_to_heatmap_with_settings(geojson_file, tmp_path):
    """Test that to_heatmap respects custom settings"""
    output_path = tmp_path / "test_heatmap_settings.html"
    settings = HeatmapSettings(radius=30, blur=20)
    result = to_heatmap(geojson_file, output_path, settings=settings)

    assert result == output_path
    content = output_path.read_text(encoding="utf-8")
    # Check if settings values are in the generated JSON configuration
    assert '"radius": 30' in content
    assert '"blur": 20' in content


def test_to_heatmap_empty_geojson(tmp_path):
    """Test that to_heatmap raises error for GeoJSON with no points"""
    empty_geojson = tmp_path / "empty.geojson"
    empty_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="No point features in GeoJSON for heatmap"):
        to_heatmap(empty_geojson, tmp_path / "output.html")
