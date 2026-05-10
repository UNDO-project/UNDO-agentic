from pathlib import Path

import folium
from folium.plugins import HeatMap
import json

from src.config.settings import HeatmapSettings


def to_heatmap(
    geojson_path: Path, output_html: Path, settings: HeatmapSettings = HeatmapSettings()
) -> Path:
    """
    Read a GeoJSON FeatureCollection of Point features, build a folium HeatMap, and save to an HTML file.
    :param geojson_path: The Path object to the geojson file
    :param output_html: The Path object the output html file
    :param settings: The Heatmap settings
    :return: The html filepath
    """
    data = json.loads(geojson_path.read_text(encoding="utf-8"))

    coords = [
        (feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0])
        for feat in data.get("features", [])
        if feat["geometry"]["type"].lower() == "point"
    ]
    if not coords:
        raise RuntimeError("No point features in GeoJSON for heatmap")

    # center map at mean lat/lon
    avg_lat = sum(lat for lat, _ in coords) / len(coords)
    avg_lon = sum(lon for _, lon in coords) / len(coords)

    m = folium.Map(location=(avg_lat, avg_lon), zoom_start=13)
    HeatMap(coords, radius=settings.radius, blur=settings.blur).add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    return output_html
