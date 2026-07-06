"""
Administrative-district aggregation of cameras by operator class.

This is the "police cameras per district" layer. It answers a question
none of the four hotspot layers do: *how are operator-classified cameras
distributed across a city's administrative units?* — the distribution the
paper's class analysis rests on.

Districts are pulled from OpenStreetMap (``boundary=administrative`` at a
configurable ``admin_level``) rather than a user-supplied boundary file,
so the layer is self-contained and city-agnostic: OSM's ``name`` tag also
gives the district names for free. Camera operators are classified via the
LLM (:mod:`src.tools.operator_classification`) into
``police`` / ``other_identified`` / ``untagged``.

Outputs (written next to the other artifacts, in EPSG:4326):

- ``<city>_districts.geojson`` — one polygon per district, carrying
  ``name``, ``admin_level``, ``total_cameras`` and the per-class counts
  plus ``untagged_share``. The citywide summary rides as a top-level
  member on the FeatureCollection.
- ``<city>_districts.csv`` — the same per-district table plus a citywide
  totals row.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config.logger import logger
from src.tools.geo_projection import pick_utm_crs, reproject_gdf
from src.tools.operator_classification import (
    OTHER_IDENTIFIED,
    POLICE,
    UNTAGGED,
    classify_camera_operator,
    classify_operators,
)

_CLASS_COLUMNS = {
    POLICE: "police_count",
    OTHER_IDENTIFIED: "other_identified_count",
    UNTAGGED: "untagged_count",
}


def fetch_admin_boundaries(
    city: str,
    admin_level: int,
    country: Optional[str] = None,
):
    """
    Fetch ``boundary=administrative`` district polygons for a city from OSM.

    :param city: City name (e.g. ``"Malmö"``).
    :param admin_level: OSM ``admin_level`` to fetch (the meaning varies
        by country — configurable, never hardcoded).
    :param country: Optional country name/code for geocoding disambiguation.
    :return: A GeoDataFrame of Polygon/MultiPolygon districts in EPSG:4326
        with ``name`` and ``admin_level`` columns. Empty when OSM has no
        matching boundaries — callers skip the layer in that case.
    """
    import osmnx as ox

    location_str = f"{city}, {country}" if country else city
    tags = {"boundary": "administrative", "admin_level": str(admin_level)}
    logger.info(
        f"Fetching OSM admin boundaries for {location_str} (admin_level={admin_level})"
    )
    try:
        gdf = ox.features_from_place(location_str, tags=tags)
    except Exception as e:
        logger.warning(f"No OSM admin boundaries for {location_str}: {e}")
        return _empty_boundaries()

    if gdf is None or gdf.empty:
        return _empty_boundaries()

    # Keep only (multi)polygons — relations occasionally resolve to points
    # or lines that can't bound anything.
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return _empty_boundaries()

    # ``features_from_place`` does NOT filter on the ``admin_level`` tag —
    # it returns every ``boundary=administrative`` relation intersecting the
    # geocoded area at once (county, municipality, districts, sub-districts).
    # Aggregating over that nested mix double-counts every camera across the
    # hierarchy, so we must keep only the requested level here; that is what
    # makes the layer a clean, non-overlapping partition.
    if "admin_level" not in gdf.columns:
        logger.warning(
            f"OSM boundaries for {location_str} carry no admin_level tag; "
            "skipping district aggregation."
        )
        return _empty_boundaries()
    gdf = gdf[gdf["admin_level"].astype(str) == str(admin_level)].copy()
    if gdf.empty:
        logger.warning(
            f"No OSM boundaries at admin_level={admin_level} for {location_str}."
        )
        return _empty_boundaries()

    if "name" not in gdf.columns:
        gdf["name"] = None
    gdf["name"] = [
        name if isinstance(name, str) and name.strip() else f"district_{i}"
        for i, name in enumerate(gdf["name"].tolist())
    ]
    gdf["admin_level"] = admin_level
    gdf = gdf[["name", "admin_level", "geometry"]].reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _empty_boundaries():
    """An empty EPSG:4326 GeoDataFrame with the district schema."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"name": [], "admin_level": [], "geometry": []},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _load_camera_points(geojson_path: Union[str, Path]):
    """
    Load camera Point features into an EPSG:4326 GeoDataFrame.

    Keeps only the ``operator`` property (flattened onto the feature by
    ``io_tools.to_geojson``) since that's all the aggregation needs.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    raw = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    geoms: List[Point] = []
    operators: List[Any] = []
    for feat in raw.get("features", []):
        geom = feat.get("geometry") or {}
        if (geom.get("type") or "").lower() != "point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        geoms.append(Point(float(lon), float(lat)))
        operators.append((feat.get("properties") or {}).get("operator"))

    return gpd.GeoDataFrame(
        {"operator": operators, "geometry": geoms},
        geometry="geometry",
        crs="EPSG:4326",
    )


def aggregate_cameras_by_district(
    geojson_path: Union[str, Path],
    boundaries,
    operator_classes: Dict[str, str],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Join cameras to districts and tally per-class counts.

    Camera points and district polygons are reprojected to a common UTM
    CRS (chosen from the camera centroid) so the point-in-polygon join is
    isotropic; the returned GeoDataFrame is back in EPSG:4326.

    :param geojson_path: Path to the enriched camera Point FeatureCollection.
    :param boundaries: District polygons in EPSG:4326 (from
        :func:`fetch_admin_boundaries`).
    :param operator_classes: ``{raw_operator: class}`` mapping from
        :func:`classify_operators`.
    :return: ``(districts_gdf, summary)``. ``districts_gdf`` has one row
        per district with the per-class counts + ``untagged_share`` and a
        WGS84 geometry; ``summary`` carries the citywide totals plus the
        ``unassigned`` count (cameras outside every district).
    """
    import geopandas as gpd

    cameras = _load_camera_points(geojson_path)
    districts = boundaries.reset_index(drop=True).copy()

    # Per-camera class from the LLM mapping (untagged handled deterministically).
    cameras["cam_class"] = [
        classify_camera_operator(op, operator_classes)
        for op in cameras["operator"].tolist()
    ]

    # Initialise per-district counters.
    for col in _CLASS_COLUMNS.values():
        districts[col] = 0

    unassigned = 0
    if not cameras.empty:
        centroid_lat = float(cameras.geometry.y.mean())
        centroid_lon = float(cameras.geometry.x.mean())
        utm = pick_utm_crs(centroid_lat, centroid_lon)

        cameras_utm = reproject_gdf(cameras, utm)
        districts_utm = reproject_gdf(districts, utm)

        joined = gpd.sjoin(
            cameras_utm,
            districts_utm[["geometry"]],
            how="left",
            predicate="within",
        )

        for _, row in joined.iterrows():
            didx = row.get("index_right")
            cls = row["cam_class"]
            if didx is None or (isinstance(didx, float) and didx != didx):
                unassigned += 1
                continue
            col = _CLASS_COLUMNS.get(cls, _CLASS_COLUMNS[OTHER_IDENTIFIED])
            districts.at[int(didx), col] += 1

    districts["total_cameras"] = (
        districts["police_count"]
        + districts["other_identified_count"]
        + districts["untagged_count"]
    )
    districts["untagged_share"] = [
        round(u / t, 4) if t else 0.0
        for u, t in zip(districts["untagged_count"], districts["total_cameras"])
    ]

    districts = districts[
        [
            "name",
            "admin_level",
            "total_cameras",
            "police_count",
            "other_identified_count",
            "untagged_count",
            "untagged_share",
            "geometry",
        ]
    ]

    total = int(districts["total_cameras"].sum())
    police = int(districts["police_count"].sum())
    other = int(districts["other_identified_count"].sum())
    untagged = int(districts["untagged_count"].sum())
    # Citywide untagged share counts unassigned cameras in the denominator
    # so it reflects the whole scan, not just those inside a district.
    citywide_total = total + unassigned
    summary = {
        "districts": int(len(districts)),
        "total_cameras": citywide_total,
        "cameras_in_districts": total,
        "police_count": police,
        "other_identified_count": other,
        "untagged_count": untagged,
        "unassigned": int(unassigned),
        "untagged_share": (
            round(untagged / citywide_total, 4) if citywide_total else 0.0
        ),
    }
    return districts, summary


def write_districts_geojson(
    districts, summary: Dict[str, Any], output_file: Union[str, Path]
) -> Path:
    """
    Write ``<city>_districts.geojson`` — district polygons + citywide summary.

    The FeatureCollection carries the citywide totals as a top-level
    ``summary`` member so the frontend can render a headline without a
    second request.
    """
    fc = json.loads(districts.to_json())
    fc["summary"] = summary
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return out_path


def write_districts_csv(
    districts, summary: Dict[str, Any], output_file: Union[str, Path]
) -> Path:
    """
    Write ``<city>_districts.csv`` — one row per district plus a citywide
    ``__ALL__`` totals row.
    """
    import csv

    columns = [
        "name",
        "admin_level",
        "total_cameras",
        "police_count",
        "other_identified_count",
        "untagged_count",
        "untagged_share",
    ]
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for _, row in districts.iterrows():
            writer.writerow({col: row[col] for col in columns})
        # Citywide totals row (name sentinel keeps it sortable/filterable).
        writer.writerow(
            {
                "name": "__ALL__",
                "admin_level": "",
                "total_cameras": summary["total_cameras"],
                "police_count": summary["police_count"],
                "other_identified_count": summary["other_identified_count"],
                "untagged_count": summary["untagged_count"],
                "untagged_share": summary["untagged_share"],
            }
        )
    return out_path


def build_district_artifacts(
    geojson_path: Union[str, Path],
    city: str,
    admin_level: int,
    llm,
    geojson_out: Union[str, Path],
    csv_out: Union[str, Path],
    country: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    End-to-end: fetch boundaries → classify operators → aggregate → write.

    :return: The citywide ``summary`` dict on success; ``None`` when OSM
        has no boundaries at ``admin_level`` (the layer is then skipped).
    """
    boundaries = fetch_admin_boundaries(city, admin_level, country=country)
    if boundaries.empty:
        logger.warning(
            f"District aggregation skipped for {city}: no OSM admin "
            f"boundaries at admin_level={admin_level}."
        )
        return None

    cameras = _load_camera_points(geojson_path)
    operator_classes = classify_operators(cameras["operator"].tolist(), llm)

    districts, summary = aggregate_cameras_by_district(
        geojson_path, boundaries, operator_classes
    )
    write_districts_geojson(districts, summary, geojson_out)
    write_districts_csv(districts, summary, csv_out)
    logger.info(
        f"District aggregation for {city}: {summary['districts']} districts, "
        f"{summary['police_count']} police cameras, "
        f"untagged_share={summary['untagged_share']}"
    )
    return summary
