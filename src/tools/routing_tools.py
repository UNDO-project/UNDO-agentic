"""
Routing tools for computing low-surveillance walking routes.

This module provides composable functions for building pedestrian networks,
snapping coordinates to graph nodes, computing paths, and scoring routes based
on surveillance camera exposure.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point

from src.config.logger import logger
from src.config.models.route_models import CameraFilter, RouteMetrics
from src.config.settings import RouteSettings


def load_camera_points(
    geojson_path: Path,
    camera_filter: Optional[CameraFilter] = None,
) -> List[Tuple[float, float]]:
    """
    Extract camera coordinates from enriched GeoJSON FeatureCollection.

    When a ``camera_filter`` is supplied, only features whose properties
    satisfy ``CameraFilter.matches`` are returned. The default
    (``camera_filter=None``) preserves pre-Issue-#6 behaviour — every Point
    feature is loaded — so existing call sites and cache keys remain stable
    until they explicitly opt in.

    :param geojson_path: Path to the enriched camera GeoJSON file.
    :param camera_filter: Optional filter applied per-feature against the
        feature's ``properties`` block before its coordinates are returned.
    :return: List of (latitude, longitude) tuples for each matching camera.
    :raises FileNotFoundError: If geojson_path does not exist.
    :raises ValueError: If GeoJSON contains no point features at all (the
        filter producing zero results does **not** raise — that's a valid
        empty exposure case the caller may want to surface to the user).
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    data = json.loads(geojson_path.read_text(encoding="utf-8"))

    apply_filter = camera_filter is not None and not camera_filter.is_inert()

    coords: List[Tuple[float, float]] = []
    point_features = 0
    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type", "").lower() != "point":
            continue
        point_features += 1
        if apply_filter and not camera_filter.matches(feat.get("properties", {})):
            continue
        # GeoJSON: [longitude, latitude]; we return (latitude, longitude).
        coords.append(
            (feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0])
        )

    if point_features == 0:
        raise ValueError(f"No point features found in GeoJSON: {geojson_path}")

    if apply_filter:
        logger.info(
            f"Loaded {len(coords)}/{point_features} camera points from "
            f"{geojson_path.name} after camera filter"
        )
    else:
        logger.info(f"Loaded {len(coords)} camera points from {geojson_path.name}")
    return coords


def count_camera_features(geojson_path: Path) -> int:
    """
    Count Point features in an enriched camera GeoJSON without filtering.

    Lets the routing agent report ``camera_count_total`` (pre-filter) next
    to the filtered set fed into scoring. Cheaper than re-loading all coords
    just for a length.

    :param geojson_path: Path to the enriched camera GeoJSON file.
    :return: Number of Point features present (0 if the file has none).
    :raises FileNotFoundError: If geojson_path does not exist.
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    return sum(
        1
        for feat in data.get("features", [])
        if feat.get("geometry", {}).get("type", "").lower() == "point"
    )


def build_pedestrian_graph(
    city: str,
    country: Optional[str],
    settings: RouteSettings,
    cache_dir: Path = Path("overpass_data/.graph_cache"),
) -> nx.MultiDiGraph:
    """
    Build or load a cached pedestrian network graph for a city.

    Uses OSMnx to download OpenStreetMap data and construct a routable graph.
    Results are cached to disk to avoid repeated downloads for the same city.

    :param city: City name.
    :param country: Optional ISO country code for disambiguation.
    :param settings: RouteSettings instance containing network_type.
    :param cache_dir: Directory for caching OSM graphs.
    :return: NetworkX MultiDiGraph representing the pedestrian network.
    :raises ValueError: If osmnx cannot find the specified location.
    """
    # Create cache key from city, country, and network type
    location_str = f"{city}, {country}" if country else city
    cache_key_input = f"{city}_{country}_{settings.network_type}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.graphml"

    # Check cache first
    if cache_file.exists():
        logger.debug(f"Loading cached graph for {location_str} from {cache_file.name}")
        return ox.load_graphml(cache_file)

    # Cache miss - download from OSM
    logger.info(
        f"Downloading pedestrian network for {location_str} "
        f"(network_type={settings.network_type})"
    )

    try:
        G = ox.graph_from_place(location_str, network_type=settings.network_type)
    except Exception as e:
        raise ValueError(f"Failed to build graph for '{location_str}': {str(e)}") from e

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_file)
    logger.info(f"Cached graph to {cache_file}")

    return G


def snap_to_graph(
    G: nx.MultiDiGraph, lat: float, lon: float, settings: RouteSettings
) -> int:
    """
    Snap a latitude/longitude coordinate to the nearest graph node.

    :param G: NetworkX graph representing the street network.
    :param lat: Latitude of the point to snap.
    :param lon: Longitude of the point to snap.
    :param settings: RouteSettings instance containing snap_distance_threshold_m.
    :return: Node ID of the nearest node in the graph.
    :raises ValueError: If the nearest node is farther than the threshold distance.
    """
    # osmnx expects (longitude, latitude) order
    nearest_node = ox.distance.nearest_nodes(G, lon, lat, return_dist=False)

    # Calculate actual distance to verify it's within threshold
    node_data = G.nodes[nearest_node]
    node_lat = node_data["y"]
    node_lon = node_data["x"]

    # Simple haversine-like distance approximation (good enough for short distances)
    # More accurate would be to use osmnx.distance.great_circle, but this is simpler
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Earth radius in meters

    lat1, lon1 = radians(lat), radians(lon)
    lat2, lon2 = radians(node_lat), radians(node_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_m = R * c

    if distance_m > settings.snap_distance_threshold_m:
        raise ValueError(
            f"Cannot snap ({lat}, {lon}) to walkable network: "
            f"nearest node is {distance_m:.1f}m away "
            f"(threshold: {settings.snap_distance_threshold_m}m)"
        )

    logger.debug(
        f"Snapped ({lat:.6f}, {lon:.6f}) to node {nearest_node} "
        f"at distance {distance_m:.1f}m"
    )

    return nearest_node


def compute_shortest_path(G: nx.MultiDiGraph, src: int, dst: int) -> List[int]:
    """
    Compute the shortest path between two nodes by distance.

    :param G: NetworkX graph representing the street network.
    :param src: Source node ID.
    :param dst: Destination node ID.
    :return: List of node IDs representing the shortest path.
    :raises ValueError: If no path exists between the nodes.
    """
    try:
        path = nx.shortest_path(G, src, dst, weight="length")
    except nx.NetworkXNoPath as e:
        raise ValueError(
            f"No walkable path exists between nodes {src} and {dst}"
        ) from e

    logger.debug(f"Computed shortest path: {len(path)} nodes from {src} to {dst}")
    return path


def generate_candidate_paths(
    G: nx.MultiDiGraph, src: int, dst: int, k: int
) -> List[List[int]]:
    """
    Generate up to k candidate paths between nodes.

    Uses NetworkX k-shortest simple paths algorithm to find alternative routes.
    If fewer than k simple paths exist, returns all available paths.
    Always returns at least one path (the shortest) if any path exists.

    :param G: NetworkX graph representing the street network.
    :param src: Source node ID.
    :param dst: Destination node ID.
    :param k: Maximum number of candidate paths to generate.
    :return: List of paths, where each path is a list of node IDs.
    :raises ValueError: If no path exists between the nodes.
    """
    try:
        # Convert MultiDiGraph to DiGraph for k-shortest paths algorithm
        # (shortest_simple_paths doesn't support multigraphs)
        # Keep minimum length edge between any two nodes
        G_simple = nx.DiGraph()

        # Copy nodes first
        G_simple.add_nodes_from(G.nodes())

        # Copy edges, keeping minimum length between any two nodes
        for u, v, data in G.edges(data=True):
            if G_simple.has_edge(u, v):
                # Keep edge with minimum length
                if data.get("length", float("inf")) < G_simple[u][v].get(
                    "length", float("inf")
                ):
                    G_simple[u][v]["length"] = data.get("length")
            else:
                G_simple.add_edge(u, v, length=data.get("length", 1.0))

        # nx.shortest_simple_paths returns a generator
        path_generator = nx.shortest_simple_paths(G_simple, src, dst, weight="length")

        # Collect up to k paths
        paths = []
        for path in path_generator:
            paths.append(path)
            if len(paths) >= k:
                break

        if not paths:
            raise ValueError(f"No paths found between nodes {src} and {dst}")

        logger.debug(
            f"Generated {len(paths)} candidate path(s) between {src} and {dst} "
            f"(requested {k})"
        )
        return paths

    except nx.NetworkXNoPath as e:
        raise ValueError(
            f"No walkable path exists between nodes {src} and {dst}"
        ) from e


def compute_exposure_for_path(
    G: nx.MultiDiGraph,
    path_nodes: List[int],
    cameras: List[Tuple[float, float]],
    settings: RouteSettings,
    camera_gdf: Optional[gpd.GeoDataFrame] = None,
) -> RouteMetrics:
    """
    Compute exposure metrics for a path based on nearby surveillance cameras.

    The function samples the path geometry at regular intervals, buffers it by
    the configured radius, and counts cameras within the buffer using spatial
    indexing for performance.

    :param G: NetworkX graph representing the street network.
    :param path_nodes: List of node IDs representing the path.
    :param cameras: List of (latitude, longitude) tuples for camera positions.
    :param settings: RouteSettings instance containing buffer_radius_m.
    :param camera_gdf: Existing GeoDataFrame containing camera positions.
    :return: RouteMetrics instance with exposure score and camera count.
    """
    # Handle edge case: zero cameras
    if len(cameras) == 0:
        logger.warning("No cameras in dataset - route has zero exposure")
        # Calculate path length
        path_length_m = _calculate_path_length(G, path_nodes)
        return RouteMetrics(
            length_m=path_length_m,
            exposure_score=0.0,
            camera_count_near_route=0,
        )

    # Build path geometry from node coordinates
    path_coords = []
    for node in path_nodes:
        node_data = G.nodes[node]
        path_coords.append((node_data["x"], node_data["y"]))  # (lon, lat)

    # Calculate total path length in meters
    path_length_m = _calculate_path_length(G, path_nodes)

    # Handle single-node path (no LineString can be created)
    if len(path_coords) < 2:
        # Use a point buffer instead
        if len(path_coords) == 1:
            point = Point(path_coords[0])
            buffer_radius_deg = settings.buffer_radius_m / 111000.0
            buffered_path = point.buffer(buffer_radius_deg)
        else:
            # Empty path - return zero metrics
            return RouteMetrics(
                length_m=0.0,
                exposure_score=0.0,
                camera_count_near_route=0,
            )
    else:
        # Create LineString for the path
        path_line = LineString(path_coords)
        buffer_radius_deg = settings.buffer_radius_m / 111000.0
        buffered_path = path_line.buffer(buffer_radius_deg)

    # Build or reuse GeoDataFrame for cameras with spatial index
    if camera_gdf is None:
        camera_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in cameras],
            crs="EPSG:4326",
        )

    # Create GeoDataFrame for buffered path
    path_gdf = gpd.GeoDataFrame([{"geometry": buffered_path}], crs="EPSG:4326")

    # Spatial join to find cameras within buffer
    cameras_in_buffer = gpd.sjoin(camera_gdf, path_gdf, predicate="within", how="inner")

    camera_count = len(cameras_in_buffer)

    # Compute exposure score
    # Simple model: cameras per km of route
    if path_length_m > 0:
        exposure_score = (camera_count / path_length_m) * 1000.0  # per km
    else:
        exposure_score = 0.0

    logger.debug(
        f"Path exposure: {camera_count} cameras along {path_length_m:.1f}m "
        f"(score: {exposure_score:.2f} cameras/km)"
    )

    return RouteMetrics(
        length_m=path_length_m,
        exposure_score=exposure_score,
        camera_count_near_route=camera_count,
    )


def _calculate_path_length(G: nx.MultiDiGraph, path_nodes: List[int]) -> float:
    """
    Calculate total path length in meters from edge weights.

    :param G: NetworkX graph representing the street network.
    :param path_nodes: List of node IDs representing the path.
    :return: Total path length in meters.
    """
    if len(path_nodes) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]

        # Get edge data (there may be multiple edges between nodes)
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            # Take the first edge's length (key=0)
            length = edge_data[0].get("length", 0.0)
            total_length += length

    return total_length


def build_route_geojson(
    G: nx.MultiDiGraph,
    path_nodes: List[int],
    metrics: RouteMetrics,
    cameras: List[Tuple[float, float]],
    city: str,
    output_path: Path,
    settings: RouteSettings,
) -> Path:
    """
    Build a GeoJSON FeatureCollection representing the computed route.

    Emits a LineString feature with route metrics in properties, including
    IDs of nearby cameras for drill-down analysis.

    :param G: NetworkX graph representing the street network.
    :param path_nodes: List of node IDs representing the route path.
    :param metrics: RouteMetrics instance with exposure and length data.
    :param cameras: List of (latitude, longitude) tuples for all cameras.
    :param city: City name for metadata.
    :param output_path: Path where the GeoJSON file will be saved.
    :param settings: RouteSettings instance containing buffer_radius_m.
    :return: Path to the saved GeoJSON file.
    """
    # Build path geometry from node coordinates
    path_coords = []
    for node in path_nodes:
        node_data = G.nodes[node]
        # GeoJSON format: [longitude, latitude]
        path_coords.append([node_data["x"], node_data["y"]])

    # Handle edge cases
    if len(path_coords) == 0:
        # Empty path - create empty FeatureCollection
        feature_collection = {"type": "FeatureCollection", "features": []}
    elif len(path_coords) == 1:
        # Single point - use Point geometry instead of LineString
        geometry = {"type": "Point", "coordinates": path_coords[0]}
        properties = {
            "city": city,
            "length_m": metrics.length_m,
            "exposure_score": metrics.exposure_score,
            "camera_count": metrics.camera_count_near_route,
            "baseline_length_m": metrics.baseline_length_m,
            "baseline_exposure_score": metrics.baseline_exposure_score,
            "nearby_camera_ids": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": geometry, "properties": properties}
            ],
        }
    else:
        # Normal case: LineString
        # Find cameras near the route for drill-down
        path_line = LineString([(lon, lat) for lon, lat in path_coords])
        buffer_radius_deg = settings.buffer_radius_m / 111000.0
        buffered_path = path_line.buffer(buffer_radius_deg)

        # Build GeoDataFrame for cameras
        camera_gdf = gpd.GeoDataFrame(
            {"camera_id": list(range(len(cameras)))},
            geometry=[Point(lon, lat) for lat, lon in cameras],
            crs="EPSG:4326",
        )

        # Create GeoDataFrame for buffered path
        path_gdf = gpd.GeoDataFrame([{"geometry": buffered_path}], crs="EPSG:4326")

        # Spatial join to find cameras within buffer
        cameras_in_buffer = gpd.sjoin(
            camera_gdf, path_gdf, predicate="within", how="inner"
        )
        nearby_camera_ids = cameras_in_buffer["camera_id"].tolist()

        # Build properties
        properties = {
            "city": city,
            "length_m": metrics.length_m,
            "exposure_score": metrics.exposure_score,
            "camera_count": metrics.camera_count_near_route,
            "baseline_length_m": metrics.baseline_length_m,
            "baseline_exposure_score": metrics.baseline_exposure_score,
            "nearby_camera_ids": nearby_camera_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Build GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": path_coords},
            "properties": properties,
        }

        feature_collection = {"type": "FeatureCollection", "features": [feature]}

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")

    logger.info(f"Saved route GeoJSON to {output_path}")
    return output_path


def render_route_map(
    route_geojson_path: Path,
    cameras_geojson_path: Path,
    output_html: Path,
) -> Path:
    """
    Render an interactive Folium map with the route and nearby cameras.

    Creates an HTML visualization showing:
    - The computed route as a colored line
    - Start and end markers
    - Surveillance cameras as semi-transparent circle markers

    :param route_geojson_path: Path to the route GeoJSON file.
    :param cameras_geojson_path: Path to the cameras GeoJSON file.
    :param output_html: Path where the HTML map will be saved.
    :return: Path to the saved HTML file.
    :raises FileNotFoundError: If input GeoJSON files don't exist.
    :raises ValueError: If route GeoJSON contains no features.
    """
    if not route_geojson_path.exists():
        raise FileNotFoundError(f"Route GeoJSON not found: {route_geojson_path}")
    if not cameras_geojson_path.exists():
        raise FileNotFoundError(f"Cameras GeoJSON not found: {cameras_geojson_path}")

    # Load route data
    route_data = json.loads(route_geojson_path.read_text(encoding="utf-8"))
    features = route_data.get("features", [])

    if not features:
        raise ValueError(f"No features in route GeoJSON: {route_geojson_path}")

    route_feature = features[0]
    route_geom = route_feature["geometry"]
    route_props = route_feature.get("properties", {})

    # Extract coordinates for centering map
    if route_geom["type"] == "LineString":
        coords = route_geom["coordinates"]
        # Calculate center
        avg_lon = sum(lon for lon, _ in coords) / len(coords)
        avg_lat = sum(lat for _, lat in coords) / len(coords)

        # Create start/end points
        start_coords = coords[0]  # [lon, lat]
        end_coords = coords[-1]
    elif route_geom["type"] == "Point":
        # Single point route
        coords = [route_geom["coordinates"]]
        avg_lon, avg_lat = route_geom["coordinates"]
        start_coords = end_coords = route_geom["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {route_geom['type']}")

    # Create base map centered on route
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

    # Add route line (if LineString)
    if route_geom["type"] == "LineString":
        # Convert coordinates from [lon, lat] to [lat, lon] for Folium
        route_coords_folium = [[lat, lon] for lon, lat in coords]

        # Add route as a colored line
        folium.PolyLine(
            locations=route_coords_folium,
            color="blue",
            weight=5,
            opacity=0.7,
            tooltip=f"Low-surveillance route: {route_props.get('length_m', 0):.0f}m, "
            f"Exposure: {route_props.get('exposure_score', 0):.2f} cameras/km",
        ).add_to(m)

    # Add start marker (green)
    folium.Marker(
        location=[start_coords[1], start_coords[0]],  # [lat, lon]
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    # Add end marker (red)
    folium.Marker(
        location=[end_coords[1], end_coords[0]],  # [lat, lon]
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Load and add cameras
    cameras_data = json.loads(cameras_geojson_path.read_text(encoding="utf-8"))
    camera_features = cameras_data.get("features", [])

    for feat in camera_features:
        if feat["geometry"]["type"].lower() == "point":
            lon, lat = feat["geometry"]["coordinates"]
            camera_props = feat.get("properties", {})

            # Add camera as semi-transparent circle
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.3,
                opacity=0.5,
                tooltip=f"Camera: {camera_props.get('surveillance:type', 'unknown')}",
            ).add_to(m)

    # Save map
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))

    logger.info(f"Saved route map to {output_html}")
    return output_html
