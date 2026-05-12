"""
Headline density metric: cameras per road-km.

A single citable number per city (e.g. "Lund: 0.4 cameras per road-km") is
the comparison Stanford's *Surveilling Surveillance* (2021) made canonical
for cross-city work. ``cameras / km²`` is biased by parks, water, and
industrial zones that contain few cameras *because nobody walks there*;
``cameras / road-km`` normalises by the network humans actually use.

The road length is taken from the same pedestrian graph the
``RouteFinderAgent`` caches under ``overpass_data/.graph_cache/<sha>.graphml``
(see :func:`src.tools.routing_tools.build_pedestrian_graph`). When the cache
is warm the metric is essentially free; otherwise the first call downloads
the graph and seeds the cache for routing too.

Area for the secondary ``cameras / km²`` figure is the convex hull of the
graph nodes, projected to UTM. We deliberately do *not* re-query OSM for a
city polygon — that's a second network call with its own ambiguity (city
limits vs. metropolitan area vs. nominatim's best guess), and the hull is
reproducible from the same graph hash the road-km number is anchored to.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from scipy.spatial import ConvexHull

from src.config.logger import logger
from src.config.settings import RouteSettings
from src.tools.geo_projection import project_to_utm
from src.tools.routing_tools import build_pedestrian_graph, count_camera_features


@dataclass
class DensityMetrics:
    """
    The four numbers that survive into the headline report.

    ``cameras_per_road_km`` is the load-bearing figure — the others are
    there so a reader who knows their methodology can sanity-check it
    against the cameras/km² they may already have in mind.
    """

    total_cameras: int = 0
    total_road_km: float = 0.0
    cameras_per_road_km: float = 0.0
    area_km2: float = 0.0
    cameras_per_km2: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


def _street_length_km(G: nx.MultiDiGraph) -> float:
    """
    Sum of undirected street lengths in kilometres.

    OSMnx's ``edge_length_total`` double-counts both directions of two-way
    streets — fine for routing-cost work, wrong for "how much road is
    there." We iterate the underlying undirected view so each street
    segment contributes exactly once.
    """
    if G.number_of_edges() == 0:
        return 0.0
    undirected = nx.Graph(G) if G.is_multigraph() else G.to_undirected()
    total_m = 0.0
    for _, _, data in undirected.edges(data=True):
        length = data.get("length")
        if length is None:
            continue
        total_m += float(length)
    return total_m / 1000.0


def _hull_area_km2(G: nx.MultiDiGraph) -> float:
    """
    Convex-hull area of the graph nodes in km², via UTM projection.

    A convex hull overstates area for crescent-shaped cities (coastal,
    river-bisected) but is reproducible from the graph alone — no
    second OSM call, no ambiguity over which administrative polygon
    "Lund" refers to. The number's job is back-of-envelope context for
    the road-km figure, not policy-grade land-use accounting.
    """
    coords = [
        (float(data["y"]), float(data["x"]))
        for _, data in G.nodes(data=True)
        if data.get("x") is not None and data.get("y") is not None
    ]
    if len(coords) < 3:
        return 0.0

    points_m, _ = project_to_utm(coords)
    if points_m.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(points_m)
    except Exception as e:
        logger.warning(f"Convex-hull area failed; returning 0 km²: {e}")
        return 0.0
    return float(hull.volume) / 1_000_000.0  # ``volume`` is area in 2D


def compute_road_km_density(
    cameras_geojson: Path,
    city: str,
    country: Optional[str],
    route_settings: Optional[RouteSettings] = None,
    cache_dir: Path = Path("overpass_data/.graph_cache"),
) -> DensityMetrics:
    """
    Compute the four headline numbers for one city.

    Reuses the cached pedestrian graph from
    :func:`src.tools.routing_tools.build_pedestrian_graph` when present —
    same cache key (``sha256(city + country + network_type)[:16]``), same
    file. On a cold cache this triggers an OSMnx download, which is slow
    but seeds the cache so the routing agent doesn't pay the price again.

    :param cameras_geojson: Path to the enriched-camera GeoJSON. Points
        are counted; geometry types other than ``Point`` are ignored
        (matches ``load_camera_points`` semantics).
    :param city: City name passed through to ``build_pedestrian_graph``.
    :param country: Optional ISO country code (disambiguation for OSMnx).
    :param route_settings: Reuses the routing agent's settings so the
        ``network_type`` and therefore the graph hash align. Defaults to
        ``RouteSettings()`` when not supplied.
    :param cache_dir: Graph cache directory; matches the routing agent's
        default so a single graph backs both pipelines.
    :return: Populated :class:`DensityMetrics`. A failure to count
        cameras (missing file) or to build the graph re-raises — these
        are upstream errors the chain layer surfaces via the standard
        ``visualization_errors`` path.
    """
    settings = route_settings or RouteSettings()

    total_cameras = count_camera_features(cameras_geojson)
    graph = build_pedestrian_graph(city, country, settings, cache_dir=cache_dir)

    total_road_km = _street_length_km(graph)
    area_km2 = _hull_area_km2(graph)

    cameras_per_road_km = total_cameras / total_road_km if total_road_km > 0 else 0.0
    cameras_per_km2 = total_cameras / area_km2 if area_km2 > 0 else 0.0

    import hashlib

    graph_hash = hashlib.sha256(
        f"{city}_{country}_{settings.network_type}".encode()
    ).hexdigest()[:16]

    metrics = DensityMetrics(
        total_cameras=total_cameras,
        total_road_km=round(total_road_km, 3),
        cameras_per_road_km=round(cameras_per_road_km, 4),
        area_km2=round(area_km2, 3),
        cameras_per_km2=round(cameras_per_km2, 4),
        provenance={
            "city": city,
            "country": country,
            "network_type": settings.network_type,
            "graph_hash": graph_hash,
            "area_source": "convex_hull_utm",
        },
    )
    logger.info(
        f"Density metrics for {city}: {metrics.cameras_per_road_km} cameras/road-km "
        f"({total_cameras} cameras / {metrics.total_road_km} km of road)"
    )
    return metrics


def write_density_metrics_json(metrics: DensityMetrics, out_path: Path) -> Path:
    """
    Serialise a :class:`DensityMetrics` to ``<city>_density_metrics.json``.

    The file is intentionally tiny (six numbers + provenance) — a stable
    surface for the dashboard's headline tile and the LLM report's
    opening line.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return out_path
