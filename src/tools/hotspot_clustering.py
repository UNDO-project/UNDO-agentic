"""
HDBSCAN-based hotspot detection for surveillance camera point sets.

Replaces the previous DBSCAN-on-degrees implementation. HDBSCAN
picks density locally rather than globally, so a single threshold no
longer fuses dense downtowns into one blob while losing sparser
suburban clusters. It also yields a per-cluster persistence score that
the dashboard surfaces as a confidence band on each polygon.

Inputs are projected to the local UTM zone via
:func:`src.tools.geo_projection.project_to_utm` before clustering, so
``min_cluster_size`` and the implicit minimum spanning tree are
expressed in metres rather than degrees of latitude.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import hdbscan
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.config.settings import HotspotSettings
from src.tools.geo_projection import project_to_utm, unproject_from_utm


@dataclass
class Cluster:
    """One discrete hotspot detected by HDBSCAN."""

    cluster_id: int
    members_latlon: List[Tuple[float, float]]
    centroid_latlon: Tuple[float, float]
    count: int
    persistence: float
    hull_latlon: List[Tuple[float, float]]


@dataclass
class ClusterResult:
    """Outcome of a single :func:`cluster_hdbscan` call."""

    clusters: List[Cluster] = field(default_factory=list)
    noise_count: int = 0
    epsg: str = ""

    def is_empty(self) -> bool:
        return not self.clusters


def _read_point_features(geojson_path: Union[str, Path]) -> List[Tuple[float, float]]:
    """Extract ``[(lat, lon), ...]`` from a Point FeatureCollection."""
    raw = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    points: List[Tuple[float, float]] = []
    for feat in raw.get("features", []):
        geom = feat.get("geometry") or {}
        if (geom.get("type") or "").lower() != "point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        points.append((float(lat), float(lon)))
    return points


def _convex_hull_latlon(member_utm: np.ndarray, epsg: str) -> List[Tuple[float, float]]:
    """
    Compute the convex hull of a cluster in UTM and return its vertices
    back in lat/lon. Falls back to the member points themselves when
    fewer than three points (a degenerate hull) are available.
    """
    if len(member_utm) < 3:
        latlon = unproject_from_utm(member_utm, epsg)
        return [(float(lat), float(lon)) for lat, lon in latlon]

    try:
        hull = ConvexHull(member_utm)
    except QhullError:
        # Collinear points → no 2-D hull. Return the input as a polyline
        # of vertices so the consumer still has something to draw.
        latlon = unproject_from_utm(member_utm, epsg)
        return [(float(lat), float(lon)) for lat, lon in latlon]

    vertex_coords = member_utm[hull.vertices]
    latlon = unproject_from_utm(vertex_coords, epsg)
    return [(float(lat), float(lon)) for lat, lon in latlon]


def cluster_hdbscan(
    geojson_path: Union[str, Path],
    settings: HotspotSettings,
) -> ClusterResult:
    """
    Cluster the points in ``geojson_path`` with HDBSCAN.

    :param geojson_path: Path to a Point FeatureCollection of cameras.
    :param settings: :class:`HotspotSettings` providing
        ``hdbscan_min_cluster_size`` and ``hdbscan_min_samples``.
    :return: :class:`ClusterResult` with one :class:`Cluster` per
        discovered group plus a noise count.
    """
    points = _read_point_features(geojson_path)
    if not points:
        return ClusterResult()

    coords_utm, epsg = project_to_utm(points)

    if len(points) < settings.hdbscan_min_cluster_size:
        # Not enough points for HDBSCAN to form even one cluster — every
        # point is implicitly noise. Return cleanly rather than raising.
        return ClusterResult(noise_count=len(points), epsg=epsg)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_utm)
    persistences = getattr(clusterer, "cluster_persistence_", np.array([]))

    clusters: List[Cluster] = []
    for raw_label in sorted(set(labels)):
        if raw_label == -1:
            continue
        member_mask = labels == raw_label
        member_utm = coords_utm[member_mask]
        member_latlon_arr = unproject_from_utm(member_utm, epsg)
        member_latlon: List[Tuple[float, float]] = [
            (float(lat), float(lon)) for lat, lon in member_latlon_arr
        ]

        centroid_utm = member_utm.mean(axis=0, keepdims=True)
        centroid_latlon_arr = unproject_from_utm(centroid_utm, epsg)
        centroid_latlon = (
            float(centroid_latlon_arr[0, 0]),
            float(centroid_latlon_arr[0, 1]),
        )

        persistence = (
            float(persistences[raw_label])
            if 0 <= raw_label < len(persistences)
            else 0.0
        )

        clusters.append(
            Cluster(
                cluster_id=int(raw_label),
                members_latlon=member_latlon,
                centroid_latlon=centroid_latlon,
                count=int(member_mask.sum()),
                persistence=persistence,
                hull_latlon=_convex_hull_latlon(member_utm, epsg),
            )
        )

    noise_count = int((labels == -1).sum())
    return ClusterResult(clusters=clusters, noise_count=noise_count, epsg=epsg)


def write_centroids_geojson(
    result: ClusterResult, output_file: Union[str, Path]
) -> Path:
    """
    Write ``<city>_hotspots.geojson`` — a Point FeatureCollection of
    cluster centroids with ``cluster_id``, ``count``, ``persistence``.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                # GeoJSON is (lon, lat); centroid is stored (lat, lon).
                "coordinates": [c.centroid_latlon[1], c.centroid_latlon[0]],
            },
            "properties": {
                "cluster_id": c.cluster_id,
                "count": c.count,
                "persistence": c.persistence,
            },
        }
        for c in result.clusters
    ]
    out = {"type": "FeatureCollection", "features": features}
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def write_polygons_geojson(
    result: ClusterResult, output_file: Union[str, Path]
) -> Path:
    """
    Write ``<city>_hotspot_polygons.geojson`` — a Polygon
    FeatureCollection of convex hulls with the same properties as the
    centroid file.

    Clusters with fewer than three points (degenerate hull) are emitted
    with whatever vertices they have; the GeoJSON polygon is still
    closed by repeating the first coordinate so consumers can render
    them as zero-area markers without special-casing.
    """
    features = []
    for c in result.clusters:
        if not c.hull_latlon:
            continue
        # GeoJSON polygons need at least one linear ring closed by
        # repeating the first vertex.
        ring = [(lon, lat) for lat, lon in c.hull_latlon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "cluster_id": c.cluster_id,
                    "count": c.count,
                    "persistence": c.persistence,
                },
            }
        )
    out = {"type": "FeatureCollection", "features": features}
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path
