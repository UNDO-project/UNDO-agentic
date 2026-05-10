"""
Getis-Ord Gi* hot-spot analysis on an H3 hexagonal grid.

This is the statistical layer of the four-layer hotspot architecture.
HDBSCAN says "here is a group" and KDE says "this region is dense";
Gi* says "this hex is *statistically* hotter than its neighbours under a permutation null,
even after correcting for multiple testing." That's the layer
researchers and journalists recognise from ArcGIS / QGIS "Hot Spot
Analysis" — the move that lets a paper cite a per-hex z-score and
FDR-adjusted p-value rather than an opaque cluster polygon.

Pipeline:

1. Bin cameras to H3 hexes at ``settings.h3_resolution``.
2. Build a distance-band spatial weights matrix on hex centroids
   (in UTM metres) so adjacency is isotropic regardless of latitude.
3. Run :class:`esda.getisord.G_Local` (the Gi* variant — ``star=True``
   includes each hex in its own neighbourhood, which is the convention
   for ArcGIS-style hot-spot analysis).
4. Apply Benjamini–Hochberg FDR adjustment to the per-hex p-values.
5. Classify each hex as ``hot_99`` / ``hot_95`` / ``not_significant``
   / ``cold_95`` / ``cold_99`` based on the FDR-adjusted p and the
   sign of the z-score.

The output is a Polygon FeatureCollection with one feature per
non-empty hex, carrying ``count``, ``gi_star_z``, ``p_fdr``, and
``category`` properties.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h3
import numpy as np
from esda.getisord import G_Local
from libpysal.weights import DistanceBand

from src.config.settings import HotspotSettings
from src.tools.geo_projection import project_to_utm


# Hot/cold band cutoffs. ``p_threshold`` (default 0.05 from settings)
# is the 95% level; the 99% level is fixed at 0.01 — the convention
# every "Hot Spot Analysis" colour ramp uses.
_P_THRESHOLD_99 = 0.01


@dataclass
class HexCell:
    """One non-empty H3 hex with its Gi* outcome."""

    h3_index: str
    boundary_latlon: List[Tuple[float, float]]
    centroid_latlon: Tuple[float, float]
    count: int
    gi_star_z: float
    p_fdr: float
    category: str


@dataclass
class HexGrid:
    """Outcome of a single :func:`compute_gi_star` call."""

    cells: List[HexCell] = field(default_factory=list)
    epsg: str = ""
    h3_resolution: int = 0

    def is_empty(self) -> bool:
        return not self.cells


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


def _bin_to_hexes(points: List[Tuple[float, float]], resolution: int) -> Dict[str, int]:
    """Return ``{h3_index: count}`` for the given lat/lon points."""
    counts: Dict[str, int] = {}
    for lat, lon in points:
        idx = h3.latlng_to_cell(lat, lon, resolution)
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjustment.

    For sorted p-values ``p_(1) ≤ … ≤ p_(n)``: adjusted
    ``q_(i) = min_{k ≥ i} p_(k) * n / k``, clipped to ``[0, 1]``. The
    monotone step ensures the adjusted series is non-decreasing — a
    requirement for a sensible "smaller p_fdr ⇒ stronger evidence"
    interpretation downstream.

    Returns adjusted p-values in the *original* input order so
    callers can pair them with the row that produced each test
    without re-sorting.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1)
    adj_sorted = ranked * n / ranks
    # Enforce monotone non-decrease across sorted positions.
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def _classify(z: float, p_fdr: float, p_threshold: float) -> str:
    """Map ``(z, p_fdr)`` to the ArcGIS-style hot/cold category."""
    if not np.isfinite(z) or not np.isfinite(p_fdr):
        return "not_significant"
    if p_fdr <= _P_THRESHOLD_99:
        return "hot_99" if z > 0 else "cold_99"
    if p_fdr <= p_threshold:
        return "hot_95" if z > 0 else "cold_95"
    return "not_significant"


def _distance_band_threshold_m(resolution: int) -> float:
    """
    Pick a distance-band threshold in metres that connects each hex to
    its immediate H3 ring-1 neighbours.

    Adjacent H3 hexes have centroids ~``√3 * edge_length`` apart. We
    pad by 30% so floating-point noise around that distance doesn't
    sever expected neighbours from the weights matrix.
    """
    edge_m = h3.average_hexagon_edge_length(resolution, unit="km") * 1000.0
    return float(edge_m * np.sqrt(3.0) * 1.3)


def compute_gi_star(
    geojson_path: Union[str, Path],
    settings: HotspotSettings,
) -> HexGrid:
    """
    Run Gi* hot-spot analysis on the cameras in ``geojson_path``.

    :param geojson_path: Path to a Point FeatureCollection of cameras.
    :param settings: :class:`HotspotSettings` providing
        ``h3_resolution`` and ``gi_star_p_threshold``.
    :return: :class:`HexGrid` with one :class:`HexCell` per non-empty
        hex, carrying its Gi* z-score and FDR-adjusted p-value.
    """
    points = _read_point_features(geojson_path)
    if not points:
        return HexGrid(h3_resolution=settings.h3_resolution)

    counts = _bin_to_hexes(points, settings.h3_resolution)
    if not counts:
        return HexGrid(h3_resolution=settings.h3_resolution)

    # A single non-empty hex has no neighbours and Gi* is undefined.
    # Emit it as ``not_significant`` so the artifact is still useful
    # as a "where are the cameras" footprint without the stat layer.
    if len(counts) < 2:
        only_idx = next(iter(counts))
        boundary = list(h3.cell_to_boundary(only_idx))
        lat, lon = h3.cell_to_latlng(only_idx)
        cell = HexCell(
            h3_index=only_idx,
            boundary_latlon=[(float(la), float(lo)) for la, lo in boundary],
            centroid_latlon=(float(lat), float(lon)),
            count=int(counts[only_idx]),
            gi_star_z=float("nan"),
            p_fdr=1.0,
            category="not_significant",
        )
        return HexGrid(cells=[cell], h3_resolution=settings.h3_resolution)

    hex_indices = list(counts.keys())
    centroids_latlon = [h3.cell_to_latlng(idx) for idx in hex_indices]
    centroids_utm, epsg = project_to_utm(
        [(float(lat), float(lon)) for lat, lon in centroids_latlon]
    )
    y = np.array([counts[idx] for idx in hex_indices], dtype=float)

    threshold = _distance_band_threshold_m(settings.h3_resolution)
    weights = DistanceBand(
        centroids_utm,
        threshold=threshold,
        binary=True,
        silence_warnings=True,
    )

    g = G_Local(
        y,
        weights,
        transform="B",
        permutations=999,
        star=True,
        seed=42,
    )

    z_scores = np.asarray(g.Zs, dtype=float)
    # We use the analytical normal-approximation p-value
    # (``g.p_norm``, derived from |z|) rather than the empirical
    # ``p_sim``. The permutation-based p floors at ``1/(perms+1)``,
    # which after BH-adjustment for hundreds of hexes can never
    # cross the 0.05 threshold even for clear hot-spots. The
    # analytical p has no such floor; ArcGIS Hot Spot Analysis
    # uses the same approach.
    raw_p = np.asarray(g.p_norm, dtype=float)
    p_fdr = _bh_adjust(raw_p)

    cells: List[HexCell] = []
    for i, idx in enumerate(hex_indices):
        boundary = list(h3.cell_to_boundary(idx))
        lat, lon = centroids_latlon[i]
        z_raw = float(z_scores[i])
        # Disconnected hexes (no neighbours under the distance band)
        # produce a non-finite z. ``_classify`` treats those as
        # ``not_significant``; we coerce the stored value to 0.0 so
        # the GeoJSON stays JSON-valid (NaN is not legal JSON).
        cells.append(
            HexCell(
                h3_index=idx,
                boundary_latlon=[(float(la), float(lo)) for la, lo in boundary],
                centroid_latlon=(float(lat), float(lon)),
                count=int(y[i]),
                gi_star_z=z_raw if np.isfinite(z_raw) else 0.0,
                p_fdr=float(p_fdr[i]),
                category=_classify(
                    z_raw,
                    float(p_fdr[i]),
                    settings.gi_star_p_threshold,
                ),
            )
        )

    return HexGrid(cells=cells, epsg=epsg, h3_resolution=settings.h3_resolution)


def write_gi_star_geojson(grid: HexGrid, output_file: Union[str, Path]) -> Path:
    """
    Write ``<city>_gi_star.geojson`` — a Polygon FeatureCollection of
    H3 hexes with ``count``, ``gi_star_z``, ``p_fdr``, ``category``.
    """
    features = []
    for cell in grid.cells:
        # GeoJSON expects (lon, lat); h3 returns (lat, lon).
        ring = [(lon, lat) for lat, lon in cell.boundary_latlon]
        if not ring:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "h3_index": cell.h3_index,
                    "count": cell.count,
                    "gi_star_z": cell.gi_star_z,
                    "p_fdr": cell.p_fdr,
                    "category": cell.category,
                },
            }
        )
    out = {"type": "FeatureCollection", "features": features}
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path
