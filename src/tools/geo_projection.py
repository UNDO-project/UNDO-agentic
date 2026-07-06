"""
Geographic projection utilities for hotspot computations.

All clustering, density estimation, and spatial-statistics methods used
by the hotspot pipeline assume an isotropic metric coordinate system.
Working in WGS84 degrees silently produces anisotropic results — at
60° latitude one degree of longitude is roughly half the length of one
degree of latitude — which warps cluster shapes and KDE bandwidths.

This module centralises the WGS84 ↔ local UTM round-trip so every
downstream tool can request "give me my points in metres" without
reasoning about EPSG codes.
"""

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from pyproj import Transformer

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def pick_utm_crs(lat: float, lon: float) -> str:
    """
    Return the EPSG code for the UTM zone covering ``(lat, lon)``.

    Northern-hemisphere zones live at 32601–32660; southern-hemisphere
    zones at 32701–32760. Zone numbers are 1–60, derived from longitude.

    :param lat: WGS84 latitude in degrees.
    :param lon: WGS84 longitude in degrees.
    :return: EPSG identifier as ``"EPSG:NNNNN"``.
    """
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def project_to_utm(
    points: List[Tuple[float, float]],
) -> Tuple[np.ndarray, str]:
    """
    Project ``[(lat, lon), ...]`` to a local UTM zone in metres.

    The UTM zone is selected from the centroid of the input so all
    points share a single isotropic frame. For inputs spanning multiple
    zones (rare in city-scale work) the centroid choice is a reasonable
    compromise and the residual distortion at the edges is small
    enough for clustering and KDE to remain valid.

    :param points: Iterable of ``(lat, lon)`` pairs in WGS84 degrees.
    :return: ``(coords, epsg)`` where ``coords`` is an ``(N, 2)`` array
        of ``(easting, northing)`` in metres and ``epsg`` is the chosen
        UTM EPSG identifier.
    :raises ValueError: If ``points`` is empty.
    """
    if not points:
        raise ValueError("project_to_utm requires at least one point")

    arr = np.asarray(points, dtype=float)
    centroid_lat = float(arr[:, 0].mean())
    centroid_lon = float(arr[:, 1].mean())
    epsg = pick_utm_crs(centroid_lat, centroid_lon)

    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    eastings, northings = transformer.transform(arr[:, 1], arr[:, 0])
    coords = np.column_stack([eastings, northings])
    return coords, epsg


def unproject_from_utm(coords: np.ndarray, epsg: str) -> np.ndarray:
    """
    Convert ``(N, 2)`` UTM eastings/northings back to WGS84 degrees.

    :param coords: ``(N, 2)`` array of ``(easting, northing)`` in metres.
    :param epsg: EPSG identifier the UTM coords are in (e.g. ``"EPSG:32633"``).
    :return: ``(N, 2)`` array of ``(lat, lon)`` in WGS84 degrees.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)

    transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([lats, lons])


def reproject_gdf(gdf: "GeoDataFrame", target_crs: str) -> "GeoDataFrame":
    """
    Reproject a GeoDataFrame of *any* declared CRS to ``target_crs``.

    Unlike :func:`project_to_utm` (which assumes WGS84 lat/lon tuples),
    this works on GeoDataFrames carrying their own CRS — needed for the
    district layer, where administrative boundaries can arrive in a
    national grid (e.g. Sweden's EPSG:3006) rather than WGS84.

    :param gdf: A GeoDataFrame with a defined ``.crs``.
    :param target_crs: Destination CRS as an ``"EPSG:NNNNN"`` string
        (or any pyproj-accepted identifier).
    :return: The GeoDataFrame reprojected to ``target_crs``.
    :raises ValueError: If ``gdf`` has no CRS set (reprojection would be
        ambiguous).
    """
    if gdf.crs is None:
        raise ValueError(
            "reproject_gdf requires a GeoDataFrame with a defined CRS; got None"
        )
    return gdf.to_crs(target_crs)
