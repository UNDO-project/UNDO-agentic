"""
Tests for the WGS84 - UTM projection utility used by the hotspot
pipeline.

The contract that downstream tools rely on:

- ``pick_utm_crs`` returns the correct EPSG for both hemispheres and
  the international date line.
- ``project_to_utm`` produces a single isotropic frame (one EPSG, in
  metres) for any city-scale point set.
- The round-trip ``project_to_utm`` → ``unproject_from_utm`` is accurate
  to within a metre for points near the chosen zone's centroid, which
  is the precision the clustering and KDE layers require.
"""

import numpy as np
import pytest

from src.tools.geo_projection import (
    pick_utm_crs,
    project_to_utm,
    reproject_gdf,
    unproject_from_utm,
)


class TestPickUTMCRS:
    def test_northern_hemisphere_western_europe(self):
        # Lund, Sweden ≈ (55.7, 13.2) → UTM zone 33N
        assert pick_utm_crs(55.7, 13.2) == "EPSG:32633"

    def test_northern_hemisphere_north_america(self):
        # New York City ≈ (40.7, -74.0) → UTM zone 18N
        assert pick_utm_crs(40.7, -74.0) == "EPSG:32618"

    def test_southern_hemisphere(self):
        # Buenos Aires ≈ (-34.6, -58.4) → UTM zone 21S
        assert pick_utm_crs(-34.6, -58.4) == "EPSG:32721"

    def test_equator_is_northern(self):
        # Convention: lat == 0 falls into the northern series.
        assert pick_utm_crs(0.0, 0.0) == "EPSG:32631"

    def test_extreme_west_clamped_to_zone_1(self):
        assert pick_utm_crs(0.0, -180.0) == "EPSG:32601"

    def test_extreme_east_clamped_to_zone_60(self):
        # Just under the dateline.
        assert pick_utm_crs(0.0, 179.99) == "EPSG:32660"


class TestProjectToUTM:
    def test_returns_metric_coordinates_and_epsg(self):
        # Two points roughly 1 km apart in Lund.
        points = [(55.7000, 13.2000), (55.7090, 13.2000)]
        coords, epsg = project_to_utm(points)

        assert epsg == "EPSG:32633"
        assert coords.shape == (2, 2)

        distance_m = float(np.linalg.norm(coords[1] - coords[0]))
        # 0.009° latitude ≈ 1000 m. Allow ±5 m for projection drift.
        assert 995 <= distance_m <= 1005

    def test_isotropic_distance_matches_haversine(self):
        # 1 km north of Lund and 1 km east of Lund, projected, should
        # both be ~1 km from the origin point — degrees would have made
        # the east/west leg shorter.
        origin = (55.7000, 13.2000)
        north = (55.7090, 13.2000)  # ~1 km N
        east = (55.7000, 13.2160)  # ~1 km E at this latitude

        coords, _ = project_to_utm([origin, north, east])
        north_dist = float(np.linalg.norm(coords[1] - coords[0]))
        east_dist = float(np.linalg.norm(coords[2] - coords[0]))

        # Both legs within 1 % of each other proves we're not in degrees.
        assert abs(north_dist - east_dist) / north_dist < 0.01

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            project_to_utm([])

    def test_single_point(self):
        coords, epsg = project_to_utm([(55.7, 13.2)])
        assert coords.shape == (1, 2)
        assert epsg == "EPSG:32633"


class TestUnprojectFromUTM:
    def test_round_trip_accuracy_under_one_metre(self):
        # Cities chosen to span hemispheres and a few zones each.
        cities = {
            "Lund": [(55.7000, 13.2000), (55.7050, 13.2050)],
            "NYC": [(40.7128, -74.0060), (40.7580, -73.9855)],
            "Buenos Aires": [(-34.6037, -58.3816), (-34.5800, -58.4000)],
        }
        for name, points in cities.items():
            coords, epsg = project_to_utm(points)
            roundtrip = unproject_from_utm(coords, epsg)

            assert roundtrip.shape == (len(points), 2)
            for original, recovered in zip(points, roundtrip):
                # 1e-5 degrees ≈ 1 m at any latitude.
                assert abs(original[0] - recovered[0]) < 1e-5, name
                assert abs(original[1] - recovered[1]) < 1e-5, name

    def test_empty_array_returns_empty(self):
        result = unproject_from_utm(np.empty((0, 2)), "EPSG:32633")
        assert result.shape == (0, 2)


class TestReprojectGdf:
    """``reproject_gdf`` handles arbitrary-CRS GeoDataFrames (district layer)."""

    def test_reproject_from_national_grid_to_wgs84(self):
        import geopandas as gpd
        from shapely.geometry import Point

        # A point in SWEREF99 TM (EPSG:3006, Sweden) — the CRS the Malmö
        # RegSO example ships in — should land near Malmö in WGS84.
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(373000, 6164000)]},
            geometry="geometry",
            crs="EPSG:3006",
        )
        out = reproject_gdf(gdf, "EPSG:4326")
        assert str(out.crs).upper().endswith("4326")
        lon, lat = out.geometry.iloc[0].x, out.geometry.iloc[0].y
        assert 12.5 < lon < 13.5
        assert 55.3 < lat < 55.9

    def test_missing_crs_raises(self):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry")
        with pytest.raises(ValueError):
            reproject_gdf(gdf, "EPSG:4326")
