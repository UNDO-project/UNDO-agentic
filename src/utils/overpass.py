from __future__ import annotations
from typing import Dict, Any
import textwrap
import requests
from src.config.settings import OverpassSettings
from src.utils.decorators import with_retry


def best_area_candidate(results: list[Dict[str, Any]]) -> tuple[int, str]:
    """
    Select the best boundary candidate from Nominatim search results.

    Priority:
      1. A relation with `admin_level` 8 or 9.
      2. Any relation.
      3. Any way (closed ways are area-able in Overpass).

    Nodes are never returned: Overpass does not derive areas from nodes, so a
    node candidate would yield a syntactically valid query against a
    non-existent area, returning zero elements with no error.

    :param results: A list of Nominatim result dictionaries.
    :returns: A tuple containing the OSM ID and type ('relation' or 'way').
    :raises RuntimeError: If no relation or way candidate is available.
    """
    for item in results:
        if item["osm_type"] == "relation" and item.get("extratags", {}).get(
            "admin_level"
        ) in {"8", "9"}:
            return int(item["osm_id"]), "relation"

    for item in results:
        if item["osm_type"] == "relation":
            return int(item["osm_id"]), "relation"

    for item in results:
        if item["osm_type"] == "way":
            return int(item["osm_id"]), "way"

    raise RuntimeError(
        "No usable boundary candidate (relation/way) in Nominatim results; "
        f"got {len(results)} result(s) of types "
        f"{sorted({r.get('osm_type') for r in results})}."
    )


def nominatim_city(
    osm_query: str,
    settings: OverpassSettings = OverpassSettings(),
    country: str | None = None,
) -> tuple[int, str]:
    """
    Query Nominatim to find the best OSM boundary match for a city name.

    :param settings: The Overpass base settings.
    :param osm_query: The name of the city to search for.
    :param country: Optional 2-letter ISO country code to narrow the search.
    :returns: A tuple containing the OSM ID and type.
    :raises RuntimeError: If no results are returned from Nominatim.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": osm_query,
        "format": "jsonv2",
        "limit": 100,
        "polygon_geojson": 0,
        "extratags": 1,
    }
    if country:
        params["countrycodes"] = country.lower()

    try:
        r = requests.get(url, params=params, headers=settings.headers, timeout=30)
        r.raise_for_status()
        results = r.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Nominatim request failed for {osm_query!r}") from e

    if not results:
        raise RuntimeError(f"No Nominatim result for {osm_query!r}")

    return best_area_candidate(results)


def area_id(osm_id: int, osm_type: str) -> int:
    """
    Convert an OSM object ID into an Overpass area ID.

    :param osm_id: The OSM ID of the object.
    :param osm_type: The type of the OSM object ('way' or 'relation').
    :returns: The corresponding Overpass area ID.
    :raises ValueError: If `osm_type` is not 'way' or 'relation'.
    """
    base = {
        "way": 2_400_000_000,
        "relation": 3_600_000_000,
    }.get(osm_type)
    if base is None:
        raise ValueError(
            f"Cannot derive Overpass area from osm_type={osm_type!r}; "
            "only 'relation' and 'way' are supported."
        )
    return base + osm_id


def build_query(
    city: str,
    *,
    country: str | None = None,
    settings: OverpassSettings = OverpassSettings(),
) -> str:
    """
    Build an Overpass QL query to find all `man_made=surveillance` features in a city.

    :param city: The name of the city to query.
    :param country: Optional ISO country code to disambiguate city name.
    :param settings: The Overpass base settings.
    :returns: A formatted Overpass QL query string.
    """
    try:
        osm_id, osm_type = nominatim_city(city, country=country, settings=settings)
        a_id = area_id(osm_id, osm_type)
    except Exception as e:
        raise RuntimeError(f"Failed to construct query for city '{city}': {e}") from e

    return textwrap.dedent(
        f"""
            [out:json][timeout:{settings.query_timeout}];
            area({a_id})->.searchArea;
            (
              nwr["man_made"="surveillance"](area.searchArea);
            );
            out geom;
            """
    ).strip()


@with_retry
def run_query(
    query: str, settings: OverpassSettings = OverpassSettings()
) -> Dict[str, Any]:
    """
    Submit an Overpass QL query and return the parsed JSON result.

    :param query: The Overpass query to execute.
    :param settings: The Overpass base settings.
    :returns: The JSON-decoded response from the Overpass API.
    :raises RuntimeError: If the Overpass API returns an error.
    """
    try:
        resp = requests.post(
            settings.endpoint,
            data=query.encode("utf-8"),
            timeout=settings.timeout,
            headers=settings.headers,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Overpass error {exc.response.status_code}: {exc.response.text.strip()}"
        ) from exc
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError("Failed to execute Overpass query") from e
