"""
Output file serving endpoints.

This module provides endpoints for accessing generated files (GeoJSON, maps, etc.)
with proper validation, MIME types, and error handling.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from src.config.logger import logger

router = APIRouter(prefix="/outputs")

# Base directory for all outputs
OUTPUT_BASE_DIR = Path(os.getenv("OVERPASS_DIR", "overpass_data"))


def resolve_city_base(city: str) -> Path:
    """
    If a per-city subdirectory exists (e.g., overpass_data/{city}), use it;
    otherwise fall back to the base directory.
    """
    city_dir = OUTPUT_BASE_DIR / city
    if city_dir.exists() and city_dir.is_dir():
        return city_dir
    return OUTPUT_BASE_DIR


def validate_path(file_path: Path) -> None:
    """
    Validate that a file path is safe and exists.

    Prevents directory traversal attacks and ensures files exist.

    :param file_path: Path to validate
    :raises HTTPException: 400 if path is invalid, 404 if not found
    """
    # Resolve to absolute path to prevent directory traversal
    try:
        resolved = file_path.resolve()
        base_resolved = OUTPUT_BASE_DIR.resolve()

        # Ensure the resolved path is within the output directory
        if not str(resolved).startswith(str(base_resolved)):
            raise HTTPException(
                status_code=400,
                detail="Invalid file path: directory traversal not allowed",
            )

        # Check if file exists
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Check if it's a file (not a directory)
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

    except (ValueError, OSError) as e:
        logger.error(f"Path validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")


def get_mime_type(file_path: Path) -> str:
    """
    Determine MIME type based on file extension.

    :param file_path: Path to file
    :return: MIME type string
    """
    extension = file_path.suffix.lower()

    mime_types = {
        ".json": "application/json",
        ".geojson": "application/geo+json",
        ".html": "text/html",
        ".md": "text/markdown",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }

    return mime_types.get(extension, "application/octet-stream")


@router.get("/{city}/geojson")
async def get_city_geojson(city: str, enriched: bool = True):
    """
    Get GeoJSON file for a city.

    :param city: City name
    :param enriched: Return enriched GeoJSON (default) or raw scraped data
    :return: GeoJSON file
    :raises HTTPException: 404 if file not found
    """
    base = resolve_city_base(city)
    if enriched:
        file_path = base / f"{city}_enriched.geojson"
    else:
        file_path = base / f"{city}.json"

    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type=get_mime_type(file_path),
        filename=file_path.name,
    )


@router.get("/{city}/map")
async def get_city_map(city: str, map_type: str = "heatmap"):
    """
    Get interactive map HTML for a city.

    :param city: City name
    :param map_type: Type of map (heatmap, hotspots)
    :return: HTML map file
    :raises HTTPException: 404 if file not found
    """
    # Map file paths — match the standardized ``<city>_<artifact>`` convention.
    base = resolve_city_base(city)
    map_files = {
        "heatmap": f"{city}_heatmap.html",
        "hotspots": f"{city}_hotspots.png",
    }

    if map_type not in map_files:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid map_type. Choose from: {list(map_files.keys())}",
        )

    file_path = base / map_files[map_type]
    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type=get_mime_type(file_path),
        headers={"Content-Disposition": f"inline; filename={file_path.name}"},
    )


@router.get("/{city}/route/{route_id}")
async def get_route_by_id(city: str, route_id: str, filetype: str = "map"):
    """
    Get a specific route by route_id.

    :param city: City name
    :param route_id: Unique route identifier (hash)
    :param filetype: Output format (map, geojson)
    :return: Route file (HTML map or GeoJSON)
    :raises HTTPException: 404 if file not found
    """
    base = resolve_city_base(city)
    routes_dir = base / "routes"

    if filetype == "map":
        file_path = routes_dir / f"route_{route_id}.html"
    elif filetype == "geojson":
        file_path = routes_dir / f"route_{route_id}.geojson"
    else:
        raise HTTPException(
            status_code=400, detail="Invalid filetype. Choose from: map, geojson"
        )

    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type=get_mime_type(file_path),
        headers={"Content-Disposition": f"inline; filename={file_path.name}"},
    )


@router.get("/{city}/charts")
async def get_city_charts(city: str, chart: str):
    """
    Get statistics for a city.

    :param city: City name
    :param chart: The desired chart (privacy, sensitivity)
    :return: Statistics file (PNG chart)
    :raises HTTPException: 404 if file not found
    """
    base = resolve_city_base(city)
    if chart == "sensitivity":
        file_path = base / f"{city}_sensitivity_reasons.png"
    elif chart == "privacy":
        file_path = base / f"{city}_privacy.png"
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid chart type. Choose from: privacy, sensitivity",
        )

    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type=get_mime_type(file_path),
        filename=file_path.name,
    )


@router.get("/{city}/report")
async def get_city_report(city: str):
    """
    Get the LLM-generated markdown city report.

    Served as ``text/markdown`` so the frontend (or curl) can render or
    download it directly. Looks for ``<city>_report.md`` in the city's
    output directory; returns 404 when the report wasn't generated for
    this run.

    :param city: City name
    :return: Markdown file
    :raises HTTPException: 404 if the report file does not exist
    """
    base = resolve_city_base(city)
    file_path = base / f"{city}_report.md"
    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type="text/markdown",
        filename=file_path.name,
    )


#: Sidecar files written next to every cached visualisation artifact.
#: These are internal to the per-artifact cache (Architecture Proposal
#: #5) and must never reach the user-facing outputs list.
_INTERNAL_SUFFIX = ".cache.json"


@router.get("/{city}/list")
async def list_city_files(city: str):
    """
    List all available files for a city.

    Lists every regular file in the per-city output directory except
    the internal ``*.cache.json`` sidecars. The directory itself is
    already per-city (``resolve_city_base`` returns ``<base>/<city>``
    when present), so a directory listing is safe — and necessary,
    because some artifact filenames carry the city stem at the end
    rather than the start (e.g. ``operator_distribution_<city>.png``)
    and would be missed by a prefix glob.

    :param city: City name
    :return: JSON list of available files with metadata
    """
    # ``/list`` is the only route that returns *every* file in a
    # directory rather than a specific filename, so it must not fall
    # back to ``OUTPUT_BASE_DIR`` when the per-city directory is
    # missing — that would leak files from sibling cities or stray
    # flat-layout artifacts. Specific-filename routes (``/geojson``,
    # ``/map``, etc.) keep their fallback intact via ``resolve_city_base``.
    city_dir = OUTPUT_BASE_DIR / city
    city_files: list = []
    if city_dir.exists() and city_dir.is_dir():
        for file_path in city_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.name.endswith(_INTERNAL_SUFFIX):
                continue
            stat = file_path.stat()
            city_files.append(
                {
                    "name": file_path.name,
                    "path": f"/outputs/{file_path.name}",
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": get_mime_type(file_path),
                }
            )

    return JSONResponse(
        content={
            "city": city,
            "file_count": len(city_files),
            "files": sorted(city_files, key=lambda x: x["modified"], reverse=True),
        }
    )


@router.get("/file/{filename}")
async def get_file_by_name(city: str, filename: str):
    """
    Get any file by filename from the output directory.

    This is a generic endpoint for accessing any generated file.

    :param city: The name of the city for which the file was generated
    :param filename: Name of the file to retrieve
    :return: Requested file
    :raises HTTPException: 404 if file not found
    """
    # Validate filename doesn't contain path separators
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = resolve_city_base(city)
    file_path = base / filename
    validate_path(file_path)

    return FileResponse(
        path=file_path,
        media_type=get_mime_type(file_path),
        filename=file_path.name,
    )
