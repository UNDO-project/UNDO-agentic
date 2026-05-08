from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from src.config.logger import logger


# -- Per-artifact visualisation cache (Architecture Proposal #5) --
#
# Each visualisation step in ``AnalysisChain.generate_visualizations``
# writes a sidecar ``<artifact_path>.cache.json`` carrying a stable key
# derived from ``(raw_hash, vis_name, options)``. On rerun we skip the
# step if the artifact + matching sidecar are still on disk; this saves
# the (sometimes-multi-second) chart renders and the LLM-driven report
# call when nothing changed.


def visualization_cache_key(
    raw_hash: str, vis_name: str, options: Dict[str, Any]
) -> str:
    """
    Compute a stable cache key for one visualisation artifact.

    The key is a SHA-256 of the canonicalised tuple
    ``(raw_hash, vis_name, options)``. Different option values (e.g.
    a different ``top_n`` for the zone chart) produce different keys
    so the artifact is regenerated; identical inputs produce the
    same key so the cache is reused.

    :param raw_hash: Hash of the underlying enriched data — same value
        the analyzer chain uses for its own enrichment cache.
    :param vis_name: Stable artifact name (``"heatmap"``, ``"report"``,
        etc.). Used so two different artifacts derived from the same
        ``raw_hash`` don't collide.
    :param options: JSON-serialisable mapping of parameters that affect
        the rendered output. Empty for parameter-less artifacts.
    :return: 64-char hex digest.
    """
    payload = json.dumps(
        {"raw_hash": raw_hash, "vis_name": vis_name, "options": options},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sidecar_path(artifact_path: Path) -> Path:
    """Return the conventional ``<artifact>.cache.json`` next to ``artifact_path``."""
    return artifact_path.with_name(artifact_path.name + ".cache.json")


def cache_hit(artifact_path: Path, key: str) -> bool:
    """
    Return ``True`` when the artifact and its sidecar both exist and the
    sidecar's recorded key matches ``key``.

    A missing artifact, missing sidecar, malformed sidecar, or mismatched
    key all yield ``False`` so the caller re-renders. The function never
    raises — a corrupt cache is treated as a miss.
    """
    if not artifact_path.exists():
        return False
    sidecar = _sidecar_path(artifact_path)
    if not sidecar.exists():
        return False
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("key") == key


def write_sidecar(artifact_path: Path, key: str) -> None:
    """
    Write ``<artifact_path>.cache.json`` with the cache key and a UTC
    timestamp so the cache is auditable from disk.

    Failures are logged but never raised — a sidecar-write failure must
    not abort the visualisation that just succeeded; it only forces a
    re-render on the next run.
    """
    sidecar = _sidecar_path(artifact_path)
    payload = {"key": key, "ts": datetime.now(timezone.utc).isoformat()}
    try:
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as e:
        logger.warning(f"Failed to write sidecar {sidecar}: {e}")


def load_overpass_elements(path: Path | str) -> List[Dict[str, Any]]:
    """
    Read an Overpass dump and return its elements list.
    Agents can call this tool by the name "load_json".

    :param path: The Path object to the Overpass dump
    :return: The extracted list of elements
    """
    p = Path(path).expanduser().resolve()
    logger.debug(f"Loading {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("elements", [])


def save_enriched_elements(elements: List[Dict[str, Any]], path: Path | str) -> str:
    """
    Write the enriched elements list next to the source file.
    :param elements: The enriched elements
    :param path: The path of the source file
    :return: The absolute path to the new file
    """
    p = Path(path).expanduser().resolve()
    destination = p.with_name(p.stem + "_enriched.json")
    logger.debug(f"Saving {destination}")
    destination.write_text(json.dumps({"elements": elements}, indent=2), "utf-8")
    return str(destination)


def save_overpass_dump(data: Dict[str, Any], city: str, dest: Union[Path, str]) -> Path:
    """
    Save the Overpass API response to a JSON file in a specified directory.

    :param data: The JSON data to write.
    :param city: The name of the city used to name the file.
    :param dest: The output directory where the file will be saved.
    :returns: The full path to the saved file.
    """
    try:
        dest = Path(dest).expanduser()
        # if dest ends in '.json' or has a suffix, treat as full filepath
        if dest.suffix.lower() == ".json":
            filepath = dest.resolve()
        else:
            # treat as directory: ensure it exists, then name file by city
            dest.mkdir(parents=True, exist_ok=True)
            filename = f"{city.lower().replace(' ', '_')}.json"
            filepath = (dest / filename).resolve()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filepath

    except Exception as e:
        raise RuntimeError(
            f"Failed to save JSON for city '{city}' at '{dest}': {e}"
        ) from e


def to_geojson(
    enriched_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convert an enriched Overpass JSON (with `elements`) into a GeoJSON FeatureCollection.
    :param enriched_file: Path to the enriched JSON file
    :param output_file: Optional path where to write the GeoJSON. If omitted, no file is written.
    :return: A dict representing a GeoJSON FeatureCollection.
    """
    enriched_path = Path(enriched_file)
    data = json.loads(enriched_path.read_text(encoding="utf-8"))
    features: List[Dict[str, Any]] = []

    for element in data.get("elements", []):
        # Skip elements without lon, lan
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is None or lon is None:
            continue

        # Merge OSM tags and analysis metadata into properties
        props: Dict[str, Any] = {}
        props.update(element.get("tags", {}))
        # flatten analysis dict on level
        analysis = element.get("analysis", {})
        props.update(analysis)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        }

        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    if output_file:
        out_path = Path(output_file)
        out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    return geojson
