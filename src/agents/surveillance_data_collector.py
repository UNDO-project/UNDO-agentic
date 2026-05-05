"""
Deterministic surveillance data collector.

Drives a four-step workflow (build query, cache lookup, run query, save) without
an LLM in the loop.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config.logger import logger
from src.config.settings import LangChainSettings
from src.memory.store import MemoryStore
from src.tools.io_tools import save_overpass_dump
from src.tools.surveillance_data_collector_tools import (
    create_surveillance_data_collector_tools,
)
from src.utils.db import payload_hash, query_hash
from src.utils.overpass import build_query, run_query


class ScrapeError(RuntimeError):
    """
    Raised when the scrape step fails permanently.

    Carries the originating stage and detail so the orchestrator and the API
    layer can surface a precise error to the caller. Transient failures
    (HTTP 429 / 5xx, connection errors, timeouts) are retried inside
    `src.utils.overpass.run_query` via `with_retry` and never reach this
    exception.
    """

    def __init__(
        self,
        message: str,
        *,
        city: str,
        stage: str,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.city = city
        self.stage = stage  # one of: "build_query", "run_query", "save"
        self.detail = detail


class SurveillanceDataCollector:
    """
    Deterministic Overpass scraper for `man_made=surveillance` features.

    Workflow per call to :py:meth:`scrape`:

    1. ``build_query(city, country)`` — construct the Overpass QL query.
    2. Cache lookup against ``MemoryStore`` (verifies file existence and
       payload hash).
    3. On miss, ``run_query(query)`` — POST to Overpass; transient errors are
       retried by the underlying ``with_retry`` decorator.
    4. Persist the response with ``save_overpass_dump`` and record the cache
       entry.
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        settings: Optional[LangChainSettings] = None,
    ) -> None:
        """
        :param name: Agent identifier (used as the memory namespace).
        :param memory: Shared :class:`MemoryStore` for cache entries.
        :param settings: Retained for API compatibility with the previous
            implementation; unused on the deterministic path.
        """
        self.name = name
        self.memory = memory
        self.settings = settings or LangChainSettings()

        # Tools are kept available so that callers introspecting `self.tools`
        # (e.g. existing tests, external integrations) keep working. The
        # scrape() flow below does not invoke them.
        self.tools = create_surveillance_data_collector_tools(memory)

        logger.info(
            f"Initialized {self.name} (deterministic scrape path, no LLM in loop)"
        )

    def scrape(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape surveillance data for a city.

        :param input_data: Dict with ``city`` (required), optional ``country``
            and ``overpass_dir``.
        :return: Result dict with at minimum ``city``, ``success``,
            ``elements_count``, and either ``filepath`` (fresh save),
            ``cached_path`` (cache hit), or ``error`` / ``empty`` on failure.
        :raises ScrapeError: On permanent failure at the build, run, or save
            stage. The orchestrator catches this and converts it to a failed
            pipeline result.
        """
        city: str = input_data["city"]
        country: Optional[str] = input_data.get("country")
        overpass_dir = input_data.get("overpass_dir", "overpass_data")

        city_dir = Path(overpass_dir) / city.lower().replace(" ", "_")
        city_dir.mkdir(parents=True, exist_ok=True)

        country_info = f", {country}" if country else ""
        logger.info(f"Starting scrape for {city}{country_info}")

        query = self._build_query(city, country)

        cached = self._lookup_cache(query)
        if cached is not None:
            cached_path, elements_count = cached
            logger.info(
                f"Cache hit for {city}: {elements_count} elements at {cached_path}"
            )
            return {
                "city": city,
                "country": country,
                "city_dir": str(city_dir),
                "success": True,
                "cache_hit": True,
                "cached_path": str(cached_path),
                "elements_count": elements_count,
            }

        data = self._run_query(city, query)
        elements_count = len(data.get("elements", []))

        if elements_count == 0:
            logger.warning(f"No surveillance elements found for {city}")
            self.memory.store(
                self.name,
                "empty",
                f"{city}|{country or ''}|{query_hash(query)}",
            )
            return {
                "city": city,
                "country": country,
                "city_dir": str(city_dir),
                "success": False,
                "cache_hit": False,
                "empty": True,
                "elements_count": 0,
                "error": f"No surveillance data found for {city}",
            }

        saved_path = self._save(city, city_dir, data)

        self.memory.store(
            self.name,
            "cache",
            f"{query_hash(query)}|{saved_path}|{payload_hash(data)}",
        )
        logger.info(f"Saved {elements_count} elements to {saved_path}")

        return {
            "city": city,
            "country": country,
            "city_dir": str(city_dir),
            "success": True,
            "cache_hit": False,
            "filepath": str(saved_path),
            "elements_count": elements_count,
        }

    def _build_query(self, city: str, country: Optional[str]) -> str:
        try:
            return build_query(city, country=country)
        except Exception as e:
            detail = str(e)
            logger.error(f"Failed to build Overpass query for {city}: {detail}")
            raise ScrapeError(
                f"Failed to build Overpass query for {city}: {detail}",
                city=city,
                stage="build_query",
                detail=detail,
            ) from e

    @staticmethod
    def _run_query(city: str, query: str) -> Dict[str, Any]:
        try:
            return run_query(query)
        except Exception as e:
            detail = str(e)
            logger.error(f"Overpass query failed for {city}: {detail}")
            raise ScrapeError(
                f"Overpass query failed for {city}: {detail}",
                city=city,
                stage="run_query",
                detail=detail,
            ) from e

    @staticmethod
    def _save(city: str, city_dir: Path, data: Dict[str, Any]) -> Path:
        try:
            return save_overpass_dump(data, city, city_dir)
        except Exception as e:
            detail = str(e)
            logger.error(f"Failed to save scrape data for {city}: {detail}")
            raise ScrapeError(
                f"Failed to save scrape data for {city}: {detail}",
                city=city,
                stage="save",
                detail=detail,
            ) from e

    def _lookup_cache(self, query: str) -> Optional[Tuple[Path, int]]:
        q_hash = query_hash(query)
        for mem in self.memory.load(self.name):
            if mem.step != "cache" or not mem.content.startswith(q_hash):
                continue
            try:
                _, filepath_str, p_hash = mem.content.split("|")
            except ValueError:
                logger.debug(f"Skipping malformed cache entry: {mem.content[:80]}")
                continue
            filepath = Path(filepath_str)
            if not filepath.exists():
                logger.debug(f"Cache file missing on disk: {filepath}")
                continue
            try:
                with filepath.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read cache file {filepath}: {e}")
                continue
            if payload_hash(data) != p_hash:
                logger.warning(f"Cache integrity check failed for {filepath}")
                continue
            return filepath, len(data.get("elements", []))
        return None

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility shim with the older ``Agent`` interface used elsewhere.

        Converts a raised :class:`ScrapeError` into a failure dict so callers
        that don't expect exceptions still get a structured response.
        """
        try:
            return self.scrape(input_data)
        except ScrapeError as e:
            return {
                "city": e.city,
                "country": input_data.get("country"),
                "success": False,
                "error": str(e),
                "stage": e.stage,
            }
