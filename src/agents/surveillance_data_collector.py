"""
Deterministic surveillance data collector.

Drives a four-step workflow (build query, cache lookup, run query, save) without
an LLM in the loop.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.logger import logger
from src.config.settings import LangChainSettings, OverpassSettings
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
        overpass_settings: Optional[OverpassSettings] = None,
    ) -> None:
        """
        :param name: Agent identifier (used as the memory namespace).
        :param memory: Shared :class:`MemoryStore` for cache entries.
        :param settings: Retained for API compatibility with the previous
            implementation; unused on the deterministic path.
        :param overpass_settings: Overpass-side configuration, including the
            cache TTL. Defaults to a fresh :class:`OverpassSettings` instance.
        """
        self.name = name
        self.memory = memory
        self.settings = settings or LangChainSettings()
        self.overpass_settings = overpass_settings or OverpassSettings()

        # Tools are kept available so that callers introspecting `self.tools`
        # (e.g. existing tests, external integrations) keep working. The
        # scrape() flow below does not invoke them.
        self.tools = create_surveillance_data_collector_tools(memory)

        logger.info(
            f"Initialized {self.name} "
            f"(cache_ttl_hours={self.overpass_settings.cache_ttl_hours})"
        )

    def scrape(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape surveillance data for a city via probe-and-compare.

        Decision tree:

        1. **Fresh cache hit** — a cache row younger than ``cache_ttl_hours``
           exists for this query and ``force_refresh`` is False. Return its
           contents without contacting Overpass. ``cache_hit=True``,
           ``probed=False``, ``changed=False``.
        2. **Probe** — TTL expired *or* ``force_refresh=True``. POST to
           Overpass and compute ``payload_hash`` of the response.

           - **No prior row**: save fresh, write a new cache row.
             ``cache_hit=False``, ``probed=True``, ``changed=True``,
             ``previous_elements_count=None``.
           - **Prior row, same hash**: data is unchanged. Skip the file
             rewrite, ``MemoryStore.touch()`` the prior row to extend its
             TTL another ``cache_ttl_hours``, return the prior file's path.
             ``cache_hit=True``, ``probed=True``, ``changed=False``,
             ``delta=0``. The orchestrator uses ``changed`` to skip the
             analyzer.
           - **Prior row, different hash**: data has changed. Save new file,
             write a new cache row. ``cache_hit=False``, ``probed=True``,
             ``changed=True``, ``delta=new-prev``.

        :param input_data: Dict with ``city`` (required), optional ``country``,
            ``overpass_dir``, and ``force_refresh`` (bool; default False).
        :return: Result dict; see decision tree above for fields. Always
            includes cache metadata (``cache_hit``, ``cached_at``,
            ``data_age_hours``, ``cache_ttl_hours``, ``cache_expires_at``)
            and probe metadata (``probed``, ``changed``,
            ``previous_elements_count``, ``delta``).
        :raises ScrapeError: On permanent failure at the build, run, or save
            stage.
        """
        city: str = input_data["city"]
        country: Optional[str] = input_data.get("country")
        overpass_dir = input_data.get("overpass_dir", "overpass_data")
        force_refresh: bool = bool(input_data.get("force_refresh", False))
        ttl_hours: float = self.overpass_settings.cache_ttl_hours

        city_dir = Path(overpass_dir) / city.lower().replace(" ", "_")
        city_dir.mkdir(parents=True, exist_ok=True)

        country_info = f", {country}" if country else ""
        refresh_info = " (force_refresh)" if force_refresh else ""
        logger.info(f"Starting scrape for {city}{country_info}{refresh_info}")

        query = self._build_query(city, country)

        # 1. Fresh cache: short-circuit, no Overpass call.
        if not force_refresh:
            cached = self._lookup_cache(query, ttl_hours=ttl_hours)
            if cached is not None:
                cached_path, elements_count, cached_at = cached
                age_hours = self._age_hours(cached_at)
                logger.info(
                    f"Cache hit (fresh) for {city}: {elements_count} elements "
                    f"({age_hours:.1f}h old) at {cached_path}"
                )
                return {
                    "city": city,
                    "country": country,
                    "city_dir": str(city_dir),
                    "success": True,
                    "cached_path": str(cached_path),
                    "elements_count": elements_count,
                    "probed": False,
                    "changed": False,
                    "previous_elements_count": elements_count,
                    "delta": 0,
                    **self._cache_metadata(
                        cache_hit=True,
                        cached_at=cached_at,
                        ttl_hours=ttl_hours,
                    ),
                }

        # 2. Probe: contact Overpass and compare with the freshest stored row.
        prior = self._find_latest_row(query)
        data = self._run_query(city, query)
        elements_count = len(data.get("elements", []))
        new_hash = payload_hash(data)

        if prior is not None and prior["payload_hash"] == new_hash:
            # Probe-no-change: extend prior row's TTL, reuse the file.
            prior_path = prior["filepath"]
            if prior_path.exists():
                self.memory.touch(prior["row_id"])
                logger.info(
                    f"Probe for {city}: data unchanged ({elements_count} elements); "
                    f"refreshed cache row id={prior['row_id']}, kept {prior_path}"
                )
                # Refresh the in-memory cached_at to "now" since we touched.
                cached_at = datetime.now(timezone.utc)
                return {
                    "city": city,
                    "country": country,
                    "city_dir": str(city_dir),
                    "success": True,
                    "cached_path": str(prior_path),
                    "elements_count": elements_count,
                    "probed": True,
                    "changed": False,
                    "previous_elements_count": elements_count,
                    "delta": 0,
                    **self._cache_metadata(
                        cache_hit=True,
                        cached_at=cached_at,
                        ttl_hours=ttl_hours,
                    ),
                }
            # Prior file vanished from disk despite hash match → fall through
            # and save fresh, then keep the same `changed` accounting as a
            # full refetch.
            logger.warning(
                f"Probe match but prior file {prior_path} is gone; saving fresh"
            )

        # Probe-changed (or first-ever scan, or prior file missing).
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
                "probed": True,
                "changed": True,
                "previous_elements_count": prior["elements_count"] if prior else None,
                "delta": None,
                "empty": True,
                "elements_count": 0,
                "error": f"No surveillance data found for {city}",
            }

        saved_path = self._save(city, city_dir, data)
        cache_row = self.memory.store(
            self.name,
            "cache",
            f"{query_hash(query)}|{saved_path}|{new_hash}",
        )
        cached_at = self._coerce_utc(cache_row.timestamp)

        prev_count = prior["elements_count"] if prior else None
        delta = (elements_count - prev_count) if prev_count is not None else None
        delta_msg = (
            f" (Δ {delta:+d} vs previous {prev_count})"
            if delta is not None
            else " (first scan)"
        )
        logger.info(f"Saved {elements_count} elements to {saved_path}{delta_msg}")

        return {
            "city": city,
            "country": country,
            "city_dir": str(city_dir),
            "success": True,
            "filepath": str(saved_path),
            "elements_count": elements_count,
            "probed": True,
            "changed": True,
            "previous_elements_count": prev_count,
            "delta": delta,
            **self._cache_metadata(
                cache_hit=False,
                cached_at=cached_at,
                ttl_hours=ttl_hours,
            ),
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

    def _find_latest_row(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Return the freshest cache row matching ``query`` regardless of TTL.

        Used by the probe-and-compare flow to compare a fresh Overpass payload
        against the most recent stored payload, so we can decide whether to
        rewrite the file or just touch the existing row.

        :returns: ``{"row_id", "filepath", "elements_count", "payload_hash",
            "cached_at"}`` for the freshest row whose content is parseable and
            whose file is readable. ``None`` if no usable row exists.
        """
        q_hash = query_hash(query)
        rows = [
            m
            for m in self.memory.load(self.name)
            if m.step == "cache" and m.content.startswith(q_hash)
        ]
        rows.sort(key=lambda m: m.timestamp, reverse=True)

        for mem in rows:
            try:
                _, filepath_str, p_hash = mem.content.split("|")
            except ValueError:
                continue
            filepath = Path(filepath_str)
            elements_count = 0
            if filepath.exists():
                try:
                    with filepath.open(encoding="utf-8") as f:
                        data = json.load(f)
                    elements_count = len(data.get("elements", []))
                except (OSError, json.JSONDecodeError):
                    # File unreadable — treat as if it's gone but keep hash
                    # so the probe can still detect "data unchanged" via
                    # payload_hash. The scrape() caller checks file existence
                    # before honoring the no-change path.
                    pass
            return {
                "row_id": mem.id,
                "filepath": filepath,
                "elements_count": elements_count,
                "payload_hash": p_hash,
                "cached_at": self._coerce_utc(mem.timestamp),
            }
        return None

    def _lookup_cache(
        self,
        query: str,
        *,
        ttl_hours: float,
    ) -> Optional[tuple[Path, int, datetime]]:
        """
        Find the freshest valid cache row for ``query``.

        Iterates rows newest-first and returns the first that passes file
        existence, payload-hash integrity, and TTL checks. Stale or corrupt
        rows are skipped so a newer row (or a fresh fetch) can supersede them.

        :returns: ``(filepath, elements_count, cached_at_utc)`` or ``None``.
        """
        q_hash = query_hash(query)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

        rows = [
            m
            for m in self.memory.load(self.name)
            if m.step == "cache" and m.content.startswith(q_hash)
        ]
        # Newest-first so a fresh entry supersedes any older expired rows
        # without us having to delete them from the DB.
        rows.sort(key=lambda m: m.timestamp, reverse=True)

        for mem in rows:
            cached_at = self._coerce_utc(mem.timestamp)
            if cached_at < cutoff:
                logger.debug(
                    f"Cache row expired (age "
                    f"{self._age_hours(cached_at):.1f}h > TTL {ttl_hours}h); "
                    f"skipping {mem.content[:32]}…"
                )
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
            return filepath, len(data.get("elements", [])), cached_at
        return None

    @staticmethod
    def _coerce_utc(ts: datetime) -> datetime:
        # SQLModel/SQLite may hand back a naive datetime even when we wrote a
        # tz-aware one. Treat naive values as UTC (the writer's intent).
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)

    @staticmethod
    def _age_hours(cached_at: datetime) -> float:
        return (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600.0

    @staticmethod
    def _cache_metadata(
        *,
        cache_hit: bool,
        cached_at: datetime,
        ttl_hours: float,
    ) -> Dict[str, Any]:
        expires_at = cached_at + timedelta(hours=ttl_hours)
        return {
            "cache_hit": cache_hit,
            "cached_at": cached_at.isoformat(),
            "data_age_hours": round(
                (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600.0,
                2,
            ),
            "cache_ttl_hours": ttl_hours,
            "cache_expires_at": expires_at.isoformat(),
        }

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
