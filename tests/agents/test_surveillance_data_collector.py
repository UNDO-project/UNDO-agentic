"""
Tests for the deterministic SurveillanceDataCollector.

These cover the post-fix scrape path that bypasses the LLM. The legacy
ReAct-based path is gone; see bug_fixes.md (Issue #1). Cache TTL and
force_refresh behavior covers Issue #2.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from src.agents.surveillance_data_collector import (
    ScrapeError,
    SurveillanceDataCollector,
)
from src.config.settings import OverpassSettings
from src.utils.db import payload_hash, query_hash


PAYLOAD_OK = {"elements": [{"id": 1}, {"id": 2}]}
PAYLOAD_EMPTY = {"elements": []}
FAKE_QUERY = "[out:json];area(3600000007)->.s;(nwr['man_made'='surveillance'](area.s););out geom;"


@pytest.fixture
def collector(mem_fake):
    # The collector eagerly creates LLM-backed tools in __init__; the
    # autouse `patch_client` fixture in tests/conftest.py replaces the Ollama
    # client with a stub, so this stays cheap.
    return SurveillanceDataCollector(
        name="ScraperAgent",
        memory=mem_fake,
        overpass_settings=OverpassSettings(cache_ttl_hours=24.0),
    )


def _seed_cache_row(mem_fake, *, content: str, age_hours: float):
    """
    Append a cache row whose `timestamp` is `age_hours` in the past, so
    TTL behavior can be exercised without sleeping or freezing the clock.

    Routes through the fake's `store()` so the row gets a synthetic `id`
    (matching the real `Memory` table) and the touch path works.
    """
    row = mem_fake.store("ScraperAgent", "cache", content)
    row.timestamp = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return row


def _patch_build_query(monkeypatch, query: str = FAKE_QUERY):
    monkeypatch.setattr(
        "src.agents.surveillance_data_collector.build_query",
        lambda *_, **__: query,
    )


def _patch_run_query(monkeypatch, *, returning=None, raising=None):
    def _run(_query):
        if raising is not None:
            raise raising
        return returning

    monkeypatch.setattr(
        "src.agents.surveillance_data_collector.run_query",
        _run,
    )


def test_first_run_saves_and_caches(collector, tmp_path, monkeypatch):
    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is False
    assert result["elements_count"] == 2

    saved = Path(result["filepath"])
    assert saved.exists()
    assert json.loads(saved.read_text()) == PAYLOAD_OK

    cache_rows = [r for r in collector.memory.rows if r.step == "cache"]
    assert len(cache_rows) == 1
    expected = f"{query_hash(FAKE_QUERY)}|{saved}|{payload_hash(PAYLOAD_OK)}"
    assert cache_rows[0].content == expected

    # Frontend-visible cache metadata is present even on a fresh fetch so
    # the UI can render "fresh, expires in N hours" without special-casing.
    assert "cached_at" in result
    assert result["data_age_hours"] == pytest.approx(0.0, abs=0.05)
    assert result["cache_ttl_hours"] == 24.0
    assert "cache_expires_at" in result


def test_second_run_hits_cache(collector, tmp_path, monkeypatch):
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
        age_hours=2.0,
    )

    _patch_build_query(monkeypatch)

    def _run(_):
        pytest.fail("run_query must not be called on cache hit")

    monkeypatch.setattr("src.agents.surveillance_data_collector.run_query", _run)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is True
    assert result["cached_path"] == str(cached)
    assert result["elements_count"] == 2
    assert result["data_age_hours"] == pytest.approx(2.0, abs=0.1)
    assert result["cache_ttl_hours"] == 24.0


def test_expired_cache_triggers_refetch(collector, tmp_path, monkeypatch):
    """
    Issue #2: a cache row older than ttl_hours must be ignored so the next
    scan picks up newly-tagged cameras instead of returning stale data.
    """
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
        age_hours=48.0,  # > default TTL of 24h
    )

    _patch_build_query(monkeypatch)
    fresh_payload = {"elements": [{"id": 1}, {"id": 2}, {"id": 3}]}
    _patch_run_query(monkeypatch, returning=fresh_payload)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is False, "expired entry must not be served"
    assert result["elements_count"] == 3, "fresh payload should be returned"


def test_force_refresh_bypasses_valid_cache(collector, tmp_path, monkeypatch):
    """
    Issue #2: explicit force_refresh must skip cache lookup even when the
    cached entry is still within TTL.
    """
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
        age_hours=1.0,  # well within TTL
    )

    _patch_build_query(monkeypatch)
    fresh_payload = {"elements": [{"id": 9}, {"id": 10}, {"id": 11}, {"id": 12}]}
    _patch_run_query(monkeypatch, returning=fresh_payload)

    result = collector.scrape(
        {
            "city": "Lund",
            "country": "SE",
            "overpass_dir": str(tmp_path),
            "force_refresh": True,
        }
    )

    assert result["cache_hit"] is False
    assert result["elements_count"] == 4

    # Two cache rows should now exist: the original (within TTL) and the
    # newly-written one. Lookup prefers timestamp DESC, so subsequent scans
    # see the freshest data.
    cache_rows = [r for r in collector.memory.rows if r.step == "cache"]
    assert len(cache_rows) == 2


def test_probe_no_change_touches_row_and_skips_save(collector, tmp_path, monkeypatch):
    """
    Issue #2 (probe-and-compare): when TTL has expired but Overpass returns
    an identical payload, we must NOT rewrite the file or insert a new
    cache row. We touch the existing row (extending its TTL) and signal
    `changed=False` so the orchestrator can skip the analyzer.
    """
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    seeded = _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
        age_hours=48.0,  # past TTL → must probe
    )
    original_timestamp = seeded.timestamp

    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)  # same payload

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is True
    assert result["probed"] is True
    assert result["changed"] is False
    assert result["delta"] == 0
    assert result["previous_elements_count"] == 2
    assert result["cached_path"] == str(cached)

    # File contents must not have been re-written; same row count, fresher
    # timestamp.
    cache_rows = [r for r in collector.memory.rows if r.step == "cache"]
    assert len(cache_rows) == 1, "no duplicate cache row should be inserted"
    assert cache_rows[0].timestamp > original_timestamp


def test_probe_changed_writes_new_row_and_signals_delta(
    collector, tmp_path, monkeypatch
):
    """
    Issue #2 (probe-and-compare): when Overpass returns a different payload,
    save a new file, write a new cache row, and report the delta so the
    frontend can show "+6 cameras since last scan" and the orchestrator can
    re-run the analyzer.
    """
    old_file = tmp_path / "lund_old.json"
    old_file.write_text(json.dumps(PAYLOAD_OK))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{old_file}|{payload_hash(PAYLOAD_OK)}",
        age_hours=48.0,  # past TTL → must probe
    )

    _patch_build_query(monkeypatch)
    fresh = {"elements": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]}
    _patch_run_query(monkeypatch, returning=fresh)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["cache_hit"] is False
    assert result["probed"] is True
    assert result["changed"] is True
    assert result["elements_count"] == 5
    assert result["previous_elements_count"] == 2
    assert result["delta"] == 3

    cache_rows = [r for r in collector.memory.rows if r.step == "cache"]
    assert len(cache_rows) == 2, "new row should be appended on probe-changed"


def test_force_refresh_probe_unchanged_skips_save(collector, tmp_path, monkeypatch):
    """
    force_refresh forces a probe even within TTL, but identical payload still
    avoids the file rewrite and the analyzer rerun. The whole point is to let
    users hit "Refresh" without paying the LLM cost when nothing changed.
    """
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
        age_hours=1.0,  # within TTL
    )

    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)

    result = collector.scrape(
        {
            "city": "Lund",
            "country": "SE",
            "overpass_dir": str(tmp_path),
            "force_refresh": True,
        }
    )

    assert result["probed"] is True
    assert result["changed"] is False
    assert result["cache_hit"] is True

    cache_rows = [r for r in collector.memory.rows if r.step == "cache"]
    assert len(cache_rows) == 1, "force_refresh must not duplicate row on no-change"


def test_first_scan_has_no_previous_count(collector, tmp_path, monkeypatch):
    """
    On a first ever scan there is no prior row, so previous_elements_count
    and delta must be None (not 0) — the frontend uses the null to render
    "First scan" instead of "Δ 0".
    """
    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["changed"] is True
    assert result["probed"] is True
    assert result["previous_elements_count"] is None
    assert result["delta"] is None


def test_freshest_cache_row_wins(collector, tmp_path, monkeypatch):
    """
    When multiple cache rows exist for the same query (e.g. after a
    force_refresh), the newest row must be picked so we never serve a
    stale superseded entry.
    """
    old_file = tmp_path / "lund_old.json"
    new_file = tmp_path / "lund_new.json"
    old_payload = {"elements": [{"id": 1}]}
    new_payload = {"elements": [{"id": 1}, {"id": 2}, {"id": 3}]}
    old_file.write_text(json.dumps(old_payload))
    new_file.write_text(json.dumps(new_payload))

    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{old_file}|{payload_hash(old_payload)}",
        age_hours=10.0,
    )
    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{new_file}|{payload_hash(new_payload)}",
        age_hours=1.0,
    )

    _patch_build_query(monkeypatch)

    def _run(_):
        pytest.fail("run_query must not be called when a fresh cache row exists")

    monkeypatch.setattr("src.agents.surveillance_data_collector.run_query", _run)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["cache_hit"] is True
    assert result["cached_path"] == str(new_file)
    assert result["elements_count"] == 3


def test_empty_result_is_failure_not_silent_success(collector, tmp_path, monkeypatch):
    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_EMPTY)

    result = collector.scrape(
        {"city": "Nowhere", "country": "ZZ", "overpass_dir": str(tmp_path)}
    )

    # Pre-fix this returned success=True with no filepath, which let the
    # pipeline limp on and fail with a confusing "No data path" message
    # downstream. Now an empty scrape is an explicit failure.
    assert result["success"] is False
    assert result["empty"] is True
    assert result["elements_count"] == 0
    assert "Nowhere" in result["error"]

    empty_rows = [r for r in collector.memory.rows if r.step == "empty"]
    assert empty_rows, "empty marker should be persisted to avoid re-fetch"


def test_overpass_4xx_raises_scrape_error(collector, tmp_path, monkeypatch):
    """
    The Malmö regression: Overpass returns 400 (e.g. malformed query). The
    underlying RuntimeError from `run_query` must be wrapped in ScrapeError
    with stage='run_query' so the orchestrator/API can surface it cleanly,
    rather than the agent silently completing with zero elements.
    """
    _patch_build_query(monkeypatch)
    _patch_run_query(
        monkeypatch,
        raising=RuntimeError("Overpass error 400: line 2: parse error"),
    )

    with pytest.raises(ScrapeError) as exc_info:
        collector.scrape(
            {"city": "Malmö", "country": "SE", "overpass_dir": str(tmp_path)}
        )

    err = exc_info.value
    assert err.stage == "run_query"
    assert err.city == "Malmö"
    assert "Overpass error 400" in str(err)


def test_build_query_failure_raises_scrape_error(collector, tmp_path, monkeypatch):
    def _broken_build(*_, **__):
        raise RuntimeError("No Nominatim result for 'Atlantis'")

    monkeypatch.setattr(
        "src.agents.surveillance_data_collector.build_query", _broken_build
    )

    with pytest.raises(ScrapeError) as exc_info:
        collector.scrape({"city": "Atlantis", "overpass_dir": str(tmp_path)})

    assert exc_info.value.stage == "build_query"


def test_achieve_goal_converts_scrape_error_to_dict(collector, tmp_path, monkeypatch):
    _patch_build_query(monkeypatch)
    _patch_run_query(
        monkeypatch,
        raising=requests.ConnectionError("connection reset"),
    )

    result = collector.achieve_goal(
        {"city": "Malmö", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is False
    assert result["stage"] == "run_query"
    assert result["city"] == "Malmö"
    assert "connection reset" in result["error"]


def test_corrupt_cache_entry_falls_through_to_fresh_fetch(
    collector, tmp_path, monkeypatch
):
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    # Tampered payload hash — integrity check must reject this entry and
    # force a re-fetch instead of returning stale data silently.
    _seed_cache_row(
        collector.memory,
        content=f"{query_hash(FAKE_QUERY)}|{cached}|deadbeef",
        age_hours=1.0,
    )

    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is False
