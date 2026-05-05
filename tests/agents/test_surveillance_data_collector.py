"""
Tests for the deterministic SurveillanceDataCollector.

These cover the post-fix scrape path that bypasses the LLM. The legacy
ReAct-based path is gone; see bug_fixes.md (Issue #1).
"""

import json
from pathlib import Path

import pytest
import requests

from src.agents.surveillance_data_collector import (
    ScrapeError,
    SurveillanceDataCollector,
)
from src.utils.db import payload_hash, query_hash


PAYLOAD_OK = {"elements": [{"id": 1}, {"id": 2}]}
PAYLOAD_EMPTY = {"elements": []}
FAKE_QUERY = "[out:json];area(3600000007)->.s;(nwr['man_made'='surveillance'](area.s););out geom;"


@pytest.fixture
def collector(mem_fake):
    # The collector eagerly creates LLM-backed tools in __init__; the
    # autouse `patch_client` fixture in tests/conftest.py replaces the Ollama
    # client with a stub, so this stays cheap.
    return SurveillanceDataCollector(name="ScraperAgent", memory=mem_fake)


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


def test_second_run_hits_cache(collector, tmp_path, monkeypatch):
    cached = tmp_path / "lund.json"
    cached.write_text(json.dumps(PAYLOAD_OK))

    collector.memory.store(
        "ScraperAgent",
        "cache",
        f"{query_hash(FAKE_QUERY)}|{cached}|{payload_hash(PAYLOAD_OK)}",
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
    collector.memory.store(
        "ScraperAgent",
        "cache",
        f"{query_hash(FAKE_QUERY)}|{cached}|deadbeef",
    )

    _patch_build_query(monkeypatch)
    _patch_run_query(monkeypatch, returning=PAYLOAD_OK)

    result = collector.scrape(
        {"city": "Lund", "country": "SE", "overpass_dir": str(tmp_path)}
    )

    assert result["success"] is True
    assert result["cache_hit"] is False
