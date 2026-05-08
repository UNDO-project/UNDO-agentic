"""
Tests for the orchestrator's on-scrape-complete hook.

The hook fires once between scrape and analyzer dispatch with the
element count, so the API layer can surface "Analyzing N cameras…" on
the polled task response.

We exercise the helper directly to avoid needing a real scraper /
analyzer / Ollama for these tests.
"""

import json
from typing import Any, Dict

from src.orchestration.langchain_pipeline import SurveillancePipeline


class _StubPipeline(SurveillancePipeline):
    """Bypass the heavy ``__init__`` so we can test the hook in isolation."""

    def __init__(self, *, on_scrape_complete=None):  # noqa: D401, WPS110
        # Skip the parent __init__ entirely — we only exercise the hook.
        self.on_scrape_complete = on_scrape_complete


def _write_dummy_data(tmp_path):
    p = tmp_path / "lund.json"
    p.write_text(json.dumps({"elements": []}), encoding="utf-8")
    return p


def test_notify_fires_once_with_run_payload(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    scrape_result = {"changed": True, "elements_count": 152}

    captured: list[Dict[str, Any]] = []
    pipe = _StubPipeline(on_scrape_complete=captured.append)

    pipe._notify_scrape_complete(str(data_path), scrape_result)

    assert len(captured) == 1
    payload = captured[0]
    assert payload["elements_count"] == 152
    assert payload["data_path"] == str(data_path)
    assert payload["scrape_result"] == scrape_result


def test_notify_fires_when_scrape_unchanged(tmp_path):
    """
    Even when scrape reports unchanged data, the analyzer always runs
    (the chain decides what to reuse via its enrichment cache and the
    per-artifact visualisation cache). The hook fires identically.
    """
    data_path = _write_dummy_data(tmp_path)
    data_path.with_name("lund_enriched.geojson").write_text("{}", encoding="utf-8")
    scrape_result = {"changed": False, "elements_count": 152}

    captured: list[Dict[str, Any]] = []
    pipe = _StubPipeline(on_scrape_complete=captured.append)

    pipe._notify_scrape_complete(str(data_path), scrape_result)

    assert len(captured) == 1
    assert captured[0]["elements_count"] == 152
    assert "will_skip_analyzer" not in captured[0]


def test_notify_no_listener_is_noop(tmp_path):
    """Without a listener the helper still runs (just for the loguru log)."""
    data_path = _write_dummy_data(tmp_path)
    scrape_result = {"changed": True, "elements_count": 1}

    pipe = _StubPipeline(on_scrape_complete=None)

    # No exception, no return value contract — just must not raise.
    pipe._notify_scrape_complete(str(data_path), scrape_result)


def test_notify_swallows_listener_exceptions(tmp_path):
    """A bad listener must not crash the pipeline mid-run."""
    data_path = _write_dummy_data(tmp_path)
    scrape_result = {"changed": True, "elements_count": 1}

    def _raises(_payload):
        raise RuntimeError("listener boom")

    pipe = _StubPipeline(on_scrape_complete=_raises)
    pipe._notify_scrape_complete(str(data_path), scrape_result)  # no exception


def test_notify_handles_missing_elements_count(tmp_path):
    """
    A scrape result without ``elements_count`` (degenerate case) still
    reaches the listener with ``elements_count=None`` — the API can
    decide whether to render a generic caption.
    """
    data_path = _write_dummy_data(tmp_path)
    scrape_result = {"changed": True}  # no elements_count

    captured: list[Dict[str, Any]] = []
    pipe = _StubPipeline(on_scrape_complete=captured.append)

    pipe._notify_scrape_complete(str(data_path), scrape_result)

    assert captured[0]["elements_count"] is None
