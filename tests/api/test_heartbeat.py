"""
Tests for the pipeline-task heartbeat loop.

The heartbeat is an asyncio task spawned right after the initial
``progress: 0`` broadcast and cancelled in the route's ``finally:``
clause before the terminal completed/failed/cancelled event. It carries
``stage`` + ``elapsed_s`` + ``timestamp`` and **no** ``progress`` field.

We exercise ``execute_pipeline_task`` directly (no TestClient) so we can:
- compress the heartbeat interval to ~0.1s for fast tests
- stub ``create_pipeline`` to control runtime
- capture every WebSocket broadcast in order
"""

import asyncio
from typing import Any, Dict, List

import pytest

from src.api.routes import pipeline as pipeline_module
from src.api.services.task_manager import task_manager


class _StubPipeline:
    """Minimal pipeline stand-in honoring the route's calling contract."""

    def __init__(
        self,
        *,
        sleep_s: float = 0.0,
        respect_cancellation: bool = False,
    ) -> None:
        self.sleep_s = sleep_s
        self.respect_cancellation = respect_cancellation
        # The route assigns these after construction.
        self.cancellation_check = lambda: False
        self.on_scrape_complete = None

    def run(self, city: str, **kwargs: Any) -> Dict[str, Any]:
        # Sleep on the worker thread (asyncio.to_thread). Respect
        # cancellation by polling the check at small intervals.
        import time

        slept = 0.0
        step = 0.05
        while slept < self.sleep_s:
            if self.respect_cancellation and self.cancellation_check():
                return {"city": city, "status": "cancelled"}
            time.sleep(step)
            slept += step
        return {"city": city, "status": "completed"}


class _CapturingWS:
    """Records every ``broadcast_progress`` call in order."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def broadcast_progress(self, task_id: str, data: dict) -> None:
        self.events.append({"task_id": task_id, **data})


def _make_request(city: str = "TestCity"):
    from src.api.models.requests import PipelineRequest
    from src.config.pipeline_config import AnalysisScenario

    return PipelineRequest(
        city=city,
        scenario=AnalysisScenario.BASIC,
        force_refresh=False,
    )


@pytest.fixture
def fast_heartbeat(monkeypatch):
    """
    Drive the heartbeat at ~0.1s so tests stay fast.

    We patch the interval function directly rather than setting the
    env var: production clamps below 1.0s back up to 1.0s on purpose
    (no accidental flooding), but tests need sub-second cadence to stay
    quick. The clamp itself is verified separately in
    ``test_heartbeat_interval_clamped_to_min_1s``.
    """
    monkeypatch.setattr(pipeline_module, "_heartbeat_interval_s", lambda: 0.1)


@pytest.fixture
def captured_ws(monkeypatch):
    """Patch the module-level ws_manager with a capturing stub."""
    capture = _CapturingWS()
    monkeypatch.setattr(pipeline_module, "ws_manager", capture)
    return capture


def _run_pipeline_task(task_id: str, request) -> None:
    """Helper: drive the async coroutine via ``asyncio.run``."""
    asyncio.run(pipeline_module.execute_pipeline_task(task_id, request))


def test_heartbeats_emitted_during_long_pipeline_run(
    fast_heartbeat, captured_ws, monkeypatch
):
    """
    A pipeline that sleeps for ~0.6s with a 0.1s heartbeat interval
    should emit at least 2 heartbeats with monotonically increasing
    ``elapsed_s``, and no heartbeat after the terminal ``completed`` event.
    """
    stub = _StubPipeline(sleep_s=0.6)
    monkeypatch.setattr(pipeline_module, "create_pipeline", lambda *a, **kw: stub)

    task_id = task_manager.create_task("pipeline", metadata={"city": "TestCity"})
    try:
        _run_pipeline_task(task_id, _make_request())
    finally:
        task_manager.delete_task(task_id)

    types = [e["type"] for e in captured_ws.events]
    heartbeats = [e for e in captured_ws.events if e["type"] == "heartbeat"]

    assert len(heartbeats) >= 2, f"expected ≥2 heartbeats, got {types}"

    # Each heartbeat carries the contract fields; no progress field.
    for hb in heartbeats:
        assert hb["stage"] in {"initializing", "scraping", "analyzing"}
        assert isinstance(hb["elapsed_s"], (int, float))
        assert "timestamp" in hb
        assert "progress" not in hb

    # Monotonic non-decreasing elapsed_s
    elapsed = [hb["elapsed_s"] for hb in heartbeats]
    assert elapsed == sorted(elapsed), f"elapsed_s not monotonic: {elapsed}"

    # No heartbeat after the terminal event.
    terminal_indices = [
        i
        for i, e in enumerate(captured_ws.events)
        if e["type"] in {"completed", "failed", "cancelled"}
    ]
    assert terminal_indices, f"no terminal event in {types}"
    last_terminal = terminal_indices[-1]
    assert all(
        e["type"] != "heartbeat" for e in captured_ws.events[last_terminal + 1 :]
    ), f"heartbeat emitted after terminal: {types}"


def test_no_heartbeats_after_cancellation(fast_heartbeat, captured_ws, monkeypatch):
    """
    Cancelling the task mid-flight: no heartbeat is emitted after the
    ``cancelled`` broadcast lands.
    """
    stub = _StubPipeline(sleep_s=2.0, respect_cancellation=True)
    monkeypatch.setattr(pipeline_module, "create_pipeline", lambda *a, **kw: stub)

    task_id = task_manager.create_task("pipeline", metadata={"city": "TestCity"})

    async def _drive():
        async def _cancel_after_delay():
            await asyncio.sleep(0.3)
            task_manager.mark_cancelled(task_id)

        await asyncio.gather(
            pipeline_module.execute_pipeline_task(task_id, _make_request()),
            _cancel_after_delay(),
        )

    try:
        asyncio.run(_drive())
    finally:
        task_manager.delete_task(task_id)

    types = [e["type"] for e in captured_ws.events]
    assert "cancelled" in types, f"expected cancelled event in {types}"

    last_cancelled = max(
        i for i, e in enumerate(captured_ws.events) if e["type"] == "cancelled"
    )
    after_cancel = captured_ws.events[last_cancelled + 1 :]
    assert all(e["type"] != "heartbeat" for e in after_cancel), (
        f"heartbeat emitted after cancellation: {[e['type'] for e in after_cancel]}"
    )


def test_short_run_emits_no_heartbeat(fast_heartbeat, captured_ws, monkeypatch):
    """
    A pipeline that completes in < interval_s should produce zero
    heartbeats — the loop's first ``await asyncio.sleep`` is preempted
    by cancellation in the route's ``finally:``.
    """
    stub = _StubPipeline(sleep_s=0.0)
    monkeypatch.setattr(pipeline_module, "create_pipeline", lambda *a, **kw: stub)

    task_id = task_manager.create_task("pipeline", metadata={"city": "TestCity"})
    try:
        _run_pipeline_task(task_id, _make_request())
    finally:
        task_manager.delete_task(task_id)

    heartbeats = [e for e in captured_ws.events if e["type"] == "heartbeat"]
    assert heartbeats == [], (
        f"unexpected heartbeats on a fast run: "
        f"{[e['type'] for e in captured_ws.events]}"
    )


def test_heartbeat_interval_clamped_to_min_1s(monkeypatch):
    """An env value below 1.0 is clamped up; a non-numeric value defaults to 5.0."""
    monkeypatch.setenv("API_HEARTBEAT_INTERVAL_S", "0.01")
    assert pipeline_module._heartbeat_interval_s() == 1.0

    monkeypatch.setenv("API_HEARTBEAT_INTERVAL_S", "not-a-number")
    assert pipeline_module._heartbeat_interval_s() == 5.0

    monkeypatch.setenv("API_HEARTBEAT_INTERVAL_S", "12.5")
    assert pipeline_module._heartbeat_interval_s() == 12.5

    monkeypatch.delenv("API_HEARTBEAT_INTERVAL_S", raising=False)
    assert pipeline_module._heartbeat_interval_s() == 5.0
