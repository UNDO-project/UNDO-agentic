"""
Tests for the per-artifact visualisation cache wired through
``AnalysisChain.generate_visualizations`` (Architecture Proposal #5).

We exercise the cache against real chart-helper invocations using the
heatmap and report steps as the most independent, stub-friendly
choices: heatmap renders from a small geojson, and the report runs
through a stub LLM. Each test pins one of:

- Cache hit: a second run with identical inputs does **not** call the
  underlying renderer (mock-call count stays at 1).
- Force-rerender: the cache is bypassed and the renderer fires twice.
- Option invalidation: a synthetic ``raw_hash`` change ⇒ a fresh
  render, while the same key reuses the artifact.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.chains.analysis_chain import AnalysisChain


class _StubLLM:
    """Mirror of the report-step stub used in test_analysis_chain_report."""

    def __init__(self, response: str = "## Overview\nstub.\n") -> None:
        self.settings = SimpleNamespace(batch_size=8)
        self.calls: list = []
        self._response = response

    def generate_city_report(self, stats: dict, sample: list) -> str:
        self.calls.append({"stats": stats, "sample": sample})
        return self._response


def _make_chain(llm: _StubLLM) -> AnalysisChain:
    chain = AnalysisChain.__new__(AnalysisChain)
    chain.llm = llm
    chain.memory = None
    chain.agent_name = "TestAgent"
    chain.on_progress = None
    return chain


def _ctx(tmp_path: Path, raw_hash: str = "deadbeef") -> dict:
    raw_path = tmp_path / "lund.json"
    raw_path.write_text("{}", encoding="utf-8")
    geojson_path = tmp_path / "lund_enriched.geojson"
    geojson_path.write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
    return {
        "path": str(raw_path),
        "enriched_path": str(tmp_path / "lund_enriched.json"),
        "geojson_path": str(geojson_path),
        "raw_hash": raw_hash,
        "enriched": [
            {
                "id": 1,
                "analysis": {
                    "sensitive": True,
                    "operator": "Polismyndigheten",
                    "zone": "town",
                    "sensitive_reason": "police operator",
                },
            }
        ],
        "stats": {
            "total": 1,
            "sensitive_count": 1,
            "public_count": 1,
            "private_count": 0,
        },
    }


# -- Report step (no real LLM, exercises the full cache path) --


def test_report_step_caches_after_first_run(tmp_path):
    """Second invocation with identical inputs does not call the LLM."""
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)
    options = {"compute_stats": False, "generate_report": True}

    out1 = chain.generate_visualizations(dict(ctx), options)
    assert "report_path" in out1
    assert len(llm.calls) == 1

    # Second run with the same context shouldn't re-invoke the LLM.
    out2 = chain.generate_visualizations(dict(ctx), options)
    assert out2["report_path"] == out1["report_path"]
    assert len(llm.calls) == 1, "Cached step must not call generate_city_report again"


def test_force_rerender_bypasses_cache(tmp_path):
    """``force_rerender=True`` re-invokes the renderer even with a fresh sidecar."""
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    chain.generate_visualizations(
        dict(ctx), {"compute_stats": False, "generate_report": True}
    )
    assert len(llm.calls) == 1

    chain.generate_visualizations(
        dict(ctx),
        {"compute_stats": False, "generate_report": True, "force_rerender": True},
    )
    assert len(llm.calls) == 2


def test_raw_hash_change_invalidates_cache(tmp_path):
    """A different ``raw_hash`` produces a different cache key — re-render fires."""
    llm = _StubLLM()
    chain = _make_chain(llm)

    ctx_a = _ctx(tmp_path, raw_hash="aaa")
    chain.generate_visualizations(
        dict(ctx_a), {"compute_stats": False, "generate_report": True}
    )
    assert len(llm.calls) == 1

    # Same disk path, different raw_hash → cache miss.
    ctx_b = _ctx(tmp_path, raw_hash="bbb")
    chain.generate_visualizations(
        dict(ctx_b), {"compute_stats": False, "generate_report": True}
    )
    assert len(llm.calls) == 2


def test_writes_sidecar_next_to_artifact(tmp_path):
    """Successful render produces ``<artifact>.cache.json`` for the next run."""
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    chain.generate_visualizations(
        dict(ctx), {"compute_stats": False, "generate_report": True}
    )

    expected_artifact = tmp_path / "lund_report.md"
    expected_sidecar = tmp_path / "lund_report.md.cache.json"
    assert expected_artifact.exists()
    assert expected_sidecar.exists()


# -- Heatmap step (cache reused even when underlying helper is mocked) --


def test_heatmap_step_caches_after_first_run(tmp_path):
    """
    The chart helper is patched so we can count invocations directly.
    First run calls it once and writes the sidecar; second run sees the
    cache hit and skips the call entirely.

    The expected filename is ``<city>_heatmap.html`` (HF#1) — that's
    what ``/api/v1/outputs/{city}/map?map_type=heatmap`` serves.
    """
    chain = _make_chain(_StubLLM())
    ctx = _ctx(tmp_path)
    options = {"compute_stats": False, "generate_heatmap": True}

    expected_path = tmp_path / "lund_heatmap.html"

    def _fake_heatmap(geojson_path, output_html, *args, **kwargs):
        Path(output_html).write_text("<html>fake heatmap</html>", encoding="utf-8")
        return Path(output_html)

    with patch(
        "src.tools.mapping_tools.to_heatmap", side_effect=_fake_heatmap
    ) as mock_heatmap:
        out1 = chain.generate_visualizations(dict(ctx), options)
        assert out1.get("heatmap_path") == str(expected_path)
        assert mock_heatmap.call_count == 1
        # Filename contract honoured: not ``lund_enriched.html``.
        assert expected_path.exists()
        assert not (tmp_path / "lund_enriched.html").exists()

        out2 = chain.generate_visualizations(dict(ctx), options)
        assert out2.get("heatmap_path") == str(expected_path)
        assert mock_heatmap.call_count == 1, (
            "Cache hit: to_heatmap must not run on the second invocation"
        )
