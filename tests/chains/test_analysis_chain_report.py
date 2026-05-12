"""
Tests for the ``generate_report`` step inside
``AnalysisChain.generate_visualizations`` (Architecture Proposal #2).

These tests pin two contracts:

- When ``options["generate_report"]`` is True the chain writes
  ``<city>_report.md`` and surfaces ``report_path`` in the context.
- When ``llm.generate_city_report`` raises, the failure lands in
  ``visualization_errors`` and the rest of the run continues.
"""

from pathlib import Path
from types import SimpleNamespace

from src.chains.analysis_chain import AnalysisChain


class _StubLLM:
    """Stand-in for ``SurveillanceLLM`` exposing only what the report path reads."""

    def __init__(
        self, *, response: str = "## Overview\nstubbed.\n", raise_on_call: bool = False
    ) -> None:
        self.settings = SimpleNamespace(batch_size=8)
        self.calls: list = []
        self._response = response
        self._raise = raise_on_call

    def generate_city_report(
        self, stats: dict, sample: list, density_metrics: dict = None
    ) -> str:
        self.calls.append(
            {"stats": stats, "sample": sample, "density_metrics": density_metrics}
        )
        if self._raise:
            raise RuntimeError("City report generation failed: ollama unreachable")
        return self._response


def _make_chain(llm: _StubLLM) -> AnalysisChain:
    chain = AnalysisChain.__new__(AnalysisChain)
    chain.llm = llm
    chain.memory = None
    chain.agent_name = "TestAgent"
    chain.on_progress = None
    return chain


def _ctx(tmp_path: Path) -> dict:
    """
    Minimal context the report step needs. ``stats`` is required (the
    step is gated on it) and ``enriched`` carries the sensitive sample.
    """
    raw_path = tmp_path / "lund.json"
    raw_path.write_text("{}", encoding="utf-8")

    return {
        "path": str(raw_path),
        "enriched_path": str(tmp_path / "lund_enriched.json"),
        "geojson_path": str(tmp_path / "lund_enriched.geojson"),
        "enriched": [
            {
                "id": 1,
                "analysis": {
                    "sensitive": True,
                    "operator": "Polismyndigheten",
                    "zone": "town",
                    "sensitive_reason": "police operator",
                },
            },
            {
                "id": 2,
                "analysis": {
                    "sensitive": False,
                    "operator": "Acme",
                    "zone": "shop",
                },
            },
        ],
        "stats": {
            "total": 2,
            "sensitive_count": 1,
            "public_count": 1,
            "private_count": 1,
        },
    }


def test_generate_report_writes_markdown_and_sets_path(tmp_path):
    llm = _StubLLM(response="## Overview\n2 cameras analysed.\n")
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    out = chain.generate_visualizations(
        ctx, options={"compute_stats": False, "generate_report": True}
    )

    expected_path = tmp_path / "lund_report.md"
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8").startswith("## Overview")
    assert out["report_path"] == str(expected_path)
    assert "visualization_errors" not in out


def test_generate_report_filters_sample_to_sensitive_only(tmp_path):
    """The LLM call receives only sensitive cameras in its ``sample`` arg."""
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    chain.generate_visualizations(
        ctx, options={"compute_stats": False, "generate_report": True}
    )

    assert len(llm.calls) == 1
    sample = llm.calls[0]["sample"]
    assert len(sample) == 1
    assert sample[0]["analysis"]["operator"] == "Polismyndigheten"


def test_generate_report_failure_lands_in_visualization_errors(tmp_path):
    """A raising LLM is contained to ``visualization_errors``; run continues."""
    llm = _StubLLM(raise_on_call=True)
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    out = chain.generate_visualizations(
        ctx, options={"compute_stats": False, "generate_report": True}
    )

    assert (tmp_path / "lund_report.md").exists() is False
    assert "report_path" not in out
    errors = out.get("visualization_errors", [])
    assert any("City report generation failed" in e for e in errors)


def test_generate_report_skipped_when_option_off(tmp_path):
    """No file written, no LLM call, no error when toggle is False."""
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)

    out = chain.generate_visualizations(
        ctx, options={"compute_stats": False, "generate_report": False}
    )

    assert llm.calls == []
    assert "report_path" not in out
    assert (tmp_path / "lund_report.md").exists() is False


def test_generate_report_skipped_when_stats_absent(tmp_path):
    """
    The report step is nested under the ``stats`` block — without stats
    in the context, the step never runs even if the toggle is on.
    """
    llm = _StubLLM()
    chain = _make_chain(llm)
    ctx = _ctx(tmp_path)
    del ctx["stats"]

    out = chain.generate_visualizations(
        ctx, options={"compute_stats": False, "generate_report": True}
    )

    assert llm.calls == []
    assert "report_path" not in out
