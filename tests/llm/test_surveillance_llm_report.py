"""
Tests for ``SurveillanceLLM.generate_city_report`` (Architecture
Proposal #2).

We replace ``self.llm`` with a fake whose ``invoke`` returns a known
markdown string so the test pins the prompt-formatting and result
shape without round-tripping to Ollama.
"""

from collections import Counter
from unittest.mock import patch

import pytest
from langchain_core.runnables import Runnable

from src.config.settings import LangChainSettings
from src.llm.surveillance_llm import SurveillanceLLM


class _FakeOllama(Runnable):
    """
    Minimal ``Runnable`` stand-in for ``OllamaLLM`` so ``prompt_template
    | self.llm`` composes via LangChain's normal runnable plumbing.
    Records each ``invoke`` payload for prompt-rendering assertions.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list = []

    def invoke(self, value, config=None, **kwargs):
        self.calls.append(value)
        return self.response


def _make_llm(response: str = "## Overview\n10 cameras.\n") -> SurveillanceLLM:
    settings = LangChainSettings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_temperature=0.0,
        ollama_timeout=30.0,
        batch_size=8,
    )
    with patch("src.llm.surveillance_llm.OllamaLLM"):
        llm = SurveillanceLLM(settings)

    # The chain composition ``prompt_template | self.llm`` calls ``__or__``
    # on the PromptTemplate (which returns a RunnableSequence). We swap
    # the OllamaLLM out here so the sequence's terminal step is our fake.
    llm.llm = _FakeOllama(response)
    return llm


def _stats(**overrides):
    base = {
        "total": 100,
        "sensitive_count": 12,
        "public_count": 30,
        "private_count": 50,
        "zone_counts": Counter({"town": 60, "airport": 25, "transit": 15}),
        "zone_sensitivity_counts": Counter({"airport": 8, "town": 4}),
        "camera_type_counts": Counter({"dome": 70, "fixed": 30}),
        "operator_counts": Counter({"Police": 20, "Transit Co": 15, "Acme": 5}),
    }
    base.update(overrides)
    return base


def test_generate_city_report_returns_markdown_string():
    """Happy path: invoke succeeds, method returns the LLM's markdown verbatim."""
    llm = _make_llm("## Overview\nN cameras analysed.\n")
    out = llm.generate_city_report(_stats(), sample=[])

    assert "## Overview" in out
    assert isinstance(out, str)


def test_generate_city_report_passes_stats_summary_into_prompt():
    """The rendered prompt sees the totals, top operators, and zones."""
    llm = _make_llm("## Overview\nok\n")
    llm.generate_city_report(_stats(), sample=[])

    # The fake Ollama records the PromptValue handed to ``invoke``.
    rendered = str(llm.llm.calls[0])

    assert "total_cameras: 100" in rendered
    assert "sensitive: 12" in rendered
    # top_operators serialisation includes the leading entries
    assert "Police" in rendered
    assert "Transit Co" in rendered
    # Zone counts are surfaced
    assert "airport" in rendered


def test_generate_city_report_renders_sensitive_sample_inline():
    """Cameras flagged sensitive in the sample land in the prompt body."""
    sample = [
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
                "sensitive": True,
                "operator": None,
                "zone": "airport",
                "sensitive_reason": "airport zone",
            },
        },
        # Non-sensitive entry — must not appear in the rendered sample.
        {
            "id": 3,
            "analysis": {
                "sensitive": False,
                "operator": "Acme Corp",
                "zone": "shop",
                "sensitive_reason": None,
            },
        },
    ]

    llm = _make_llm("## Overview\nok\n")
    llm.generate_city_report(_stats(), sample=sample)
    rendered = str(llm.llm.calls[0])

    assert "Polismyndigheten" in rendered
    assert "police operator" in rendered
    assert "airport zone" in rendered
    # The non-sensitive entry's operator must NOT leak into the prompt.
    assert "Acme Corp" not in rendered


def test_generate_city_report_handles_empty_sensitive_sample():
    """An empty sample renders as ``(none)`` so the prompt has explicit data."""
    llm = _make_llm("## Overview\nok\n")
    llm.generate_city_report(_stats(), sample=[])
    rendered = str(llm.llm.calls[0])

    assert "(none)" in rendered


def test_generate_city_report_truncates_sample_to_ten():
    """No more than ten sensitive rows are emitted into the prompt body."""
    sample = [
        {
            "id": i,
            "analysis": {
                "sensitive": True,
                "operator": f"Op{i}",
                "zone": "town",
                "sensitive_reason": "reason",
            },
        }
        for i in range(20)
    ]

    llm = _make_llm("## Overview\nok\n")
    llm.generate_city_report(_stats(), sample=sample)
    rendered = str(llm.llm.calls[0])

    # First 10 must be in; entries 10..19 must not.
    for i in range(10):
        assert f"Op{i}" in rendered
    for i in range(10, 20):
        assert f"Op{i}" not in rendered


def test_generate_city_report_propagates_llm_exception_as_runtime_error():
    """A bad LLM call raises ``RuntimeError`` so the chain-level try/except can catch it."""
    llm = _make_llm("ignored")

    class _Boom(Runnable):
        def invoke(self, value, config=None, **kwargs):
            raise ConnectionError("ollama unreachable")

    llm.llm = _Boom()

    with pytest.raises(RuntimeError, match="City report generation failed"):
        llm.generate_city_report(_stats(), sample=[])
