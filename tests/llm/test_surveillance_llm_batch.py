"""
Tests for ``SurveillanceLLM.analyze_surveillance_elements_batch``.

The chain itself is replaced with a fake whose ``batch`` method returns
a pre-configured list (mixing successful ``SurveillanceMetadata``
instances and Exception objects to exercise ``return_exceptions=True``).
"""

from unittest.mock import patch

import pytest

from src.config.models.surveillance_metadata import SurveillanceMetadata
from src.config.settings import LangChainSettings
from src.llm.surveillance_llm import SurveillanceLLM


class _FakeChain:
    """Fake LangChain runnable that records ``.batch`` calls."""

    def __init__(self, results: list) -> None:
        self.results = results
        self.batch_calls: list = []

    def batch(self, inputs, return_exceptions: bool = False):
        self.batch_calls.append(
            {"inputs": inputs, "return_exceptions": return_exceptions}
        )
        return list(self.results)


def _make_llm() -> SurveillanceLLM:
    """Build a SurveillanceLLM with chain bookkeeping in place."""
    settings = LangChainSettings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_temperature=0.0,
        ollama_timeout=30.0,
        batch_size=8,
    )
    with patch("src.llm.surveillance_llm.OllamaLLM"):
        llm = SurveillanceLLM(settings)
    # Initialise the prompt template + parser so format_instructions works,
    # then stub the chain.
    llm._ensure_chain_initialized()
    return llm


@pytest.fixture
def llm() -> SurveillanceLLM:
    return _make_llm()


def test_empty_input_returns_empty_list(llm: SurveillanceLLM) -> None:
    """No batch call is issued for an empty input list."""
    fake = _FakeChain(results=[])
    llm.chain = fake

    out = llm.analyze_surveillance_elements_batch([])

    assert out == []
    assert fake.batch_calls == []  # No round-trip to Ollama for empty input


def test_batch_returns_aligned_dump_dicts(llm: SurveillanceLLM) -> None:
    """Each successful result is dumped to a dict aligned to input order."""
    elements = [
        {"id": 1, "tags": {"camera:type": "dome"}},
        {"id": 2, "tags": {"camera:type": "fixed"}},
    ]
    results = [
        SurveillanceMetadata(camera_type="dome"),
        SurveillanceMetadata(camera_type="fixed"),
    ]
    fake = _FakeChain(results=results)
    llm.chain = fake

    out = llm.analyze_surveillance_elements_batch(elements)

    assert len(out) == 2
    assert out[0]["camera_type"] == "dome"
    assert out[1]["camera_type"] == "fixed"
    # Single batch call with return_exceptions=True so per-result errors
    # don't abort the batch.
    assert len(fake.batch_calls) == 1
    assert fake.batch_calls[0]["return_exceptions"] is True
    assert len(fake.batch_calls[0]["inputs"]) == 2


def test_per_result_exception_becomes_error_dict(llm: SurveillanceLLM) -> None:
    """
    Exception at index ``i`` in the batch results becomes
    ``{"error": ...}`` at index ``i``; other indices are dumped normally.
    """
    elements = [
        {"id": 1, "tags": {"camera:type": "dome"}},
        {"id": 2, "tags": {"camera:type": "broken"}},
        {"id": 3, "tags": {"camera:type": "fixed"}},
    ]
    results = [
        SurveillanceMetadata(camera_type="dome"),
        ValueError("unparsable LLM output"),
        SurveillanceMetadata(camera_type="fixed"),
    ]
    fake = _FakeChain(results=results)
    llm.chain = fake

    out = llm.analyze_surveillance_elements_batch(elements)

    assert out[0]["camera_type"] == "dome"
    assert out[1] == {"error": "unparsable LLM output"}
    assert out[2]["camera_type"] == "fixed"


def test_from_raw_failure_isolated_to_index(llm: SurveillanceLLM) -> None:
    """
    A post-LLM ``from_raw`` failure (e.g. validator rejects start_date)
    is annotated only at the affected index. We force this by stubbing
    a result with a malformed ``start_date`` so the validator raises
    inside ``from_raw``.
    """

    class _BadDump:
        """Stand-in for a chain result whose ``model_dump`` triggers a
        downstream validator failure when fed to ``SurveillanceMetadata.from_raw``."""

        def model_dump(self):
            return {"camera_type": "dome", "start_date": "not-a-date"}

    elements = [
        {"id": 1, "tags": {}},
        {"id": 2, "tags": {}},
    ]
    results = [SurveillanceMetadata(camera_type="dome"), _BadDump()]
    fake = _FakeChain(results=results)
    llm.chain = fake

    out = llm.analyze_surveillance_elements_batch(elements)

    assert out[0]["camera_type"] == "dome"
    assert "error" in out[1]


def test_batch_call_propagates_top_level_exception(
    llm: SurveillanceLLM,
) -> None:
    """
    A failure of ``chain.batch`` itself (network down, model not loaded)
    is allowed to propagate so the chain-level caller can wrap each
    chunk in try/except and annotate every element in that chunk.
    """

    class _RaisingChain:
        def batch(self, inputs, return_exceptions: bool = False):
            raise RuntimeError("ollama unreachable")

    llm.chain = _RaisingChain()

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        llm.analyze_surveillance_elements_batch([{"id": 1, "tags": {}}])
