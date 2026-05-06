"""
Tests for the chunked-batch enrichment in ``AnalysisChain._enrich_data``.

We exercise ``_enrich_data`` directly against a stub LLM so the tests
don't need a real Ollama, prompt template, or output parser. The stub
records each batch's input list so we can pin chunking behaviour.
"""

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from src.chains.analysis_chain import AnalysisChain


class _StubLLM:
    """
    Minimal stand-in for ``SurveillanceLLM`` exposing only what
    ``_enrich_data`` reads:

    - ``settings.batch_size``
    - ``analyze_surveillance_elements_batch(elements) -> list[dict]``
    """

    def __init__(
        self,
        batch_size: int = 8,
        *,
        results_factory=None,
        raise_on_chunk: int | None = None,
    ) -> None:
        self.settings = SimpleNamespace(batch_size=batch_size)
        self.batches: list[list[Dict[str, Any]]] = []
        self._results_factory = results_factory or (
            lambda chunk: [{"camera_type": "dome"} for _ in chunk]
        )
        self._raise_on_chunk = raise_on_chunk

    def analyze_surveillance_elements_batch(self, elements: list) -> list:
        idx = len(self.batches)
        self.batches.append(list(elements))
        if self._raise_on_chunk is not None and idx == self._raise_on_chunk:
            raise RuntimeError("ollama unreachable")
        return self._results_factory(elements)


def _make_chain(llm: _StubLLM) -> AnalysisChain:
    """Build an AnalysisChain bypassing the heavy ``__init__``."""
    chain = AnalysisChain.__new__(AnalysisChain)
    chain.llm = llm
    chain.memory = None  # not used by _enrich_data on the non-cache path
    chain.agent_name = "TestAgent"
    chain.on_progress = None  # default in __init__; tests override per-test
    return chain


def _ctx(n: int) -> Dict[str, Any]:
    """Build a context dict mimicking what ``_load_data`` produces."""
    return {
        "elements": [{"id": i, "tags": {"camera:type": "dome"}} for i in range(n)],
        "raw_hash": "abc",
        "element_count": n,
        "path": "/tmp/x.json",
        "cache_hit": False,
        "enriched_exists": False,
    }


def test_chunking_17_elements_at_batch_8_issues_3_calls():
    """17 elements at batch_size=8 → batch sizes (8, 8, 1), order preserved."""
    llm = _StubLLM(batch_size=8)
    chain = _make_chain(llm)

    out = chain._enrich_data(_ctx(17))

    assert [len(b) for b in llm.batches] == [8, 8, 1]
    assert len(out["enriched"]) == 17
    # Output ordering is preserved: enriched[i] corresponds to elements[i].
    for i, el in enumerate(out["enriched"]):
        assert el["id"] == i
        assert el["analysis"] == {"camera_type": "dome"}


def test_chunk_exception_annotates_all_elements_in_chunk():
    """
    A top-level exception from the batch call annotates every element in
    that chunk with ``{"error": ...}``; subsequent chunks are unaffected.
    """
    # 20 elements, batch_size=8 → chunks of (8, 8, 4). Raise on the
    # second chunk only.
    llm = _StubLLM(batch_size=8, raise_on_chunk=1)
    chain = _make_chain(llm)

    out = chain._enrich_data(_ctx(20))

    enriched = out["enriched"]
    assert len(enriched) == 20

    # Chunk 0 (indices 0-7): success
    for i in range(0, 8):
        assert enriched[i]["analysis"] == {"camera_type": "dome"}

    # Chunk 1 (indices 8-15): every element has {"error": ...}
    for i in range(8, 16):
        assert "error" in enriched[i]["analysis"]
        assert "ollama unreachable" in enriched[i]["analysis"]["error"]

    # Chunk 2 (indices 16-19): success
    for i in range(16, 20):
        assert enriched[i]["analysis"] == {"camera_type": "dome"}


def test_per_result_error_dict_passed_through_unchanged():
    """
    A single error dict in the batch result list lands at exactly that
    index of the enriched output; the rest are dumped normally.
    """
    elements = [{"id": i, "tags": {}} for i in range(3)]

    def _results(chunk):
        return [
            {"camera_type": "dome"},
            {"error": "validation failed: bad start_date"},
            {"camera_type": "fixed"},
        ]

    llm = _StubLLM(batch_size=8, results_factory=_results)
    chain = _make_chain(llm)

    out = chain._enrich_data(
        {
            "elements": elements,
            "raw_hash": "abc",
            "element_count": 3,
            "path": "/tmp/x.json",
            "cache_hit": False,
            "enriched_exists": False,
        }
    )

    assert out["enriched"][0]["analysis"] == {"camera_type": "dome"}
    assert out["enriched"][1]["analysis"] == {
        "error": "validation failed: bad start_date"
    }
    assert out["enriched"][2]["analysis"] == {"camera_type": "fixed"}


def test_empty_elements_list_issues_no_batch_calls():
    """An empty input is a no-op — no batch round-trip to Ollama."""
    llm = _StubLLM(batch_size=8)
    chain = _make_chain(llm)

    out = chain._enrich_data(_ctx(0))

    assert out["enriched"] == []
    assert llm.batches == []


def test_cache_hit_path_short_circuits_without_calling_llm(tmp_path):
    """
    On a cache hit the enriched list is loaded from disk and the batch
    method is never called — preserves today's caching contract.
    """
    import json

    enriched_path = tmp_path / "lund_enriched.json"
    cached = {
        "elements": [
            {"id": 1, "analysis": {"camera_type": "dome"}},
            {"id": 2, "analysis": {"camera_type": "fixed"}},
        ]
    }
    enriched_path.write_text(json.dumps(cached), encoding="utf-8")

    llm = _StubLLM(batch_size=8)
    chain = _make_chain(llm)

    out = chain._enrich_data(
        {
            "elements": [],  # ignored when cache_hit=True
            "raw_hash": "abc",
            "path": str(tmp_path / "lund.json"),
            "enriched_path": str(enriched_path),
            "cache_hit": True,
            "enriched_exists": True,
        }
    )

    assert len(out["enriched"]) == 2
    assert llm.batches == []  # No LLM round-trip on cache hit


@pytest.mark.parametrize("size", [1, 2, 3, 7, 8, 16])
def test_batch_size_respected_across_sizes(size: int):
    """At any batch_size in [1, 16], 16 elements split into ceil(16/size) chunks."""
    llm = _StubLLM(batch_size=size)
    chain = _make_chain(llm)

    chain._enrich_data(_ctx(16))

    expected_chunks = (16 + size - 1) // size
    assert len(llm.batches) == expected_chunks
    assert sum(len(b) for b in llm.batches) == 16


# -- on_progress hook (Issue #6) --


def test_on_progress_fires_once_per_chunk_with_running_total():
    """
    10 elements at batch_size=4 → progress hook fires (4, 10), (8, 10),
    (10, 10) in order.
    """
    llm = _StubLLM(batch_size=4)
    chain = _make_chain(llm)
    calls: list[tuple[int, int]] = []
    chain.on_progress = lambda done, total: calls.append((done, total))

    chain._enrich_data(_ctx(10))

    assert calls == [(4, 10), (8, 10), (10, 10)]


def test_on_progress_counts_failed_chunks_too():
    """
    A chunk whose batch call raises still gets its elements appended
    (with ``{"error": ...}`` annotations) and still triggers the progress
    hook — count semantics match ``len(enriched)``.
    """
    llm = _StubLLM(batch_size=4, raise_on_chunk=1)  # chunk index 1 raises
    chain = _make_chain(llm)
    calls: list[tuple[int, int]] = []
    chain.on_progress = lambda done, total: calls.append((done, total))

    chain._enrich_data(_ctx(10))

    # All three chunks reported, including the one that raised.
    assert calls == [(4, 10), (8, 10), (10, 10)]


def test_on_progress_listener_exception_swallowed():
    """A bad listener must not interrupt the run; subsequent chunks continue."""
    llm = _StubLLM(batch_size=4)
    chain = _make_chain(llm)
    calls: list[tuple[int, int]] = []

    def _listener(done: int, total: int) -> None:
        calls.append((done, total))
        if done == 4:
            raise RuntimeError("listener boom on first chunk")

    chain.on_progress = _listener

    out = chain._enrich_data(_ctx(10))

    # Listener was called for every chunk despite raising on the first.
    assert calls == [(4, 10), (8, 10), (10, 10)]
    assert len(out["enriched"]) == 10


def test_no_on_progress_listener_is_noop():
    """``on_progress = None`` (default) is a clean no-op; run still completes."""
    llm = _StubLLM(batch_size=4)
    chain = _make_chain(llm)
    chain.on_progress = None  # explicit for the test

    out = chain._enrich_data(_ctx(10))

    assert len(out["enriched"]) == 10
    assert len(llm.batches) == 3  # 4 + 4 + 2


def test_on_progress_does_not_fire_on_cache_hit_path(tmp_path):
    """The cache-hit short-circuit returns before the chunk loop, so the hook never fires."""
    import json

    enriched_path = tmp_path / "lund_enriched.json"
    cached = {"elements": [{"id": 1, "analysis": {}}]}
    enriched_path.write_text(json.dumps(cached), encoding="utf-8")

    llm = _StubLLM(batch_size=4)
    chain = _make_chain(llm)
    calls: list = []
    chain.on_progress = lambda done, total: calls.append((done, total))

    chain._enrich_data(
        {
            "elements": [],
            "raw_hash": "abc",
            "path": str(tmp_path / "lund.json"),
            "enriched_path": str(enriched_path),
            "cache_hit": True,
            "enriched_exists": True,
        }
    )

    assert calls == []
    assert llm.batches == []
