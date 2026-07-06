"""
Tests for LLM-driven operator classification.

The LLM is stubbed so these run offline and deterministically. The stub
inspects only the ``Operator:`` line of the prompt (a real model sees the
whole prompt, but keying on the operator keeps the test's intent explicit
and avoids matching the word "police" in the instructions themselves).
"""

import re

from src.tools.operator_classification import (
    OTHER_IDENTIFIED,
    POLICE,
    UNTAGGED,
    classify_camera_operator,
    classify_operators,
    normalize_operator,
)

_POLICE_TOKENS = ("polis", "politi", "polizei", "police", "guardia")


class StubLLM:
    """Answers YES when the operator name looks like police in any language."""

    def __init__(self):
        self.calls = 0

    def generate_batch(self, prompts):
        self.calls += 1
        out = []
        for p in prompts:
            m = re.search(r"Operator:\s*(.*)", p)
            name = (m.group(1) if m else "").lower()
            out.append("YES" if any(tok in name for tok in _POLICE_TOKENS) else "NO")
        return out


class RaisingLLM:
    def generate_batch(self, prompts):
        raise RuntimeError("ollama down")


def test_normalize_operator():
    assert normalize_operator("  Polis  myndigheten ") == "polis myndigheten"
    assert normalize_operator("POLIS") == "polis"
    assert normalize_operator("") == ""
    assert normalize_operator(None) == ""
    assert normalize_operator(123) == ""


def test_classify_operators_multilingual_and_misspelling():
    ops = [
        "Polismyndigheten",
        "Polismyndgheten",  # real-data misspelling
        "Politi",
        "Polizei",
        "Stockholm Stad",
        "ACME Security",
    ]
    mapping = classify_operators(ops, StubLLM())
    assert mapping["Polismyndigheten"] == POLICE
    assert mapping["Polismyndgheten"] == POLICE
    assert mapping["Politi"] == POLICE
    assert mapping["Polizei"] == POLICE
    assert mapping["Stockholm Stad"] == OTHER_IDENTIFIED
    assert mapping["ACME Security"] == OTHER_IDENTIFIED


def test_classify_operators_dedupes_and_skips_blanks():
    llm = StubLLM()
    ops = ["Polisen", "polisen", " POLISEN ", "", None, "  "]
    mapping = classify_operators(ops, llm)
    # All three "polisen" variants collapse to a single classified entry.
    assert len(mapping) == 1
    assert next(iter(mapping.values())) == POLICE
    # Blanks never reach the LLM; one batch call for the single distinct op.
    assert llm.calls == 1


def test_classify_operators_empty_returns_empty_and_skips_llm():
    llm = StubLLM()
    assert classify_operators(["", None, "   "], llm) == {}
    assert llm.calls == 0  # no LLM call when there's nothing to classify


def test_classify_operators_batch_failure_defaults_to_other():
    # A whole-batch failure must not raise — it degrades to a conservative
    # other_identified undercount rather than aborting the layer.
    mapping = classify_operators(["Polismyndigheten", "ACME"], RaisingLLM())
    assert mapping == {
        "ACME": OTHER_IDENTIFIED,
        "Polismyndigheten": OTHER_IDENTIFIED,
    }


def test_classify_camera_operator_untagged_is_deterministic():
    mapping = {"Polismyndigheten": POLICE}
    # No LLM object passed — untagged is decided structurally.
    assert classify_camera_operator("", mapping) == UNTAGGED
    assert classify_camera_operator(None, mapping) == UNTAGGED
    assert classify_camera_operator("   ", mapping) == UNTAGGED
    assert classify_camera_operator("Polismyndigheten", mapping) == POLICE


def test_classify_camera_operator_normalised_match():
    mapping = {"Polismyndigheten": POLICE}
    # A casing/spacing variant not seen at classification time still resolves.
    assert classify_camera_operator("  polismyndigheten ", mapping) == POLICE
    # An operator absent from the map falls back to other_identified.
    assert classify_camera_operator("Unknown Co", mapping) == OTHER_IDENTIFIED
