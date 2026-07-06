"""
LLM-driven operator classification for the district-aggregation layer.

The paper's class analysis rests on how many cameras each district is
*police*-operated. Operators are free-text OSM strings and are
language-specific — ``Polismyndigheten`` (Swedish, plus the real-data
misspelling ``Polismyndgheten``), ``Politi`` (Norwegian/Danish),
``Polizei`` (German), ``Police`` (English), ``Politie`` (Dutch), and so
on. A hardcoded alias list would never generalise across the cities the
tool is meant to scan, so classification is delegated to the same local
Ollama LLM the enrichment step already uses.

Three classes, matching the acceptance criteria:

- ``untagged`` — no operator string (missing / empty / whitespace).
  Decided **deterministically**, without an LLM call.
- ``police`` — the operator is a police / law-enforcement authority.
- ``other_identified`` — any other named operator (municipality,
  transit authority, private company, …).

The classifier works on the *distinct* operator strings across all
cameras (usually a handful), so one batched call covers a whole city.
The resulting ``operator -> class`` mapping is logged for audit; callers
cache it implicitly by wrapping the whole district artifact in the
analyzer's per-artifact cache, so a re-run makes no LLM call.
"""

import re
from typing import Dict, Iterable, List

from src.config.logger import logger

#: The three operator classes. ``untagged`` is assigned deterministically;
#: the LLM only ever chooses between ``police`` and ``other_identified``.
POLICE = "police"
OTHER_IDENTIFIED = "other_identified"
UNTAGGED = "untagged"

_CLASSIFY_PROMPT = (
    "You are classifying the operator of a public surveillance camera.\n"
    "Is the organisation named below a police force or law-enforcement "
    "authority (e.g. national/state/municipal police, gendarmerie, sheriff)?\n"
    "Consider names in any language (for example Polismyndigheten, Politi, "
    "Polizei, Police, Politie, Guardia Civil).\n"
    "Answer with exactly one word: YES or NO.\n\n"
    "Operator: {operator}\n"
    "Answer:"
)


def normalize_operator(operator: object) -> str:
    """
    Normalise a raw operator value for presence-testing and de-duplication.

    Lower-cases, strips, and collapses internal whitespace. Non-string
    inputs (``None``, numbers) become ``""``.

    :param operator: Raw ``operator`` property from a camera feature.
    :return: Normalised string; ``""`` when there is no usable operator.
    """
    if not isinstance(operator, str):
        return ""
    return re.sub(r"\s+", " ", operator).strip().lower()


def _distinct_operators(operators: Iterable[object]) -> List[str]:
    """Return the sorted distinct *raw* operator strings that are non-empty."""
    seen: Dict[str, str] = {}
    for op in operators:
        if isinstance(op, str) and normalize_operator(op):
            # Key on the normalised form so casing/spacing variants collapse
            # to one LLM call; keep the first raw spelling as the label.
            seen.setdefault(normalize_operator(op), op)
    return [seen[k] for k in sorted(seen)]


def _parse_yes_no(response: str) -> bool:
    """
    Interpret an LLM YES/NO answer as ``True`` when it means police.

    Small instruction-tuned models often wrap the answer in extra prose,
    so we look for a ``yes``/``no`` token rather than requiring an exact
    match. Ambiguous or empty responses default to ``False`` (i.e.
    ``other_identified``) — the conservative choice for a police tally.
    """
    text = (response or "").strip().lower()
    if re.search(r"\byes\b", text):
        return True
    return False


def classify_operators(distinct_operators: Iterable[object], llm) -> Dict[str, str]:
    """
    Classify raw operator strings into ``police`` / ``other_identified``.

    ``untagged`` is not returned here — it is a per-camera property of
    *absence*, decided by :func:`classify_camera_operator`. This function
    only answers, for each *named* operator, whether it is police.

    :param distinct_operators: Iterable of raw operator strings (may
        contain duplicates / blanks; both are filtered out).
    :param llm: A ``SurveillanceLLM``-like object exposing
        ``generate_batch(prompts) -> list[str]``.
    :return: Mapping ``{raw_operator: class}`` for every distinct
        non-empty operator. Empty when there are no named operators.
    """
    operators = _distinct_operators(distinct_operators)
    if not operators:
        return {}

    prompts = [_CLASSIFY_PROMPT.format(operator=op) for op in operators]
    try:
        responses = llm.generate_batch(prompts)
    except Exception as e:
        # A whole-batch failure shouldn't abort the district layer; fall
        # back to ``other_identified`` for every operator so the police
        # tally is a conservative undercount rather than missing entirely.
        logger.warning(
            f"Operator classification batch failed ({e}); "
            f"defaulting {len(operators)} operators to '{OTHER_IDENTIFIED}'"
        )
        responses = ["no"] * len(operators)

    mapping: Dict[str, str] = {}
    for op, resp in zip(operators, responses):
        mapping[op] = POLICE if _parse_yes_no(resp) else OTHER_IDENTIFIED

    logger.info(f"Operator classification: {mapping}")
    return mapping


def classify_camera_operator(operator: object, operator_classes: Dict[str, str]) -> str:
    """
    Resolve one camera's operator to its class.

    :param operator: Raw ``operator`` property of the camera.
    :param operator_classes: The ``{raw_operator: class}`` mapping from
        :func:`classify_operators`.
    :return: ``untagged`` when there is no operator, otherwise the class
        from ``operator_classes`` (``other_identified`` if the operator
        somehow isn't in the map).
    """
    if not normalize_operator(operator):
        return UNTAGGED
    if isinstance(operator, str) and operator in operator_classes:
        return operator_classes[operator]
    # Match on the normalised form so a casing/spacing variant not seen at
    # classification time still resolves.
    norm = normalize_operator(operator)
    for raw, cls in operator_classes.items():
        if normalize_operator(raw) == norm:
            return cls
    return OTHER_IDENTIFIED
