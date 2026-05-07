"""
Tests for ``compute_statistics``.

Covers the operator + manufacturer counters introduced in
Architecture Proposal #3, plus a small smoke check on the rest of
the response shape so a regression in the existing fields surfaces
here too.
"""

from collections import Counter

from src.tools.stat_tools import compute_statistics


def _enriched(elements):
    """Wrap raw analysis dicts as enriched elements (the stats input shape)."""
    return [{"id": i, "analysis": a} for i, a in enumerate(elements)]


def test_manufacturer_counts_aggregates_present_values():
    enriched = _enriched(
        [
            {"manufacturer": "Acme"},
            {"manufacturer": "Acme"},
            {"manufacturer": "Bosch"},
            {"manufacturer": None},  # ignored
            {},  # ignored — no manufacturer key
        ]
    )

    stats = compute_statistics(enriched)

    assert stats["manufacturer_counts"] == Counter({"Acme": 2, "Bosch": 1})


def test_manufacturer_counts_empty_when_no_data():
    enriched = _enriched([{}, {"manufacturer": None}])
    stats = compute_statistics(enriched)
    assert stats["manufacturer_counts"] == Counter()


def test_operator_counts_unchanged_alongside_manufacturer():
    """Sanity: adding ``manufacturer_counts`` doesn't disturb ``operator_counts``."""
    enriched = _enriched(
        [
            {"operator": "Police", "manufacturer": "Acme"},
            {"operator": "Police", "manufacturer": "Bosch"},
            {"operator": "Transit", "manufacturer": None},
        ]
    )

    stats = compute_statistics(enriched)

    assert stats["operator_counts"] == Counter({"Police": 2, "Transit": 1})
    assert stats["manufacturer_counts"] == Counter({"Acme": 1, "Bosch": 1})
