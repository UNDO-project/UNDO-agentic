"""
Tests for ``compute_statistics``.

Covers the operator + manufacturer counters introduced in
Architecture Proposal #3, plus a small smoke check on the rest of
the response shape so a regression in the existing fields surfaces
here too.
"""

from collections import Counter

import pytest

from src.tools.stat_tools import compute_statistics, _start_year


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


# -- _start_year helper --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2018-04-12", "2018"),
        ("2019", "2019"),
        ("2019?", "2019"),
        ("2024-12", "2024"),
        ("2024-12-31", "2024"),
        (None, None),
        ("", None),
        ("badvalue", None),
        ("12", None),  # Too short to extract a year
        ("?", None),  # Just the fuzzy marker, no digits
    ],
)
def test_start_year_parser(raw, expected):
    assert _start_year(raw) == expected


# -- start_year_counts in compute_statistics --


def test_start_year_counts_buckets_known_years_and_unknowns():
    """
    The acceptance-criterion mix: ``2018-04-12``, ``2019``, ``2019?``,
    ``None``, ``"badvalue"`` should yield ``2018: 1, 2019: 2, unknown: 2``.
    """
    enriched = _enriched(
        [
            {"start_date": "2018-04-12"},
            {"start_date": "2019"},
            {"start_date": "2019?"},
            {"start_date": None},
            {"start_date": "badvalue"},
        ]
    )

    stats = compute_statistics(enriched)

    assert stats["start_year_counts"] == Counter({"2018": 1, "2019": 2, "unknown": 2})


def test_start_year_counts_missing_field_counts_as_unknown():
    """Cameras without a ``start_date`` key at all also land in ``unknown``."""
    enriched = _enriched([{}, {}, {"start_date": "2020"}])
    stats = compute_statistics(enriched)
    assert stats["start_year_counts"] == Counter({"unknown": 2, "2020": 1})


def test_start_year_counts_total_matches_input_count():
    """Every camera lands in exactly one bin — total preserved."""
    enriched = _enriched(
        [
            {"start_date": "2020"},
            {"start_date": None},
            {"start_date": "garbage"},
            {},
        ]
    )
    stats = compute_statistics(enriched)
    assert sum(stats["start_year_counts"].values()) == len(enriched)
