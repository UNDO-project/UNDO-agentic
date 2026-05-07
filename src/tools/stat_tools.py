from collections import Counter
from typing import List, Dict, Any, Optional


def _start_year(value: Optional[str]) -> Optional[str]:
    """
    Extract the four-digit year prefix from a ``start_date`` string.

    Accepts the formats validated by ``SurveillanceMetadata.start_date``:
    ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD``, and the fuzzy ``YYYY?`` form.
    Returns ``None`` for missing or unparseable input — the caller is
    responsible for bucketing those into the ``"unknown"`` bin so the
    histogram still surfaces gaps in the OSM tagging.

    :param value: Raw ``start_date`` from the enriched analysis dict.
    :return: 4-digit year string, or ``None`` when the input doesn't
        carry a recognisable year prefix.
    """
    if not value:
        return None
    stripped = value.rstrip("?")
    if len(stripped) >= 4 and stripped[:4].isdigit():
        return stripped[:4]
    return None


def compute_statistics(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of enriched element dicts (each having element["analysis"]),
    compute summary stats: total, sensitive/public/private counts,
    counts per zone, camera_type, and top operators.
    :param elements: The element dict
    :return: Summary statistics
    """
    total = len(elements)
    analysis = [el["analysis"] for el in elements]

    sensitive_count = sum(1 for a in analysis if a.get("sensitive"))
    public_count = sum(1 for a in analysis if a.get("public") is True)
    private_count = sum(1 for a in analysis if a.get("public") is False)

    zone_counts = Counter()
    zone_sensitivity = Counter()
    for e in elements:
        z = e["analysis"].get("zone") or "unknown"
        zone_counts[z] += 1
        if e["analysis"].get("sensitive"):
            zone_sensitivity[z] += 1

    camera_type_counts = Counter(
        a.get("camera_type") for a in analysis if a.get("camera_type")
    )
    operator_counts = Counter(a.get("operator") for a in analysis if a.get("operator"))
    manufacturer_counts = Counter(
        a.get("manufacturer") for a in analysis if a.get("manufacturer")
    )

    # Install-year histogram. Cameras with missing or non-parseable
    # ``start_date`` land in the "unknown" bin so the chart visualises
    # gaps in OSM tagging rather than dropping them silently.
    start_year_counts: Counter = Counter()
    for a in analysis:
        year = _start_year(a.get("start_date"))
        start_year_counts[year if year else "unknown"] += 1

    return {
        "total": total,
        "sensitive_count": sensitive_count,
        "public_count": public_count,
        "private_count": private_count,
        "zone_counts": zone_counts,
        "zone_sensitivity_counts": zone_sensitivity,
        "camera_type_counts": camera_type_counts,
        "operator_counts": operator_counts,
        "manufacturer_counts": manufacturer_counts,
        "start_year_counts": start_year_counts,
    }
