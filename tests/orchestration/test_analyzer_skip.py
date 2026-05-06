"""
Tests for the orchestrator's probe-and-compare analyzer-skip path.

`_maybe_skip_analyzer` is a static helper, so we test it directly without
instantiating a full SurveillancePipeline.
"""

import json

from src.orchestration.langchain_pipeline import SurveillancePipeline


def _write_dummy_data(tmp_path):
    data_path = tmp_path / "lund.json"
    data_path.write_text(json.dumps({"elements": [{"id": 1}]}), encoding="utf-8")
    return data_path


def test_skip_when_unchanged_and_geojson_exists(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    enriched_geojson = data_path.with_name("lund_enriched.geojson")
    enriched_json = data_path.with_name("lund_enriched.json")
    enriched_geojson.write_text("{}", encoding="utf-8")
    enriched_json.write_text("{}", encoding="utf-8")

    scrape_result = {
        "changed": False,
        "elements_count": 152,
        "cache_hit": True,
    }

    skip = SurveillancePipeline._maybe_skip_analyzer(str(data_path), scrape_result)

    assert skip is not None
    assert skip["success"] is True
    assert skip["skipped"] is True
    assert skip["skipped_reason"] == "scrape_unchanged"
    assert skip["element_count"] == 152
    assert skip["geojson_path"] == str(enriched_geojson)
    assert skip["enriched_path"] == str(enriched_json)


def test_no_skip_when_changed_is_true(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    data_path.with_name("lund_enriched.geojson").write_text("{}", encoding="utf-8")

    scrape_result = {"changed": True, "elements_count": 158}

    assert (
        SurveillancePipeline._maybe_skip_analyzer(str(data_path), scrape_result) is None
    )


def test_no_skip_when_geojson_missing(tmp_path):
    """
    If the prior enriched geojson is gone (user deleted overpass_data,
    fresh checkout, etc.) we cannot honor the skip — the analyzer must
    rebuild it even though scrape says "unchanged".
    """
    data_path = _write_dummy_data(tmp_path)
    # No enriched files written.

    scrape_result = {"changed": False, "elements_count": 152}

    assert (
        SurveillancePipeline._maybe_skip_analyzer(str(data_path), scrape_result) is None
    )


def test_no_skip_without_scrape_result(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    assert SurveillancePipeline._maybe_skip_analyzer(str(data_path), None) is None
    assert SurveillancePipeline._maybe_skip_analyzer(str(data_path), {}) is None


def test_skip_includes_optional_visualizations_if_present(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    data_path.with_name("lund_enriched.geojson").write_text("{}", encoding="utf-8")
    data_path.with_name("lund_heatmap.html").write_text("<html/>", encoding="utf-8")
    (tmp_path / "stats_chart_lund.png").write_bytes(b"\x89PNG")

    scrape_result = {"changed": False, "elements_count": 152}
    skip = SurveillancePipeline._maybe_skip_analyzer(str(data_path), scrape_result)

    assert skip is not None
    assert skip["heatmap_path"].endswith("lund_heatmap.html")
    assert skip["pie_chart_path"].endswith("stats_chart_lund.png")
    assert "hotspots_path" not in skip  # not on disk → not surfaced


# -- will_skip_analyzer (predicate used by the on-scrape-complete hook) --


def test_will_skip_true_when_unchanged_and_geojson_exists(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    data_path.with_name("lund_enriched.geojson").write_text("{}", encoding="utf-8")
    scrape_result = {"changed": False, "elements_count": 152}

    assert (
        SurveillancePipeline.will_skip_analyzer(str(data_path), scrape_result) is True
    )


def test_will_skip_false_when_changed(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    data_path.with_name("lund_enriched.geojson").write_text("{}", encoding="utf-8")
    scrape_result = {"changed": True, "elements_count": 158}

    assert (
        SurveillancePipeline.will_skip_analyzer(str(data_path), scrape_result) is False
    )


def test_will_skip_false_when_geojson_missing(tmp_path):
    data_path = _write_dummy_data(tmp_path)
    scrape_result = {"changed": False, "elements_count": 152}

    assert (
        SurveillancePipeline.will_skip_analyzer(str(data_path), scrape_result) is False
    )


def test_will_skip_false_without_scrape_result(tmp_path):
    data_path = _write_dummy_data(tmp_path)

    assert SurveillancePipeline.will_skip_analyzer(str(data_path), None) is False
    assert SurveillancePipeline.will_skip_analyzer(str(data_path), {}) is False
