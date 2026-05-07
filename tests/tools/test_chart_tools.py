import json
from collections import Counter

import pytest

from src.tools.chart_tools import (
    private_public_pie,
    plot_zone_sensitivity,
    plot_sensitivity_reasons,
    plot_hotspots,
    plot_operator_distribution,
    plot_manufacturer_distribution,
    plot_install_timeline,
)


def test_private_public_pie(sample_stats, tmp_path):
    """Test that private_public_pie creates a chart file"""
    result = private_public_pie(sample_stats, tmp_path)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "privacy_distribution.png"


def test_plot_zone_sensitivity(sample_stats, tmp_path):
    """Test that plot_zone_sensitivity creates a chart file"""
    result = plot_zone_sensitivity(sample_stats, tmp_path)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "zone_sensitivity.png"


def test_plot_zone_sensitivity_custom_filename(sample_stats, tmp_path):
    """Test plot_zone_sensitivity with custom filename"""
    result = plot_zone_sensitivity(sample_stats, tmp_path, filename="custom_zones.png")

    assert result.exists()
    assert result.name == "custom_zones.png"


def test_plot_sensitivity_reasons(sample_enriched_data, tmp_path):
    """Test that plot_sensitivity_reasons creates a chart file"""
    # Create input file from the enriched data
    input_file = tmp_path / "enriched.json"
    input_file.write_text(json.dumps(sample_enriched_data), encoding="utf-8")

    output_file = tmp_path / "sensitivity_reasons.png"
    result = plot_sensitivity_reasons(input_file, output_file, top_n=2)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "sensitivity_reasons.png"


def test_plot_sensitivity_reasons_empty_data(tmp_path):
    """Test plot_sensitivity_reasons with no sensitive cameras"""
    empty_data = {
        "elements": [
            {"id": 1, "analysis": {"sensitive": False}},
            {"id": 2, "analysis": {"sensitive": False}},
        ]
    }

    input_file = tmp_path / "empty.json"
    input_file.write_text(json.dumps(empty_data))

    output_file = tmp_path / "empty_reasons.png"

    with pytest.raises(ValueError, match="No sensitive_reason data found"):
        plot_sensitivity_reasons(input_file, output_file)


def test_plot_operator_distribution_writes_default_filename(tmp_path):
    """Default filename is ``operator_distribution.png`` next to ``output_dir``."""
    stats = {
        "operator_counts": Counter({"Police": 12, "Transit": 7, "Acme": 3}),
    }
    result = plot_operator_distribution(stats, tmp_path)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "operator_distribution.png"


def test_plot_operator_distribution_custom_city_filename(tmp_path):
    """The chain passes a per-city filename so artifacts are self-identifying."""
    stats = {"operator_counts": Counter({"Police": 1})}
    result = plot_operator_distribution(
        stats, tmp_path, filename="operator_distribution_lund.png"
    )

    assert result.exists()
    assert result.name == "operator_distribution_lund.png"


def test_plot_operator_distribution_buckets_other(tmp_path):
    """Top-N is honoured; everything past the cutoff lands in an ``other`` bucket."""
    stats = {
        "operator_counts": Counter({f"Op{i}": 100 - i for i in range(15)}),
    }
    # No assertion on the chart contents — matplotlib renders to PNG. We
    # just confirm the function completes (i.e. the "other" code path
    # didn't blow up) and emits a file when there's overflow.
    result = plot_operator_distribution(stats, tmp_path, top_n=5)
    assert result.exists()


def test_plot_operator_distribution_handles_empty_counts(tmp_path):
    """Empty input yields a placeholder chart, not a missing file."""
    stats = {"operator_counts": Counter()}
    result = plot_operator_distribution(stats, tmp_path)
    assert result.exists()


def test_plot_operator_distribution_handles_missing_key(tmp_path):
    """A stats dict without ``operator_counts`` is a no-op, not a KeyError."""
    result = plot_operator_distribution({}, tmp_path)
    assert result.exists()


def test_plot_manufacturer_distribution_writes_default_filename(tmp_path):
    stats = {"manufacturer_counts": Counter({"Acme": 5, "Bosch": 3})}
    result = plot_manufacturer_distribution(stats, tmp_path)

    assert result.exists()
    assert result.name == "manufacturer_distribution.png"


def test_plot_manufacturer_distribution_handles_empty_counts(tmp_path):
    """Manufacturer is sparse in OSM data; the empty path must work cleanly."""
    stats = {"manufacturer_counts": Counter()}
    result = plot_manufacturer_distribution(stats, tmp_path)
    assert result.exists()


def test_plot_install_timeline_writes_default_filename(tmp_path):
    stats = {
        "start_year_counts": Counter({"2018": 1, "2019": 2, "unknown": 2}),
    }
    result = plot_install_timeline(stats, tmp_path)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "install_timeline.png"


def test_plot_install_timeline_custom_city_filename(tmp_path):
    stats = {"start_year_counts": Counter({"2020": 1})}
    result = plot_install_timeline(
        stats, tmp_path, filename="install_timeline_lund.png"
    )
    assert result.exists()
    assert result.name == "install_timeline_lund.png"


def test_plot_install_timeline_handles_only_unknowns(tmp_path):
    """A dataset with no known dates still produces a chart (single 'unknown' bar)."""
    stats = {"start_year_counts": Counter({"unknown": 5})}
    result = plot_install_timeline(stats, tmp_path)
    assert result.exists()


def test_plot_install_timeline_handles_empty_counts(tmp_path):
    """No data at all → placeholder chart, not a missing file."""
    stats = {"start_year_counts": Counter()}
    result = plot_install_timeline(stats, tmp_path)
    assert result.exists()


def test_plot_install_timeline_handles_missing_key(tmp_path):
    """A stats dict without ``start_year_counts`` is a no-op, not a KeyError."""
    result = plot_install_timeline({}, tmp_path)
    assert result.exists()


def test_plot_hotspots(sample_hotspots, tmp_path):
    """Test that plot_hotspots creates a map visualization"""
    # Create input GeoJSON file for hotspots
    input_file = tmp_path / "hotspots.geojson"
    input_file.write_text(json.dumps(sample_hotspots), encoding="utf-8")

    output_file = tmp_path / "hotspots_viz.png"
    result = plot_hotspots(input_file, output_file)

    assert result.exists()
    assert result.suffix == ".png"
    assert result.name == "hotspots_viz.png"
