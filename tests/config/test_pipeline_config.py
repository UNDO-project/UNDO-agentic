"""
Tests for ``AnalysisScenario`` and ``PipelineConfig.from_scenario`` after
the scenario-collapse work (Architecture Proposal #1).

The scenario enum was trimmed to two members (``BASIC`` and ``FULL``);
``QUICK``, ``REPORT``, and ``MAPPING`` were removed in favour of explicit
toggle overrides.
"""

import pytest

from src.config.pipeline_config import AnalysisScenario, PipelineConfig


def test_scenario_enum_has_exactly_basic_and_full():
    assert {s.value for s in AnalysisScenario} == {"basic", "full"}


@pytest.mark.parametrize("removed", ["quick", "report", "mapping"])
def test_removed_scenarios_no_longer_construct(removed: str):
    """The dropped scenario strings must not coerce back into the enum."""
    with pytest.raises(ValueError):
        AnalysisScenario(removed)


def test_basic_baseline_only_geojson_and_stats():
    """``BASIC`` enables the data + summary stats; no charts, no maps."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.BASIC)

    assert cfg.scenario is AnalysisScenario.BASIC
    assert cfg.generate_geojson is True
    assert cfg.compute_stats is True

    assert cfg.generate_heatmap is False
    assert cfg.generate_hotspots is False
    assert cfg.generate_chart is False
    assert cfg.plot_zone_sensitivity is False
    assert cfg.plot_sensitivity_reasons is False
    assert cfg.plot_hotspots is False


def test_full_flips_every_output_toggle_on():
    """``FULL`` is the strict superset — every output flag True."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.FULL)

    assert cfg.scenario is AnalysisScenario.FULL
    assert cfg.generate_geojson is True
    assert cfg.compute_stats is True
    assert cfg.generate_heatmap is True
    assert cfg.generate_hotspots is True
    assert cfg.generate_chart is True
    assert cfg.plot_zone_sensitivity is True
    assert cfg.plot_sensitivity_reasons is True
    assert cfg.plot_hotspots is True
