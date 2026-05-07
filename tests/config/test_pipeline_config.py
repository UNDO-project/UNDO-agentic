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
    assert cfg.plot_operator_distribution is True
    assert cfg.plot_manufacturer_distribution is True
    assert cfg.plot_install_timeline is True
    assert cfg.generate_report is True


def test_basic_does_not_enable_report():
    """The narrative report is opt-in on BASIC; it costs an extra LLM call."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.BASIC)
    assert cfg.generate_report is False


def test_to_analyzer_options_carries_generate_report():
    """``to_analyzer_options`` propagates ``generate_report`` to the chain."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.FULL)
    options = cfg.to_analyzer_options()
    assert options["generate_report"] is True

    cfg2 = PipelineConfig.from_scenario(AnalysisScenario.BASIC)
    assert cfg2.to_analyzer_options()["generate_report"] is False


def test_to_analyzer_options_carries_distribution_chart_toggles():
    """Operator + manufacturer distribution toggles propagate via the options dict."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.FULL)
    options = cfg.to_analyzer_options()
    assert options["plot_operator_distribution"] is True
    assert options["plot_manufacturer_distribution"] is True

    cfg2 = PipelineConfig.from_scenario(AnalysisScenario.BASIC)
    options2 = cfg2.to_analyzer_options()
    assert options2["plot_operator_distribution"] is False
    assert options2["plot_manufacturer_distribution"] is False


def test_to_analyzer_options_carries_install_timeline_toggle():
    """The install-timeline toggle propagates via the options dict."""
    cfg = PipelineConfig.from_scenario(AnalysisScenario.FULL)
    assert cfg.to_analyzer_options()["plot_install_timeline"] is True

    cfg2 = PipelineConfig.from_scenario(AnalysisScenario.BASIC)
    assert cfg2.to_analyzer_options()["plot_install_timeline"] is False
