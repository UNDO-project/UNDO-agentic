"""
Tests for ``OutputOverrides`` and the request-level layering of toggles
on top of an ``AnalysisScenario`` preset (Architecture Proposal #1).

These tests do not spin up a real pipeline — they verify the model
shape and the kwargs handoff to ``create_pipeline``, which is the
contract the route depends on.
"""

import pytest
from pydantic import ValidationError

from src.api.models.requests import OutputOverrides, PipelineRequest
from src.config.pipeline_config import AnalysisScenario, PipelineConfig


def test_output_overrides_all_fields_optional():
    """An empty overrides object is valid and dumps to {}."""
    o = OutputOverrides()
    assert o.model_dump(exclude_none=True) == {}


def test_output_overrides_partial_set():
    """Only set fields land in the dump."""
    o = OutputOverrides(generate_heatmap=True, plot_hotspots=False)
    assert o.model_dump(exclude_none=True) == {
        "generate_heatmap": True,
        "plot_hotspots": False,
    }


def test_pipeline_request_accepts_overrides_block():
    req = PipelineRequest(
        city="Berlin",
        scenario=AnalysisScenario.BASIC,
        overrides={"generate_heatmap": True},
    )
    assert req.overrides is not None
    assert req.overrides.generate_heatmap is True
    assert req.overrides.generate_hotspots is None


def test_pipeline_request_overrides_default_none():
    req = PipelineRequest(city="Berlin", scenario=AnalysisScenario.BASIC)
    assert req.overrides is None


def test_pipeline_request_rejects_dropped_scenarios():
    """``quick``, ``report``, ``mapping`` no longer pass validation."""
    for dropped in ("quick", "report", "mapping"):
        with pytest.raises(ValidationError):
            PipelineRequest(city="Berlin", scenario=dropped)


def _layer_overrides(
    scenario: AnalysisScenario, overrides_kwargs: dict
) -> PipelineConfig:
    """
    Mirror of the production layering performed by ``create_pipeline`` —
    extracted here so the test does not need to construct the heavy
    ``SurveillancePipeline`` (LLM, MemoryStore, agents). The contract under
    test is that the same kwargs the route hands to ``create_pipeline``
    yield the same final ``PipelineConfig``.
    """
    cfg = PipelineConfig.from_scenario(scenario)
    for key, value in overrides_kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def test_overrides_layer_on_top_of_basic_preset():
    """
    Simulates the route path: dump the overrides into kwargs and assert
    the resulting config has the BASIC baseline plus the overridden field.
    """
    overrides = OutputOverrides(generate_heatmap=True).model_dump(exclude_none=True)
    cfg = _layer_overrides(AnalysisScenario.BASIC, overrides)

    # Override applied
    assert cfg.generate_heatmap is True
    # BASIC baseline preserved for everything else
    assert cfg.generate_geojson is True
    assert cfg.compute_stats is True
    assert cfg.generate_hotspots is False
    assert cfg.generate_chart is False


def test_overrides_can_disable_flags_in_full_preset():
    """A ``False`` override flips a FULL-preset toggle off."""
    overrides = OutputOverrides(generate_heatmap=False).model_dump(exclude_none=True)
    cfg = _layer_overrides(AnalysisScenario.FULL, overrides)

    assert cfg.generate_heatmap is False
    # Other FULL flags untouched
    assert cfg.generate_hotspots is True
    assert cfg.plot_hotspots is True
    assert cfg.generate_chart is True


def test_unset_overrides_leave_preset_untouched():
    """``None`` overrides do not surface in the dump and never touch the config."""
    overrides = OutputOverrides().model_dump(exclude_none=True)
    cfg = _layer_overrides(AnalysisScenario.FULL, overrides)

    # Identical to a plain FULL preset
    full = PipelineConfig.from_scenario(AnalysisScenario.FULL)
    assert cfg.model_dump() == full.model_dump()
