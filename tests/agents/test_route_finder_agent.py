"""Tests for RouteFinderAgent."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from src.agents.route_finder_agent import RouteFinderAgent
from src.config.models.route_models import (
    CameraFilter,
    RouteMetrics,
    RouteRequest,
    RouteResult,
)
from src.config.settings import RouteSettings


@pytest.fixture
def route_settings():
    """Create default RouteSettings."""
    return RouteSettings()


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    # Create a simple synthetic graph
    G = nx.MultiDiGraph()
    G.add_node(0, x=13.40, y=52.52)
    G.add_node(1, x=13.41, y=52.53)
    G.add_edge(0, 1, length=100.0)

    return {
        "load_cameras": MagicMock(return_value=[(52.52, 13.40), (52.53, 13.41)]),
        "build_graph": MagicMock(return_value=G),
        "snap_start": MagicMock(return_value=0),
        "snap_end": MagicMock(return_value=1),
        "generate_paths": MagicMock(return_value=[[0, 1]]),
        "compute_exposure": MagicMock(
            return_value=RouteMetrics(
                length_m=100.0, exposure_score=5.0, camera_count_near_route=2
            )
        ),
        "build_geojson": MagicMock(return_value=Path("/tmp/route.geojson")),
        "render_map": MagicMock(return_value=Path("/tmp/map.html")),
    }


@pytest.fixture
def route_request(tmp_path):
    """Create a sample route request."""
    # Create mock cameras GeoJSON
    cameras_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.40, 52.52]},
                "properties": {},
            }
        ],
    }

    cameras_path = tmp_path / "cameras.geojson"
    cameras_path.write_text(json.dumps(cameras_geojson), encoding="utf-8")

    return RouteRequest(
        city="Berlin",
        country="DE",
        start_lat=52.52,
        start_lon=13.40,
        end_lat=52.53,
        end_lon=13.41,
        data_path=cameras_path,
    )


def test_route_finder_agent_init(mem_fake, route_settings):
    """Test RouteFinderAgent initialization."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    assert agent.name == "test_agent"
    assert agent.memory == mem_fake
    assert agent.settings == route_settings
    assert len(agent.tools) == 8  # Default tools


def test_perceive_no_cache(mem_fake, route_settings, route_request):
    """Test perceive method without cached result."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    observation = agent.perceive(route_request)

    assert observation["city"] == "Berlin"
    assert observation["country"] == "DE"
    assert observation["start_lat"] == 52.52
    assert observation["start_lon"] == 13.40
    assert observation["end_lat"] == 52.53
    assert observation["end_lon"] == 13.41
    assert "cache_key" in observation
    assert observation["cache_hit"] is False
    assert observation["cached_result"] is None


def test_plan_cache_miss(mem_fake, route_settings):
    """Test plan method when cache is missed."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    observation = {"cache_hit": False}
    plan = agent.plan(observation)

    expected_steps = [
        "load_cameras",
        "build_graph",
        "snap_start",
        "snap_end",
        "generate_paths",
        "score_paths",
        "build_geojson",
        "render_map",
    ]

    assert plan == expected_steps


def test_plan_cache_hit(mem_fake, route_settings):
    """Test plan method when cache is hit."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    observation = {"cache_hit": True}
    plan = agent.plan(observation)

    assert plan == []  # No steps when using cache


def test_act_load_cameras(mem_fake, route_settings, mock_tools, tmp_path):
    """Test act method for load_cameras action."""
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )

    # Create mock cameras file
    cameras_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.40, 52.52]},
                "properties": {},
            }
        ],
    }
    cameras_path = tmp_path / "cameras.geojson"
    cameras_path.write_text(json.dumps(cameras_geojson), encoding="utf-8")

    context = {"cameras_geojson_path": cameras_path}

    agent.act("load_cameras", context)

    mock_tools["load_cameras"].assert_called_once_with(cameras_path)
    assert "cameras" in context


def test_act_score_paths(mem_fake, route_settings, mock_tools):
    """Test act method for score_paths action."""
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )

    # Setup context with candidate paths
    G = mock_tools["build_graph"].return_value
    context = {
        "graph": G,
        "cameras": [(52.52, 13.40)],
        "candidate_paths": [[0, 1], [0, 1]],  # Two paths
    }

    agent.act("score_paths", context)

    # Should have called compute_exposure for each path
    assert mock_tools["compute_exposure"].call_count == 2

    # Should have selected best path and metrics
    assert "best_path" in context
    assert "best_metrics" in context
    assert "baseline_metrics" in context

    # Best metrics should have baseline comparison
    assert context["best_metrics"].baseline_length_m is not None
    assert context["best_metrics"].baseline_exposure_score is not None


def test_act_unknown_action(mem_fake, route_settings):
    """Test that unknown action raises ValueError."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    with pytest.raises(ValueError, match="Unknown action"):
        agent.act("unknown_action", {})


def test_achieve_goal_full_workflow(
    mem_fake, route_settings, mock_tools, route_request, tmp_path
):
    """Test achieve_goal method with full workflow."""
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )

    # Create output paths for mocked tools
    route_geojson_path = tmp_path / "route.geojson"
    route_map_path = tmp_path / "map.html"

    route_geojson_path.write_text('{"type": "FeatureCollection", "features": []}')
    route_map_path.write_text("<html></html>")

    mock_tools["build_geojson"].return_value = route_geojson_path
    mock_tools["render_map"].return_value = route_map_path

    result = agent.achieve_goal(route_request)

    # Verify result
    assert isinstance(result, RouteResult)
    assert result.city == "Berlin"
    assert result.from_cache is False
    assert isinstance(result.metrics, RouteMetrics)

    # Verify all tools were called
    mock_tools["load_cameras"].assert_called_once()
    mock_tools["build_graph"].assert_called_once()
    mock_tools["snap_start"].assert_called_once()
    mock_tools["snap_end"].assert_called_once()
    mock_tools["generate_paths"].assert_called_once()
    mock_tools["build_geojson"].assert_called_once()
    mock_tools["render_map"].assert_called_once()


def test_achieve_goal_with_cache(mem_fake, route_settings, route_request, tmp_path):
    """Test achieve_goal method when result is cached."""
    # Pre-populate memory with cached result
    route_geojson_path = tmp_path / "cached_route.geojson"
    route_map_path = tmp_path / "cached_map.html"

    route_geojson_path.write_text('{"type": "FeatureCollection", "features": []}')
    route_map_path.write_text("<html></html>")

    cached_metrics = RouteMetrics(
        length_m=100.0,
        exposure_score=5.0,
        camera_count_near_route=2,
        baseline_length_m=90.0,
        baseline_exposure_score=8.0,
    )

    # Create cache key (must match agent's key generation logic)
    cache_key_input = (
        f"{route_request.city}_{route_request.country}_"
        f"{route_request.start_lat}_{route_request.start_lon}_"
        f"{route_request.end_lat}_{route_request.end_lon}_"
        f"{route_settings.max_candidates}_{route_settings.buffer_radius_m}"
    )
    import hashlib

    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()[:16]

    cache_content = (
        f"{cache_key}|"
        f"{route_geojson_path}|"
        f"{route_map_path}|"
        f"{cached_metrics.model_dump_json()}"
    )

    mem_fake.store("test_agent", "route_cache", cache_content)

    # Create agent and run
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )

    result = agent.achieve_goal(route_request)

    # Verify cache was used
    assert result.from_cache is True
    assert result.city == "Berlin"
    assert result.route_geojson_path == route_geojson_path
    assert result.route_map_path == route_map_path
    assert result.metrics.length_m == 100.0


def test_achieve_goal_stores_in_cache(
    mem_fake, route_settings, mock_tools, route_request, tmp_path
):
    """Test that achieve_goal stores result in cache."""
    # Create output paths for mocked tools
    route_geojson_path = tmp_path / "route.geojson"
    route_map_path = tmp_path / "map.html"

    route_geojson_path.write_text('{"type": "FeatureCollection", "features": []}')
    route_map_path.write_text("<html></html>")

    mock_tools["build_geojson"].return_value = route_geojson_path
    mock_tools["render_map"].return_value = route_map_path

    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )

    # Run once to populate cache
    result1 = agent.achieve_goal(route_request)
    assert result1.from_cache is False

    # Check that result was stored in memory
    memories = list(mem_fake.load("test_agent"))
    cache_memories = [m for m in memories if m.step == "route_cache"]

    assert len(cache_memories) > 0

    # Verify cache content format
    cache_content = cache_memories[0].content
    parts = cache_content.split("|", 3)
    assert len(parts) == 4
    # Verify cache structure: cache_key|geojson_path|map_path|metrics_json
    assert len(parts[0]) == 16  # Cache key is 16-char hash
    assert parts[1].endswith(".geojson")  # Route GeoJSON path
    assert parts[2].endswith(".html")  # Route map path
    # Verify metrics JSON can be parsed
    metrics_json = json.loads(parts[3])
    assert "length_m" in metrics_json


# Tests for camera filter integration


def _request_with_filter(base, camera_filter):
    """Build a RouteRequest mirroring ``base`` but with ``camera_filter``."""
    return RouteRequest(
        city=base.city,
        country=base.country,
        start_lat=base.start_lat,
        start_lon=base.start_lon,
        end_lat=base.end_lat,
        end_lon=base.end_lon,
        data_path=base.data_path,
        camera_filter=camera_filter,
    )


def test_perceive_no_filter_keeps_legacy_cache_key(
    mem_fake, route_settings, route_request
):
    """Acceptance criterion: default behaviour (no filter) shares the cache
    slot with pre-Issue-#6 entries. The cache key for ``camera_filter=None``
    and an inert filter must match."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )
    key_none = agent.perceive(route_request)["cache_key"]
    key_inert = agent.perceive(_request_with_filter(route_request, CameraFilter()))[
        "cache_key"
    ]
    assert key_none == key_inert


def test_perceive_active_filter_changes_cache_key(
    mem_fake, route_settings, route_request
):
    """Different filters must hash to different cache slots so they don't
    collide."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )
    baseline = agent.perceive(route_request)["cache_key"]
    sensitive = agent.perceive(
        _request_with_filter(route_request, CameraFilter(sensitive_only=True))
    )["cache_key"]
    police = agent.perceive(
        _request_with_filter(route_request, CameraFilter(operators=["police"]))
    )["cache_key"]
    assert baseline != sensitive != police
    assert baseline != police


def test_perceive_carries_camera_filter_into_observation(
    mem_fake, route_settings, route_request
):
    """The filter must travel into the act() context so ``load_cameras``
    can read it."""
    agent = RouteFinderAgent(
        name="test_agent", memory=mem_fake, settings=route_settings
    )
    cf = CameraFilter(sensitive_only=True)
    observation = agent.perceive(_request_with_filter(route_request, cf))
    assert observation["camera_filter"] is cf


def test_act_load_cameras_forwards_filter(
    mem_fake, route_settings, mock_tools, route_request, tmp_path
):
    """When the context carries an active filter, load_cameras receives
    it as the ``camera_filter`` keyword."""
    cameras_path = route_request.data_path
    # Drop a sensitive-camera-only fixture into place; the existing
    # mock_tools["load_cameras"] is a MagicMock so we just inspect the
    # arguments it was called with.
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )
    cf = CameraFilter(operators=["police"])
    context = {"cameras_geojson_path": cameras_path, "camera_filter": cf}
    agent.act("load_cameras", context)

    mock_tools["load_cameras"].assert_called_once_with(cameras_path, camera_filter=cf)
    # The unfiltered total denominator must also be captured.
    assert context["camera_count_total"] == 1  # route_request fixture has one feature


def test_act_load_cameras_omits_filter_kwarg_when_none(
    mem_fake, route_settings, mock_tools, route_request
):
    """Backward-compat: with no filter we call load_cameras(path) — the
    one-arg signature matches the legacy contract for any tool override
    a test passed in pre-Issue-#6."""
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )
    context = {
        "cameras_geojson_path": route_request.data_path,
        "camera_filter": None,
    }
    agent.act("load_cameras", context)
    mock_tools["load_cameras"].assert_called_once_with(route_request.data_path)


def test_score_paths_propagates_camera_count_total(
    mem_fake, route_settings, mock_tools
):
    """``score_paths`` must stamp the unfiltered total onto best_metrics
    so the orchestrator can surface it on the routing response."""
    agent = RouteFinderAgent(
        name="test_agent",
        memory=mem_fake,
        settings=route_settings,
        tools=mock_tools,
    )
    G = mock_tools["build_graph"].return_value
    context = {
        "graph": G,
        "cameras": [(52.52, 13.40)],
        "candidate_paths": [[0, 1]],
        "camera_count_total": 7,  # was 7 before filtering down to 1
    }
    agent.act("score_paths", context)
    assert context["best_metrics"].camera_count_total == 7
