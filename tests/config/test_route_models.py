"""Tests for CameraFilter on RouteRequest / RouteMetrics (Backend #6)."""

from src.config.models.route_models import (
    CameraFilter,
    RouteMetrics,
    RouteRequest,
)


def test_camera_filter_inert_by_default():
    """Default-constructed filter must be a no-op so omitting the field
    on a RouteRequest preserves pre-Issue-#6 behaviour byte-for-byte."""
    f = CameraFilter()
    assert f.is_inert() is True
    # And every plausible camera passes
    assert f.matches({"sensitive": True, "operator": "police"}) is True
    assert f.matches({}) is True


def test_camera_filter_sensitive_only_drops_non_sensitive():
    f = CameraFilter(sensitive_only=True)
    assert f.is_inert() is False
    assert f.matches({"sensitive": True}) is True
    assert f.matches({"sensitive": False}) is False
    # Falsy / missing both fail the constraint
    assert f.matches({}) is False


def test_camera_filter_operators_whitelist():
    f = CameraFilter(operators=["police", "transit"])
    assert f.matches({"operator": "police"}) is True
    assert f.matches({"operator": "private_retailer"}) is False
    assert f.matches({"operator": None}) is False
    assert f.matches({}) is False


def test_camera_filter_surveillance_types_whitelist():
    f = CameraFilter(surveillance_types=["camera"])
    assert f.matches({"surveillance_type": "camera"}) is True
    assert f.matches({"surveillance_type": "guard"}) is False


def test_camera_filter_combined_constraints():
    f = CameraFilter(
        sensitive_only=True,
        operators=["police"],
        surveillance_types=["camera"],
    )
    # All three must pass
    assert (
        f.matches(
            {"sensitive": True, "operator": "police", "surveillance_type": "camera"}
        )
        is True
    )
    # Any one failing rejects
    assert (
        f.matches(
            {"sensitive": False, "operator": "police", "surveillance_type": "camera"}
        )
        is False
    )
    assert (
        f.matches(
            {"sensitive": True, "operator": "private", "surveillance_type": "camera"}
        )
        is False
    )
    assert (
        f.matches(
            {"sensitive": True, "operator": "police", "surveillance_type": "guard"}
        )
        is False
    )


def test_camera_filter_empty_operators_list_rejects_all():
    """An explicit empty list is a constraint that nothing satisfies —
    distinct from ``None`` which disables the constraint entirely."""
    f = CameraFilter(operators=[])
    assert f.is_inert() is False
    assert f.matches({"operator": "police"}) is False


def test_route_request_camera_filter_default_is_none():
    """Default RouteRequest has no filter so existing call sites keep
    the legacy 'consider every camera' semantics."""
    req = RouteRequest(
        city="Berlin",
        country="DE",
        start_lat=52.52,
        start_lon=13.40,
        end_lat=52.53,
        end_lon=13.41,
    )
    assert req.camera_filter is None


def test_route_request_camera_filter_round_trips():
    req = RouteRequest(
        city="Berlin",
        country="DE",
        start_lat=52.52,
        start_lon=13.40,
        end_lat=52.53,
        end_lon=13.41,
        camera_filter=CameraFilter(sensitive_only=True, operators=["police"]),
    )
    assert req.camera_filter.sensitive_only is True
    assert req.camera_filter.operators == ["police"]


def test_route_metrics_camera_count_total_optional():
    """RouteMetrics gains an optional camera_count_total without breaking
    pre-Issue-#6 callers that omit it."""
    m = RouteMetrics(length_m=100.0, exposure_score=2.0, camera_count_near_route=3)
    assert m.camera_count_total is None

    m2 = RouteMetrics(
        length_m=100.0,
        exposure_score=2.0,
        camera_count_near_route=3,
        camera_count_total=10,
    )
    assert m2.camera_count_total == 10
