import json
from collections import Counter
from types import SimpleNamespace

import pytest
import requests

from src.config.settings import OllamaSettings, DatabaseSettings, OverpassSettings
from src.utils.decorators import log_action


@pytest.fixture
def ollama_settings():
    return OllamaSettings(
        base_url="http://test-server/api/generate",
        timeout_seconds=0.1,
        model="test-model",
    )


@pytest.fixture
def db_settings(tmp_path):
    return DatabaseSettings(
        url=f"sqlite:///{tmp_path / 'test_memory.db'}",
        echo=False,
    )


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    # Replace LangChain Ollama client used inside SurveillanceLLM with our stub
    monkeypatch.setattr(
        "src.llm.surveillance_llm.OllamaLLM", lambda **kwargs: StubClient(kwargs)
    )


class StubClient:
    def __init__(self, settings):
        pass

    def invoke(self, prompt, **kwargs):
        # LangChain's Ollama returns a string directly, not a dict
        if hasattr(self, "_response") and "response" in self._response:
            return self._response["response"]
        return json.dumps({"error": "no response set"})

    def batch(self, prompts, **kwargs):
        # Return list of string responses for batch processing
        return [self.invoke(prompt, **kwargs) for prompt in prompts]


@pytest.fixture
def fake_sleep(monkeypatch):
    """
    Replace time.sleep so tests run instantly and record delays.
    """
    delays = []

    def _fake_sleep(seconds: float):
        delays.append(seconds)

    monkeypatch.setattr("time.sleep", _fake_sleep)
    return delays


def http_error(status: int) -> requests.HTTPError:
    r = requests.Response()
    r.status_code = status
    return requests.HTTPError(response=r)


@pytest.fixture
def ovp_settings(tmp_path):
    return OverpassSettings(
        endpoint="https://overpass.test/api",
        headers={"User-Agent": "pytest"},
        query_timeout=10,
        timeout=2,
        dir=str(tmp_path / "ovp"),
        max_attempts=2,
        base_delay=0.0,
    )


class _DummyResp(SimpleNamespace):
    def raise_for_status(self):
        if 400 <= self.status_code:
            raise requests.HTTPError(response=self)

    def json(self):
        return json.loads(self._text) if isinstance(self._text, str) else self._text

    text = property(lambda self: self._text)


@pytest.fixture
def patch_requests(monkeypatch):
    """
    Fixture that lets each test prepare responses:

        store["get"]  = _DummyResp(...)
        store["post"] = _DummyResp(...)
    """
    store = {"get": None, "post": None}

    def fake_get(*_, **__):
        return store["get"]

    def fake_post(*_, **__):
        return store["post"]

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return store


class MemoryStoreFake:
    def __init__(self):
        self.rows = []
        self._next_id = 1

    def store(self, agent_id: str, step: str, content: str):
        # Mirror the real `Memory` SQLModel default_factory so cache TTL
        # checks can read `row.timestamp` regardless of which store impl
        # the code under test happens to use. Each row also gets a synthetic
        # `id` so `touch(memory_id)` works against the fake.
        from datetime import datetime, timezone

        row = SimpleNamespace(
            id=self._next_id,
            agent_id=agent_id,
            step=step,
            content=content,
            timestamp=datetime.now(timezone.utc),
        )
        self._next_id += 1
        self.rows.append(row)
        return row

    def load(self, agent_id: str):
        return [r for r in self.rows if r.agent_id == agent_id]

    def touch(self, memory_id: int) -> None:
        from datetime import datetime, timezone

        for row in self.rows:
            if row.id == memory_id:
                row.timestamp = datetime.now(timezone.utc)
                return


@pytest.fixture
def mem_fake():
    return MemoryStoreFake()


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.debugs = []
        self.exceptions = []

    def info(self, msg):
        self.infos.append(msg)

    def debug(self, msg):
        self.debugs.append(msg)

    def exception(self, msg):
        self.exceptions.append(msg)


@pytest.fixture(autouse=True)
def swap_logger(monkeypatch):
    stub = DummyLogger()
    # patch both logger and any module-local imports
    monkeypatch.setattr("src.config.logger.logger", stub)
    monkeypatch.setattr("src.utils.decorators.logger", stub)
    return stub


class DummyAgent:
    name = "MyAgent"

    @log_action
    def simple(self, x, context=None):
        """just return x * 2"""
        return x * 2

    @log_action
    def make_list(self, n):
        return list(range(n))

    @log_action
    def save_file(self, path: str):
        return f"{path}.json"

    @log_action
    def blows_up(self):
        raise ValueError("oops")


# -------------------- Helpers for testing LLM/Chain functionality --------------------#
def make_raw_dump(tmp_path):
    """Return Path to a tiny overpass dump with one element."""
    raw = {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 55.0,
                "lon": 13.0,
                "tags": {"man_made": "surveillance"},
            }
        ]
    }
    p = tmp_path / "lund.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p, raw


def set_stub_response(data):
    """
    Tell StubClient (patched in conftest) to return a canned Ollama reply.
    """
    # from tests.conftest import StubClient

    StubClient._response = {
        # LangChainLLM expects {"response": "<json>"} which gets returned as string
        "response": json.dumps(data, separators=(",", ":")),
    }


@pytest.fixture
def sample_points():
    """Fixture providing sample GeoJSON points data"""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [12.994158, 55.607726],  # lon, lat
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [12.994268, 55.607826]},
            },
        ],
    }


@pytest.fixture
def geojson_file(tmp_path, sample_points):
    """Fixture creating a temporary GeoJSON file"""
    file_path = tmp_path / "test_points.geojson"
    file_path.write_text(json.dumps(sample_points), encoding="utf-8")
    return file_path


@pytest.fixture
def sample_stats():
    """Basic statistics fixture for testing charts"""
    return {
        "total": 100,
        "public_count": 30,
        "private_count": 50,
        "zone_counts": Counter({"downtown": 40, "suburbs": 30, "industrial": 30}),
        "zone_sensitivity_counts": Counter(
            {"downtown": 20, "suburbs": 10, "industrial": 15}
        ),
    }


@pytest.fixture
def sample_enriched_data():
    """Sample enriched data for testing sensitivity reasons"""
    return {
        "elements": [
            {
                "id": 1,
                "analysis": {"sensitive": True, "sensitive_reason": "near_school"},
            },
            {
                "id": 2,
                "analysis": {"sensitive": True, "sensitive_reason": "near_school"},
            },
            {
                "id": 3,
                "analysis": {"sensitive": True, "sensitive_reason": "residential_area"},
            },
        ]
    }


@pytest.fixture
def sample_hotspots():
    """Sample hotspots GeoJSON data"""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [12.994158, 55.607726]},
                "properties": {"cluster_id": 1, "count": 5},
            }
        ],
    }
