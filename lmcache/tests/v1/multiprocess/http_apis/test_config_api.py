# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``/config`` endpoint exposed by config_api.

Covers:
- Dataclass configs serialized with nested values.
- Plain-dict configs merged via ``make_json_safe``.
- Missing ``app.state.configs`` returns HTTP 503.
- Response body is valid JSON and is indented for readability.
"""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
import json

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_apis.config_api import router as config_router
from lmcache.v1.multiprocess.http_apis.dependencies import build_context


@dataclass
class _FakeMPConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    path: Path = Path("/tmp/mp")


@dataclass
class _FakeStorageConfig:
    backend: str = "local"
    capacity: int = 1024
    tags: list = field(default_factory=lambda: ["a", "b"])


def _make_app(configs):
    app = FastAPI()
    app.include_router(config_router)
    if configs is not None:
        app.state.configs = configs
    return app


class TestConfigEndpoint:
    def test_dataclass_configs_serialized(self):
        app = _make_app(
            {
                "mp": _FakeMPConfig(host="1.2.3.4", port=8000),
                "storage": _FakeStorageConfig(backend="redis"),
            }
        )
        client = TestClient(app)
        resp = client.get("/config")

        assert resp.status_code == 200
        body = resp.json()
        assert body["mp"] == {
            "host": "1.2.3.4",
            "port": 8000,
            "path": "/tmp/mp",
        }
        assert body["storage"]["backend"] == "redis"
        assert body["storage"]["capacity"] == 1024
        assert body["storage"]["tags"] == ["a", "b"]

    def test_plain_dict_config_merged(self):
        """Non-dataclass values still go through make_json_safe."""
        app = _make_app({"extra": {"k": Path("/v")}})
        client = TestClient(app)

        body = client.get("/config").json()
        assert body == {"extra": {"k": "/v"}}

    def test_returns_503_when_configs_missing(self):
        """/config returns 503 if app.state.configs is absent."""
        client = TestClient(_make_app(configs=None))
        resp = client.get("/config")

        assert resp.status_code == 503
        assert resp.json() == {"error": "configs not initialized"}

    def test_response_is_indented_json(self):
        """Indented JSON renderer keeps the payload human-readable."""
        app = _make_app({"mp": _FakeMPConfig()})
        client = TestClient(app)

        raw = client.get("/config").text
        # Indented output has newlines and 2-space indentation.
        assert "\n" in raw
        assert '  "mp"' in raw
        # And the body is still valid JSON.
        assert json.loads(raw)["mp"]["host"] == "0.0.0.0"

    def test_empty_configs_returns_empty_object(self):
        client = TestClient(_make_app(configs={}))
        resp = client.get("/config")

        assert resp.status_code == 200
        assert resp.json() == {}


@pytest.mark.parametrize(
    "configs,expected_key",
    [
        ({"only": _FakeMPConfig()}, "only"),
        ({"a": _FakeMPConfig(), "b": _FakeStorageConfig()}, "b"),
    ],
)
def test_arbitrary_config_keys_round_trip(configs, expected_key):
    client = TestClient(_make_app(configs))
    body = client.get("/config").json()
    assert expected_key in body


# =============================================================================
# Adapters -- ``GET /config/adapters`` (listing configured cache adapters is a
# configuration-inspection concern, so it lives in the config group).
# =============================================================================


@dataclass
class _FakeDescriptor:
    """Minimal adapter descriptor -- only ``type_name`` is read."""

    type_name: str = "s3"


@dataclass
class _FakeStorageManager:
    """Implements ``l2_adapters()`` and ``reconfigurable_l2_backends()``. An
    empty ``adapters`` list reproduces the "no L2 adapters configured"
    condition; ``reconfigurable`` names the type_names that report runtime
    reconfiguration support."""

    adapters: list = field(default_factory=list)
    reconfigurable: set = field(default_factory=set)

    def l2_adapters(self):
        return [(_FakeDescriptor(type_name=n), a) for n, a in self.adapters]

    def reconfigurable_l2_backends(self) -> set:
        return set(self.reconfigurable)


class _FakeEngine:
    def __init__(self, sm: _FakeStorageManager):
        self.storage_manager = sm


def _make_adapters_app(sm) -> FastAPI:
    """Build an app with the config router and a context wrapping a fake engine
    (or no context, reproducing the not-initialized condition)."""
    app = FastAPI()
    app.include_router(config_router)
    if sm is not None:
        app.state.context = build_context(_FakeEngine(sm))
    return app


class TestConfigAdaptersEndpoint:
    def test_empty_adapter_list_returns_empty_array(self):
        client = TestClient(_make_adapters_app(_FakeStorageManager()))
        resp = client.get("/config/adapters")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"adapters": []}

    def test_lists_all_adapters_with_primary_and_reconfigurable_flags(self):
        sm = _FakeStorageManager(
            adapters=[("s3", object()), ("dax", object())],
            reconfigurable={"dax"},
        )
        client = TestClient(_make_adapters_app(sm))
        resp = client.get("/config/adapters")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "adapters": [
                {
                    "index": 0,
                    "type_name": "s3",
                    "tier": "l2",
                    "primary": True,
                    "reconfigurable": False,
                },
                {
                    "index": 1,
                    "type_name": "dax",
                    "tier": "l2",
                    "primary": False,
                    "reconfigurable": True,
                },
            ]
        }

    def test_503_when_not_initialized(self):
        client = TestClient(_make_adapters_app(None))
        assert client.get("/config/adapters").status_code == 503
