# SPDX-License-Identifier: Apache-2.0
"""
Tests for common_api.py — the aggregation layer that pulls in
``internal_api_server/common`` routers while excluding
vLLM-specific modules.

Covers:
- ``run_script_api`` IS now registered on the mp HTTP server.
- Other common endpoints (env, loglevel, metrics, …) ARE present.
- /run_script works in mp mode (no lmcache_adapter on app.state).
- /run_script works in inProcess mode (lmcache_adapter on app.state).
"""

# Standard
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.internal_api_server.common.run_script_api import (
    router as run_script_router,
)
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.http_api_registry import HTTPAPIRegistry


def _app_with_all_apis() -> FastAPI:
    app = FastAPI()
    registry = HTTPAPIRegistry(app)
    registry.register_all_apis()
    return app


class TestCommonApiAggregation:
    def test_run_script_endpoint_present(self):
        """/run_script must be registered on the mp server."""
        app = _app_with_all_apis()
        paths = set(app.openapi()["paths"])
        assert "/run_script" in paths

    def test_common_env_endpoint_present(self):
        """/env from env_api should be registered."""
        app = _app_with_all_apis()
        paths = set(app.openapi()["paths"])
        assert "/env" in paths

    def test_common_loglevel_endpoint_present(self):
        """/loglevel from loglevel_api should be registered."""
        app = _app_with_all_apis()
        paths = set(app.openapi()["paths"])
        assert "/loglevel" in paths

    def test_config_endpoint_present(self):
        """/config from config_api should be registered."""
        app = _app_with_all_apis()
        paths = set(app.openapi()["paths"])
        assert "/config" in paths


@dataclass
class _FakeAdapterConfig:
    script_allowed_imports: Optional[list] = None


class _FakeAdapter:
    def __init__(self, allowed=None):
        self.config = _FakeAdapterConfig(script_allowed_imports=allowed)


def _make_run_script_app(state_kwargs: dict) -> FastAPI:
    app = FastAPI()
    app.include_router(run_script_router)
    for k, v in state_kwargs.items():
        setattr(app.state, k, v)
    return app


class TestRunScriptMpMode:
    """Tests for /run_script in mp mode (no lmcache_adapter)."""

    def test_mp_mode_basic_script(self):
        """Script executes and returns result in mp mode."""
        app = _make_run_script_app({"configs": {"mp": MPServerConfig()}})
        client = TestClient(app)
        script = b"result = 1 + 2"
        resp = client.post(
            "/run_script",
            files={"script": ("test.py", BytesIO(script), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.text == "3"

    def test_mp_mode_no_script_file(self):
        """Missing script file returns 400."""
        app = _make_run_script_app({"configs": {"mp": MPServerConfig()}})
        client = TestClient(app)
        resp = client.post("/run_script", data={})
        assert resp.status_code == 400

    def test_mp_mode_allowed_imports(self):
        """script_allowed_imports in MPServerConfig is respected."""
        cfg = MPServerConfig(script_allowed_imports=["math"])
        app = _make_run_script_app({"configs": {"mp": cfg}})
        client = TestClient(app)
        script = b"math = __import__('math'); result = math.floor(3.9)"
        resp = client.post(
            "/run_script",
            files={"script": ("test.py", BytesIO(script), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.text == "3"

    def test_mp_mode_no_state(self):
        """No app.state set — allowed_imports falls back to empty list."""
        app = FastAPI()
        app.include_router(run_script_router)
        client = TestClient(app)
        script = b"result = 'ok'"
        resp = client.post(
            "/run_script",
            files={"script": ("test.py", BytesIO(script), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.text == "ok"


class TestRunScriptInProcessMode:
    """Tests for /run_script in inProcess mode (lmcache_adapter present)."""

    def test_inprocess_mode_basic_script(self):
        """Script executes and returns result in inProcess mode."""
        app = _make_run_script_app({"lmcache_adapter": _FakeAdapter()})
        client = TestClient(app)
        script = b"result = 'hello'"
        resp = client.post(
            "/run_script",
            files={"script": ("test.py", BytesIO(script), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.text == "hello"

    def test_inprocess_mode_default_result(self):
        """Script with no result variable returns default message."""
        app = _make_run_script_app({"lmcache_adapter": _FakeAdapter()})
        client = TestClient(app)
        script = b"x = 1"
        resp = client.post(
            "/run_script",
            files={"script": ("test.py", BytesIO(script), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.text == "Script executed successfully"
