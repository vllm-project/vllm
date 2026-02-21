# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the admin control plane API router."""

import argparse
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.admin.api_router import attach_router, router


@pytest.fixture
def admin_app():
    """Create a FastAPI app with the admin router attached."""
    app = FastAPI()

    # Set up mock app state
    app.state.args = argparse.Namespace(
        enable_admin_api=True,
        admin_readonly=False,
    )
    app.state.engine_client = MagicMock()
    app.state.engine_client.check_health = AsyncMock()
    app.state.engine_client.is_running = True
    app.state.engine_client.is_paused = AsyncMock(return_value=False)
    app.state.engine_client.pause_generation = AsyncMock()
    app.state.engine_client.resume_generation = AsyncMock()

    # Mock model config
    model_config = MagicMock()
    model_config.model = "test-model"
    model_config.dtype = "float16"
    model_config.max_model_len = 4096
    app.state.engine_client.model_config = model_config

    # Mock parallel config
    parallel_config = MagicMock()
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.enable_expert_parallel = False
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    app.state.engine_client.vllm_config = vllm_config

    # Mock model serving
    models_mock = MagicMock()
    models_mock.show_available_models = AsyncMock(
        return_value=MagicMock(
            model_dump=MagicMock(
                return_value={"object": "list", "data": [{"id": "test-model"}]}
            )
        )
    )
    app.state.openai_serving_models = models_mock

    # Mock server load
    app.state.server_load_metrics = 5
    app.state.enable_server_load_tracking = True

    app.include_router(router)
    return app


@pytest.fixture
def admin_readonly_app(admin_app):
    """Create an admin app in read-only mode."""
    admin_app.state.args.admin_readonly = True
    return admin_app


@pytest.fixture
def client(admin_app):
    return TestClient(admin_app)


@pytest.fixture
def readonly_client(admin_readonly_app):
    return TestClient(admin_readonly_app)


class TestAdminHealth:
    def test_health_returns_200_when_healthy(self, client):
        resp = client.get("/v1/admin/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["is_running"] is True
        assert data["is_paused"] is False

    def test_health_returns_503_when_dead(self, admin_app):
        from vllm.v1.engine.exceptions import EngineDeadError

        admin_app.state.engine_client.check_health = AsyncMock(
            side_effect=EngineDeadError("dead")
        )
        client = TestClient(admin_app)
        resp = client.get("/v1/admin/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"


class TestAdminModels:
    def test_models_returns_model_list(self, client):
        resp = client.get("/v1/admin/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1


class TestAdminQueue:
    def test_queue_returns_load_metrics(self, client):
        resp = client.get("/v1/admin/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_load"] == 5
        assert data["load_tracking_enabled"] is True

    def test_queue_returns_null_when_tracking_disabled(self, admin_app):
        admin_app.state.enable_server_load_tracking = False
        client = TestClient(admin_app)
        resp = client.get("/v1/admin/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_load"] is None
        assert data["load_tracking_enabled"] is False


class TestAdminDrain:
    def test_drain_pauses_engine(self, client, admin_app):
        resp = client.post("/v1/admin/drain")
        assert resp.status_code == 200
        assert resp.json()["status"] == "drained"
        admin_app.state.engine_client.pause_generation.assert_called_once()

    def test_drain_blocked_in_readonly(self, readonly_client):
        resp = readonly_client.post("/v1/admin/drain")
        assert resp.status_code == 403
        assert "read-only" in resp.json()["error"]


class TestAdminResume:
    def test_resume_resumes_engine(self, client, admin_app):
        resp = client.post("/v1/admin/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resumed"
        admin_app.state.engine_client.resume_generation.assert_called_once()

    def test_resume_blocked_in_readonly(self, readonly_client):
        resp = readonly_client.post("/v1/admin/resume")
        assert resp.status_code == 403


class TestAdminConfig:
    def test_config_returns_engine_config(self, client):
        resp = client.get("/v1/admin/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test-model"
        assert data["tensor_parallel_size"] == 1


class TestAdminReloadModel:
    def test_reload_returns_501(self, client):
        resp = client.post("/v1/admin/reload_model")
        assert resp.status_code == 501


class TestAttachRouter:
    def test_router_not_attached_when_disabled(self):
        app = FastAPI()
        app.state.args = argparse.Namespace(enable_admin_api=False)
        attach_router(app)
        # No admin routes should be registered
        admin_routes = [
            r for r in app.routes if hasattr(r, "path") and "/v1/admin" in r.path
        ]
        assert len(admin_routes) == 0

    def test_router_attached_when_enabled(self):
        app = FastAPI()
        app.state.args = argparse.Namespace(
            enable_admin_api=True,
            admin_readonly=False,
        )
        attach_router(app)
        admin_routes = [
            r for r in app.routes if hasattr(r, "path") and "/v1/admin" in r.path
        ]
        assert len(admin_routes) > 0

    def test_router_not_attached_when_no_args(self):
        app = FastAPI()
        attach_router(app)
        admin_routes = [
            r for r in app.routes if hasattr(r, "path") and "/v1/admin" in r.path
        ]
        assert len(admin_routes) == 0
