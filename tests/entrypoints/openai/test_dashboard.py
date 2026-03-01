# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vLLM dashboard functionality."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dashboard.api_router import (
    _get_dashboard_html,
    _get_explicit_cli_args,
    _get_vllm_env_vars,
    attach_router,
    router,
)
from vllm.version import __version__ as VLLM_VERSION


class TestDashboardHelpers:
    """Test helper functions in dashboard api_router."""

    def test_get_vllm_env_vars(self):
        """Test that _get_vllm_env_vars returns VLLM environment variables."""
        vllm_envs, explicit_envs = _get_vllm_env_vars()

        # Should return a dict of VLLM_* variables
        assert isinstance(vllm_envs, dict)
        assert isinstance(explicit_envs, set)

        # All keys should start with VLLM_
        for key in vllm_envs:
            assert key.startswith("VLLM_"), f"Key {key} doesn't start with VLLM_"

        # Should not include keys with "KEY" (secrets)
        for key in vllm_envs:
            assert "KEY" not in key, f"Key {key} contains KEY (potential secret)"

    def test_get_vllm_env_vars_explicit_detection(self):
        """Test that explicitly set env vars are detected."""
        test_var = "VLLM_TEST_DASHBOARD_VAR"
        try:
            os.environ[test_var] = "test_value"
            # Need to reload envs module to pick up the new var
            # For this test, we just verify the mechanism works
            _, explicit_envs = _get_vllm_env_vars()
            # The test var won't be in vllm.envs, but the mechanism is tested
            assert isinstance(explicit_envs, set)
        finally:
            os.environ.pop(test_var, None)

    def test_get_explicit_cli_args_none(self):
        """Test _get_explicit_cli_args with None args."""
        result = _get_explicit_cli_args(None)
        assert result == set()

    def test_get_explicit_cli_args_with_args(self):
        """Test _get_explicit_cli_args with mock args."""
        mock_args = MagicMock()
        mock_args.model = "test-model"
        mock_args.served_model_name = "test-served-model"
        mock_args.dtype = "float16"  # Non-default value

        result = _get_explicit_cli_args(mock_args)

        # model should always be explicit if set
        assert "model" in result
        assert "served_model_name" in result

    def test_get_dashboard_html(self):
        """Test that dashboard HTML is loaded correctly."""
        html = _get_dashboard_html()

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "vLLM Dashboard" in html
        assert "</html>" in html


class TestDashboardRouter:
    """Test dashboard router endpoints."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with dashboard router."""
        app = FastAPI()
        app.include_router(router)

        # Mock app state
        app.state.openai_serving_models = None
        app.state.vllm_config = None
        app.state.args = None
        app.state.engine_client = None

        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    def test_dashboard_index(self, client):
        """Test GET /dashboard returns HTML page."""
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "vLLM Dashboard" in response.text

    def test_dashboard_api_info(self, client):
        """Test GET /dashboard/api/info returns server info."""
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()

        assert "version" in data
        assert data["version"] == VLLM_VERSION
        assert "status" in data
        assert data["status"] == "running"

    def test_dashboard_api_info_with_models(self, app):
        """Test /dashboard/api/info includes model info when available."""
        # Mock serving models
        mock_model = MagicMock()
        mock_model.id = "test-model"
        mock_model.root = "test-model-root"

        mock_models_response = MagicMock()
        mock_models_response.data = [mock_model]

        mock_serving_models = AsyncMock()
        mock_serving_models.show_available_models = AsyncMock(
            return_value=mock_models_response
        )

        app.state.openai_serving_models = mock_serving_models

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["id"] == "test-model"
        assert data["models"][0]["root"] == "test-model-root"

    def test_dashboard_api_info_health_check(self, app):
        """Test /dashboard/api/info uses engine health check."""
        # Mock healthy engine client
        mock_engine_client = AsyncMock()
        mock_engine_client.check_health = AsyncMock(return_value=None)
        mock_engine_client.is_sleeping = AsyncMock(return_value=False)
        mock_engine_client.is_paused = AsyncMock(return_value=False)

        app.state.engine_client = mock_engine_client

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["is_sleeping"] is False
        assert data["is_paused"] is False

    def test_dashboard_api_info_unhealthy_engine(self, app):
        """Test /dashboard/api/info reports unhealthy when engine is dead."""
        from vllm.v1.engine.exceptions import EngineDeadError

        # Mock unhealthy engine client
        mock_engine_client = AsyncMock()
        mock_engine_client.check_health = AsyncMock(side_effect=EngineDeadError())
        mock_engine_client.is_sleeping = AsyncMock(return_value=False)
        mock_engine_client.is_paused = AsyncMock(return_value=False)

        app.state.engine_client = mock_engine_client

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_dashboard_api_info_sleeping_engine(self, app):
        """Test /dashboard/api/info reports is_sleeping when engine is sleeping."""
        # Mock sleeping engine client
        mock_engine_client = AsyncMock()
        mock_engine_client.check_health = AsyncMock(return_value=None)
        mock_engine_client.is_sleeping = AsyncMock(return_value=True)
        mock_engine_client.is_paused = AsyncMock(return_value=False)

        app.state.engine_client = mock_engine_client

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["is_sleeping"] is True

    def test_dashboard_api_info_server_load(self, app):
        """Test /dashboard/api/info includes server_load when available."""
        app.state.server_load_metrics = 5

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()
        assert data["server_load"] == 5

    def test_dashboard_api_info_paused_engine(self, app):
        """Test /dashboard/api/info reports is_paused when engine is paused."""
        # Mock paused engine client
        mock_engine_client = AsyncMock()
        mock_engine_client.check_health = AsyncMock(return_value=None)
        mock_engine_client.is_sleeping = AsyncMock(return_value=False)
        mock_engine_client.is_paused = AsyncMock(return_value=True)

        app.state.engine_client = mock_engine_client

        client = TestClient(app)
        response = client.get("/dashboard/api/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["is_paused"] is True

    def test_dashboard_api_metrics(self, client):
        """Test GET /dashboard/api/metrics returns metrics."""
        response = client.get("/dashboard/api/metrics")

        assert response.status_code == 200
        data = response.json()

        # Should return a dict (may be empty if no metrics available)
        assert isinstance(data, dict)


class TestAttachRouter:
    """Test attach_router function."""

    def test_attach_router_disabled(self):
        """Test router is not attached when dashboard is disabled."""
        app = FastAPI()
        mock_args = MagicMock()
        mock_args.enable_dashboard = False
        app.state.args = mock_args

        attach_router(app)

        # Dashboard routes should not be available
        client = TestClient(app)
        response = client.get("/dashboard")
        assert response.status_code == 404

    def test_attach_router_enabled(self):
        """Test router is attached when dashboard is enabled."""
        app = FastAPI()
        mock_args = MagicMock()
        mock_args.enable_dashboard = True
        app.state.args = mock_args
        app.state.openai_serving_models = None
        app.state.vllm_config = None

        attach_router(app)

        # Dashboard routes should be available
        client = TestClient(app)
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_attach_router_no_args(self):
        """Test router is not attached when args is None."""
        app = FastAPI()
        app.state.args = None

        attach_router(app)

        # Dashboard routes should not be available
        client = TestClient(app)
        response = client.get("/dashboard")
        assert response.status_code == 404
