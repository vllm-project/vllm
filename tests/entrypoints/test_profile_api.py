# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for profiler API router."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config import ProfilerConfig
from vllm.entrypoints.serve.profile.api_router import attach_router


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client with profiling methods."""
    client = AsyncMock()
    client.start_profile = AsyncMock()
    client.stop_profile = AsyncMock()
    client.start_mem_profile = AsyncMock()
    client.stop_mem_profile = AsyncMock()
    return client


@pytest.fixture
def app_with_torch_profiler(mock_engine_client):
    """Create app with torch profiler enabled."""
    app = FastAPI()
    app.state.args = MagicMock()
    app.state.args.profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/torch_profile",
    )
    app.state.engine_client = mock_engine_client
    attach_router(app)
    return app


@pytest.fixture
def app_with_mem_profiler(mock_engine_client):
    """Create app with memory profiler enabled."""
    app = FastAPI()
    app.state.args = MagicMock()
    app.state.args.profiler_config = ProfilerConfig(
        memory_profiler_dir="/tmp/mem_profile",
    )
    app.state.engine_client = mock_engine_client
    attach_router(app)
    return app


@pytest.fixture
def app_with_both_profilers(mock_engine_client):
    """Create app with both profilers enabled."""
    app = FastAPI()
    app.state.args = MagicMock()
    app.state.args.profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/torch_profile",
        memory_profiler_dir="/tmp/mem_profile",
    )
    app.state.engine_client = mock_engine_client
    attach_router(app)
    return app


@pytest.fixture
def app_no_profiler(mock_engine_client):
    """Create app with no profilers enabled."""
    app = FastAPI()
    app.state.args = MagicMock()
    app.state.args.profiler_config = ProfilerConfig()
    app.state.engine_client = mock_engine_client
    attach_router(app)
    return app


class TestProfilerEndpoints:
    """Tests for /start_profile and /stop_profile endpoints."""

    def test_start_profile_endpoint(self, app_with_torch_profiler):
        """Test /start_profile endpoint calls engine client."""
        client = TestClient(app_with_torch_profiler)
        response = client.post("/start_profile")

        assert response.status_code == 200
        app_with_torch_profiler.state.engine_client.start_profile.assert_called_once()

    def test_stop_profile_endpoint(self, app_with_torch_profiler):
        """Test /stop_profile endpoint calls engine client."""
        client = TestClient(app_with_torch_profiler)
        response = client.post("/stop_profile")

        assert response.status_code == 200
        app_with_torch_profiler.state.engine_client.stop_profile.assert_called_once()


class TestMemoryProfilerEndpoints:
    """Tests for /start_mem_profile and /stop_mem_profile endpoints."""

    def test_start_mem_profile_endpoint(self, app_with_mem_profiler):
        """Test /start_mem_profile endpoint calls engine client."""
        client = TestClient(app_with_mem_profiler)
        response = client.post("/start_mem_profile")

        assert response.status_code == 200
        app_with_mem_profiler.state.engine_client.start_mem_profile.assert_called_once()

    def test_stop_mem_profile_endpoint(self, app_with_mem_profiler):
        """Test /stop_mem_profile endpoint calls engine client."""
        client = TestClient(app_with_mem_profiler)
        response = client.post("/stop_mem_profile")

        assert response.status_code == 200
        app_with_mem_profiler.state.engine_client.stop_mem_profile.assert_called_once()


class TestRouterAttachment:
    """Tests for router attachment logic."""

    def test_router_attached_with_torch_profiler(self, app_with_torch_profiler):
        """Test router is attached when torch profiler is enabled."""
        client = TestClient(app_with_torch_profiler)
        response = client.post("/start_profile")
        assert response.status_code == 200

    def test_router_attached_with_mem_profiler(self, app_with_mem_profiler):
        """Test router is attached when memory profiler is enabled."""
        client = TestClient(app_with_mem_profiler)
        response = client.post("/start_mem_profile")
        assert response.status_code == 200

    def test_router_attached_with_both_profilers(self, app_with_both_profilers):
        """Test router is attached when both profilers are enabled."""
        client = TestClient(app_with_both_profilers)

        # Both endpoints should work
        response1 = client.post("/start_profile")
        assert response1.status_code == 200

        response2 = client.post("/start_mem_profile")
        assert response2.status_code == 200

    def test_router_not_attached_without_profiler(self, app_no_profiler):
        """Test router is not attached when no profiler is enabled."""
        client = TestClient(app_no_profiler)

        # Endpoints should return 404
        response1 = client.post("/start_profile")
        assert response1.status_code == 404

        response2 = client.post("/start_mem_profile")
        assert response2.status_code == 404

    def test_router_not_attached_with_none_config(self, mock_engine_client):
        """Test router is not attached when profiler_config is None."""
        app = FastAPI()
        app.state.args = MagicMock()
        app.state.args.profiler_config = None
        app.state.engine_client = mock_engine_client
        attach_router(app)

        client = TestClient(app)
        response = client.post("/start_profile")
        assert response.status_code == 404
