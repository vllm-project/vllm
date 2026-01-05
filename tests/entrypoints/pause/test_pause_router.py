# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the step-barrier pause endpoints in vllm/entrypoints/serve/pause/.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.pause import api_router
from vllm.entrypoints.serve.pause.api_router import attach_router


@pytest.fixture
def mock_engine_core():
    """Create a mock engine core with utility call methods."""
    engine_core = MagicMock()
    engine_core.core_engines = [b"\x00\x00"]  # Single engine identity

    async def call_utility_async(method, *args, **kwargs):
        if method == "pause":
            return 42  # Mock step_counter
        if method == "resume":
            return None
        if method == "run_until_target_step_count":
            return None
        if method == "get_step_counter":
            # Return a high value so barrier polling succeeds immediately
            return 1000
        if method == "is_engine_paused":
            return True
        raise ValueError(f"Unknown utility method: {method}")

    async def _call_utility_async(method, *args, engine=None, **kwargs):
        return await call_utility_async(method, *args, **kwargs)

    engine_core.call_utility_async = AsyncMock(side_effect=call_utility_async)
    engine_core._call_utility_async = AsyncMock(side_effect=_call_utility_async)
    return engine_core


@pytest.fixture
def mock_async_llm(mock_engine_core):
    """Create a mock AsyncLLM client that passes the isinstance check."""
    # Create a mock that will be returned by _require_async_llm
    client = MagicMock()
    client.engine_core = mock_engine_core
    return client


@pytest.fixture
def app(mock_async_llm):
    """Create a FastAPI app with the pause router attached."""
    app = FastAPI()
    app.state.engine_client = mock_async_llm
    attach_router(app)
    return app


@pytest.fixture
def client(app, mock_async_llm):
    """Create a test client for the app, patching the AsyncLLM check."""
    # Patch _require_async_llm to return our mock directly
    with patch.object(
        api_router, "_require_async_llm", return_value=mock_async_llm
    ):
        yield TestClient(app)


class TestPauseStepEndpoint:
    """Tests for POST /pause/step with various query parameters."""

    def test_pause_step_default_with_barrier(self, client, mock_engine_core):
        """Default behavior: pause + wait for barrier at step+1."""
        response = client.post("/pause/step")
        assert response.status_code == 200
        data = response.json()
        assert data["paused"] is True
        # Final step should be from the barrier (1000 from mock)
        assert data["step_counter"] == 1000
        # Verify barrier was called with step_counter + 1 = 43
        mock_engine_core.call_utility_async.assert_any_call(
            "run_until_target_step_count", 43
        )

    def test_pause_step_no_barrier(self, client, mock_engine_core):
        """With no_barrier=true: fast pause, immediate return."""
        response = client.post("/pause/step?no_barrier=true")
        assert response.status_code == 200
        data = response.json()
        assert data["paused"] is True
        assert data["step_counter"] == 42  # Step counter at pause time
        # Verify run_until_target_step_count was NOT called
        for call in mock_engine_core.call_utility_async.call_args_list:
            assert call[0][0] != "run_until_target_step_count"

    def test_pause_step_custom_barrier(self, client, mock_engine_core):
        """With barrier=<value>: wait until specified step."""
        response = client.post("/pause/step?barrier=100")
        assert response.status_code == 200
        data = response.json()
        assert data["paused"] is True
        # Verify barrier was called with custom target
        mock_engine_core.call_utility_async.assert_any_call(
            "run_until_target_step_count", 100
        )

    def test_pause_step_barrier_overrides_default(self, client, mock_engine_core):
        """Explicit barrier value overrides the default step+1."""
        response = client.post("/pause/step?barrier=50")
        assert response.status_code == 200
        # Verify barrier was called with 50, not 43 (step_counter + 1)
        mock_engine_core.call_utility_async.assert_any_call(
            "run_until_target_step_count", 50
        )
