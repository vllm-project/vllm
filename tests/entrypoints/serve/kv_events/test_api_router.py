# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, Request

from vllm.entrypoints.serve.kv_events.api_router import (
    attach_router,
    get_kv_event_sources,
)


@pytest.mark.asyncio
async def test_get_kv_event_sources_returns_engine_client_result():
    """The route returns the engine client's discovery list directly, as a
    bare JSON array (not wrapped in an envelope object)."""
    expected = [
        {
            "data_parallel_rank": 0,
            "endpoint": "tcp://10.0.0.1:5557",
            "replay_endpoint": None,
            "topic": "",
        }
    ]
    mock_engine_client = AsyncMock()
    mock_engine_client.get_kv_event_sources.return_value = expected

    mock_app_state = Mock()
    mock_app_state.engine_client = mock_engine_client
    mock_request = Mock(spec=Request)
    mock_request.app.state = mock_app_state

    response = await get_kv_event_sources(mock_request)

    assert response.status_code == 200
    assert json.loads(response.body) == expected


def test_kv_event_sources_route_registered_unconditionally():
    """attach_router must register the route with no dev-mode gate, unlike
    e.g. the tokenizer_info or dev-only routes."""
    app = FastAPI()
    attach_router(app)

    paths = {route.path for route in app.routes}
    assert "/kv_event_sources" in paths
