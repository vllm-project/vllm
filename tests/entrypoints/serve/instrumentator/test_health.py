# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request

from vllm.v1.engine.exceptions import EngineDeadError, EngineUnhealthyError


@pytest.mark.asyncio
async def test_health_ready_ok():
    from vllm.entrypoints.serve.instrumentator.health import health_ready

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_ready(mock_request)

    assert response.status_code == 200
    mock_engine_client.check_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_ready_unhealthy_error():
    from vllm.entrypoints.serve.instrumentator.health import health_ready

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.check_ready.side_effect = EngineUnhealthyError()
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_ready(mock_request)

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_ready_dead_error():
    from vllm.entrypoints.serve.instrumentator.health import health_ready

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.check_ready.side_effect = EngineDeadError()
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_ready(mock_request)

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_ready_no_engine():
    from vllm.entrypoints.serve.instrumentator.health import health_ready

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_app_state.engine_client = None
    mock_request.app.state = mock_app_state

    response = await health_ready(mock_request)

    assert response.status_code == 200
