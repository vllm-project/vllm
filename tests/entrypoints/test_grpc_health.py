# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from grpc_health.v1 import health_pb2

from vllm.entrypoints.grpc_server import VllmHealthServicer

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@pytest.fixture
def async_llm():
    mock = MagicMock()
    mock.check_health = AsyncMock()
    return mock


@pytest.fixture
def context():
    return MagicMock(spec=grpc.aio.ServicerContext)


@pytest.fixture
def servicer(async_llm):
    return VllmHealthServicer(async_llm)


@pytest.fixture
def request_msg():
    msg = MagicMock()
    msg.service = ""
    return msg


@pytest.mark.asyncio
async def test_check_returns_serving_when_engine_healthy(
    servicer, request_msg, context, async_llm
):
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVING
    async_llm.check_health.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_returns_serving_for_vllm_service(
    servicer, request_msg, context, async_llm
):
    request_msg.service = "vllm.grpc.engine.VllmEngine"
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVING
    async_llm.check_health.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_returns_not_serving_when_engine_errored(
    servicer, request_msg, context, async_llm
):
    async_llm.check_health = AsyncMock(side_effect=Exception("engine dead"))
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == NOT_SERVING


@pytest.mark.asyncio
async def test_check_returns_not_serving_when_shutting_down(
    servicer, request_msg, context, async_llm
):
    servicer.set_not_serving()
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == NOT_SERVING
    async_llm.check_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_returns_not_found_for_unknown_service(
    servicer, request_msg, context
):
    request_msg.service = "nonexistent.Service"
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVICE_UNKNOWN
    context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


@pytest.mark.asyncio
async def test_watch_yields_current_status(servicer, request_msg, context, async_llm):
    request_msg.service = ""
    results = []
    async for response in servicer.Watch(request_msg, context):
        results.append(response)
    assert len(results) == 1
    assert results[0].status == SERVING
    async_llm.check_health.assert_awaited_once()
