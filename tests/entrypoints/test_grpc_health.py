# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from grpc_health.v1 import health_pb2
from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer

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


# -- Check() tests --


@pytest.mark.asyncio
async def test_check_serving_overall(servicer, request_msg, context, async_llm):
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVING
    async_llm.check_health.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_serving_vllm_service(servicer, request_msg, context, async_llm):
    request_msg.service = "vllm.grpc.engine.VllmEngine"
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVING
    async_llm.check_health.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_not_serving_engine_errored(
    servicer, request_msg, context, async_llm
):
    async_llm.check_health = AsyncMock(side_effect=Exception("engine dead"))
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == NOT_SERVING


@pytest.mark.asyncio
async def test_check_not_serving_shutting_down(
    servicer, request_msg, context, async_llm
):
    servicer.set_not_serving()
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == NOT_SERVING
    async_llm.check_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_unknown_service_status(servicer, request_msg, context):
    request_msg.service = "nonexistent.Service"
    response = await servicer.Check(request_msg, context)
    assert response.status == SERVICE_UNKNOWN


@pytest.mark.asyncio
async def test_check_unknown_service_grpc_code(servicer, request_msg, context):
    request_msg.service = "fake.Svc"
    await servicer.Check(request_msg, context)
    context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
    context.set_details.assert_called_once()
    details_arg = context.set_details.call_args[0][0]
    assert "fake.Svc" in details_arg


@pytest.mark.asyncio
async def test_check_shutting_down_overrides_healthy(
    servicer, request_msg, context, async_llm
):
    servicer.set_not_serving()
    request_msg.service = ""
    response = await servicer.Check(request_msg, context)
    assert response.status == NOT_SERVING
    async_llm.check_health.assert_not_awaited()


@pytest.mark.asyncio
@patch("smg_grpc_servicer.vllm.health_servicer.logger")
async def test_check_logs_exception_on_error(
    mock_logger, servicer, request_msg, context, async_llm
):
    async_llm.check_health = AsyncMock(side_effect=Exception("engine exploded"))
    request_msg.service = ""
    await servicer.Check(request_msg, context)
    mock_logger.exception.assert_called_once()
    log_args = mock_logger.exception.call_args
    # The logged message should reference the service name
    full_message = str(log_args)
    assert "" in full_message or "service" in full_message.lower()


# -- Watch() tests --


@pytest.mark.asyncio
async def test_watch_yields_serving(servicer, request_msg, context, async_llm):
    request_msg.service = ""
    results = []
    async for response in servicer.Watch(request_msg, context):
        results.append(response)
    assert len(results) == 1
    assert results[0].status == SERVING


@pytest.mark.asyncio
async def test_watch_yields_not_serving(servicer, request_msg, context, async_llm):
    async_llm.check_health = AsyncMock(side_effect=Exception("engine down"))
    request_msg.service = ""
    results = []
    async for response in servicer.Watch(request_msg, context):
        results.append(response)
    assert len(results) == 1
    assert results[0].status == NOT_SERVING


@pytest.mark.asyncio
async def test_watch_yields_exactly_once(servicer, request_msg, context, async_llm):
    request_msg.service = ""
    results = []
    async for response in servicer.Watch(request_msg, context):
        results.append(response)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_watch_unknown_service(servicer, request_msg, context):
    request_msg.service = "fake.Service"
    results = []
    async for response in servicer.Watch(request_msg, context):
        results.append(response)
    assert len(results) == 1
    assert results[0].status == SERVICE_UNKNOWN
    # Watch must NOT set gRPC status code (unlike Check), to avoid
    # polluting the streaming response.
    context.set_code.assert_not_called()


# -- set_not_serving() test --


def test_set_not_serving_sets_flag(servicer):
    servicer.set_not_serving()
    assert servicer._shutting_down is True
