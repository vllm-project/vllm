# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gRPC RenderGrpcServicer."""

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from starlette.datastructures import State

from vllm.entrypoints.grpc.render_servicer import RenderGrpcServicer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    cfg = MagicMock()
    cfg.model = overrides.get("model", "test-model")
    cfg.runner_type = overrides.get("runner_type", "generate")
    cfg.max_model_len = overrides.get("max_model_len", 4096)
    cfg.get_vocab_size.return_value = overrides.get("vocab_size", 32000)
    cfg.is_multimodal_model = overrides.get("is_multimodal_model", False)
    return cfg


def _make_state(*, openai_serving_render=None, model_config=None):
    state = State()
    vllm_config = MagicMock()
    vllm_config.model_config = model_config or _make_model_config()
    state.vllm_config = vllm_config
    state.openai_serving_render = openai_serving_render
    return state


def _make_context():
    ctx = AsyncMock()
    ctx.abort = AsyncMock(
        side_effect=grpc.aio.AbortError(grpc.StatusCode.INTERNAL, "aborted")
    )
    return ctx


def _make_generate_request(**overrides):
    """Create a mock GenerateRequest with model_dump support."""
    defaults = {
        "request_id": "chatcmpl-test-123",
        "token_ids": [1, 2, 3, 4],
        "model": "test-model",
        "stream": False,
        "sampling_params": {"temperature": 0.7, "max_tokens": 100},
    }
    defaults.update(overrides)
    mock = MagicMock()
    mock.model_dump.return_value = defaults
    return mock


START_TIME = 1000.0


@pytest.fixture
def state():
    return _make_state()


@pytest.fixture
def servicer(state):
    return RenderGrpcServicer(state, START_TIME)


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(servicer):
    resp = await servicer.HealthCheck(None, _make_context())
    assert resp.healthy is True
    assert resp.message == "Healthy"


# ---------------------------------------------------------------------------
# GetModelInfo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_info_defaults(servicer):
    resp = await servicer.GetModelInfo(None, _make_context())
    assert resp.model_path == "test-model"
    assert resp.is_generation is True
    assert resp.max_context_length == 4096
    assert resp.vocab_size == 32000
    assert resp.supports_vision is False


@pytest.mark.asyncio
async def test_get_model_info_custom():
    state = _make_state(
        model_config=_make_model_config(
            model="custom/model",
            runner_type="embed",
            max_model_len=2048,
            vocab_size=50000,
            is_multimodal_model=True,
        )
    )
    servicer = RenderGrpcServicer(state, START_TIME)
    resp = await servicer.GetModelInfo(None, _make_context())
    assert resp.model_path == "custom/model"
    assert resp.is_generation is False
    assert resp.max_context_length == 2048
    assert resp.vocab_size == 50000
    assert resp.supports_vision is True


# ---------------------------------------------------------------------------
# GetServerInfo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_server_info(servicer):
    with patch("vllm.entrypoints.grpc.render_servicer.time") as mock_time:
        mock_time.time.return_value = START_TIME + 60.0
        resp = await servicer.GetServerInfo(None, _make_context())
    assert resp.server_type == "vllm-render-grpc"
    assert abs(resp.uptime_seconds - 60.0) < 0.1


# ---------------------------------------------------------------------------
# RenderChat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_chat_unimplemented_when_render_is_none():
    """When openai_serving_render is None, RenderChat should abort."""
    state = _make_state(openai_serving_render=None)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with pytest.raises(grpc.aio.AbortError):
        await servicer.RenderChat(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    call_args = ctx.abort.call_args
    assert call_args[0][0] == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_render_chat_success():
    """RenderChat returns a RenderChatResponse with GenerateRequestProto."""
    mock_render = AsyncMock()
    gen_req = _make_generate_request(request_id="chatcmpl-abc")
    mock_render.render_chat_request.return_value = gen_req

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto:
        mock_from_proto.return_value = MagicMock()
        result = await servicer.RenderChat(MagicMock(), ctx)

    assert result.generate_request.request_id == "chatcmpl-abc"
    assert list(result.generate_request.token_ids) == [1, 2, 3, 4]
    mock_render.render_chat_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_render_chat_error_response():
    """RenderChat aborts with INVALID_ARGUMENT on ErrorResponse."""
    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

    mock_render = AsyncMock()
    error = ErrorResponse(
        error=ErrorInfo(message="bad request", type="invalid_request", code=400)
    )
    mock_render.render_chat_request.return_value = error

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with (
        patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto,
        pytest.raises(grpc.aio.AbortError),
    ):
        mock_from_proto.return_value = MagicMock()
        await servicer.RenderChat(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_render_chat_internal_error():
    """RenderChat aborts with INTERNAL on unexpected exception."""
    mock_render = AsyncMock()
    mock_render.render_chat_request.side_effect = RuntimeError("boom")

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with (
        patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto,
        pytest.raises(grpc.aio.AbortError),
    ):
        mock_from_proto.return_value = MagicMock()
        await servicer.RenderChat(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.INTERNAL


# ---------------------------------------------------------------------------
# RenderCompletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_completion_unimplemented_when_render_is_none():
    state = _make_state(openai_serving_render=None)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with pytest.raises(grpc.aio.AbortError):
        await servicer.RenderCompletion(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_render_completion_success():
    """RenderCompletion returns a list of GenerateRequestProto."""
    mock_render = AsyncMock()
    gen_reqs = [
        _make_generate_request(request_id="cmpl-1"),
        _make_generate_request(request_id="cmpl-2"),
    ]
    mock_render.render_completion_request.return_value = gen_reqs

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto:
        mock_from_proto.return_value = MagicMock()
        result = await servicer.RenderCompletion(MagicMock(), ctx)

    assert len(result.generate_requests) == 2
    assert result.generate_requests[0].request_id == "cmpl-1"
    assert result.generate_requests[1].request_id == "cmpl-2"
    mock_render.render_completion_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_render_completion_error_response():
    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

    mock_render = AsyncMock()
    error = ErrorResponse(
        error=ErrorInfo(message="invalid prompt", type="invalid_request", code=400)
    )
    mock_render.render_completion_request.return_value = error

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with (
        patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto,
        pytest.raises(grpc.aio.AbortError),
    ):
        mock_from_proto.return_value = MagicMock()
        await servicer.RenderCompletion(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_render_completion_internal_error():
    mock_render = AsyncMock()
    mock_render.render_completion_request.side_effect = ValueError("oops")

    state = _make_state(openai_serving_render=mock_render)
    servicer = RenderGrpcServicer(state, START_TIME)
    ctx = _make_context()

    with (
        patch("vllm.entrypoints.grpc.render_servicer.from_proto") as mock_from_proto,
        pytest.raises(grpc.aio.AbortError),
    ):
        mock_from_proto.return_value = MagicMock()
        await servicer.RenderCompletion(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.INTERNAL
