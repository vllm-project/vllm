# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /render endpoints that expose prompt preprocessing."""

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import SpanKind, StatusCode
from prometheus_client import REGISTRY

from tests.utils import RemoteLaunchRenderServer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.render.api_router import router
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import _request_tracer
from vllm.renderers.online_renderer import OnlineRenderer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
SERVED_MODEL_NAME = "render-test"
TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture(params=["recording", "unsampled", "disabled"])
def render_spans(request, monkeypatch):
    exporter = InMemorySpanExporter()
    provider = (
        TracerProvider(sampler=ALWAYS_OFF)
        if request.param == "unsampled"
        else TracerProvider()
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "vllm.tracing.otel.trace.get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.render"),
    )
    if request.param == "disabled":
        monkeypatch.setattr("vllm.tracing.otel._IS_OTEL_AVAILABLE", False)
    # Each test installs its own provider, so its tracer must not be reused.
    _request_tracer.cache_clear()
    yield exporter, request.param == "recording"
    _request_tracer.cache_clear()
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
@pytest.mark.parametrize("outcome", ["rendered", "returned_error", "raised_error"])
@pytest.mark.parametrize(
    "tool_request", [False, True, None], ids=["plain", "named", "null"]
)
async def test_render_tool_outcomes(api_surface, outcome, tool_request, render_spans):
    """Count each tool render once without exposing request or error values."""
    app = FastAPI()
    app.include_router(router)
    serving = _build_serving_render()
    serving.model_config.model = MODEL_NAME
    serving.model_config.served_model_name = SERVED_MODEL_NAME
    app.state.serving_render = serving
    result = {"token_ids": [7, 8, 9]}
    handler = AsyncMock(return_value=SimpleNamespace(model_dump=lambda: result))
    if outcome == "returned_error":
        handler.return_value = serving.create_error_response("private-render-error")
    elif outcome == "raised_error":
        handler.side_effect = RuntimeError("private-render-error")
    payload: dict[str, object] = {
        "model": "private-request-model",
        "messages": [{"role": "user", "content": "private-prompt"}],
    }
    if api_surface == "messages":
        serving.render_messages_request = handler
        payload["max_tokens"] = 1
        route = "/v1/messages/render"
        tool: dict[str, object] = {
            "name": "private-tool",
            "input_schema": {"type": "object"},
        }
        choice: dict[str, object] = {"type": "tool", "name": "private-tool"}
    else:
        serving.render_chat_request = handler
        route = "/v1/chat/completions/render"
        tool = {
            "type": "function",
            "function": {"name": "private-tool", "parameters": {"type": "object"}},
        }
        choice = {"type": "function", "function": {"name": "private-tool"}}
    if tool_request:
        payload.update(tools=[tool], tool_choice=choice)
    elif tool_request is None:
        payload.update(tools=None, tool_choice=None)
    labels = {
        "model_name": SERVED_MODEL_NAME,
        "api_surface": api_surface,
        "status": "rendered" if outcome == "rendered" else "error",
    }
    metric = "vllm:tool_call_render_requests_total"
    before = REGISTRY.get_sample_value(metric, labels) or 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        if outcome == "raised_error":
            with pytest.raises(RuntimeError, match="private-render-error"):
                await client.post(
                    route, json=payload, headers={"traceparent": TRACEPARENT}
                )
        else:
            response = await client.post(
                route, json=payload, headers={"traceparent": TRACEPARENT}
            )
            assert response.status_code == (200 if outcome == "rendered" else 400)
            if outcome == "rendered":
                assert response.json() == result
    handler.assert_awaited_once()
    request, _ = handler.call_args.args
    assert request.model == payload["model"]
    assert request.tools is not None if tool_request else request.tools is None
    assert (REGISTRY.get_sample_value(metric, labels) or 0) - before == int(
        bool(tool_request)
    )
    for collector in REGISTRY.collect():
        for sample in collector.samples:
            if sample.name == metric:
                assert set(sample.labels) == {"model_name", "api_surface", "status"}
                assert sample.labels["model_name"] == SERVED_MODEL_NAME
                assert "private" not in str(sample.labels)
    exporter, recording = render_spans
    spans = exporter.get_finished_spans()
    if recording:
        assert len(spans) == 1
        span = spans[0]
        assert span.kind == SpanKind.SERVER
        assert span.context.trace_id == int(TRACEPARENT.split("-")[1], 16)
        assert span.parent.span_id == int(TRACEPARENT.split("-")[2], 16)
        assert dict(span.attributes) == (
            {
                "vllm.tool_calling.present": True,
                "vllm.tool_calling.tool_count": 1,
                "vllm.tool_calling.tool_choice": "named",
                "vllm.tool_calling.api_surface": api_surface,
                "vllm.tool_calling.render_status": labels["status"],
            }
            if tool_request
            else {}
        )
        assert not span.events
        assert span.status.status_code == (
            StatusCode.UNSET if outcome == "rendered" else StatusCode.ERROR
        )
        assert not span.status.description
    else:
        assert not spans


def _build_serving_render() -> ServingRender:
    serving = ServingRender.__new__(ServingRender)
    serving.model_config = SimpleNamespace(
        max_model_len=100,
        is_encoder_decoder=False,
    )
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.tool_server = MagicMock()
    serving.online_renderer = MagicMock()
    serving.online_renderer.create_error_response = (
        OnlineRenderer.create_error_response.__get__(serving.online_renderer)
    )
    serving.online_renderer.validate_chat_template = (
        OnlineRenderer.validate_chat_template.__get__(serving.online_renderer)
    )
    serving.online_renderer.trust_request_chat_template = True
    serving.online_renderer.parser = None
    serving._check_model = AsyncMock(return_value=None)
    return serving


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_completion_rejects_beam_search():
    serving = _build_serving_render()
    serving.online_renderer.render_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [7, 8, 9]}]
    )
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
        use_beam_search=True,
        n=2,
        length_penalty=0.7,
    )

    response = await serving.render_completion_request(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert (
        response.error.message == "Beam search is not supported by the render endpoint"
    )
    serving.online_renderer.render_completion.assert_not_awaited()


@pytest.mark.skip_global_cleanup
def test_responses_render_route_is_registered():
    assert any(route.path == "/v1/responses/render" for route in router.routes)


@pytest.mark.skip_global_cleanup
def test_responses_render_route_documents_not_implemented():
    route = next(
        route for route in router.routes if route.path == "/v1/responses/render"
    )

    assert HTTPStatus.NOT_IMPLEMENTED.value in route.responses


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_responses_returns_generate_request_without_stored_state():
    serving = _build_serving_render()
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(
            messages=[{"role": "user", "content": "Test prompt"}],
            engine_input={"prompt_token_ids": [7, 8, 9]},
        )
    )
    request = ResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        max_output_tokens=12,
        stream=True,
        cache_salt="request-salt",
        priority=3,
        kv_transfer_params={"do_remote_prefill": True},
        ec_transfer_params={"remote": True},
    )

    response = await serving.render_responses_request(request)

    assert response.token_ids == [7, 8, 9]
    assert response.request_id == request.request_id
    assert response.sampling_params.max_tokens == 12
    assert response.model == MODEL_NAME
    assert response.stream is True
    assert response.cache_salt == "request-salt"
    assert response.priority == 3
    assert response.kv_transfer_params == {"do_remote_prefill": True}
    assert response.ec_transfer_params == {"remote": True}
    assert response.sampling_params.extra_args is None
    serving.online_renderer.render_responses.assert_awaited_once_with(
        request,
        previous_messages=None,
        previous_response_outputs=None,
        tool_server=serving.tool_server,
        skip_mm_cache=True,
    )


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_chat_hoists_transfer_params_out_of_extra_args():
    serving = _build_serving_render()
    serving.online_renderer.render_chat = AsyncMock(
        return_value=([], [{"prompt_token_ids": [7, 8, 9]}])
    )
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        kv_transfer_params={"do_remote_prefill": True},
        ec_transfer_params={"remote": True},
        vllm_xargs={"foo": 1},
    )

    response = await serving.render_chat_request(request)

    assert response.kv_transfer_params == {"do_remote_prefill": True}
    assert response.ec_transfer_params == {"remote": True}
    assert response.sampling_params.extra_args == {"foo": 1}


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_completion_hoists_transfer_params_for_every_prompt():
    serving = _build_serving_render()
    serving.online_renderer.render_completion = AsyncMock(
        return_value=[
            {"prompt_token_ids": [7, 8, 9]},
            {"prompt_token_ids": [4, 5]},
        ]
    )
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt=["a", "b"],
        kv_transfer_params={"do_remote_prefill": True},
        ec_transfer_params={"remote": True},
    )

    responses = await serving.render_completion_request(request)

    assert len(responses) == 2
    for response in responses:
        assert response.kv_transfer_params == {"do_remote_prefill": True}
        assert response.ec_transfer_params == {"remote": True}
        assert response.sampling_params.extra_args is None


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("return_token_ids", [None, True])
async def test_render_forwards_return_token_ids(return_token_ids):
    """The engine only returns prompt_token_ids which derender needs to
    answer `return_token_ids` when the GenerateRequest asks for them."""
    serving = _build_serving_render()
    serving.online_renderer.render_chat = AsyncMock(
        return_value=([], [{"prompt_token_ids": [7, 8, 9]}])
    )
    serving.online_renderer.render_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [7, 8, 9]}]
    )

    chat = await serving.render_chat_request(
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Test prompt"}],
            return_token_ids=return_token_ids,
        )
    )
    (completion,) = await serving.render_completion_request(
        CompletionRequest(
            model=MODEL_NAME, prompt="a", return_token_ids=return_token_ids
        )
    )

    assert chat.return_token_ids == return_token_ids
    assert completion.return_token_ids == return_token_ids


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_responses_rejects_previous_response_id():
    serving = _build_serving_render()
    serving.online_renderer.render_responses = AsyncMock()
    request = ResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        previous_response_id="resp_previous",
    )

    response = await serving.render_responses_request(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert response.error.param == "previous_response_id"
    serving.online_renderer.render_responses.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_responses_rejects_empty_token_ids():
    serving = _build_serving_render()
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(
            messages=[],
            engine_input={"prompt_token_ids": []},
        )
    )

    response = await serving.render_responses_request(
        ResponsesRequest(model=MODEL_NAME, input="Test prompt")
    )

    assert isinstance(response, ErrorResponse)
    assert response.error.message == "No token_ids rendered"


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_render_responses_carries_reasoning_parser_kwargs():
    """/v1/responses gives the engine's reasoning parser the effective template
    kwargs, so the rendered request must carry them to generate."""
    serving = _build_serving_render()
    online = serving.online_renderer
    online.parser = MagicMock(reasoning_parser_cls=object)
    online.chat_template = None
    online.chat_template_content_format = "auto"
    online.default_chat_template_kwargs = {"enable_thinking": True}
    online.effective_chat_template_kwargs = (
        OnlineRenderer.effective_chat_template_kwargs.__get__(online)
    )
    online.render_responses = AsyncMock(
        return_value=MagicMock(messages=[], engine_input={"prompt_token_ids": [7]})
    )
    request = ResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        chat_template_kwargs={"enable_thinking": False},
    )

    response = await serving.render_responses_request(request)

    assert not isinstance(response, ErrorResponse)
    assert response.reasoning_parser_kwargs is not None
    kwargs = response.reasoning_parser_kwargs.chat_template_kwargs
    assert kwargs["enable_thinking"] is False


@pytest.fixture(scope="module")
def server():
    args: list[str] = ["--trust-request-chat-template"]

    with RemoteLaunchRenderServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_responses_render_basic(client):
    response = await client.post(
        "/v1/responses/render",
        json={
            "model": MODEL_NAME,
            "input": "When should a Responses handler return an empty string?",
            "max_output_tokens": 7,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == MODEL_NAME
    assert data["request_id"].startswith("resp_")
    assert data["sampling_params"]["max_tokens"] == 7
    assert data["token_ids"]


@pytest.mark.asyncio
async def test_responses_render_includes_prior_input_items(client):
    current_turn = {
        "role": "user",
        "content": "Which color should I remember?",
    }
    current_turn_response = await client.post(
        "/v1/responses/render",
        json={"model": MODEL_NAME, "input": [current_turn]},
    )
    multi_turn_response = await client.post(
        "/v1/responses/render",
        json={
            "model": MODEL_NAME,
            "input": [
                {"role": "user", "content": "Remember the color cobalt."},
                {"role": "assistant", "content": "I will remember cobalt."},
                current_turn,
            ],
        },
    )

    assert current_turn_response.status_code == 200
    assert multi_turn_response.status_code == 200
    current_turn_token_ids = current_turn_response.json()["token_ids"]
    multi_turn_token_ids = multi_turn_response.json()["token_ids"]
    assert len(multi_turn_token_ids) > len(current_turn_token_ids)


@pytest.mark.asyncio
async def test_responses_render_rejects_previous_response_id_over_http(client):
    response = await client.post(
        "/v1/responses/render",
        json={
            "model": MODEL_NAME,
            "input": "Continue the previous response.",
            "previous_response_id": "resp_previous",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "previous_response_id"


@pytest.mark.asyncio
async def test_completion_render_basic(client):
    """Test basic completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "When should a chat-completions handler return an empty string?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - list of GenerateRequest
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify first prompt is a GenerateRequest
    first_prompt = data[0]
    assert "token_ids" in first_prompt
    assert "sampling_params" in first_prompt
    assert "model" in first_prompt
    assert "request_id" in first_prompt
    assert isinstance(first_prompt["token_ids"], list)
    assert len(first_prompt["token_ids"]) > 0
    assert first_prompt["model"] == MODEL_NAME
    assert first_prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_chat_completion_render_basic(client):
    """Test basic chat completion render endpoint."""
    # Make request to render endpoint
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Returning an empty string for the prompt may be confusing."
                    ),
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure - should be a GenerateRequest
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0

    # Verify token IDs are integers and BOS token is present
    token_ids = data["token_ids"]
    assert all(isinstance(tid, int) for tid in token_ids)
    assert token_ids[0] == 1


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts(client):
    """Test completion render with multiple prompts."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return two GenerateRequest items
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify both prompts have GenerateRequest fields
    for prompt in data:
        assert "token_ids" in prompt
        assert "sampling_params" in prompt
        assert "model" in prompt
        assert "request_id" in prompt
        assert len(prompt["token_ids"]) > 0
        assert prompt["request_id"].startswith("cmpl-")


CHAT_BODY = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.asyncio
async def test_chat_render_honors_body_request_id(client):
    response = await client.post(
        "/v1/chat/completions/render", json={**CHAT_BODY, "request_id": "abc"}
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "chatcmpl-abc"


@pytest.mark.asyncio
async def test_chat_render_header_request_id_wins(client):
    response = await client.post(
        "/v1/chat/completions/render",
        json={**CHAT_BODY, "request_id": "abc"},
        headers={"X-Request-Id": "hdr"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "chatcmpl-hdr"


@pytest.mark.asyncio
async def test_chat_render_generates_unique_request_ids(client):
    ids = set()
    for _ in range(2):
        response = await client.post("/v1/chat/completions/render", json=CHAT_BODY)
        assert response.status_code == 200
        request_id = response.json()["request_id"]
        assert request_id.startswith("chatcmpl-")
        ids.add(request_id)

    assert len(ids) == 2


@pytest.mark.asyncio
async def test_completion_render_request_id_single_prompt(client):
    response = await client.post(
        "/v1/completions/render",
        json={"model": MODEL_NAME, "prompt": "Hello world", "request_id": "abc"},
    )

    assert response.status_code == 200
    assert [r["request_id"] for r in response.json()] == ["cmpl-abc"]


@pytest.mark.asyncio
async def test_completion_render_request_id_multiple_prompts(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
            "request_id": "abc",
        },
        headers={"X-Request-Id": "hdr"},
    )

    assert response.status_code == 200
    assert [r["request_id"] for r in response.json()] == ["cmpl-hdr-0", "cmpl-hdr-1"]


@pytest.mark.asyncio
async def test_chat_completion_render_multi_turn(client):
    """Test chat completion render with multi-turn conversation."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify tokenization occurred
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_render_with_stream_true(client):
    """Render accepts stream params but still returns JSON (non-streamed)."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Stream options should be accepted by /render.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0

    # /render should preserve stream fields on the returned token-in request.
    assert data.get("stream") is True
    assert isinstance(data.get("stream_options"), dict)
    assert data["stream_options"].get("include_usage") is True
    assert data["stream_options"].get("continuous_usage_stats") is True


@pytest.mark.asyncio
async def test_completion_render_error_invalid_model(client):
    """Test completion render with invalid model returns error."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": "invalid-model-name",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completion_render_error_invalid_model(client):
    """Test chat completion render with invalid model returns error."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": "invalid-model-name",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_completion_render_no_generation(client):
    """Verify render endpoint does not generate text."""
    # This test verifies that calling render is fast (no generation)
    import time

    start = time.perf_counter()
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Tell me a very long story about " * 10,
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    # Render should be fast (< 1 second) since no generation
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_chat_completion_render_with_sampling_params(client):
    """Verify sampling params are correctly returned by /render."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Test sampling params"}],
            "temperature": 0.123,
            "top_p": 0.456,
            "frequency_penalty": 1.1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "sampling_params" in data
    sampling_params = data["sampling_params"]

    assert sampling_params.get("temperature") == 0.123
    assert sampling_params.get("top_p") == 0.456
    assert sampling_params.get("frequency_penalty") == 1.1

    # Check that internal fields are not present
    assert "_all_stop_token_ids" not in sampling_params


@pytest.mark.asyncio
async def test_completion_render_emits_token_offsets(client):
    """With return_token_offsets, /v1/completions/render returns per-token
    (start, end) char offsets aligned with token_ids."""
    prompt = "Hello, world."
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    offsets = data[0]["token_offsets"]
    assert offsets is not None
    assert len(offsets) == len(data[0]["token_ids"])
    for start, end in offsets:
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start <= end <= len(prompt)


@pytest.mark.asyncio
async def test_completion_render_default_no_token_offsets(client):
    """Without the flag, token_offsets must be null (existing responses
    unchanged)."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Hello, world.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data[0]["token_offsets"] is None


@pytest.mark.asyncio
async def test_chat_render_emits_token_offsets(client):
    """With return_token_offsets, /v1/chat/completions/render returns
    per-token offsets relative to the templated prompt string."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, world."}],
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    offsets = data["token_offsets"]
    assert offsets is not None
    assert len(offsets) == len(data["token_ids"])
    for start, end in offsets:
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start <= end


@pytest.mark.asyncio
async def test_chat_render_default_no_token_offsets(client):
    """Without the flag, chat render token_offsets must be null."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, world."}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["token_offsets"] is None


@pytest.mark.asyncio
async def test_completion_render_truncated_token_offsets(client):
    """Truncation must shorten token_offsets together with token_ids.

    An explicit truncation_side turns off tokenizer-level truncation, so the
    tokenizer returns offsets for the whole prompt and they are reduced
    afterwards -- separately from token_ids. GenerateRequest documents that the
    two lists have equal length.
    """
    prompt = "The quick brown fox jumps over the lazy dog."
    keep = 4
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "return_token_offsets": True,
            "truncate_prompt_tokens": keep,
            "truncation_side": "left",
        },
    )

    assert response.status_code == 200
    data = response.json()
    token_ids = data[0]["token_ids"]
    offsets = data[0]["token_offsets"]

    assert len(token_ids) == keep
    assert len(offsets) == len(token_ids)

    # Equal length is not enough: truncating from the left keeps the *last*
    # tokens, so the surviving offsets must cover the end of the prompt.
    assert offsets[-1][1] == len(prompt)
    assert offsets[0][0] > 0


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts_token_offsets(client):
    """Each prompt in a batch gets its own offsets aligned with its tokens."""
    prompts = ["Hello, world.", "Goodbye, world."]
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": prompts,
            "return_token_offsets": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(prompts)
    for item, prompt in zip(data, prompts):
        offsets = item["token_offsets"]
        assert offsets is not None
        assert len(offsets) == len(item["token_ids"])
        for start, end in offsets:
            assert 0 <= start <= end <= len(prompt)


@pytest.mark.asyncio
async def test_messages_render_basic(client):
    """Test basic Anthropic Messages render endpoint."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Render this Anthropic message."}],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Single GenerateRequest, like chat render.
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert "sampling_params" in data
    assert "model" in data
    assert data["model"] == MODEL_NAME

    token_ids = data["token_ids"]
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(tid, int) for tid in token_ids)
    assert token_ids[0] == 1  # BOS


@pytest.mark.asyncio
async def test_messages_render_system_and_multi_turn(client):
    """System field + multi-turn messages render to a single prompt."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data["token_ids"]) > 0
    assert data["token_ids"][0] == 1  # BOS


@pytest.mark.asyncio
async def test_messages_render_merges_inline_system(client):
    """Inline system messages merge into the leading system block.

    The model's chat template rejects non-leading system messages, so the
    /v1/messages server path detects merge_inline_system=True and render must
    produce the same tokens as the manually pre-merged request.
    """
    inline = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )
    assert inline.status_code == 200

    merged = await client.post(
        "/v1/messages/render",
        json={
            "model": MODEL_NAME,
            "max_tokens": 16,
            "system": "You are a helpful assistant.Be brief.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )
    assert merged.status_code == 200

    assert inline.json()["token_ids"] == merged.json()["token_ids"]


@pytest.mark.asyncio
async def test_messages_render_error_invalid_model(client):
    """Messages render with an invalid model returns an error."""
    response = await client.post(
        "/v1/messages/render",
        json={
            "model": "invalid-model-name",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
