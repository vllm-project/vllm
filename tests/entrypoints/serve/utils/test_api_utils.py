# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.generate.base.protocol import StreamOptions
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    should_include_usage,
)


def _traced_request(*, enabled=True):
    app = FastAPI()
    app.state.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            otlp_traces_endpoint="test" if enabled else None
        )
    )

    async def receive():
        await asyncio.Event().wait()

    return Request({"type": "http", "app": app, "headers": []}, receive)


@pytest.mark.asyncio
@pytest.mark.parametrize("api,num_choices", [("chat", 1), ("chat", 2), ("messages", 1)])
@pytest.mark.parametrize(
    "tool_fields",
    [
        "plain",
        "null",
        "tools",
        "empty",
        "auto",
        "none",
        "required",
        "named",
        "parallel",
    ],
)
@pytest.mark.parametrize("sampled", [False, True])
async def test_inference_spans_wait_for_api_conversion(
    inference_spans, api, num_choices, tool_fields, sampled, monkeypatch
):
    """Both native routes finish their conversions before exporting engine spans."""
    from vllm.entrypoints.anthropic.api_router import create_messages
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    from vllm.entrypoints.openai.chat_completion.api_router import (
        create_chat_completion,
    )
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.engine.protocol import UsageInfo
    from vllm.exceptions import VLLMValidationError
    from vllm.tracing import get_request_trace

    exporter, finish_engine = inference_spans
    raw_request = _traced_request()
    headers = {
        "traceparent": f"00-{'1' * 32}-{'2' * 16}-{'01' if sampled else '00'}",
        "tracestate": "test=parent",
        "vllm.tool_calling.tool_count": "999",
        "vllm.tool_calling.tool_choice": "private-spoof",
    }
    original_headers = headers.copy()

    async def complete(request, raw_request):
        # Rendering/parsing may mutate the request after native intent is captured.
        request.tools = []
        request.tool_choice = "none"
        spans = [finish_engine(str(index), headers) for index in range(num_choices)]
        assert all(span.is_recording() == sampled for span in spans)
        assert not exporter.get_finished_spans()
        trace = get_request_trace()
        assert trace is not None
        trace.record_tool_response()
        response = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=index,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                )
                for index in range(num_choices)
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        return response

    fields: dict[str, Any] = {}
    choice_label = "unset"
    tool_count = 0
    if tool_fields == "null":
        fields = {"tools": None, "tool_choice": None}
    elif tool_fields == "empty":
        fields = {"tools": []}
    elif tool_fields == "parallel":
        # Native Messages has no parallel_tool_calls field.
        fields = {"parallel_tool_calls": False} if api == "chat" else {"tools": []}
        if api == "chat":
            choice_label = "none"
    elif tool_fields == "none":
        fields = {"tool_choice": "none" if api == "chat" else {"type": "none"}}
        choice_label = "none"
    elif tool_fields != "plain":
        tool_count = 1
        fields = {
            "tools": [{"type": "function", "function": {"name": "private-tool"}}]
            if api == "chat"
            else [{"name": "private-tool", "input_schema": {"type": "object"}}]
        }
        if tool_fields != "tools":
            choice_label = tool_fields
            if tool_fields == "named":
                fields["tool_choice"] = (
                    {"type": "function", "function": {"name": "private-tool"}}
                    if api == "chat"
                    else {"type": "tool", "name": "private-tool"}
                )
            else:
                fields["tool_choice"] = (
                    tool_fields
                    if api == "chat"
                    else {"type": "any" if tool_fields == "required" else tool_fields}
                )
        if api == "chat" and tool_fields == "tools":
            choice_label = "auto"  # Chat normalizes the default during validation.

    if api == "chat":
        handler = OpenAIServingChat.__new__(OpenAIServingChat)
        handler.has_kv_connector = False
        monkeypatch.setattr(handler, "_create_chat_completion", complete)
        raw_request.app.state.openai_serving_chat = handler
        if tool_fields == "empty":
            with pytest.raises(VLLMValidationError, match="must not be an empty array"):
                ChatCompletionRequest(messages=[], **fields)
            assert not exporter.get_finished_spans()
            return
        request = ChatCompletionRequest(messages=[], n=num_choices, **fields)
        response = await create_chat_completion(request, raw_request)
    else:
        handler = AnthropicServingMessages.__new__(AnthropicServingMessages)
        handler.has_kv_connector = False
        handler._merge_inline_system = False
        monkeypatch.setattr(handler, "_create_chat_completion", complete)
        raw_request.app.state.anthropic_serving_messages = handler
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1,
            messages=[],
            **fields,
        )
        response = await create_messages(request, raw_request)

    assert response.status_code == 200
    assert b"hello" in response.body
    finished = exporter.get_finished_spans()
    assert len(finished) == (num_choices if sampled else 0)
    for span in finished:
        attributes = {
            key: value
            for key, value in span.attributes.items()
            if key.startswith("vllm.tool_calling.")
        }
        assert attributes == (
            {}
            if tool_fields in ("plain", "null")
            else {
                "vllm.tool_calling.present": True,
                "vllm.tool_calling.tool_count": tool_count,
                "vllm.tool_calling.tool_choice": choice_label,
                "vllm.tool_calling.api_surface": "chat_completions"
                if api == "chat"
                else "messages",
                "vllm.tool_calling.response_has_tool_calls": False,
                "vllm.tool_calling.tool_calls_count": 0,
                "vllm.tool_calling.parser_status": "no_tool_calls",
            }
        )
        assert span.context.trace_id == int("1" * 32, 16)
        assert span.parent.span_id == int("2" * 16, 16)
        assert "private" not in str(span.attributes)
    assert headers == original_headers
    assert get_request_trace() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("has_tools", [False, True])
@pytest.mark.parametrize(
    "outcome",
    ["completed", "disconnect", "send_error", "cancel", "api_error", "cleanup_error"],
)
async def test_streaming_inference_span_completion(inference_spans, outcome, has_tools):
    """The ASGI response owns spans even when interruption happens during send."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tracing import StatusCode, get_request_trace

    exporter, finish_engine = inference_spans
    raw_request = _traced_request()

    @api_utils.with_cancellation
    async def handler(body, raw_request):
        trace = get_request_trace()
        assert trace is not None
        trace.capture_tool_request(body, "chat_completions")

        async def stream():
            span = finish_engine()
            assert span.is_recording()
            try:
                yield b"data: hello\n\n"
                if outcome == "disconnect":
                    await asyncio.Event().wait()
                if outcome == "api_error":
                    from vllm.entrypoints.generate.base.serving import (
                        GenerateBaseServing,
                    )

                    serving = GenerateBaseServing.__new__(GenerateBaseServing)
                    yield serving.create_streaming_error_response("sensitive error")
            finally:
                assert span.is_recording()
                trace.record_tool_response()
                if outcome == "cleanup_error":
                    raise RuntimeError("cleanup failed")

        return StreamingResponse(stream(), headers={"x-test": "kept"})

    response = await handler(
        ChatCompletionRequest(messages=[], parallel_tool_calls=False)
        if has_tools
        else ChatCompletionRequest(messages=[]),
        raw_request,
    )
    from starlette.background import BackgroundTask

    background_calls: list[bool] = []
    response.background = BackgroundTask(background_calls.append, True)
    response.status_code = 201
    response.headers["x-added"] = "also-kept"
    first_chunk = asyncio.Event()
    sent = []

    async def receive():
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message["body"]:
            assert not exporter.get_finished_spans()
            first_chunk.set()
            if outcome in ("send_error", "cleanup_error"):
                raise OSError("client disappeared")
            if outcome == "cancel":
                raise asyncio.CancelledError()

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.0" if outcome == "disconnect" else "2.4"},
    }
    if outcome in ("completed", "disconnect", "api_error"):
        await asyncio.wait_for(response(scope, receive, send), timeout=5)
    else:
        from starlette.requests import ClientDisconnect

        with pytest.raises(
            ClientDisconnect
            if outcome in ("send_error", "cleanup_error")
            else asyncio.CancelledError
        ):
            await response(scope, receive, send)

    (span,) = exporter.get_finished_spans()
    assert (span.status.status_code == StatusCode.ERROR) == (outcome != "completed")
    if has_tools:
        assert span.attributes["vllm.tool_calling.parser_status"] == (
            "no_tool_calls" if outcome == "completed" else "interrupted"
        )
    else:
        assert not any(key.startswith("vllm.tool_calling.") for key in span.attributes)
    assert (b"x-test", b"kept") in sent[0]["headers"]
    assert (b"x-added", b"also-kept") in sent[0]["headers"]
    assert sent[0]["status"] == 201
    assert bool(background_calls) == (
        outcome in ("completed", "disconnect", "api_error")
    )
    assert get_request_trace() is None


@pytest.mark.asyncio
async def test_background_inference_outlives_initial_response(inference_spans):
    """An independent worker owns its spans after the HTTP request has returned."""
    exporter, finish_engine = inference_spans
    started = asyncio.Event()
    finish = asyncio.Event()
    tasks = []

    async def work():
        span = finish_engine()
        started.set()
        await finish.wait()
        assert span.is_recording()

    @api_utils.with_cancellation
    async def handler(body, raw_request):
        tasks.append(
            asyncio.create_task(
                api_utils.run_with_inference_trace(work(), enabled=True)
            )
        )
        await started.wait()
        return JSONResponse({"status": "queued"})

    response = await handler({}, _traced_request())
    assert response.status_code == 200
    assert not exporter.get_finished_spans()
    finish.set()
    await tasks[0]
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["response", "exception"])
async def test_inference_conversion_failure_closes_spans(inference_spans, failure):
    from vllm.tracing import StatusCode

    exporter, finish_engine = inference_spans

    @api_utils.with_cancellation
    async def handler(body, raw_request):
        finish_engine()
        if failure == "exception":
            raise ValueError("sensitive conversion error")
        return JSONResponse({}, status_code=400)

    if failure == "exception":
        with pytest.raises(ValueError, match="sensitive conversion error"):
            await handler({}, _traced_request())
    else:
        await handler({}, _traced_request())
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is None
    assert not span.events


@pytest.mark.asyncio
async def test_without_api_owner_keeps_engine_completion(inference_spans):
    from vllm.tracing import get_request_trace

    exporter, finish_engine = inference_spans

    @api_utils.with_cancellation
    async def handler(body, raw_request):
        assert get_request_trace() is None
        finish_engine()
        assert len(exporter.get_finished_spans()) == 1
        return JSONResponse({})

    await handler({}, _traced_request(enabled=False))


@pytest.mark.asyncio
async def test_unconsumed_stream_closes_admitted_spans(inference_spans):
    import gc

    from vllm.tracing import StatusCode

    exporter, finish_engine = inference_spans

    @api_utils.with_cancellation
    async def handler(body, raw_request):
        finish_engine()

        async def stream():
            yield b"unused"

        return StreamingResponse(stream())

    response = await handler({}, _traced_request())
    assert not exporter.get_finished_spans()
    del response
    gc.collect()
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_with_cancellation_disconnect_skips_response_validation():
    """A disconnect must cancel the handler without a response_model 500."""
    app = FastAPI()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @app.post("/", response_model=str)
    @api_utils.with_cancellation
    async def handler(body: dict, raw_request: Request):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    messages = []
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert messages[0]["status"] == 200
    assert messages[1]["body"] == b""
    await asyncio.wait_for(cancelled.wait(), timeout=5)


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )
