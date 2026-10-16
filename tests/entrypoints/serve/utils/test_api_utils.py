# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.generate.base.protocol import StreamOptions
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    redact_sensitive_args,
    should_include_usage,
)


@pytest.fixture
def inference_spans(monkeypatch):
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from vllm.sampling_params import SamplingParams
    from vllm.tracing import otel
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.metrics.stats import IterationStats

    if not otel.is_otel_available():
        pytest.skip("OTel required")
    provider = sdk.TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel.trace, "get_tracer", provider.get_tracer)
    processor = OutputProcessor(None, log_stats=False, tracing_enabled=True)

    def finish_engine(request_id="request"):
        request = EngineCoreRequest(
            request_id=request_id,
            external_req_id=request_id,
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=SamplingParams(detokenize=False),
            pooling_params=None,
            arrival_time=1,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        processor.add_request(request, None)
        span = processor.request_states[request_id].trace_span
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[2],
                    finish_reason=FinishReason.STOP,
                )
            ],
            1,
            IterationStats(),
        )
        return span

    yield exporter, finish_engine
    processor.shutdown()
    provider.shutdown()


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
@pytest.mark.parametrize("has_tools", [False, True])
async def test_inference_spans_wait_for_api_conversion(
    inference_spans, api, num_choices, has_tools
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
    from vllm.entrypoints.serve.engine.protocol import UsageInfo
    from vllm.tracing import get_request_trace

    exporter, finish_engine = inference_spans
    raw_request = _traced_request()

    async def complete(request, raw_request):
        spans = [finish_engine(str(index)) for index in range(num_choices)]
        assert all(span.is_recording() for span in spans)
        assert not exporter.get_finished_spans()
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
        if api == "messages":
            response = AnthropicServingMessages.__new__(
                AnthropicServingMessages
            ).messages_full_converter(response)
            assert all(span.is_recording() for span in spans)
        return response

    if api == "chat":
        raw_request.app.state.openai_serving_chat = SimpleNamespace(
            create_chat_completion=complete
        )
        request = ChatCompletionRequest(
            messages=[],
            n=num_choices,
            tools=[{"type": "function", "function": {"name": "test"}}]
            if has_tools
            else None,
        )
        response = await create_chat_completion(request, raw_request)
    else:
        raw_request.app.state.anthropic_serving_messages = SimpleNamespace(
            create_messages=complete
        )
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1,
            messages=[],
            tools=[{"name": "test", "input_schema": {"type": "object"}}]
            if has_tools
            else None,
        )
        response = await create_messages(request, raw_request)

    assert response.status_code == 200
    assert b"hello" in response.body
    assert len(exporter.get_finished_spans()) == num_choices
    assert get_request_trace() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["completed", "disconnect", "send_error", "cancel", "api_error", "cleanup_error"],
)
async def test_streaming_inference_span_completion(inference_spans, outcome):
    """The ASGI response owns spans even when interruption happens during send."""
    from vllm.tracing import StatusCode, get_request_trace

    exporter, finish_engine = inference_spans
    raw_request = _traced_request()

    @api_utils.with_cancellation
    async def handler(body, raw_request):
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
                if outcome == "cleanup_error":
                    raise RuntimeError("cleanup failed")

        return StreamingResponse(stream(), headers={"x-test": "kept"})

    response = await handler({}, raw_request)
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


class TestRedactSensitiveArgs:
    API_KEY = "sk-test-secret-12345"

    def test_redact_replaces_sensitive_values_only(self):
        args = {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }
        redacted = redact_sensitive_args(args)
        assert redacted == {
            "api_key": "***",
            "hf_token": "***",
            "watermark_config": "***",
            "other": "visible",
        }
        # original dict must not be mutated
        assert args == {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }

    def test_no_sensitive_fields_returns_original(self):
        args = {"model_tag": "org/model", "other": "visible"}
        assert redact_sensitive_args(args) is args

    def test_api_key_not_in_log(self, monkeypatch, caplog):
        non_default = {
            "model_tag": "org/model",
            "default_chat_template_kwargs": {"enable_thinking": False},
            "api_key": self.API_KEY,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(api_utils, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.serve.utils.api_utils"):
            api_utils.log_non_default_args(args=Namespace())
        message = caplog.text
        assert self.API_KEY not in message
        assert "'api_key': '***'" in message
        # non-sensitive args are still logged
        assert "org/model" in message
        assert "qwen3_coder" in message

    def test_hf_token_not_in_log(self, monkeypatch, caplog):
        hf_token = "hf_TESTTOKENVALUE123"
        non_default = {
            "model_tag": "org/model",
            "hf_token": hf_token,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(api_utils, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.serve.utils.api_utils"):
            api_utils.log_non_default_args(args=Namespace())
        message = caplog.text
        assert hf_token not in message
        assert "'hf_token': '***'" in message
        assert "org/model" in message
