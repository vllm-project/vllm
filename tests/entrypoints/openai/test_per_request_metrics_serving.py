# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-level tests for the per-request timing metrics double gate.

These drive ``OpenAIServingChat.chat_completion_full_generator`` directly on an
instance created via ``__new__`` so we exercise the real gating + response
wiring without standing up a renderer, tokenizer, or engine.
"""

import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.metrics.stats import RequestStateStats

MODEL_NAME = "test-model"


def _build_serving_chat(enable_per_request_metrics: bool) -> OpenAIServingChat:
    """Create a minimally-initialized serving instance for the metrics path."""
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.response_role = "assistant"
    serving.tool_call_id_type = "default"
    serving.parser_cls = None
    serving.enable_auto_tools = False
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.enable_force_include_usage = False
    serving.request_logger = None
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = enable_per_request_metrics
    return serving


def _build_serving_completion(
    enable_per_request_metrics: bool,
) -> OpenAIServingCompletion:
    """Create a minimally-initialized completion serving instance."""
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = enable_per_request_metrics
    return serving


def _make_request_output(
    metrics: RequestStateStats | None,
    token_ids: tuple[int, ...] = (100, 101),
) -> RequestOutput:
    completion_output = CompletionOutput(
        index=0,
        text="Hello",
        token_ids=list(token_ids),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    return RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=metrics,
    )


async def _as_generator(
    request_output: RequestOutput,
) -> AsyncIterator[RequestOutput]:
    yield request_output


async def _run_full_generator(
    serving: OpenAIServingChat,
    request: ChatCompletionRequest,
    request_output: RequestOutput,
):
    request_id = "chatcmpl-test-id"
    return await serving.chat_completion_full_generator(
        request,
        _as_generator(request_output),
        request_id,
        MODEL_NAME,
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id=request_id),
    )


def _make_request(include_metrics: bool, n: int = 1) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
        include_metrics=include_metrics,
        n=n,
    )


def _make_stream_request(
    include_metrics: bool, include_usage: bool, n: int = 1
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        stream_options={"include_usage": include_usage},
        include_metrics=include_metrics,
        n=n,
    )


async def _collect_stream_chunks(
    serving: OpenAIServingChat,
    request: ChatCompletionRequest,
    request_output: RequestOutput,
) -> list[dict]:
    """Run the streaming generator and return the parsed SSE JSON chunks."""
    request_id = "chatcmpl-test-id"
    chunks: list[dict] = []
    async for line in serving.chat_completion_stream_generator(
        request,
        _as_generator(request_output),
        request_id,
        MODEL_NAME,
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id=request_id),
    ):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


def _make_completion_request(
    include_metrics: bool, n: int = 1, num_prompts: int = 1
) -> CompletionRequest:
    prompt = ["p"] * num_prompts if num_prompts > 1 else "p"
    return CompletionRequest(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=10,
        include_metrics=include_metrics,
        n=n,
    )


def _run_completion_response(
    serving: OpenAIServingCompletion,
    request: CompletionRequest,
    final_res_batch: list[RequestOutput],
):
    request_id = "cmpl-test-id"
    return serving.request_output_to_completion_response(
        final_res_batch,
        request,
        request_id,
        0,
        MODEL_NAME,
        None,
        RequestResponseMetadata(request_id=request_id),
    )


_STATS = RequestStateStats(
    queued_ts=1.0,
    scheduled_ts=1.5,
    first_token_ts=2.0,
    last_token_ts=3.0,
    num_generation_tokens=2,
)


# ---------------------------------------------------------------------------
# enable_per_request_metrics=False (default): metrics never populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_disabled_include_metrics_true_returns_none():
    serving = _build_serving_chat(enable_per_request_metrics=False)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=True),
        _make_request_output(metrics=_STATS),
    )
    assert response.metrics is None


@pytest.mark.asyncio
async def test_metrics_disabled_include_metrics_false_returns_none():
    serving = _build_serving_chat(enable_per_request_metrics=False)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=False),
        _make_request_output(metrics=_STATS),
    )
    assert response.metrics is None


# ---------------------------------------------------------------------------
# enable_per_request_metrics=True: gated by request.include_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_enabled_include_metrics_false_returns_none():
    serving = _build_serving_chat(enable_per_request_metrics=True)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=False),
        _make_request_output(metrics=_STATS),
    )
    assert response.metrics is None


@pytest.mark.asyncio
async def test_metrics_enabled_include_metrics_true_returns_populated():
    serving = _build_serving_chat(enable_per_request_metrics=True)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=True),
        _make_request_output(metrics=_STATS),
    )
    assert response.metrics is not None
    # decode-only generation time: (3.0 - 2.0) * 1000 = 1000.0
    assert response.metrics.time_to_first_token_ms == pytest.approx(500.0)
    assert response.metrics.generation_time_ms == pytest.approx(1000.0)
    assert response.metrics.queue_time_ms == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_metrics_suppressed_for_n_greater_than_one():
    # For n>1 the returned stats describe only one of the n sequences, so
    # timing metrics cannot be attributed to the request and are suppressed.
    serving = _build_serving_chat(enable_per_request_metrics=True)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=True, n=2),
        _make_request_output(metrics=_STATS),
    )
    assert response.metrics is None


@pytest.mark.asyncio
async def test_metrics_enabled_no_request_output_metrics():
    serving = _build_serving_chat(enable_per_request_metrics=True)
    response = await _run_full_generator(
        serving,
        _make_request(include_metrics=True),
        _make_request_output(metrics=None),
    )
    # metrics object is returned but all fields are None when stats is None
    assert response.metrics is not None
    assert response.metrics.time_to_first_token_ms is None
    assert response.metrics.generation_time_ms is None
    assert response.metrics.queue_time_ms is None


# ---------------------------------------------------------------------------
# Streaming chat: metrics ride on the final usage chunk, which only exists when
# include_usage is set. So include_metrics alone is not enough in streaming.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_metrics_present_with_include_usage():
    serving = _build_serving_chat(enable_per_request_metrics=True)
    chunks = await _collect_stream_chunks(
        serving,
        _make_stream_request(include_metrics=True, include_usage=True),
        _make_request_output(metrics=_STATS),
    )
    # The final usage chunk is the one carrying ``usage``; metrics attach there.
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert usage_chunks, "expected a final usage chunk when include_usage=True"
    metrics = usage_chunks[-1].get("metrics")
    assert metrics is not None
    assert metrics["time_to_first_token_ms"] == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_streaming_metrics_absent_without_include_usage():
    serving = _build_serving_chat(enable_per_request_metrics=True)
    chunks = await _collect_stream_chunks(
        serving,
        _make_stream_request(include_metrics=True, include_usage=False),
        _make_request_output(metrics=_STATS),
    )
    # Without include_usage there is no final usage chunk, so no chunk carries
    # metrics even though include_metrics=True.
    assert all("metrics" not in c for c in chunks)


# ---------------------------------------------------------------------------
# Completion endpoint: metrics are suppressed for multiple prompts because the
# timing data cannot be attributed to a single prompt's generation.
# ---------------------------------------------------------------------------


def test_completion_metrics_present_for_single_prompt():
    serving = _build_serving_completion(enable_per_request_metrics=True)
    response = _run_completion_response(
        serving,
        _make_completion_request(include_metrics=True, num_prompts=1),
        [_make_request_output(metrics=_STATS)],
    )
    assert response.metrics is not None
    assert response.metrics.time_to_first_token_ms == pytest.approx(500.0)


def test_completion_metrics_suppressed_for_multiple_prompts():
    serving = _build_serving_completion(enable_per_request_metrics=True)
    response = _run_completion_response(
        serving,
        _make_completion_request(include_metrics=True, num_prompts=2),
        [_make_request_output(metrics=_STATS), _make_request_output(metrics=_STATS)],
    )
    assert response.metrics is None
