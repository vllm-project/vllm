# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokensDetails,
    PerRequestTimingMetrics,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    build_per_request_timing_metrics as _build_per_request_timing_metrics,
)
from vllm.v1.metrics.stats import RequestStateStats

# ---------------------------------------------------------------------------
# CompletionTokensDetails
# ---------------------------------------------------------------------------


def test_completion_tokens_details_all_none():
    details = CompletionTokensDetails()
    assert details.reasoning_tokens is None
    assert details.accepted_prediction_tokens is None
    assert details.rejected_prediction_tokens is None


def test_completion_tokens_details_reasoning_tokens():
    details = CompletionTokensDetails(reasoning_tokens=42)
    assert details.reasoning_tokens == 42
    assert details.accepted_prediction_tokens is None
    assert details.rejected_prediction_tokens is None


def test_completion_tokens_details_serialization():
    details = CompletionTokensDetails(reasoning_tokens=7)
    data = details.model_dump()
    assert data["reasoning_tokens"] == 7


def test_completion_tokens_details_all_fields():
    details = CompletionTokensDetails(
        reasoning_tokens=10,
        accepted_prediction_tokens=5,
        rejected_prediction_tokens=2,
    )
    data = details.model_dump()
    assert data["reasoning_tokens"] == 10
    assert data["accepted_prediction_tokens"] == 5
    assert data["rejected_prediction_tokens"] == 2


# ---------------------------------------------------------------------------
# PerRequestTimingMetrics
# ---------------------------------------------------------------------------


def test_per_request_timing_metrics_all_none():
    m = PerRequestTimingMetrics()
    assert m.time_to_first_token_ms is None
    assert m.time_to_first_content_token_ms is None
    assert m.generation_time_ms is None
    assert m.queue_time_ms is None
    assert m.mean_itl_ms is None
    assert m.tokens_per_second is None


def test_per_request_timing_metrics_all_set():
    m = PerRequestTimingMetrics(
        time_to_first_token_ms=100.0,
        time_to_first_content_token_ms=120.0,
        generation_time_ms=500.0,
        queue_time_ms=50.0,
        mean_itl_ms=44.4,
        tokens_per_second=20.0,
    )
    assert m.time_to_first_token_ms == 100.0
    assert m.time_to_first_content_token_ms == 120.0
    assert m.generation_time_ms == 500.0
    assert m.queue_time_ms == 50.0
    assert m.mean_itl_ms == 44.4
    assert m.tokens_per_second == 20.0


def test_per_request_timing_metrics_serialization():
    m = PerRequestTimingMetrics(time_to_first_token_ms=200.0)
    data = m.model_dump()
    assert data["time_to_first_token_ms"] == 200.0
    assert data["generation_time_ms"] is None


# ---------------------------------------------------------------------------
# UsageInfo with completion_tokens_details
# ---------------------------------------------------------------------------


def test_usage_info_completion_tokens_details_default_none():
    usage = UsageInfo(prompt_tokens=5, total_tokens=10, completion_tokens=5)
    assert usage.completion_tokens_details is None


def test_usage_info_completion_tokens_details_set():
    details = CompletionTokensDetails(reasoning_tokens=3)
    usage = UsageInfo(
        prompt_tokens=5,
        total_tokens=10,
        completion_tokens=5,
        completion_tokens_details=details,
    )
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 3


# ---------------------------------------------------------------------------
# _build_per_request_timing_metrics helper
# ---------------------------------------------------------------------------


def test_build_per_request_timing_metrics_none_input():
    result = _build_per_request_timing_metrics(None, num_generation_tokens=10)
    assert isinstance(result, PerRequestTimingMetrics)
    assert result.time_to_first_token_ms is None
    assert result.generation_time_ms is None
    assert result.queue_time_ms is None
    assert result.mean_itl_ms is None
    assert result.tokens_per_second is None


def test_build_per_request_timing_metrics_all_zero_timestamps():
    stats = RequestStateStats(
        queued_ts=0.0,
        scheduled_ts=0.0,
        first_token_ts=0.0,
        last_token_ts=0.0,
        num_generation_tokens=10,
    )
    result = _build_per_request_timing_metrics(stats, num_generation_tokens=10)
    assert result.time_to_first_token_ms is None
    assert result.generation_time_ms is None
    assert result.queue_time_ms is None
    assert result.mean_itl_ms is None
    assert result.tokens_per_second is None


def test_build_per_request_timing_metrics_valid_timestamps():
    # queued_ts=1.0, scheduled_ts=1.5, first_token_ts=2.0, last_token_ts=3.0, 10 tokens
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=10,
    )
    result = _build_per_request_timing_metrics(stats, num_generation_tokens=10)

    # time_to_first_token_ms = (2.0 - 1.5) * 1000 = 500.0
    assert result.time_to_first_token_ms == pytest.approx(500.0)
    # generation_time_ms = (3.0 - 1.5) * 1000 = 1500.0
    assert result.generation_time_ms == pytest.approx(1500.0)
    # queue_time_ms = (1.5 - 1.0) * 1000 = 500.0
    assert result.queue_time_ms == pytest.approx(500.0)
    # mean_itl_ms = (3.0 - 2.0) / (10 - 1) * 1000 ≈ 111.11
    assert result.mean_itl_ms == pytest.approx(1000.0 / 9, rel=1e-4)
    # tokens_per_second = 10 / 1500.0 * 1000 ≈ 6.667
    assert result.tokens_per_second == pytest.approx(10.0 / 1.5, rel=1e-4)


def test_build_per_request_timing_metrics_single_token():
    # With only 1 generation token, mean_itl_ms should be None
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=2.0,
        num_generation_tokens=1,
    )
    result = _build_per_request_timing_metrics(stats, num_generation_tokens=1)
    assert result.mean_itl_ms is None


def test_build_per_request_timing_metrics_partial_timestamps():
    # Only queued_ts and scheduled_ts set; first_token and last_token are 0
    stats = RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=2.0,
        first_token_ts=0.0,
        last_token_ts=0.0,
        num_generation_tokens=5,
    )
    result = _build_per_request_timing_metrics(stats, num_generation_tokens=5)
    assert result.queue_time_ms == pytest.approx(1000.0)
    assert result.time_to_first_token_ms is None
    assert result.generation_time_ms is None
    assert result.mean_itl_ms is None
    assert result.tokens_per_second is None


def test_build_per_request_timing_metrics_no_queue_info():
    # queued_ts is 0, so queue_time_ms should be None
    stats = RequestStateStats(
        queued_ts=0.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=5,
    )
    result = _build_per_request_timing_metrics(stats, num_generation_tokens=5)
    assert result.queue_time_ms is None
    assert result.time_to_first_token_ms == pytest.approx(500.0)
    assert result.generation_time_ms == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# ChatCompletionRequest.include_metrics field
# ---------------------------------------------------------------------------


def test_chat_completion_request_include_metrics_default():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert request.include_metrics is False


def test_chat_completion_request_include_metrics_true():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        include_metrics=True,
    )
    assert request.include_metrics is True


# ---------------------------------------------------------------------------
# ChatCompletionResponse.metrics field
# ---------------------------------------------------------------------------


def test_chat_completion_response_metrics_default_none():
    response = ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1),
    )
    assert response.metrics is None


def test_chat_completion_response_metrics_set():
    m = PerRequestTimingMetrics(time_to_first_token_ms=100.0)
    response = ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1),
        metrics=m,
    )
    assert response.metrics is not None
    assert response.metrics.time_to_first_token_ms == 100.0


# ---------------------------------------------------------------------------
# CompletionRequest.include_metrics field
# ---------------------------------------------------------------------------


def test_completion_request_include_metrics_default():
    request = CompletionRequest(model="test-model", prompt="Hello")
    assert request.include_metrics is False


def test_completion_request_include_metrics_true():
    request = CompletionRequest(
        model="test-model", prompt="Hello", include_metrics=True
    )
    assert request.include_metrics is True


# ---------------------------------------------------------------------------
# CompletionResponse.metrics field
# ---------------------------------------------------------------------------


def test_completion_response_metrics_default_none():
    response = CompletionResponse(
        model="test-model",
        choices=[
            CompletionResponseChoice(
                index=0,
                text="Hi",
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1),
    )
    assert response.metrics is None


def test_completion_response_metrics_set():
    m = PerRequestTimingMetrics(generation_time_ms=300.0)
    response = CompletionResponse(
        model="test-model",
        choices=[
            CompletionResponseChoice(
                index=0,
                text="Hi",
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1),
        metrics=m,
    )
    assert response.metrics is not None
    assert response.metrics.generation_time_ms == 300.0
