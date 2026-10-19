# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatible parser metrics and request-level tool-calling outcomes."""

from functools import partial

import pytest
from prometheus_client import CollectorRegistry, Counter, generate_latest

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import metrics

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def registry(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr(metrics, "REGISTRY", registry)
    monkeypatch.setattr(metrics, "Counter", partial(Counter, registry=registry))
    monkeypatch.setattr(metrics, "_model_name", None)
    monkeypatch.setattr(metrics, "_tool_call_parser_invocations", None)
    monkeypatch.setattr(metrics, "_tool_call_responses", None)
    monkeypatch.setattr(metrics, "_tool_calls_emitted", None)
    metrics.init_parser_metrics(model_name="test-model")
    return registry


def _record_and_read(
    registry,
    request,
    is_streaming,
    *,
    is_tool_called=True,
    outcome="tool_call",
    request_type="chat_completions",
):
    metrics.record_tool_parser_invocation(
        is_tool_called=is_tool_called, is_streaming=is_streaming, request=request
    )
    return registry.get_sample_value(
        "vllm:tool_call_parser_invocations_total",
        {
            "model_name": "test-model",
            "mode": "streaming" if is_streaming else "non_streaming",
            "outcome": outcome,
            "request_type": request_type,
        },
    )


def test_parser_counter_keeps_existing_label_set_and_series(registry):
    samples = [
        sample
        for family in registry.collect()
        for sample in family.samples
        if sample.name == "vllm:tool_call_parser_invocations_total"
    ]
    assert len(samples) == 12
    assert all(
        set(sample.labels) == {"model_name", "mode", "outcome", "request_type"}
        for sample in samples
    )
    assert {sample.labels["outcome"] for sample in samples} == {
        "tool_call",
        "no_tool_call",
    }
    assert {sample.labels["request_type"] for sample in samples} == {
        "chat_completions",
        "responses",
        "other",
    }


def test_response_counters_initialize_all_series_at_zero(registry):
    samples = [
        sample
        for family in registry.collect()
        for sample in family.samples
        if sample.name == "vllm:tool_call_responses_total"
    ]
    assert len(samples) == 3
    assert {sample.labels["status"] for sample in samples} == {
        "success",
        "parse_failure",
        "no_tool_calls",
    }
    assert all(
        set(sample.labels) == {"model_name", "status"} and sample.value == 0
        for sample in samples
    )
    emitted = [
        sample
        for family in registry.collect()
        for sample in family.samples
        if sample.name == "vllm:tool_calls_emitted_total"
    ]
    assert len(emitted) == 1
    assert emitted[0].labels == {"model_name": "test-model"}
    assert emitted[0].value == 0


def test_reinitializing_metrics_preserves_labeled_counts(registry):
    request = ChatCompletionRequest.model_construct(tool_choice="auto", tools=[])
    assert _record_and_read(registry, request, False) == 1
    metrics.init_parser_metrics(model_name="test-model")
    assert _record_and_read(registry, request, False) == 2


@pytest.mark.parametrize("is_streaming", [False, True])
def test_parser_failures_keep_existing_no_tool_call_classification(
    registry, is_streaming
):
    request = ChatCompletionRequest.model_construct(tool_choice="auto", tools=[])
    record = partial(_record_and_read, registry, request, is_streaming)
    assert record(is_tool_called=False, outcome="no_tool_call") == 1
    assert (
        record(
            is_tool_called=ValueError("private-argument"),
            outcome="no_tool_call",
        )
        == 2
    )
    assert record(is_tool_called=False, outcome="no_tool_call") == 3
    assert record() == 1
    assert b"private-argument" not in generate_latest(registry)


@pytest.mark.parametrize(
    "count, failed, completed, status",
    [
        (3, False, True, "success"),
        (0, False, True, "no_tool_calls"),
        (0, True, False, "parse_failure"),
        (2, True, False, "parse_failure"),
    ],
)
def test_response_counts_survive_reinitialization(
    registry, count, failed, completed, status
):
    metrics.record_tool_call_response(
        tool_calls_count=count, parse_failed=failed, completed=completed
    )
    metrics.init_parser_metrics(model_name="test-model")
    assert (
        registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {"model_name": "test-model", "status": status},
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model"},
        )
        == count
    )


@pytest.mark.parametrize("count", [0, 2])
def test_interrupted_responses_only_count_emitted_calls(registry, count):
    metrics.record_tool_call_response(tool_calls_count=count, completed=False)
    for status in ("success", "parse_failure", "no_tool_calls"):
        assert (
            registry.get_sample_value(
                "vllm:tool_call_responses_total",
                {"model_name": "test-model", "status": status},
            )
            == 0
        )
    assert (
        registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model"},
        )
        == count
    )


@pytest.mark.parametrize("is_streaming", [False, True])
def test_converted_messages_keep_parser_attribution(registry, is_streaming):
    request = AnthropicServingMessages.to_chat_completion_request(
        AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1,
            messages=[],
            stream=is_streaming,
            tools=[{"name": "private-function", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "private-function"},
        )
    )
    assert _record_and_read(registry, request, is_streaming) == 1
    metrics.record_tool_call_response(tool_calls_count=1)
    assert (
        registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {"model_name": "test-model", "status": "success"},
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model"},
        )
        == 1
    )
    assert b"private-function" not in generate_latest(registry)
