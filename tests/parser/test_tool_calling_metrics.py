# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatible parser metrics and bounded tool-calling response dimensions."""

from functools import partial

import pytest
from prometheus_client import CollectorRegistry, Counter, generate_latest

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
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


@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
@pytest.mark.parametrize(
    "choice, expected",
    [
        (None, "unset"),
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        (
            ChatCompletionNamedToolChoiceParam(function={"name": "private-function"}),
            "named",
        ),
        ({"type": "function", "function": {"name": "private-function"}}, "named"),
        ("private-function", "other"),
        ({"type": "private-function"}, "other"),
    ],
)
def test_tool_choice_labels_do_not_export_function_names(
    registry, api_surface, choice, expected
):
    # model_construct also exercises unknown values without API validation.
    request = ChatCompletionRequest.model_construct(tool_choice=choice, tools=None)
    request._api_surface = api_surface
    metrics.record_tool_call_response(request)
    assert (
        registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {
                "model_name": "test-model",
                "status": "no_tool_calls",
                "tool_choice": expected,
                "tool_count": "0",
                "api_surface": api_surface,
            },
        )
        == 1
    )
    assert b"private-function" not in generate_latest(registry)


@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
@pytest.mark.parametrize(
    "count, expected",
    [
        (0, "0"),
        (1, "1"),
        (2, "2-5"),
        (5, "2-5"),
        (6, "6-10"),
        (10, "6-10"),
        (11, "10+"),
        (100, "10+"),
    ],
)
def test_tool_counts_use_fixed_buckets(registry, api_surface, count, expected):
    request = ChatCompletionRequest.model_construct(
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "private-function"}}] * count,
    )
    request._api_surface = api_surface
    metrics.record_tool_call_response(request)
    assert (
        registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {
                "model_name": "test-model",
                "status": "no_tool_calls",
                "tool_choice": "auto",
                "tool_count": expected,
                "api_surface": api_surface,
            },
        )
        == 1
    )
    assert b"private-function" not in generate_latest(registry)


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
@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
def test_response_counts_use_bounded_labels_and_survive_reinitialization(
    registry, count, failed, completed, status, api_surface
):
    request = ChatCompletionRequest.model_construct(
        tool_choice={"type": "function", "function": {"name": "private-function"}}
    )
    request._api_surface = api_surface
    metrics.record_tool_call_response(
        request, tool_calls_count=count, parse_failed=failed, completed=completed
    )
    metrics.init_parser_metrics(model_name="test-model")
    assert (
        registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {
                "model_name": "test-model",
                "status": status,
                "tool_choice": "named",
                "tool_count": "0",
                "api_surface": api_surface,
            },
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model", "api_surface": api_surface},
        )
        == count
    )
    assert b"private-function" not in generate_latest(registry)


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("api_surface", ["chat_completions", "messages"])
def test_interrupted_responses_only_count_emitted_calls(registry, count, api_surface):
    request = ChatCompletionRequest.model_construct(tool_choice="auto")
    request._api_surface = api_surface
    metrics.record_tool_call_response(request, tool_calls_count=count, completed=False)
    for status in ("success", "parse_failure", "no_tool_calls"):
        assert (
            registry.get_sample_value(
                "vllm:tool_call_responses_total",
                {
                    "model_name": "test-model",
                    "status": status,
                    "tool_choice": "auto",
                    "tool_count": "0",
                    "api_surface": api_surface,
                },
            )
            is None
        )
    assert (
        registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model", "api_surface": api_surface},
        )
        == count
    )


@pytest.mark.parametrize("is_streaming", [False, True])
def test_converted_messages_keep_legacy_parser_attribution(registry, is_streaming):
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
    metrics.record_tool_call_response(request, tool_calls_count=1)
    for surface in ("chat_completions", "messages"):
        assert registry.get_sample_value(
            "vllm:tool_call_responses_total",
            {
                "model_name": "test-model",
                "status": "success",
                "tool_choice": "named",
                "tool_count": "1",
                "api_surface": surface,
            },
        ) == (1 if surface == "messages" else None)
        assert registry.get_sample_value(
            "vllm:tool_calls_emitted_total",
            {"model_name": "test-model", "api_surface": surface},
        ) == (1 if surface == "messages" else 0)
    assert b"private-function" not in generate_latest(registry)
