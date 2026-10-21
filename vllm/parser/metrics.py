# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for tool parsing and response outcomes."""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import cast

from prometheus_client import REGISTRY, Counter, Histogram

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None
_tool_call_responses: Counter | None = None
_tool_calls_emitted: Counter | None = None
_request_tools: Histogram | None = None


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"


class RequestType(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


def _register_counter(name: str, documentation: str, labelnames: list[str]) -> Counter:
    try:
        return Counter(name=name, documentation=documentation, labelnames=labelnames)
    except ValueError:
        return cast(Counter, REGISTRY._names_to_collectors[name])


def init_parser_metrics(*, model_name: str) -> None:
    """Lazily register parser metrics and cache the shared model label."""
    global _model_name
    _model_name = model_name

    global _tool_call_parser_invocations
    _tool_call_parser_invocations = _register_counter(
        _TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
        (
            "Total number of ToolParser invocations. "
            "Non-streaming increments once per choice; "
            "streaming increments once per delta."
        ),
        ["model_name", "mode", "outcome", "request_type"],
    )

    global _tool_call_responses, _tool_calls_emitted
    _tool_call_responses = _register_counter(
        "vllm:tool_call_responses_total",
        "Tool-calling response outcomes, counted once per request.",
        ["model_name", "status"],
    )
    _tool_calls_emitted = _register_counter(
        "vllm:tool_calls_emitted_total",
        "Individual tool-call objects emitted to clients.",
        ["model_name"],
    )
    for status in ("success", "parse_failure", "no_tool_calls"):
        _tool_call_responses.labels(model_name=_model_name, status=status)
    _tool_calls_emitted.labels(model_name=_model_name)

    global _request_tools
    try:
        _request_tools = Histogram(
            "vllm:request_tools",
            "Number of tool definitions supplied per request.",
            ["model_name"],
            buckets=(0, 1, 5, 10),
        )
    except ValueError:
        _request_tools = cast(
            Histogram, REGISTRY._names_to_collectors["vllm:request_tools"]
        )
    _request_tools.labels(model_name=_model_name)

    for mode, outcome, request_type in product(
        ("streaming", "non_streaming"),
        ToolCallOutcome,
        RequestType,
    ):
        _tool_call_parser_invocations.labels(
            model_name=_model_name,
            mode=mode,
            outcome=outcome.value,
            request_type=request_type.value,
        )


def record_request_tools(request: object) -> None:
    """Observe the supplied tool count before parsing can mutate the request."""
    if _request_tools is not None:
        _request_tools.labels(model_name=_model_name).observe(
            len(getattr(request, "tools", None) or ())
        )


def record_tool_call_response(
    *,
    tool_calls_count: int = 0,
    parse_failed: bool = False,
    completed: bool = True,
) -> None:
    """Record a request outcome and the calls actually returned to the client.

    Interrupted streams retain emitted counts without a success/no-tool outcome.
    """
    if _tool_call_responses is None or _tool_calls_emitted is None:
        return
    status = (
        "parse_failure"
        if parse_failed
        else "success"
        if tool_calls_count
        else "no_tool_calls"
    )
    if completed or parse_failed:
        _tool_call_responses.labels(model_name=_model_name, status=status).inc()
    _tool_calls_emitted.labels(model_name=_model_name).inc(tool_calls_count)


def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered.

    Messages requests are converted to ChatCompletionRequest before parsing,
    so they use request_type="chat_completions". Exceptions count as "no_tool_call";
    the response counter reports surfaced parsing failures separately.
    """
    if _tool_call_parser_invocations is None:
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    match request:
        case ChatCompletionRequest():
            request_type = RequestType.CHAT_COMPLETIONS
        case ResponsesRequest():
            request_type = RequestType.RESPONSES
        case _:
            request_type = RequestType.OTHER

    match is_tool_called:
        case bool():
            outcome = (
                ToolCallOutcome.TOOL_CALL
                if is_tool_called
                else ToolCallOutcome.NO_TOOL_CALL
            )
        case _:
            outcome = ToolCallOutcome.NO_TOOL_CALL

    _tool_call_parser_invocations.labels(
        model_name=_model_name,
        mode="streaming" if is_streaming else "non_streaming",
        outcome=outcome.value,
        request_type=request_type.value,
    ).inc()
