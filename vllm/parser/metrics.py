# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Literal, cast

from prometheus_client import REGISTRY, Counter

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None
_tool_call_responses: Counter | None = None
_tool_calls_emitted: Counter | None = None


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"
    PARSE_FAILURE = "parse_failure"


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
            "streaming increments once per delta. "
            "Tool choices are normalized and tool counts are bucketed."
        ),
        [
            "model_name",
            "mode",
            "outcome",
            "request_type",
            "tool_choice",
            "tool_count",
        ],
    )

    global _tool_call_responses, _tool_calls_emitted
    _tool_call_responses = _register_counter(
        "vllm:tool_call_responses_total",
        "Tool-calling response outcomes, counted once per request.",
        ["model_name", "status", "tool_choice", "api_surface"],
    )
    _tool_calls_emitted = _register_counter(
        "vllm:tool_calls_emitted_total",
        "Individual tool-call objects emitted to clients.",
        ["model_name", "api_surface"],
    )
    for status in ("success", "parse_failure", "no_tool_calls"):
        _tool_call_responses.labels(
            model_name=_model_name,
            status=status,
            tool_choice="unset",
            api_surface="chat_completions",
        )
    _tool_calls_emitted.labels(model_name=_model_name, api_surface="chat_completions")

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
            tool_choice="unset",
            tool_count="0",
        )


def _tool_choice_label(tool_choice: object) -> str:
    if tool_choice is None:
        return "unset"
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ("auto", "none", "required") else "other"
    choice_type = (
        tool_choice.get("type")
        if isinstance(tool_choice, dict)
        else getattr(tool_choice, "type", None)
    )
    return "named" if choice_type == "function" else "other"


def _tool_count_bucket(count: int) -> str:
    if count <= 1:
        return str(count)
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    return "10+"


def record_tool_call_response(
    request: object,
    *,
    tool_calls_count: int = 0,
    parse_failed: bool = False,
    api_surface: Literal["chat_completions", "messages"] = "chat_completions",
) -> None:
    """Record a request outcome and the calls actually returned to the client."""
    if _tool_call_responses is None or _tool_calls_emitted is None:
        return
    status = (
        "parse_failure"
        if parse_failed
        else "success"
        if tool_calls_count
        else "no_tool_calls"
    )
    _tool_call_responses.labels(
        model_name=_model_name,
        status=status,
        tool_choice=_tool_choice_label(getattr(request, "tool_choice", None)),
        api_surface=api_surface,
    ).inc()
    _tool_calls_emitted.labels(model_name=_model_name, api_surface=api_surface).inc(
        tool_calls_count
    )


def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered.

    Surfaced exceptions count as parse failures. Errors caught internally by a
    parser cannot be distinguished from ordinary no-tool output here.
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
        case Exception():
            outcome = ToolCallOutcome.PARSE_FAILURE
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
        tool_choice=_tool_choice_label(getattr(request, "tool_choice", None)),
        tool_count=_tool_count_bucket(len(getattr(request, "tools", None) or ())),
    ).inc()
