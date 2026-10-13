# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import cast

from prometheus_client import REGISTRY, Counter

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"
    PARSE_FAILURE = "parse_failure"


class RequestType(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


def init_parser_metrics(*, model_name: str) -> None:
    """Lazily register parser metrics and cache the shared model label."""
    global _model_name
    _model_name = model_name

    global _tool_call_parser_invocations
    try:
        _tool_call_parser_invocations = Counter(
            name=_TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
            documentation=(
                "Total number of ToolParser invocations. "
                "Non-streaming increments once per choice; "
                "streaming increments once per delta. "
                "Tool choices are normalized and tool counts are bucketed."
            ),
            labelnames=[
                "model_name",
                "mode",
                "outcome",
                "request_type",
                "tool_choice",
                "tool_count",
            ],
        )
    except ValueError:
        _tool_call_parser_invocations = cast(
            Counter,
            REGISTRY._names_to_collectors[_TOOL_CALL_PARSER_INVOCATIONS_TOTAL],
        )

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
