# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

import json
from enum import Enum
from itertools import product
from typing import Any, cast

from prometheus_client import REGISTRY, Counter

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None

_TOOL_CALLS_COMPLETED_TOTAL = "vllm:tool_calls_completed_total"
_tool_calls_completed: Counter | None = None


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"


class RequestType(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


class ValidationOutcome(Enum):
    VALID = "valid"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"


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
                "streaming increments once per delta."
            ),
            labelnames=["model_name", "mode", "outcome", "request_type"],
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
        )

    global _tool_calls_completed
    try:
        _tool_calls_completed = Counter(
            name=_TOOL_CALLS_COMPLETED_TOTAL,
            documentation=(
                "Total number of completed tool calls. "
                "Incremented once per finished call. "
                "Streaming and non-streaming share this counter."
            ),
            labelnames=["model_name", "request_type", "validation_outcome"],
        )
    except ValueError:
        _tool_calls_completed = cast(
            Counter,
            REGISTRY._names_to_collectors[_TOOL_CALLS_COMPLETED_TOTAL],
        )

    for request_type, validation_outcome in product(RequestType, ValidationOutcome):
        _tool_calls_completed.labels(
            model_name=_model_name,
            request_type=request_type.value,
            validation_outcome=validation_outcome.value,
        )


def _request_type(request: object) -> RequestType:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    match request:
        case ChatCompletionRequest():
            return RequestType.CHAT_COMPLETIONS
        case ResponsesRequest():
            return RequestType.RESPONSES
        case _:
            return RequestType.OTHER


def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered.
    Currently parser failures are treated as no tool calls.

    TODO: To accurately track parser failures, add a new ToolCallOutcome and
    more importantly, ensure exceptions are propagated out of the ToolParsers
    instead of being caught internally. This would require going through
    ToolParser implementation on a case-by-case basis.
    """
    if _tool_call_parser_invocations is None:
        return

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
        request_type=_request_type(request).value,
    ).inc()


def record_tool_call_completed(
    *,
    request: object,
    outcome: ValidationOutcome,
) -> None:
    """Increment the completed tool-call counter when registered."""
    if _tool_calls_completed is None:
        return

    _tool_calls_completed.labels(
        model_name=_model_name,
        request_type=_request_type(request).value,
        validation_outcome=outcome.value,
    ).inc()


def classify_completed_tool_call(
    name: str,
    arguments: str,
    tools: object | None,
) -> ValidationOutcome:
    """Return one validation label for a completed tool call.

    Never raises. Unexpected errors map to ``invalid_args``.
    """
    try:
        import jsonschema

        from vllm.tool_parsers.utils import find_tool_name, find_tool_parameters

        catalog = cast(Any, tools)
        if not find_tool_name(catalog, name):
            return ValidationOutcome.UNKNOWN_TOOL

        try:
            instance = json.loads(arguments)
        except json.JSONDecodeError:
            return ValidationOutcome.INVALID_ARGS

        params = find_tool_parameters(catalog, name)
        schema = params if params is not None else {"type": "object", "properties": {}}
        jsonschema.validate(instance=instance, schema=schema)
        return ValidationOutcome.VALID
    except Exception:
        return ValidationOutcome.INVALID_ARGS
