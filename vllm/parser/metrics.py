# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

from typing import Literal, cast

from prometheus_client import REGISTRY, Counter

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None


def init_parser_metrics() -> None:
    """Lazily register parser metrics in the current Prometheus registry."""
    global _tool_call_parser_invocations
    try:
        _tool_call_parser_invocations = Counter(
            name=_TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
            documentation=(
                "Total number of ToolParser invocations. "
                "Non-streaming increments once per choice; "
                "streaming increments once per delta."
            ),
            labelnames=["mode", "outcome", "request_type"],
        )
    except ValueError:
        _tool_call_parser_invocations = cast(
            Counter,
            REGISTRY._names_to_collectors[_TOOL_CALL_PARSER_INVOCATIONS_TOTAL],
        )


def record_tool_parser_invocation(
    *,
    mode: Literal["streaming", "non_streaming"],
    tools_called: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered."""
    if _tool_call_parser_invocations is not None:
        _tool_call_parser_invocations.labels(
            mode=mode,
            outcome="tool_call" if tools_called else "no_tool_call",
            request_type=request.__class__.__name__,
        ).inc()
