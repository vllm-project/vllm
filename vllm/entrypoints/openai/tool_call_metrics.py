# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for tool call parsing in the OpenAI serving layer.

This module instruments tool call parsing attempts to provide visibility into
parsing failures during model rollouts, runtime upgrades, or edge cases.
"""

from __future__ import annotations

from prometheus_client import Counter

_tool_call_total = Counter(
    name="vllm:tool_call_total",
    documentation=(
        "Total number of tool call parsing attempts at the HTTP serving layer. "
        "Tracks successes and failures to detect parsing issues during model "
        "rollouts, runtime upgrades, or edge cases."
    ),
    labelnames=["model", "status", "parser"],
)


def record_tool_call_parse_attempt(
    *,
    model: str,
    success: bool,
    parser_type: str,
) -> None:
    parser = "custom" if parser_type == "custom" else "standard"
    status = "success" if success else "error"

    _tool_call_total.labels(
        model=model,
        status=status,
        parser=parser,
    ).inc()
