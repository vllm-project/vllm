# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the reasoning effort requests are rendered with."""

from __future__ import annotations

from typing import Any, cast, get_args

from prometheus_client import REGISTRY, Counter

from vllm.renderers import ReasoningEffort

_REQUEST_REASONING_EFFORT_TOTAL = "vllm:request_reasoning_effort"
# Labels are limited to these values to bound cardinality: clients and
# --reasoning-effort-map can send arbitrary strings and numbers.
_EFFORTS = get_args(ReasoningEffort)
_LABELS = (*_EFFORTS, "other", "unset")

_model_name: str | None = None
_request_reasoning_effort: Counter | None = None


def init_reasoning_effort_metrics(*, model_name: str) -> None:
    global _model_name, _request_reasoning_effort
    _model_name = model_name
    try:
        _request_reasoning_effort = Counter(
            name=_REQUEST_REASONING_EFFORT_TOTAL,
            documentation=(
                "Number of generation requests by the reasoning_effort they "
                "were rendered with, after --reasoning-effort-map. Values "
                "outside the API efforts are reported as 'other'."
            ),
            labelnames=["model_name", "reasoning_effort"],
        )
    except ValueError:
        _request_reasoning_effort = cast(
            Counter, REGISTRY._names_to_collectors[_REQUEST_REASONING_EFFORT_TOTAL]
        )
    for label in _LABELS:
        _request_reasoning_effort.labels(model_name=model_name, reasoning_effort=label)


def record_reasoning_effort(chat_template_kwargs: dict[str, Any]) -> None:
    if _request_reasoning_effort is None:
        return

    effort = chat_template_kwargs.get("reasoning_effort")
    if effort is None:
        label = "unset"
    elif isinstance(effort, str) and effort in _EFFORTS:
        label = effort
    else:
        label = "other"
    _request_reasoning_effort.labels(
        model_name=_model_name, reasoning_effort=label
    ).inc()
