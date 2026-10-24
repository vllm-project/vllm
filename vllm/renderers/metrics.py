# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the online renderers."""

from functools import lru_cache

from prometheus_client import Histogram


@lru_cache(maxsize=1)
def _request_render_duration() -> Histogram:
    return Histogram(
        name="vllm:request_render_duration_seconds",
        documentation=(
            "Time spent rendering a request, including chat templating, "
            "tokenization, and multimodal preprocessing."
        ),
        labelnames=["model_name", "request_type"],
        buckets=(
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1,
            2.5,
            5,
            10,
        ),
    )


def observe_request_render_duration(
    model_name: str, request_type: str, duration_seconds: float
) -> None:
    _request_render_duration().labels(model_name, request_type).observe(
        duration_seconds
    )
