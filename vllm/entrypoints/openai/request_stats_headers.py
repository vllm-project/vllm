# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.metrics.stats import FinishedRequestStats


def build_request_stats_headers(stats: FinishedRequestStats) -> dict[str, str]:
    """Format computed request timings as x-vllm-* response headers.

    Times are in milliseconds, rounded to 2 decimal places. Values come
    directly from FinishedRequestStats; no arithmetic happens here.
    """
    return {
        "x-vllm-total-time": f"{stats.e2e_latency * 1000:.2f}",
        "x-vllm-queue-time": f"{stats.queued_time * 1000:.2f}",
        "x-vllm-inference-time": f"{stats.inference_time * 1000:.2f}",
        "x-vllm-prefill-time": f"{stats.prefill_time * 1000:.2f}",
        "x-vllm-decode-time": f"{stats.decode_time * 1000:.2f}",
        "x-vllm-prompt-tokens": str(stats.num_prompt_tokens),
        "x-vllm-completion-tokens": str(stats.num_generation_tokens),
        "x-vllm-cached-tokens": str(stats.num_cached_tokens),
        "x-vllm-time-per-output-token": (
            f"{stats.mean_time_per_output_token * 1000:.2f}"
        ),
    }


async def request_stats_headers_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """FastAPI middleware that attaches x-vllm-* timing headers.

    Reads request.state.request_metadata (populated by the serving layer).
    No-op if metadata or finished_stats is missing — covers streaming,
    errors, and non-OpenAI routes.
    """
    response = await call_next(request)
    metadata = getattr(request.state, "request_metadata", None)
    if metadata is None or metadata._finished_stats is None:
        return response
    headers = build_request_stats_headers(metadata._finished_stats)
    for key, value in headers.items():
        response.headers[key] = value
    return response
