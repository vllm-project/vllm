# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from vllm.entrypoints.openai.engine.protocol import UsageInfo

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.metrics.stats import RequestStateStats


def build_request_stats_headers(
    metrics: RequestStateStats,
    usage: UsageInfo,
    num_cached_tokens: int,
) -> dict[str, str]:
    """Build HTTP response headers with per-request timing and compute stats.

    Times are in milliseconds, rounded to 2 decimal places.
    Tokens-per-second is decode throughput (completion_tokens / decode_time).
    """
    total_time_ms = max(round((time.time() - metrics.arrival_time) * 1000, 2), 0)
    queue_time_ms = max(round((metrics.scheduled_ts - metrics.queued_ts) * 1000, 2), 0)
    prefill_time_ms = max(
        round((metrics.first_token_ts - metrics.scheduled_ts) * 1000, 2), 0
    )
    decode_time_ms = max(
        round((metrics.last_token_ts - metrics.first_token_ts) * 1000, 2), 0
    )
    inference_time_ms = max(
        round((metrics.last_token_ts - metrics.scheduled_ts) * 1000, 2), 0
    )

    decode_time_s = decode_time_ms / 1000.0
    completion_tokens = usage.completion_tokens or 0
    if decode_time_s > 0 and completion_tokens > 0:
        tokens_per_second = round(completion_tokens / decode_time_s, 2)
    else:
        tokens_per_second = 0.0

    return {
        "x-vllm-total-time": f"{total_time_ms:.2f}",
        "x-vllm-queue-time": f"{queue_time_ms:.2f}",
        "x-vllm-inference-time": f"{inference_time_ms:.2f}",
        "x-vllm-prefill-time": f"{prefill_time_ms:.2f}",
        "x-vllm-decode-time": f"{decode_time_ms:.2f}",
        "x-vllm-prompt-tokens": str(usage.prompt_tokens or 0),
        "x-vllm-completion-tokens": str(completion_tokens),
        "x-vllm-cached-tokens": str(num_cached_tokens),
        "x-vllm-tokens-per-second": f"{tokens_per_second:.2f}",
    }


async def request_stats_headers_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """Middleware that injects x-vllm-* timing headers into responses.

    Reads request_metadata from request.state (populated by serving layers).
    Returns response unchanged if metadata or stats are missing.
    """
    response = await call_next(request)
    metadata = getattr(request.state, "request_metadata", None)
    if (
        metadata is None
        or metadata.request_stats is None
        or metadata.final_usage_info is None
    ):
        return response
    headers = build_request_stats_headers(
        metrics=metadata.request_stats,
        usage=metadata.final_usage_info,
        num_cached_tokens=metadata.num_cached_tokens,
    )
    for key, value in headers.items():
        response.headers[key] = value
    return response
