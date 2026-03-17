# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import UsageInfo

if TYPE_CHECKING:
    from fastapi import Request

    from vllm.v1.metrics.stats import RequestStateStats


def build_request_stats_headers(
    metrics: RequestStateStats,
    usage: UsageInfo,
    num_cached_tokens: int,
) -> dict[str, str]:
    """Build HTTP response headers with per-request timing and compute stats.

    Times are in milliseconds, rounded to 2 decimal places.
    """
    total_time_ms = round((time.time() - metrics.arrival_time) * 1000, 2)
    queue_time_ms = round((metrics.scheduled_ts - metrics.queued_ts) * 1000, 2)
    prefill_time_ms = round(
        (metrics.first_token_ts - metrics.scheduled_ts) * 1000, 2
    )
    decode_time_ms = round(
        (metrics.last_token_ts - metrics.first_token_ts) * 1000, 2
    )
    inference_time_ms = round(
        (metrics.last_token_ts - metrics.scheduled_ts) * 1000, 2
    )

    return {
        "x-total-time": f"{total_time_ms:.2f}",
        "x-queue-time": f"{queue_time_ms:.2f}",
        "x-inference-time": f"{inference_time_ms:.2f}",
        "x-prefill-time": f"{prefill_time_ms:.2f}",
        "x-decode-time": f"{decode_time_ms:.2f}",
        "x-prompt-tokens": str(usage.prompt_tokens),
        "x-completion-tokens": str(usage.completion_tokens or 0),
        "x-cached-tokens": str(num_cached_tokens),
    }


def maybe_build_request_stats_headers(
    raw_request: Request,
) -> dict[str, str] | None:
    """Build stats headers if enabled and stats are available.

    Returns None if the feature is disabled or stats are not available.
    """
    if not getattr(
        raw_request.app.state.args, "enable_request_stats_headers", False
    ):
        return None
    metadata = getattr(raw_request.state, "request_metadata", None)
    if metadata is None or metadata.request_stats is None:
        return None
    return build_request_stats_headers(
        metrics=metadata.request_stats,
        usage=metadata.final_usage_info,
        num_cached_tokens=metadata.num_cached_tokens,
    )
