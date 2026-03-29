# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from vllm.entrypoints.openai.request_stats_headers import (
    build_request_stats_headers,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.v1.metrics.stats import RequestStateStats


def test_build_request_stats_headers_basic():
    """Headers are computed correctly from known timestamps."""
    now = time.time()
    stats = RequestStateStats(
        arrival_time=now - 1.0,
        queued_ts=100.0,
        scheduled_ts=100.05,
        first_token_ts=100.15,
        last_token_ts=100.45,
        num_generation_tokens=10,
    )
    usage = UsageInfo(
        prompt_tokens=50,
        completion_tokens=10,
        total_tokens=60,
    )
    headers = build_request_stats_headers(
        metrics=stats,
        usage=usage,
        num_cached_tokens=5,
    )

    assert "x-total-time" in headers
    assert "x-queue-time" in headers
    assert "x-inference-time" in headers
    assert "x-prefill-time" in headers
    assert "x-decode-time" in headers
    assert "x-prompt-tokens" in headers
    assert "x-completion-tokens" in headers
    assert "x-cached-tokens" in headers

    assert float(headers["x-queue-time"]) == round((100.05 - 100.0) * 1000, 2)
    assert float(headers["x-prefill-time"]) == round((100.15 - 100.05) * 1000, 2)
    assert float(headers["x-decode-time"]) == round((100.45 - 100.15) * 1000, 2)
    assert float(headers["x-inference-time"]) == round((100.45 - 100.05) * 1000, 2)

    assert headers["x-prompt-tokens"] == "50"
    assert headers["x-completion-tokens"] == "10"
    assert headers["x-cached-tokens"] == "5"

    total_time = float(headers["x-total-time"])
    assert 900 < total_time < 1500


def test_build_request_stats_headers_zero_timestamps():
    """When timestamps are 0 (not set), timing headers show 0."""
    stats = RequestStateStats(
        arrival_time=time.time(),
        queued_ts=0.0,
        scheduled_ts=0.0,
        first_token_ts=0.0,
        last_token_ts=0.0,
    )
    usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    headers = build_request_stats_headers(metrics=stats, usage=usage, num_cached_tokens=0)

    assert headers["x-queue-time"] == "0.00"
    assert headers["x-prefill-time"] == "0.00"
    assert headers["x-decode-time"] == "0.00"
    assert headers["x-inference-time"] == "0.00"


def test_build_request_stats_headers_partial_timestamps():
    """When scheduled but cancelled before tokens, timing values clamp to 0."""
    stats = RequestStateStats(
        arrival_time=time.time() - 0.5,
        queued_ts=100.0,
        scheduled_ts=100.05,
        first_token_ts=0.0,  # no tokens generated
        last_token_ts=0.0,
    )
    usage = UsageInfo(prompt_tokens=20, completion_tokens=0, total_tokens=20)
    headers = build_request_stats_headers(
        metrics=stats, usage=usage, num_cached_tokens=0
    )

    # These would be negative without clamping
    assert float(headers["x-prefill-time"]) == 0.0
    assert float(headers["x-decode-time"]) == 0.0
    assert float(headers["x-inference-time"]) == 0.0
    # Queue time should still be valid
    assert float(headers["x-queue-time"]) == 50.0
