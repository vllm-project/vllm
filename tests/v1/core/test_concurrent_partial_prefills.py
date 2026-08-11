# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for concurrent partial-prefill limits in the V1 scheduler."""
import pytest

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_default_limit_one_partial_prefill_at_a_time():
    """With max_num_partial_prefills=1 only one long request prefills per step."""
    scheduler = create_scheduler(
        max_num_batched_tokens=64,
        long_prefill_token_threshold=32,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        max_model_len=512,
    )
    reqs = create_requests(num_requests=2, num_tokens=200, max_tokens=4)
    for r in reqs:
        scheduler.add_request(r)

    output = scheduler.schedule()
    scheduled = list(output.num_scheduled_tokens.keys())

    assert len(scheduled) == 1, (
        f"Expected 1 prefilling request, got {len(scheduled)}: {scheduled}"
    )


def test_limit_two_allows_two_concurrent_partial_prefills():
    """With max_num_partial_prefills=2 both long requests prefill together."""
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        long_prefill_token_threshold=32,
        max_num_partial_prefills=2,
        max_long_partial_prefills=2,
        max_model_len=512,
    )
    reqs = create_requests(num_requests=2, num_tokens=200, max_tokens=4)
    for r in reqs:
        scheduler.add_request(r)

    output = scheduler.schedule()
    scheduled = list(output.num_scheduled_tokens.keys())

    assert len(scheduled) == 2, (
        f"Expected 2 prefilling requests, got {len(scheduled)}: {scheduled}"
    )


def test_max_long_partial_prefills_limits_long_subset():
    """max_long_partial_prefills=1 caps only the long-prefill subset.
    A short request (below threshold) is unaffected and scheduled alongside."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256,
        long_prefill_token_threshold=50,
        max_num_partial_prefills=2,
        max_long_partial_prefills=1,
        max_model_len=512,
    )
    long_req = create_requests(num_requests=1, num_tokens=200, max_tokens=4,
                               req_ids=["long"])
    short_req = create_requests(num_requests=1, num_tokens=20, max_tokens=4,
                                req_ids=["short"])
    for r in long_req + short_req:
        scheduler.add_request(r)

    output = scheduler.schedule()
    scheduled = set(output.num_scheduled_tokens.keys())

    # long gets 1 chunk (threshold=50), short completes in one step
    assert "long" in scheduled
    assert "short" in scheduled


def test_short_requests_not_blocked_by_partial_prefill_limit():
    """Requests shorter than long_prefill_token_threshold are never
    treated as partial prefills and are unaffected by the limit."""
    scheduler = create_scheduler(
        max_num_batched_tokens=512,
        long_prefill_token_threshold=100,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        max_model_len=512,
    )
    # short requests (10 tokens each) are well below threshold=100
    short_reqs = create_requests(num_requests=4, num_tokens=10, max_tokens=4,
                                 req_ids=["s0", "s1", "s2", "s3"])
    for r in short_reqs:
        scheduler.add_request(r)

    output = scheduler.schedule()
    scheduled = set(output.num_scheduled_tokens.keys())

    # All short requests complete in one step — none are partial prefills
    assert {"s0", "s1", "s2", "s3"} == scheduled
