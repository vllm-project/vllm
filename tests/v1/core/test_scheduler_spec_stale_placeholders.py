# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for issue #36906: stale spec_token_ids = [-1, ...] placeholders
leak into scheduled_spec_decode_tokens when a running decode request is not
scheduled due to token budget exhaustion, then re-scheduled in a later step.

This bug is NOT multimodal-specific. It affects any model using async scheduling
+ speculative decoding under high enough concurrency to exhaust the token budget.
"""

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _get_async_scheduler(
    num_spec_tokens: int = 3,
    max_num_batched_tokens: int = 30,
    num_requests: int = 10,
    prompt_len: int = 10,
) -> tuple[AsyncScheduler, list]:
    """Create an async scheduler with spec decode config and add requests."""
    scheduler = create_scheduler(
        max_num_seqs=num_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=4096,
        async_scheduling=True,
        enable_chunked_prefill=True,
    )
    assert isinstance(scheduler, AsyncScheduler)
    scheduler.num_spec_tokens = num_spec_tokens
    scheduler.num_lookahead_tokens = num_spec_tokens
    scheduler._spec_token_placeholders = [-1] * num_spec_tokens

    requests = create_requests(
        num_requests=num_requests,
        num_tokens=prompt_len,
        max_tokens=256,
    )
    for req in requests:
        scheduler.add_request(req)
    return scheduler, requests


def test_stale_spec_token_ids_cleared_for_unscheduled_requests():
    """Directly verify: after schedule(), any running request NOT in
    num_scheduled_tokens must have spec_token_ids = [].

    Simulates the production scenario:
    1. Requests start in running queue with spec_token_ids = [-1,-1,-1]
       (set by _update_after_schedule from a previous step)
    2. schedule() exhausts token budget before visiting all requests
    3. Unvisited requests must have spec_token_ids cleared

    Uses text-only requests to prove the bug is not multimodal-specific.
    """
    num_spec_tokens = 3
    # Budget=30 can fit 7 decode requests (7*4=28 < 30) but not 10 (10*4=40).
    scheduler, requests = _get_async_scheduler(
        num_spec_tokens=num_spec_tokens,
        max_num_batched_tokens=25,
        num_requests=10,
        prompt_len=10,
    )

    # Manually move all requests to running + decode state,
    # simulating the state after prefill completes.
    for req in requests:
        scheduler.waiting.pop_request()
        scheduler.running.append(req)
        req.num_computed_tokens = len(req.prompt_token_ids)
        req.status = req.status  # keep WAITING->RUNNING handled by scheduler
        # Simulate _update_after_schedule: set stale spec placeholders.
        req.spec_token_ids = list(scheduler._spec_token_placeholders)

    # Now schedule(). With budget=30, only ~7 of 10 decode requests fit.
    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens.keys())

    # Verify budget caused some requests to be unscheduled.
    unscheduled = [r for r in scheduler.running if r.request_id not in scheduled_ids]
    assert len(unscheduled) > 0, (
        f"Expected some unscheduled requests, but all were scheduled. "
        f"Tokens: {dict(output.num_scheduled_tokens)}"
    )

    # THE FIX: stale spec_token_ids must be cleared for unscheduled requests.
    for req in unscheduled:
        assert req.spec_token_ids == [], (
            f"Unscheduled request {req.request_id} has stale "
            f"spec_token_ids={req.spec_token_ids}. "
            f"Without the fix, these -1 values would leak into "
            f"scheduled_spec_decode_tokens when re-scheduled."
        )


def test_no_negative_one_in_scheduled_spec_decode_tokens():
    """Verify: scheduled_spec_decode_tokens never contains -1.

    After the fix, unscheduled requests have spec_token_ids cleared.
    When re-scheduled in a later step, they won't contribute -1 to
    scheduled_spec_decode_tokens.
    """
    num_spec_tokens = 3
    scheduler, requests = _get_async_scheduler(
        num_spec_tokens=num_spec_tokens,
        max_num_batched_tokens=25,
        num_requests=10,
        prompt_len=10,
    )

    # Move all requests to decode state with stale spec placeholders.
    for req in requests:
        scheduler.waiting.pop_request()
        scheduler.running.append(req)
        req.num_computed_tokens = len(req.prompt_token_ids)
        req.spec_token_ids = list(scheduler._spec_token_placeholders)

    # Step 0: initial schedule — some scheduled with [-1,-1,-1]
    # (expected: these are fresh placeholders for the first decode step).
    output0 = scheduler.schedule()
    first_unscheduled = {
        r.request_id
        for r in scheduler.running
        if r.request_id not in output0.num_scheduled_tokens
    }
    assert len(first_unscheduled) > 0, "Need budget pressure for this test"

    # Step 1+: the previously unscheduled requests should NOT have -1
    # when they get scheduled, because the fix cleared their spec_token_ids.
    for step in range(1, 4):
        output = scheduler.schedule()

        for req_id, spec_tokens in output.scheduled_spec_decode_tokens.items():
            if req_id in first_unscheduled:
                assert all(t != -1 for t in spec_tokens), (
                    f"Step {step}: previously unscheduled request {req_id} "
                    f"has -1 in scheduled_spec_decode_tokens={spec_tokens}. "
                    f"Stale placeholders leaked into scheduler output."
                )
