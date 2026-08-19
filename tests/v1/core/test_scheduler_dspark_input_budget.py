# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ensure DSpark scheduling does not charge drafting slots against the
target/context input budget, allowing requests that fit the separate
stages to be admitted.
"""
import pytest

from tests.v1.core.utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_dspark_input_budget_separate():
    # max_num_batched_tokens: target/context stage capacity
    # num_speculative_tokens: K for DSpark
    scheduler = create_scheduler(
        max_num_batched_tokens=16,
        num_speculative_tokens=4,
        speculative_method="dspark",
    )

    # Two requests, each with 8 prefill tokens (fits target/context stage: 8+8=16)
    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    out = scheduler.schedule()

    # Both requests should be admitted and scheduled for 8 tokens each.
    assert len(out.num_scheduled_tokens) == 2
    for v in out.num_scheduled_tokens.values():
        assert v == 8
