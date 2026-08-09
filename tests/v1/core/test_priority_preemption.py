# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vLLM V1 Priority Scheduler preemption re-admission."""

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.request import Request, RequestStatus


def create_test_request(
    request_id: str,
    priority: int = 0,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=arrival_time,
        priority=priority,
    )
    req.num_preemptions = num_preemptions
    req.status = (
        RequestStatus.PREEMPTED if num_preemptions > 0 else RequestStatus.WAITING
    )
    return req


def test_priority_queue_preemption_readmission():
    """Preempted request should be re-admitted before fresh requests."""
    queue = PriorityRequestQueue()

    req_fresh = create_test_request(
        "fresh", priority=5, arrival_time=5.0, num_preemptions=0
    )
    req_preempted = create_test_request(
        "preempted", priority=5, arrival_time=10.0, num_preemptions=1
    )

    queue.add_request(req_fresh)
    queue.prepend_request(req_preempted)

    assert queue.pop_request().request_id == "preempted"
    assert queue.pop_request().request_id == "fresh"


def test_priority_queue_starvation_cap():
    """Preemption boost should be capped at 3."""
    req_3 = create_test_request("r3", priority=5, num_preemptions=3)
    req_10 = create_test_request("r10", priority=5, num_preemptions=10)

    assert req_3.sort_key[1] == -3
    assert req_10.sort_key[1] == -3
