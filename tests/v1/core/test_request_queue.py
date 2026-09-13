# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from .utils import create_requests


@pytest.mark.parametrize("policy", list(SchedulingPolicy))
def test_iter_unordered_yields_every_request_once(policy: SchedulingPolicy):
    queue = create_request_queue(policy)
    requests = create_requests(num_requests=5)
    for request in requests:
        queue.add_request(request)

    unordered = list(queue.iter_unordered())

    assert len(unordered) == len(requests)
    assert {r.request_id for r in unordered} == {r.request_id for r in requests}
    assert list(queue) == list(queue), "iteration must not consume the queue"
