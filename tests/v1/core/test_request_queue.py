# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.request_queue import PriorityRequestQueue

from .utils import create_requests

pytestmark = pytest.mark.cpu_test


def test_priority_queue_lazy_remove_skips_deleted_requests():
    queue = PriorityRequestQueue()
    requests = create_requests(
        num_requests=5,
        req_ids=["req-3", "req-1", "req-4", "req-0", "req-2"],
    )
    for req, priority in zip(requests, [3, 1, 4, 0, 2]):
        req.priority = priority
    for req in requests:
        queue.add_request(req)

    queue.remove_request(requests[3])  # top
    queue.remove_request(requests[1])  # middle

    assert len(queue) == 3
    popped_ids = [queue.pop_request().request_id for _ in range(3)]
    assert popped_ids == ["req-2", "req-3", "req-4"]
    assert not queue


def test_priority_queue_rebuild_clears_tombstones():
    queue = PriorityRequestQueue()
    requests = create_requests(
        num_requests=40,
        req_ids=[f"req-{i}" for i in range(40)],
    )
    for i, req in enumerate(requests):
        req.priority = i
    for req in requests:
        queue.add_request(req)

    for req in requests[:33]:
        queue.remove_request(req)

    # Trigger cleanup path.
    top = queue.peek_request()
    assert top.request_id == "req-33"
    assert len(queue) == 7
    assert len(queue._heap) == 7


def test_priority_queue_iter_excludes_deleted_requests():
    queue = PriorityRequestQueue()
    requests = create_requests(
        num_requests=6,
        req_ids=[f"req-{i}" for i in range(6)],
    )
    for i, req in enumerate(requests):
        req.priority = i
    for req in requests:
        queue.add_request(req)

    queue.remove_requests([requests[1], requests[4]])
    ordered_ids = [req.request_id for req in queue]
    assert ordered_ids == ["req-0", "req-2", "req-3", "req-5"]


def test_priority_queue_remove_then_readd_same_request():
    queue = PriorityRequestQueue()
    req_a, req_b = create_requests(num_requests=2, req_ids=["a", "b"])
    req_a.priority = 1
    req_b.priority = 0

    queue.add_request(req_a)
    queue.remove_request(req_a)
    queue.add_request(req_a)
    queue.add_request(req_b)

    assert [queue.pop_request().request_id for _ in range(2)] == ["b", "a"]
    assert not queue
