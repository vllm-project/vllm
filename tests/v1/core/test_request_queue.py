# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import pytest

from vllm.v1.core.sched.request_queue import PriorityRequestQueue

pytestmark = pytest.mark.cpu_test


@dataclass(order=True, unsafe_hash=True)
class DummyRequest:
    priority: int
    arrival_time: float
    request_id: str = field(compare=False)


def test_priority_queue_lazy_remove_skips_deleted_requests():
    queue = PriorityRequestQueue()
    requests = [
        DummyRequest(priority=3, arrival_time=0.0, request_id="req-3"),
        DummyRequest(priority=1, arrival_time=0.0, request_id="req-1"),
        DummyRequest(priority=4, arrival_time=0.0, request_id="req-4"),
        DummyRequest(priority=0, arrival_time=0.0, request_id="req-0"),
        DummyRequest(priority=2, arrival_time=0.0, request_id="req-2"),
    ]
    for req in requests:
        queue.add_request(req)  # type: ignore[arg-type]

    queue.remove_request(requests[3])  # top
    queue.remove_request(requests[1])  # middle

    assert len(queue) == 3
    popped_ids = [queue.pop_request().request_id for _ in range(3)]  # type: ignore[union-attr]
    assert popped_ids == ["req-2", "req-3", "req-4"]
    assert not queue


def test_priority_queue_rebuild_clears_tombstones():
    queue = PriorityRequestQueue()
    requests = [
        DummyRequest(priority=i, arrival_time=0.0, request_id=f"req-{i}")
        for i in range(40)
    ]
    for req in requests:
        queue.add_request(req)  # type: ignore[arg-type]

    for req in requests[:33]:
        queue.remove_request(req)

    # Trigger cleanup path.
    top = queue.peek_request()  # type: ignore[assignment]
    assert top.request_id == "req-33"
    assert len(queue) == 7
    assert len(queue._removed_requests) == 0
    assert len(queue._heap) == 7


def test_priority_queue_iter_excludes_deleted_requests():
    queue = PriorityRequestQueue()
    requests = [
        DummyRequest(priority=i, arrival_time=0.0, request_id=f"req-{i}")
        for i in range(6)
    ]
    for req in requests:
        queue.add_request(req)  # type: ignore[arg-type]

    queue.remove_requests([requests[1], requests[4]])
    ordered_ids = [req.request_id for req in queue]  # type: ignore[misc]
    assert ordered_ids == ["req-0", "req-2", "req-3", "req-5"]
