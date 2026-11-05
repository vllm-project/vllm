# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for non-blocking polling of multiproc executor futures."""

from collections import deque

import pytest

from vllm.v1.executor.multiproc_executor import FutureWrapper


class Response:
    """A response that can be collected once it has arrived."""

    def __init__(self, value: object, calls: list[object]):
        self.value = value
        self.arrived = False
        self.calls = calls

    def ready(self) -> bool:
        return self.arrived

    def get(self) -> object:
        assert self.arrived, "get_response() would block"
        self.calls.append(self.value)
        return self.value


def test_poll_collects_arrived_responses_in_order():
    queue: deque[FutureWrapper] = deque()
    calls: list[object] = []
    first, second = Response("a", calls), Response("b", calls)
    f1 = FutureWrapper(queue, first.get, response_ready=first.ready)
    f2 = FutureWrapper(queue, second.get, response_ready=second.ready)

    assert not f2.poll()
    assert calls == []

    # A response for the later future can't be collected before the earlier one.
    second.arrived = True
    assert not f1.poll()
    assert calls == []

    first.arrived = True
    assert f2.poll()
    assert calls == ["a", "b"]
    assert f1.result() == "a"
    assert f2.result() == "b"
    assert not queue


def test_poll_stops_at_first_response_not_yet_arrived():
    queue: deque[FutureWrapper] = deque()
    calls: list[object] = []
    first, second = Response("a", calls), Response("b", calls)
    f1 = FutureWrapper(queue, first.get, response_ready=first.ready)
    f2 = FutureWrapper(queue, second.get, response_ready=second.ready)

    first.arrived = True
    assert not f2.poll()
    assert f1.done() and f1.result() == "a"
    assert list(queue) == [f2]

    second.arrived = True
    assert f2.poll()
    assert f2.result() == "b"


def test_poll_without_readiness_check_collects_response():
    queue: deque[FutureWrapper] = deque()
    future = FutureWrapper(queue, lambda: "value")
    assert future.poll()
    assert future.result() == "value"


def test_poll_sets_exception_from_response():
    def fail():
        raise ValueError("worker failed")

    queue: deque[FutureWrapper] = deque()
    future = FutureWrapper(queue, fail, response_ready=lambda: True)
    assert future.poll()
    with pytest.raises(ValueError, match="worker failed"):
        future.result()


def test_wait_briefly_waits_for_oldest_response():
    queue: deque[FutureWrapper] = deque()
    waited: list[str] = []

    def waiter(name: str):
        return lambda pending, deadline: waited.append(name)

    f1 = FutureWrapper(queue, lambda: "a", wait_for_response=waiter("first"))
    f2 = FutureWrapper(queue, lambda: "b", wait_for_response=waiter("second"))

    f2.wait_briefly([], 0.0)
    assert waited == ["first"]

    assert f1.poll()
    f2.wait_briefly([], 0.0)
    assert waited == ["first", "second"]
