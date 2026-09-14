# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for stale multiproc RPC deadlines."""

import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from unittest.mock import patch


def _dequeue_timeout(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response,
        aggregate=lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakeResponseMQ:
    def __init__(self, response: object = ("SUCCESS", "dummy_response")) -> None:
        self.response = response
        self.timeouts: list[float | None] = []

    def dequeue(self, timeout: float | None = None) -> object:
        assert timeout is None or timeout >= 0.0, (
            f"dequeue received negative timeout: {timeout}"
        )
        self.timeouts.append(timeout)
        return self.response


def test_future_wrapper_stale_deadline_never_passes_negative_timeout() -> None:
    clock = FakeClock(100.0)
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    deadline = clock.monotonic() + 1.0

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(deadline))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    clock.advance(5.0)

    with patch("time.monotonic", clock.monotonic):
        future.result()

    assert response_mq.timeouts == [0.0]


def test_future_wrapper_non_expired_deadline_passes_positive_timeout() -> None:
    clock = FakeClock(100.0)
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    deadline = clock.monotonic() + 5.0

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(deadline))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    clock.advance(2.0)

    with patch("time.monotonic", clock.monotonic):
        future.result()

    assert len(response_mq.timeouts) == 1
    timeout = response_mq.timeouts[0]
    assert timeout is not None
    assert timeout > 0.0
    assert timeout <= 5.0


def test_future_wrapper_deadline_none_passes_none() -> None:
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(None))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    future.result()

    assert response_mq.timeouts == [None]


def test_future_wrapper_drains_pending_before_own_get_response() -> None:
    futures_queue: deque[FutureWrapper] = deque()
    call_order: list[str] = []

    def make_get_response(label: str):
        def _get() -> object:
            call_order.append(label)
            return FakeResponseMQ().dequeue(timeout=None)

        return _get

    first = FutureWrapper(futures_queue, make_get_response("first"))
    second = FutureWrapper(futures_queue, make_get_response("second"))

    second.result()

    assert first.done()
    assert second.done()
    assert call_order == ["first", "second"]


def test_recv_timeout_ms_clamps_negative_timeout() -> None:
    def recv_timeout_ms(timeout: float | None) -> int | None:
        return None if timeout is None else max(0, int(timeout * 1000))

    assert recv_timeout_ms(None) is None
    assert recv_timeout_ms(-1.0) == 0
    assert recv_timeout_ms(-0.001) == 0
    assert recv_timeout_ms(0.0) == 0
    assert recv_timeout_ms(0.001) == 1
    assert recv_timeout_ms(2.5) == 2500
