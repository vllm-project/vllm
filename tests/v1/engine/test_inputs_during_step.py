# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EngineCoreProc handling inputs while waiting for a step's output."""

import queue
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any
from unittest import mock

import pytest

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine import core as engine_core
from vllm.v1.engine.core import EngineCoreProc

ADD = EngineCoreRequestType.ADD
ABORT = EngineCoreRequestType.ABORT
UTILITY = EngineCoreRequestType.UTILITY
START_DP_WAVE = EngineCoreRequestType.START_DP_WAVE
EXECUTOR_FAILED = EngineCoreRequestType.EXECUTOR_FAILED
WAKEUP = EngineCoreRequestType.WAKEUP


class PollableFuture(Future):
    """Becomes done on the `ready_after`-th poll; on_poll is called first."""

    def __init__(
        self,
        ready_after: int,
        on_poll: Callable[[int], None] | None = None,
    ):
        super().__init__()
        self.ready_after = ready_after
        self.on_poll = on_poll
        self.polls = 0
        self.waits = 0

    def poll(self) -> bool:
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.polls >= self.ready_after and not self.done():
            self.set_result("output")
        return self.done()

    def wait_briefly(self, pending, deadline) -> None:
        self.waits += 1

    def result(self, timeout=None):
        # Stands in for blocking until the output arrives.
        if not self.done():
            self.set_result("output")
        return super().result(timeout)


def make_engine(handle: bool = True) -> tuple[EngineCoreProc, list[Any]]:
    """An EngineCoreProc with just the state used by input handling."""
    engine = object.__new__(EngineCoreProc)
    engine.input_queue = queue.Queue()
    engine._deferred_inputs = deque()
    engine.has_work = lambda: True  # type: ignore[method-assign]
    handled: list[Any] = []
    if handle:

        def record(request_type: EngineCoreRequestType, request: Any) -> None:
            handled.append((request_type, request))

        engine._handle_client_request = record  # type: ignore[method-assign]
    return engine, handled


def test_adds_and_aborts_are_handled_while_waiting():
    engine, handled = make_engine()

    def on_poll(n: int) -> None:
        if n == 2:
            engine.input_queue.put((ADD, "a"))
        elif n == 4:
            engine.input_queue.put((ABORT, ["a"]))
            engine.input_queue.put((WAKEUP, None))

    future = PollableFuture(ready_after=10, on_poll=on_poll)
    assert engine._wait_for_output(future) == "output"
    assert future.polls == 10
    assert handled == [(ADD, "a"), (ABORT, ["a"]), (WAKEUP, None)]
    assert not engine._deferred_inputs


@pytest.mark.parametrize("barrier", [UTILITY, START_DP_WAVE])
def test_other_inputs_and_those_after_them_wait_for_step(barrier):
    engine, handled = make_engine()
    inputs = [(ADD, "a"), (ABORT, ["x"]), (barrier, "u"), (ADD, "b"), (ABORT, ["a"])]
    for req in inputs:
        engine.input_queue.put(req)

    future = PollableFuture(ready_after=1000)
    assert engine._wait_for_output(future) == "output"
    # Stopped polling at the first input that can't be handled during the step.
    assert future.polls < 10
    assert handled == inputs[:2]
    assert list(engine._deferred_inputs) == [inputs[2]]

    # After the step, the remaining inputs are handled in their original order.
    engine._process_input_queue()
    assert handled == inputs
    assert not engine._deferred_inputs
    assert engine.input_queue.empty()


def test_executor_failure_raises_while_waiting():
    engine, _ = make_engine(handle=False)
    engine.input_queue.put((EXECUTOR_FAILED, b""))
    with pytest.raises(RuntimeError, match="Executor failed"):
        engine._wait_for_output(PollableFuture(ready_after=1000))


def test_future_without_poll_blocks_for_result():
    engine, handled = make_engine()
    engine.input_queue.put((ADD, "a"))
    future: Future = Future()
    future.set_result("output")
    assert engine._wait_for_output(future) == "output"
    assert handled == []
    assert engine.input_queue.qsize() == 1


def test_blocks_for_output_after_spin_period():
    """Once the spin period has passed, pending inputs are still handled, then
    the wait blocks for the output rather than busy-looping."""
    engine, handled = make_engine()
    engine.input_queue.put((ADD, "a"))
    future = PollableFuture(ready_after=1000)
    with mock.patch.object(engine_core, "_INPUT_SPIN_DURATION_S", 0.0):
        assert engine._wait_for_output(future) == "output"
    assert future.waits == 0
    assert future.polls == 2
    assert handled == [(ADD, "a")]
