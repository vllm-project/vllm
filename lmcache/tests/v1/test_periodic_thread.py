# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lmcache.v1.periodic_thread.PeriodicThread."""

# Standard
from typing import Optional
import threading
import time

# First Party
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
)


class _CountingThread(PeriodicThread):
    """PeriodicThread that counts executions and can gate them on an event."""

    def __init__(self, interval: float, gate: Optional[threading.Event] = None):
        super().__init__(
            name="test-counting-thread",
            interval=interval,
            level=ThreadLevel.LOW,
        )
        self._runs = 0
        self._gate = gate

    def _execute(self) -> ThreadRunSummary:
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        self._runs += 1
        return ThreadRunSummary(success=True, message="ok")

    @property
    def runs(self) -> int:
        return self._runs

    def wait_for_runs(self, count: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while self._runs < count and time.time() < deadline:
            time.sleep(0.01)
        return self._runs >= count


def test_wake_triggers_immediate_run():
    """wake() should skip the interval sleep and run one cycle now."""
    PeriodicThreadRegistry.reset()
    # Long interval; without wake() the second run would never arrive
    # within the test's time budget.
    t = _CountingThread(interval=60.0)
    t.start()
    try:
        assert t.wait_for_runs(1, timeout=5.0), "first run did not happen"
        t.wake()
        assert t.wait_for_runs(2, timeout=5.0), "wake() did not trigger a run"
    finally:
        t.stop(timeout=2.0)


def test_wake_before_start_is_noop():
    """wake() before start() must not crash and must not schedule a run."""
    PeriodicThreadRegistry.reset()
    t = _CountingThread(interval=60.0)
    # Not started yet.
    t.wake()
    assert t.runs == 0
    assert not t.is_running


def test_stop_breaks_interval_sleep_promptly():
    """stop() should return quickly even with a long interval."""
    PeriodicThreadRegistry.reset()
    t = _CountingThread(interval=60.0)
    t.start()
    try:
        assert t.wait_for_runs(1, timeout=5.0)
    finally:
        started = time.time()
        t.stop(timeout=2.0)
        elapsed = time.time() - started
        assert not t.is_running
        # Was 60 s before wake support; must now be well under a second.
        assert elapsed < 2.0, "stop() waited on interval sleep: %.2fs" % elapsed


def test_wake_during_execute_schedules_next_cycle():
    """wake() called while _execute is running should short-circuit the
    following interval sleep and run once more right away."""
    PeriodicThreadRegistry.reset()
    gate = threading.Event()
    t = _CountingThread(interval=60.0, gate=gate)
    t.start()
    try:
        # Wake while the first _execute is blocked on the gate.
        t.wake()
        gate.set()
        # First cycle completes; then the wake flag makes the next
        # interval sleep return immediately, so runs reaches 2 without
        # waiting on the 60 s interval.
        assert t.wait_for_runs(2, timeout=5.0), (
            "wake() did not carry over; runs=%d" % t.runs
        )
    finally:
        gate.set()
        t.stop(timeout=2.0)
