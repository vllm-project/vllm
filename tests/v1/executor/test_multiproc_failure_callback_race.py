# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the worker-failure notification race (#69292).

`register_failure_callback` and the `MultiprocWorkerMonitor` thread form a
pair over (`is_failed`, `failure_callback`). Without synchronization, a
registration that checks the flag just before the monitor claims the
failure could store a callback that nothing ever invokes, leaving the
engine scheduling onto a dead executor.
"""

import threading
from typing import Any

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


def _make_executor() -> Any:
    """Build a bare MultiprocExecutor with only the failure-protocol state."""
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.failure_callback = None
    executor._failure_lock = threading.Lock()
    return executor


def _monitor_claim(executor: Any) -> Any:
    """Run the monitor's claim section, exactly as `monitor_workers` does."""
    with executor._failure_lock:
        executor.is_failed = True
        callback = executor.failure_callback
        executor.failure_callback = None
    return callback


def test_monitor_claims_previously_registered_callback():
    """A callback stored before the failure is fetched and invoked."""
    executor = _make_executor()
    invoked: list[str] = []

    executor.register_failure_callback(lambda: invoked.append("cb"))

    callback = _monitor_claim(executor)
    assert callback is not None
    callback()

    assert invoked == ["cb"]
    assert executor.failure_callback is None


def test_register_after_failure_claim_invokes_directly():
    """The losing schedule from #69292: the monitor claims the failure
    before the registration stores its callback. The callback must still
    be invoked instead of being orphaned."""
    executor = _make_executor()
    invoked: list[str] = []

    # The monitor wins the race: claim the flag and fetch/clear (nothing
    # registered yet).
    fetched = _monitor_claim(executor)
    assert fetched is None

    # Registration now observes the claimed failure and must invoke.
    executor.register_failure_callback(lambda: invoked.append("cb"))

    assert invoked == ["cb"]


def test_register_invokes_exactly_once_when_flag_set():
    """A registration against an already-failed executor invokes exactly
    once and does not store the callback."""
    executor = _make_executor()
    invoked: list[str] = []

    executor.is_failed = True
    executor.register_failure_callback(lambda: invoked.append("cb"))

    assert invoked == ["cb"]
    assert executor.failure_callback is None

    # The monitor's claim section finds nothing left to invoke.
    assert _monitor_claim(executor) is None
    assert invoked == ["cb"]


def test_no_lost_notification_under_concurrent_interleaving():
    """Stress the interleaving: for every ordering of one registration
    against the monitor's claim, the callback runs exactly once."""
    for _ in range(200):
        executor = _make_executor()
        calls: list[int] = []
        invoked = threading.Event()

        def callback(_calls=calls, _invoked=invoked) -> None:
            _calls.append(1)
            _invoked.set()

        def monitor(_executor=executor) -> None:
            claimed = _monitor_claim(_executor)
            if claimed is not None:
                claimed()

        thread = threading.Thread(target=monitor)
        thread.start()
        executor.register_failure_callback(callback)
        thread.join(timeout=5)

        assert invoked.wait(timeout=5), "failure notification was lost"
        assert calls == [1]
        thread.join(timeout=5)
