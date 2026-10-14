# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlConnector async completion polling (``async_completion_poll``)."""

import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorWorker,
    NixlPushConnectorWorker,
    base_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    _parse_bool_option,
)

pytestmark = pytest.mark.cpu_test

XFER_TIME_S = 0.05
MAIN_BLOCK_S = 0.3


class TimedNixlWrapper:
    """Transfers finish ``XFER_TIME_S`` after posting. Like NIXL, telemetry is
    stamped at the first check that sees DONE.

    ``states`` forces a handle's outcome: a NIXL state string, ``"RAISE"``
    (check_xfer_state raises) or ``"BAD_TELEMETRY"`` (DONE, but reading the
    telemetry raises).
    """

    def __init__(self, states: dict[int, str] | None = None):
        self._lock = threading.Lock()
        self._posted: dict[int, float] = {}
        self._duration: dict[int, float] = {}
        self._states = states or {}
        self.checks: defaultdict[int, int] = defaultdict(int)
        self.checks_after_done: list[int] = []
        self.checks_after_release: list[int] = []
        self.released: list[int] = []

    def post(self, handle: int) -> int:
        self._posted[handle] = time.perf_counter()
        return handle

    def check_xfer_state(self, handle: int) -> str:
        with self._lock:
            self.checks[handle] += 1
            if handle in self.released:
                self.checks_after_release.append(handle)
            if handle in self._duration:
                self.checks_after_done.append(handle)
            forced = self._states.get(handle)
            if forced == "RAISE":
                raise RuntimeError("check failed")
            if forced == "BAD_TELEMETRY":
                return "DONE"
            if forced is not None:
                return forced
            elapsed = time.perf_counter() - self._posted[handle]
            if elapsed < XFER_TIME_S:
                return "PROC"
            self._duration[handle] = elapsed
            return "DONE"

    def get_xfer_telemetry(self, handle: int):
        if self._states.get(handle) == "BAD_TELEMETRY":
            raise RuntimeError("telemetry failed")
        return SimpleNamespace(xferDuration=self._duration[handle])

    def release_xfer_handle(self, handle: int) -> None:
        with self._lock:
            self.released.append(handle)


def _make_worker(
    wrapper: TimedNixlWrapper, handles: list[int], async_poll: bool = True
) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker.nixl_wrapper = wrapper
    worker.xfer_stats = MagicMock()
    worker._recving_transfers = defaultdict(list)
    worker._recving_transfers["req"] = [wrapper.post(h) for h in handles]
    worker._async_poll = async_poll
    worker._async_poll_lock = threading.Lock()
    worker._async_done = {}
    worker._log_failure = MagicMock()  # type: ignore[method-assign]

    def handle_failed(req_id, handle, failed_req_ids):
        failed_req_ids.add(req_id)
        wrapper.release_xfer_handle(handle)
        return True

    worker._handle_failed_transfer = MagicMock(  # type: ignore[method-assign]
        side_effect=handle_failed
    )
    return worker


def _wait_until_recorded(worker: NixlConnectorWorker, n: int) -> None:
    deadline = time.monotonic() + 5
    while len(worker._async_done) < n:
        assert time.monotonic() < deadline, "poll thread recorded nothing"
        time.sleep(0.005)


def test_completion_stamped_while_main_thread_blocked():
    wrapper = TimedNixlWrapper()
    worker = _make_worker(wrapper, [1, 2, 3])
    worker._start_async_poll(0.001)
    try:
        # The engine step is busy well past the transfer time.
        time.sleep(MAIN_BLOCK_S)
        done, failed = worker._pop_done_transfers(worker._recving_transfers)
    finally:
        worker._stop_async_poll()

    assert done == {"req"} and failed == set()
    durations = [
        c.args[0].xferDuration for c in worker.xfer_stats.record_transfer.mock_calls
    ]
    assert len(durations) == 3
    # Stamped by the poll thread, not at the engine step (>= MAIN_BLOCK_S).
    assert all(d < MAIN_BLOCK_S for d in durations), durations
    assert wrapper.checks_after_done == []
    assert sorted(wrapper.released) == [1, 2, 3]
    assert worker._async_done == {}
    assert "req" not in worker._recving_transfers


def test_sync_path_stamps_at_engine_step():
    wrapper = TimedNixlWrapper()
    worker = _make_worker(wrapper, [1], async_poll=False)

    time.sleep(MAIN_BLOCK_S)
    done, _ = worker._pop_done_transfers(worker._recving_transfers)

    assert done == {"req"}
    (call,) = worker.xfer_stats.record_transfer.mock_calls
    assert call.args[0].xferDuration >= MAIN_BLOCK_S


@pytest.mark.parametrize("async_poll", [True, False])
def test_failures_are_reported_the_same_with_and_without_polling(async_poll):
    wrapper = TimedNixlWrapper(states={1: "ERR", 2: "RAISE", 3: "BAD_TELEMETRY"})
    worker = _make_worker(wrapper, [1, 2, 3], async_poll=async_poll)
    if async_poll:
        worker._start_async_poll(0.001)
    try:
        if async_poll:
            _wait_until_recorded(worker, 3)
        done, failed = worker._pop_done_transfers(worker._recving_transfers)
    finally:
        worker._stop_async_poll()

    assert done == {"req"} and failed == {"req"}
    failure_types = sorted(
        c.kwargs["failure_type"] for c in worker._log_failure.mock_calls
    )
    assert failure_types == [
        "transfer_exception",
        "transfer_exception",
        "transfer_failed",
    ]
    worker.xfer_stats.record_transfer.assert_not_called()
    # Each terminal state is observed exactly once.
    assert wrapper.checks == {1: 1, 2: 1, 3: 1}
    assert worker._async_done == {}


def test_poll_thread_survives_unexpected_error():
    wrapper = TimedNixlWrapper()
    worker = _make_worker(wrapper, [1])
    record = worker._record_completed_recvs
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        record()

    worker._record_completed_recvs = flaky  # type: ignore[method-assign]
    worker._start_async_poll(0.001)
    try:
        _wait_until_recorded(worker, 1)
        assert worker._async_poll_thread.is_alive()
    finally:
        worker._stop_async_poll()


def test_shutdown_stops_thread_before_releasing():
    wrapper = TimedNixlWrapper(states={1: "PROC"})
    worker = _make_worker(wrapper, [1])
    worker._handshake_initiation_executor = MagicMock()
    worker._finish_shutdown = MagicMock()  # type: ignore[method-assign]
    worker._start_async_poll(0.001)
    deadline = time.monotonic() + 5
    while wrapper.checks[1] == 0:
        assert time.monotonic() < deadline
        time.sleep(0.005)

    worker.shutdown()

    assert not worker._async_poll_thread.is_alive()
    assert wrapper.released == [1]
    assert wrapper.checks_after_release == []
    worker._finish_shutdown.assert_called_once()


@pytest.mark.parametrize(
    ("worker_cls", "agent_config", "expected"),
    [
        (NixlConnectorWorker, object(), True),
        # Push sends complete via _sending_transfers, which is not polled.
        (NixlPushConnectorWorker, object(), False),
        # Without an agent config the agent cannot be made thread-safe.
        (NixlConnectorWorker, None, False),
    ],
)
def test_async_poll_enabled_only_where_supported(
    monkeypatch, worker_cls, agent_config, expected
):
    monkeypatch.setattr(base_worker, "nixl_agent_config", agent_config)
    worker = object.__new__(worker_cls)
    assert worker._async_poll_enabled(True) is expected
    assert worker._async_poll_enabled(False) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("1", True), ("False", False)],
)
def test_async_poll_flag_parsing(value, expected):
    assert _parse_bool_option("async_completion_poll", value) is expected


@pytest.mark.parametrize("value", ["maybe", 2, None])
def test_async_poll_flag_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="async_completion_poll"):
        _parse_bool_option("async_completion_poll", value)
