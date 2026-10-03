# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ConnectorPoller: services a connector only while the engine waits."""

import threading
import time
from concurrent.futures import Future

import pytest

from vllm.v1.engine.core import ConnectorPoller


class _CountingConnector:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.thread_names: set[str] = set()

    def poll_pending_work(self) -> None:
        self.calls += 1
        self.thread_names.add(threading.current_thread().name)
        if self.fail:
            raise RuntimeError("sweep failed")


def _resolve_after(future: Future, delay_s: float, value: object = "out") -> None:
    def run():
        time.sleep(delay_s)
        future.set_result(value)

    threading.Thread(target=run, daemon=True).start()


def test_poller_sweeps_only_while_waiting():
    connector = _CountingConnector()
    poller = ConnectorPoller(connector, interval_s=0.001)
    try:
        time.sleep(0.02)
        assert connector.calls == 0

        future: Future = Future()
        _resolve_after(future, 0.05)
        assert poller.wait(future) == "out"
        assert connector.calls > 0
        assert connector.thread_names == {"kv-connector-poller"}

        settled = connector.calls
        time.sleep(0.02)
        assert connector.calls == settled
    finally:
        poller.close()


def test_poller_reraises_sweep_error_on_engine_thread():
    connector = _CountingConnector(fail=True)
    poller = ConnectorPoller(connector, interval_s=0.001)
    try:
        future: Future = Future()
        _resolve_after(future, 0.03)
        with pytest.raises(RuntimeError, match="sweep failed"):
            poller.wait(future)
        assert connector.calls == 1

        # A later wait runs sweeps again; the stored error was consumed.
        connector.fail = False
        future = Future()
        _resolve_after(future, 0.03)
        assert poller.wait(future) == "out"
        assert connector.calls > 1
    finally:
        poller.close()


def test_poller_propagates_future_exception():
    connector = _CountingConnector()
    poller = ConnectorPoller(connector, interval_s=0.001)
    try:
        future: Future = Future()
        future.set_exception(ValueError("model failed"))
        with pytest.raises(ValueError, match="model failed"):
            poller.wait(future)
    finally:
        poller.close()
