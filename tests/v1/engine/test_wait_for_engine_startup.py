# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for wait_for_engine_startup's readiness deadline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import zmq

from vllm import envs
from vllm.v1.engine import utils as engine_utils


class _StopWaiting(Exception):
    pass


class _FakePoller:
    """Poller stub that never reports events, simulating a wedged engine
    that connects to nothing and never sends READY."""

    def register(self, *args, **kwargs) -> None:
        pass

    def poll(self, _timeout_ms):
        return []


class _CountingPoller:
    """Like _FakePoller but breaks the wait after a fixed number of polls so
    a disabled (infinite) timeout can be exercised without hanging."""

    def __init__(self, max_polls: int = 50) -> None:
        self._polls = 0
        self._max_polls = max_polls

    def register(self, *args, **kwargs) -> None:
        pass

    def poll(self, _timeout_ms):
        self._polls += 1
        if self._polls > self._max_polls:
            raise _StopWaiting
        return []


def _parallel_config() -> SimpleNamespace:
    return SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=True,
    )


def test_wait_for_engine_startup_times_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(zmq, "Poller", _FakePoller)
    monkeypatch.setattr(envs, "VLLM_ENGINE_READY_TIMEOUT_S", 0.2)

    core_engines = [SimpleNamespace(identity=b"\x00", local=True)]

    with pytest.raises(TimeoutError, match="timed out after 0.2s"):
        engine_utils.wait_for_engine_startup(
            handshake_socket=None,
            addresses=None,
            core_engines=core_engines,
            parallel_config=_parallel_config(),
            coordinated_dp=False,
            cache_config=None,
            proc_manager=None,
            coord_process=None,
        )


def test_wait_for_engine_startup_disabled_timeout_keeps_waiting(
    monkeypatch: pytest.MonkeyPatch,
):
    """With the timeout disabled (0), the loop must keep waiting rather than
    raising TimeoutError. The poller breaks the wait via _StopWaiting after
    many polls, proving no premature timeout fired."""
    monkeypatch.setattr(zmq, "Poller", _CountingPoller)
    monkeypatch.setattr(envs, "VLLM_ENGINE_READY_TIMEOUT_S", 0)

    core_engines = [SimpleNamespace(identity=b"\x00", local=True)]

    with pytest.raises(_StopWaiting):
        engine_utils.wait_for_engine_startup(
            handshake_socket=None,
            addresses=None,
            core_engines=core_engines,
            parallel_config=_parallel_config(),
            coordinated_dp=False,
            cache_config=None,
            proc_manager=None,
            coord_process=None,
        )
