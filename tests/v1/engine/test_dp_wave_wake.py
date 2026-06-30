# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DPEngineCoreProc.add_request first-wave wake behavior.

In a multi-pod Wide-EP DP deployment, the first request of wave 0 arrives when
``current_wave == 0``; upstream only wakes peers when ``request_wave !=
current_wave``, so the other DP engines never start stepping and the collective
hangs. The fix wakes peers on the first request when running on ROCm, while
leaving the default (non-ROCm) behavior unchanged.

We bind ``DPEngineCoreProc.add_request`` to a bare instance and stub the heavy
``super().add_request`` (``EngineCore.add_request``) so only the wake gate runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.platforms import current_platform
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore

add_request = DPEngineCoreProc.add_request


@pytest.fixture(autouse=True)
def _stub_super(monkeypatch):
    # Prevent super().add_request() from running the real scheduler path.
    monkeypatch.setattr(EngineCore, "add_request", lambda self, req, wave=0: None)


def _proc(
    *,
    has_coordinator=True,
    current_wave=0,
    engines_running=False,
    pause_state=PauseState.UNPAUSED,
):
    obj = DPEngineCoreProc.__new__(DPEngineCoreProc)
    obj.has_coordinator = has_coordinator
    obj.current_wave = current_wave
    obj.engines_running = engines_running
    obj.scheduler = SimpleNamespace(pause_state=pause_state)
    events: list = []
    obj._events = events
    obj.output_queue = SimpleNamespace(put_nowait=events.append)
    return obj


def _set_rocm(monkeypatch, value: bool):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: value)


def test_non_rocm_same_wave_does_not_wake(monkeypatch):
    # Upstream behavior preserved: wave 0's first request must NOT wake peers
    # off-ROCm (request_wave == current_wave == 0).
    _set_rocm(monkeypatch, False)
    p = _proc(current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is False
    assert p._events == []


def test_non_rocm_new_wave_wakes(monkeypatch):
    # Upstream behavior preserved: a higher request_wave still wakes peers.
    _set_rocm(monkeypatch, False)
    p = _proc(current_wave=0)
    add_request(p, MagicMock(), request_wave=1)
    assert p.engines_running is True
    assert p.current_wave == 1
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 1


def test_rocm_same_wave_wakes(monkeypatch):
    # The fix: on ROCm, the first request of wave 0 wakes peers even though
    # request_wave == current_wave == 0.
    _set_rocm(monkeypatch, True)
    p = _proc(current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is True
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 0


def test_no_coordinator_never_wakes(monkeypatch):
    # Without a coordinator there is nothing to wake.
    _set_rocm(monkeypatch, True)
    p = _proc(has_coordinator=False, current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is False
    assert p._events == []


def test_already_running_does_not_double_wake(monkeypatch):
    _set_rocm(monkeypatch, True)
    p = _proc(current_wave=0, engines_running=True)
    add_request(p, MagicMock(), request_wave=0)
    # Stays running, but no redundant wake message is enqueued.
    assert p.engines_running is True
    assert p._events == []


def test_paused_scheduler_does_not_wake(monkeypatch):
    # If the scheduler isn't UNPAUSED, the wake is suppressed.
    _set_rocm(monkeypatch, True)
    p = _proc(current_wave=0, pause_state=object())  # any non-UNPAUSED value
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is False
    assert p._events == []
