# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DPEngineCoreProc.add_request first-wave wake behavior.

In a multi-pod Wide-EP DP deployment, the first request of wave 0 arrives when
``current_wave == 0``. If the engine only wakes peers when ``request_wave !=
current_wave`` (``0 != 0 == False``), the other DP engines never start stepping
and the EP all2all collective hangs. The engine therefore wakes peers whenever
they are idle and the scheduler is unpaused -- including the current wave's
first request -- for every LB mode with a coordinator. This is platform-
agnostic (the DP coordinator's front-end ``FIRST_REQ`` wake was already present
when the cold-start hang reproduced 100% on ROCm DP+EP, so the engine
re-broadcasts as well).

We bind ``DPEngineCoreProc.add_request`` to a bare instance and stub the heavy
``super().add_request`` (``EngineCore.add_request``) so only the wake gate runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    external_lb=False,
    current_wave=0,
    engines_running=False,
    pause_state=PauseState.UNPAUSED,
):
    obj = DPEngineCoreProc.__new__(DPEngineCoreProc)
    obj.has_coordinator = has_coordinator
    obj.external_lb = external_lb
    obj.engine_index = 0
    obj.current_wave = current_wave
    obj.engines_running = engines_running
    obj.scheduler = SimpleNamespace(pause_state=pause_state)
    events: list = []
    obj._events = events
    obj.output_queue = SimpleNamespace(put_nowait=events.append)
    return obj


def test_internal_lb_same_wave_wakes():
    # The core fix: the first request of wave 0 wakes peers even though
    # request_wave == current_wave == 0, in internal/hybrid LB (external_lb
    # False). Platform-agnostic -- no ROCm gate.
    p = _proc(external_lb=False, current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is True
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 0


def test_external_lb_same_wave_wakes():
    # External-LB: the coordinator is not in the per-request path, so the
    # engine-side wake is the only wake for the current wave's first request.
    p = _proc(external_lb=True, current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is True
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 0


def test_new_wave_wakes_and_advances():
    # A higher request_wave advances current_wave and wakes peers.
    p = _proc(current_wave=0)
    add_request(p, MagicMock(), request_wave=1)
    assert p.engines_running is True
    assert p.current_wave == 1
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 1


def test_stale_wave_wakes_for_current_wave():
    # A request for an already-completed wave (request_wave < current_wave)
    # does not roll current_wave back, but still triggers a wake so the
    # front-end starts the current wave.
    p = _proc(current_wave=2)
    add_request(p, MagicMock(), request_wave=1)
    assert p.engines_running is True
    assert p.current_wave == 2
    assert len(p._events) == 1
    _, outputs = p._events[0]
    assert outputs.start_wave == 2


def test_no_coordinator_never_wakes():
    # Without a coordinator there is nothing to wake.
    p = _proc(has_coordinator=False, current_wave=0)
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is False
    assert p._events == []


def test_already_running_does_not_double_wake():
    p = _proc(current_wave=0, engines_running=True)
    add_request(p, MagicMock(), request_wave=0)
    # Stays running, but no redundant wake message is enqueued.
    assert p.engines_running is True
    assert p._events == []


def test_paused_scheduler_does_not_wake():
    # If the scheduler isn't UNPAUSED, the wake is suppressed.
    p = _proc(current_wave=0, pause_state=object())  # any non-UNPAUSED value
    add_request(p, MagicMock(), request_wave=0)
    assert p.engines_running is False
    assert p._events == []
