# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

import vllm.v1.engine.coordinator as coordinator
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats


class SimulationDone(Exception):
    pass


def run_busy_coordinator(monkeypatch, packets, *, lockstep, quiet_after=False):
    """Drive the production loop through 400 ms of continuously readable input."""
    state = SimpleNamespace(now=0.0, polls=0, received=0)
    snapshots = []

    class Socket:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self):
            if self.name == "back_publish":
                return b"\x01"
            assert self.name == "back_output"
            packet = packets[min(state.received, len(packets) - 1)]
            state.received += 1
            engine, step, running, waiting = packet
            return msgspec.msgpack.encode(
                EngineCoreOutputs(
                    engine_index=engine,
                    scheduler_stats=SchedulerStats(
                        num_running_reqs=running,
                        num_waiting_reqs=waiting,
                        step_counter=step,
                        current_wave=0,
                    ),
                )
            )

        def send(self, payload):
            if self.name == "back_publish":
                assert payload == b"READY"
                return
            assert self.name == "front"
            snapshots.append((state.now, msgspec.msgpack.decode(payload)))

    sockets = {name: Socket(name) for name in ("front", "back_output", "back_publish")}

    class BusyPoller:
        def register(self, *args):
            pass

        def poll(self, timeout):
            state.polls += 1
            if state.polls <= 40:
                state.now += 0.010
                return [(sockets["back_output"], coordinator.zmq.POLLIN)]
            if quiet_after and state.polls == 41:
                state.now += max(timeout, 0) / 1000
                return []
            raise SimulationDone

    monkeypatch.setattr(coordinator, "time", SimpleNamespace(time=lambda: state.now))
    monkeypatch.setattr(coordinator.zmq, "Poller", BusyPoller)
    monkeypatch.setattr(coordinator, "make_zmq_socket", lambda path, **_: sockets[path])
    monkeypatch.setattr(coordinator, "logger", Mock())
    proc = object.__new__(coordinator.DPCoordinatorProc)
    proc.ctx = None
    proc.engines = [coordinator.EngineState(), coordinator.EngineState()]
    proc.stats_update_interval_ms = 100
    proc.enable_wave_coordination = lockstep
    with pytest.raises(SimulationDone):
        proc.process_input_socket("front", "back_output", "back_publish")
    return snapshots


def test_dense_publishes_while_input_remains_readable(monkeypatch):
    snapshots = run_busy_coordinator(
        monkeypatch, [(0, 0, 5, 0), (1, 0, 7, 0)], lockstep=False
    )
    assert len(snapshots) >= 3, "Busy input must not suppress periodic stats"
    previous_time = 0.0
    for publish_time, payload in snapshots:
        assert 0.100 <= publish_time - previous_time <= 0.120
        assert payload[0] == [[0, 5, 0.0], [0, 7, 0.0]]
        previous_time = publish_time


def test_moe_publishes_previous_step_while_input_remains_readable(monkeypatch):
    snapshots = run_busy_coordinator(
        monkeypatch,
        [(0, 1, 1, 0), (1, 1, 2, 0), (0, 2, 10, 0)],
        lockstep=True,
    )
    assert snapshots, "An available previous-step snapshot must not starve"
    assert snapshots[0][0] <= 0.120
    # Rank 1 has not reported step 2: retain the previous-step snapshot.
    assert snapshots[0][1][0] == [[0, 1, 0.0], [0, 2, 0.0]]


def test_moe_preserves_quiet_poll_before_fallback_snapshot(monkeypatch):
    snapshots = run_busy_coordinator(
        monkeypatch, [(0, 1, 10, 0)], lockstep=True, quiet_after=True
    )
    # With no previous-step snapshot, keep the existing 50 ms quiet-poll fallback.
    assert len(snapshots) == 1
    assert snapshots[0][0] >= 0.450
    assert snapshots[0][1][0] == [[0, 10, 0.0], [0, 0, 0.0]]
