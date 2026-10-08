# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Regression tests for termination signals that arrive while EngineCoreProc is
# still being constructed.
#
# When this fix matters
# ---------------------
# Engine shutdown is driven from the frontend. If a frontend process dies while
# EngineCore is still starting up, the process manager sends SIGTERM to its
# children and force kills whatever is still alive after a short timeout (see
# shutdown in vllm/v1/utils.py). Construction is slow - it spawns the executor
# processes, loads weights, profiles and captures CUDA graphs - so that
# shutdown often lands inside the construction window. Bigger models and more
# ranks widen the window. This is the case PR 52299 (follow up to 43417) has to
# survive: 43417 stops EngineCore from exiting silently, but the request itself
# still has to outlive the construction window.
#
# run_engine_core installs the signal handler before engine_core exists, so in
# that window the handler runs with engine_core == None. Losing the request
# there is permanent, not a delay: shutdown_state stays RUNNING, and
# wakeup_engine returns early while consuming the one-shot SignalCallback. The
# busy loop then starts RUNNING and blocks on input_queue forever, so no later
# signal can wake it. Graceful shutdown is gone for good and the process only
# stops when something force kills it, which skips executor shutdown and GPU
# release. In a log this shows up as the signal being received, followed by a
# force kill of EngineCore after the process manager timeout.
#
# What these tests pin
# --------------------
# A request arriving in the window must be remembered somewhere other than
# engine_core, applied before run_busy_loop starts, and must not consume the
# one-shot wakeup callback, so a later signal can still wake an idle loop.

import queue
import signal as signal_module
from typing import Any

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

WAKEUP_ITEM = (EngineCoreRequestType.WAKEUP, None)
CONSTRUCTION_SIGNAL = signal_module.SIGTERM


class _FakeSignal:
    # Minimal stand-in for the signal module. It records the installed handler
    # so the test can fire it by hand, and avoids touching the real process
    # signal disposition. Signals is reused from the real module so the handler
    # can still resolve signal.Signals(signum).name.
    SIGTERM = signal_module.SIGTERM
    SIGINT = signal_module.SIGINT
    SIG_DFL = signal_module.SIG_DFL
    Signals = signal_module.Signals

    def __init__(self):
        self.handlers = {}

    def signal(self, signum, handler):
        self.handlers[signum] = handler

    def fire(self):
        self.handlers[CONSTRUCTION_SIGNAL](CONSTRUCTION_SIGNAL, None)


class _FakeSignalCallback:
    # One-shot stand-in for SignalCallback. The real class runs the callback on
    # a dedicated thread and exits after a single trigger, so every later
    # trigger is a no-op. Running the callback inline keeps the test
    # deterministic instead of depending on thread scheduling.
    instances: list["_FakeSignalCallback"] = []

    def __init__(self, callback):
        self.callback = callback
        self.trigger_count = 0
        self.consumed = False
        self.stopped = False
        _FakeSignalCallback.instances.append(self)

    def trigger(self):
        self.trigger_count += 1
        if self.stopped or self.consumed:
            return
        self.consumed = True
        callback, self.callback = self.callback, None
        callback()

    def stop(self):
        self.stopped = True


class _StubParallelConfig:
    def __init__(self):
        self.data_parallel_size = 1
        self.data_parallel_rank_local = 0
        self.data_parallel_index = 0
        self.data_parallel_rank = 0
        self.numa_bind = False

    def reconfigure_for_independent_dp_rank(self):
        pass


class _StubModelConfig:
    is_moe = False


class _StubVllmConfig:
    def __init__(self):
        self.parallel_config = _StubParallelConfig()
        self.model_config = _StubModelConfig()
        self.logging_config = None
        self.kv_transfer_config = None
        self.shutdown_timeout = 0


def _run_engine_core(
    monkeypatch,
    *,
    signal_during_construction=False,
    signal_while_idle=False,
):
    _FakeSignalCallback.instances.clear()
    fake_signal = _FakeSignal()
    trace = {}

    class StubEngineCoreProc:
        def __init__(self, *args, **kwargs):
            self.shutdown_state = EngineShutdownState.RUNNING
            # Same queue type as EngineCore, so the wakeup path under test is
            # the real one.
            self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
            self.vllm_config = kwargs.get("vllm_config")
            if signal_during_construction:
                # The termination signal lands mid construction, while the
                # caller still sees engine_core == None.
                fake_signal.fire()
            trace["constructed"] = True

        def run_busy_loop(self):
            # What the real loop sees on its first _handle_shutdown() check.
            trace["loop_entered"] = True
            trace["state_at_loop_entry"] = self.shutdown_state
            if signal_while_idle:
                fake_signal.fire()
            try:
                trace["wakeup"] = self.input_queue.get_nowait()
            except queue.Empty:
                trace["wakeup"] = None

        def shutdown(self):
            pass

        def _send_engine_dead(self):
            pass

        def has_work(self):
            return False

    monkeypatch.setattr(core_module, "signal", fake_signal)
    monkeypatch.setattr(core_module, "SignalCallback", _FakeSignalCallback)
    monkeypatch.setattr(core_module, "EngineCoreProc", StubEngineCoreProc)
    monkeypatch.setattr(core_module, "DPEngineCoreProc", StubEngineCoreProc)
    monkeypatch.setattr(core_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        core_module, "maybe_register_config_serialize_by_value", lambda *a, **k: None
    )
    monkeypatch.setattr(core_module, "set_process_title", lambda *a, **k: None)
    monkeypatch.setattr(core_module, "maybe_init_worker_tracer", lambda *a, **k: None)
    monkeypatch.setattr(core_module, "decorate_logs", lambda *a, **k: None)

    EngineCoreProc.run_engine_core(vllm_config=_StubVllmConfig())
    return trace


def test_signal_during_construction_is_applied_before_loop(monkeypatch):
    trace = _run_engine_core(monkeypatch, signal_during_construction=True)

    assert trace["constructed"] is True
    assert trace["loop_entered"] is True
    # The request arrived while engine_core was None, so it could not be stored
    # there. It must still be applied before run_busy_loop starts, otherwise the
    # loop comes up as RUNNING, the shutdown is silently dropped, and force kill
    # becomes the only way out.
    assert trace["state_at_loop_entry"] == EngineShutdownState.REQUESTED, (
        "a signal received during construction must be applied before the loop "
        f"starts, got {trace['state_at_loop_entry']!r}"
    )


def test_signal_during_construction_keeps_later_wakeup_working(monkeypatch):
    trace = _run_engine_core(
        monkeypatch,
        signal_during_construction=True,
        signal_while_idle=True,
    )

    assert trace["state_at_loop_entry"] == EngineShutdownState.REQUESTED, (
        "the construction time signal must be applied before the loop starts, "
        f"got {trace['state_at_loop_entry']!r}"
    )
    # SignalCallback is one-shot. If the construction time signal consumed it,
    # no wakeup could be delivered afterwards and the idle busy loop would block
    # on input_queue forever instead of shutting down, so the process would sit
    # there holding GPU memory until something force kills it.
    assert trace["wakeup"] == WAKEUP_ITEM, (
        "a signal after construction must still wake the idle loop, "
        f"got {trace['wakeup']!r}"
    )


def test_signal_while_idle_still_wakes_the_loop(monkeypatch):
    trace = _run_engine_core(monkeypatch, signal_while_idle=True)

    assert trace["state_at_loop_entry"] == EngineShutdownState.RUNNING
    assert trace["wakeup"] == WAKEUP_ITEM


def test_no_signal_leaves_the_loop_running(monkeypatch):
    trace = _run_engine_core(monkeypatch)

    assert trace["state_at_loop_entry"] == EngineShutdownState.RUNNING
    assert trace["wakeup"] is None
