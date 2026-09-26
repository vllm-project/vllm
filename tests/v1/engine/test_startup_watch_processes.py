# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing import connection
from threading import Event
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("is_rocm", "request_timeout", "manager_timeout", "process_timeout"),
    [
        (True, 0, 0, 15.0),
        (True, 0, 7, 7),
        (True, 0, None, None),
        (False, 0, 0, 0),
        (True, 7, 0, 0),
    ],
)
def test_engine_core_process_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    request_timeout: float | None,
    manager_timeout: float | None,
    process_timeout: float | None,
):
    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = request_timeout
    manager.manager_stopped = Event()
    manager.processes = [object()]
    detach_results = iter((object(), None))
    manager._finalizer = SimpleNamespace(detach=lambda: next(detach_results))

    shutdown_calls = []
    monkeypatch.setattr(
        engine_utils,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    monkeypatch.setattr(
        engine_utils,
        "shutdown",
        lambda processes, timeout: shutdown_calls.append((processes, timeout)),
    )

    manager.shutdown(timeout=manager_timeout)
    manager.shutdown(timeout=manager_timeout)

    assert manager.manager_stopped.is_set()
    assert shutdown_calls == [(manager.processes, process_timeout)]


@pytest.mark.parametrize(
    (
        "is_rocm",
        "shutdown_state",
        "has_work",
        "shutdown_timeout",
        "exit_code",
        "expected_calls",
    ),
    [
        (
            True,
            EngineShutdownState.SHUTTING_DOWN,
            False,
            0,
            None,
            ["shutdown", "freeze"],
        ),
        (False, EngineShutdownState.SHUTTING_DOWN, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.RUNNING, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, True, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 7, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 0, 1, ["shutdown"]),
    ],
)
def test_freeze_gc_after_clean_rocm_engine_core_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    shutdown_state: EngineShutdownState,
    has_work: bool,
    shutdown_timeout: int,
    exit_code: int | None,
    expected_calls: list[str],
):
    calls: list[str] = []
    vllm_config = SimpleNamespace(shutdown_timeout=shutdown_timeout)
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
    )

    def run_busy_loop():
        proc.shutdown_state = shutdown_state
        raise SystemExit(exit_code)

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = lambda: calls.append("shutdown")
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config.parallel_config = parallel_config

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert calls == expected_calls


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


class _SpuriousThenDeadProcess:
    """A process whose sentinel is already poll-ready (its pipe's write end
    is closed), but whose exitcode only becomes real after `spurious_reads`
    readiness events -- simulating a stale/invalid descriptor that reports
    readiness before the process has actually exited."""

    name = "RustFrontend"

    def __init__(self, sentinel, spurious_reads: int):
        self.sentinel = sentinel
        self.remaining_spurious = spurious_reads

    @property
    def exitcode(self):
        if self.remaining_spurious > 0:
            self.remaining_spurious -= 1
            return None
        return 1


def test_wait_for_engine_startup_ignores_spurious_sentinel_readiness():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    proc = _SpuriousThenDeadProcess(recv, spurious_reads=2)

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [proc]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    # The first two readiness events were spurious (exitcode still None)
    # and must not raise; only the third, real exit is reported.
    assert proc.remaining_spurious == 0
    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)


def test_monitor_engine_liveness_ignores_spurious_sentinel_readiness(
    monkeypatch: pytest.MonkeyPatch,
):
    recv, send = connection.Pipe(duplex=False)
    send.close()
    proc = _SpuriousThenDeadProcess(recv, spurious_reads=2)
    proc.name = "EngineCore"

    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [proc]
    manager.manager_stopped = Event()
    manager.failed_proc_name = None

    shutdown_calls = []
    monkeypatch.setattr(manager, "shutdown", lambda: shutdown_calls.append(1))

    try:
        manager.monitor_engine_liveness()
    finally:
        recv.close()

    # A sentinel that reports readiness twice with exitcode still None must
    # not be mistaken for a dead process; only the real, non-zero exit on
    # the third readiness event should be recorded and trigger shutdown.
    assert proc.remaining_spurious == 0
    assert manager.failed_proc_name == "EngineCore"
    assert shutdown_calls == [1]


def test_monitor_engine_liveness_polls_a_persistently_invalid_sentinel(
    monkeypatch: pytest.MonkeyPatch,
):
    """A sentinel that never resolves to a real exit code -- a persistently
    stale or invalid descriptor rather than a briefly spurious one -- must
    stop being handed to connection.wait once recognized. Left in the wait
    set, it would report ready on every single call and spin this loop at
    full speed instead of blocking. A second, healthy process's real exit
    must still be caught promptly while the first sits in fallback
    polling."""
    stuck = SimpleNamespace(sentinel=1, exitcode=None, name="stuck")
    healthy = SimpleNamespace(sentinel=2, exitcode=None, name="healthy")

    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [stuck, healthy]
    manager.manager_stopped = Event()
    manager.failed_proc_name = None

    shutdown_calls = []
    monkeypatch.setattr(manager, "shutdown", lambda: shutdown_calls.append(1))

    wait_calls = []

    def fake_wait(object_list, timeout=None):
        wait_calls.append((set(object_list), timeout))
        if len(wait_calls) == 1:
            return [1]  # only the stuck sentinel is ready at first
        healthy.exitcode = 1  # the healthy process actually exits now
        return [2]

    monkeypatch.setattr(engine_utils.connection, "wait", fake_wait)
    monkeypatch.setattr(engine_utils.time, "sleep", lambda _: None)

    manager.monitor_engine_liveness()

    # The stuck sentinel was handed to connection.wait exactly once, then
    # never again -- moved to bounded polling instead of staying in the
    # wait set and reporting ready every call.
    assert wait_calls[0] == ({1, 2}, 1)
    assert all(1 not in sentinels for sentinels, _ in wait_calls[1:])
    # Once something needed polling, the remaining wait used the shorter
    # cadence instead of blocking for a full second.
    assert all(timeout == 0.1 for _, timeout in wait_calls[1:])
    assert manager.failed_proc_name == "healthy"
    assert shutdown_calls == [1]


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)
