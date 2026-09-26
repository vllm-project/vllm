# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import signal
from multiprocessing import connection, get_context
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


def _run_core_cleanup(conn: connection.Connection) -> None:
    config = SimpleNamespace(
        shutdown_timeout=0,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            numa_bind=False,
            reconfigure_for_independent_dp_rank=lambda: None,
        ),
    )

    def run_busy_loop() -> None:
        raise RuntimeError("injected engine failure")

    def cleanup() -> None:
        conn.send("cleanup started")
        conn.recv()
        conn.send("cleanup finished")

    proc = SimpleNamespace(
        vllm_config=config,
        shutdown_state=EngineShutdownState.RUNNING,
        run_busy_loop=run_busy_loop,
        _send_engine_dead=lambda: None,
        shutdown=cleanup,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in (
            "maybe_register_config_serialize_by_value",
            "set_process_title",
            "maybe_init_worker_tracer",
            "decorate_logs",
        ):
            monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(core_module, "EngineCoreProc", lambda **kwargs: proc)
        monkeypatch.setattr(
            core_module,
            "SignalCallback",
            lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
        )
        try:
            with pytest.raises(RuntimeError, match="injected engine failure"):
                EngineCoreProc.run_engine_core(vllm_config=config)
        finally:
            conn.close()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "signum",
    [signal.SIGTERM, signal.SIGINT, None],
    ids=["sigterm", "sigint", "manager-timeout"],
)
def test_engine_core_cleanup_survives_signals_but_respects_deadline(
    signum: int | None,
) -> None:
    """Finish cleanup on termination requests; permit the manager's forced kill."""
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_run_core_cleanup, args=(child_conn,))
    proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(60)
        assert parent_conn.recv() == "cleanup started"
        if signum is None:
            engine_utils.shutdown([proc], timeout=0.1)
            proc.join(5)
            assert proc.exitcode == -signal.SIGKILL
        else:
            assert proc.pid is not None
            os.kill(proc.pid, signum)
            parent_conn.send(None)
            assert parent_conn.poll(5)
            assert parent_conn.recv() == "cleanup finished"
            proc.join(5)
            assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
        parent_conn.close()


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
