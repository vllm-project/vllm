# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from multiprocessing import connection, get_context
from threading import Event
from types import SimpleNamespace

import pytest
import torch
import zmq

import vllm.platforms as platforms
from vllm.config import ParallelConfig, WatchdogConfig
from vllm.utils import torch_utils
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
from vllm.v1.executor import UniProcExecutor, multiproc_executor

pytestmark = pytest.mark.skip_global_cleanup


def _exit_when_released(release, exit_code):
    assert release.wait(timeout=15)
    raise SystemExit(exit_code)


@pytest.mark.parametrize("planned,exit_code", [(True, 0), (False, 0), (True, 3)])
def test_engine_group_monitor_preserves_retiring_siblings(planned, exit_code):
    """Planned clean exits wait for siblings; unexpected exits remain failures."""
    ctx = get_context("spawn")
    releases = [ctx.Event(), ctx.Event()]
    processes = [
        ctx.Process(
            target=_exit_when_released,
            args=(release, exit_code if index == 0 else 0),
            name=f"engine-{index}",
        )
        for index, release in enumerate(releases)
    ]
    stopped, expected_shutdown = Event(), Event()
    if planned:
        expected_shutdown.set()
    try:
        for proc in processes:
            proc.start()
        with ThreadPoolExecutor(max_workers=1) as executor:
            monitor = executor.submit(
                engine_utils._wait_for_engine_processes,
                processes,
                stopped,
                expected_shutdown,
            )
            releases[0].set()
            assert connection.wait([processes[0].sentinel], timeout=10)
            assert not connection.wait([processes[1].sentinel], timeout=0)
            if planned and exit_code == 0:
                assert not monitor.done()
                releases[1].set()
                result = monitor.result(timeout=10)
                assert result is None, (result, [proc.exitcode for proc in processes])
            else:
                assert monitor.result(timeout=10) == processes[0].name
            assert processes[0].exitcode == exit_code
    finally:
        stopped.set()
        for release in releases:
            release.set()
        for proc in processes:
            if proc.pid is not None:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                proc.close()


@pytest.mark.parametrize(
    (
        "executor_class",
        "local_engine_count",
        "initial_threads",
        "user_threads",
        "fail_start",
        "expected",
    ),
    [
        (UniProcExecutor, 1, 192, None, False, (12, "12", "1")),
        (UniProcExecutor, 2, 192, None, False, (6, "6", "1")),
        (UniProcExecutor, 2, 192, None, True, (6, "6", "1")),
        (UniProcExecutor, 2, 5, None, False, (5, "5", "1")),
        (UniProcExecutor, 2, 1, None, False, (1, "1", "1")),
        (UniProcExecutor, 2, 5, "5", False, (5, "5", None)),
        (
            multiproc_executor.MultiprocExecutor,
            2,
            192,
            None,
            False,
            (192, None, None),
        ),
    ],
)
def test_engine_core_startup_threads_are_scoped_to_launch(
    monkeypatch: pytest.MonkeyPatch,
    executor_class,
    local_engine_count: int,
    initial_threads: int,
    user_threads: str | None,
    fail_start: bool,
    expected: tuple[int, str | None, str | None],
):
    """UniProc workers inherit their CPU share and restore the parent."""
    marker = torch_utils.OMP_NUM_THREADS_SET_BY_VLLM
    monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    if user_threads is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", user_threads)
    monkeypatch.setattr(torch_utils, "available_cpu_count", lambda: 12)
    platform = SimpleNamespace(
        is_cpu=lambda: False, is_cuda_alike=lambda: True, is_xpu=lambda: False
    )
    monkeypatch.setattr(engine_utils, "current_platform", platform)
    threads = SimpleNamespace(count=initial_threads)
    setter_calls = []

    def set_threads(count):
        setter_calls.append(count)
        threads.count = count

    monkeypatch.setattr(torch, "get_num_threads", lambda: threads.count)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    start_states = []

    def thread_state():
        return (
            threads.count,
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get(marker),
        )

    def start_process():
        start_states.append(thread_state())
        if fail_start:
            raise RuntimeError("start failed")

    def get_context():
        assert thread_state() == (initial_threads, user_threads, None)
        return SimpleNamespace(
            Process=lambda **kwargs: SimpleNamespace(
                name=kwargs["name"],
                exitcode=1 if fail_start else None,
                start=start_process,
            )
        )

    monkeypatch.setattr(engine_utils, "get_mp_context", get_context)
    monkeypatch.setattr(engine_utils, "shutdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_utils.numa_utils,
        "configure_subprocess",
        lambda *args, **kwargs: nullcontext(),
    )
    with (
        pytest.raises(RuntimeError, match="start failed")
        if fail_start
        else nullcontext()
    ):
        manager = CoreEngineProcManager(
            local_engine_count=local_engine_count,
            start_index=0,
            local_start_index=0,
            vllm_config=SimpleNamespace(
                shutdown_timeout=0,
                parallel_config=SimpleNamespace(
                    data_parallel_size=local_engine_count,
                    assigned_physical_gpu_ids=None,
                    use_ray=False,
                ),
            ),
            local_client=True,
            handshake_address="unused",
            executor_class=executor_class,
            log_stats=False,
        )
        manager._finalizer.detach()
    assert start_states == [expected] * (1 if fail_start else local_engine_count)
    assert thread_state() == (initial_threads, user_threads, None)
    assert setter_calls == (
        [expected[0], initial_threads]
        if executor_class is UniProcExecutor and expected[0] < initial_threads
        else []
    )


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
    vllm_config = SimpleNamespace(
        shutdown_timeout=shutdown_timeout,
        # run_engine_core starts the watchdog from this field. The default
        # config has no dump_dir, so start_watch_dog returns immediately.
        watchdog_config=WatchdogConfig(),
    )
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
        # Mirrors EngineCore._send_engine_dead, which run_engine_core calls
        # from its fatal-error handler.
        _send_engine_dead=lambda: None,
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
    """Report a watched frontend's exit while engines are still starting."""
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = ParallelConfig()

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
                parallel_config,
                coordinated_dp=False,
                cache_config=None,
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
