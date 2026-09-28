# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from multiprocessing import connection
from threading import Event
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
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


@pytest.mark.parametrize(
    ("executor_class", "local_engine_count", "user_threads", "cpu_limits", "expected"),
    [
        (UniProcExecutor, 1, None, (192, 192, 12), (12, "12", "1")),
        (UniProcExecutor, 2, None, (192, 192, 12), (6, "6", "1")),
        (UniProcExecutor, 2, "5", (5, 192, 12), (5, "5", None)),
        (
            multiproc_executor.MultiprocExecutor,
            2,
            None,
            (192, 192, 12),
            (192, None, None),
        ),
        (UniProcExecutor, 1, None, (112, 224, None), (112, "112", "1")),
        (UniProcExecutor, 2, None, (112, 224, None), (56, "56", "1")),
        (UniProcExecutor, 2, None, (1, 224, None), (1, "1", "1")),
    ],
)
@pytest.mark.parametrize("fail_context", [False, True])
def test_engine_core_startup_threads_respect_quota_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    executor_class,
    local_engine_count: int,
    user_threads: str | None,
    cpu_limits: tuple[int, int, int | None],
    expected: tuple[int, str | None, str | None],
    fail_context: bool,
):
    """UniProc workers inherit their CPU share before context creation and start."""
    marker = torch_utils.OMP_NUM_THREADS_SET_BY_VLLM
    monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    if user_threads is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", user_threads)
    initial_threads, visible_cpus, quota = cpu_limits
    monkeypatch.setattr(torch_utils, "_cgroup_cpu_limit", lambda: quota)
    monkeypatch.setattr(
        torch_utils.os, "sched_getaffinity", lambda pid: set(range(visible_cpus))
    )
    monkeypatch.setattr(multiproc_executor, "_maybe_force_spawn", lambda: None)
    platform = SimpleNamespace(
        is_cpu=lambda: False, is_cuda_alike=lambda: True, is_xpu=lambda: False
    )
    monkeypatch.setattr(engine_utils, "current_platform", platform)
    monkeypatch.setattr(multiproc_executor, "current_platform", platform)
    threads = SimpleNamespace(count=initial_threads)
    monkeypatch.setattr(
        multiproc_executor.torch, "get_num_threads", lambda: threads.count
    )
    monkeypatch.setattr(
        multiproc_executor.torch,
        "set_num_threads",
        lambda count: setattr(threads, "count", count),
    )
    observed = []

    def record_startup_state():
        observed.append(
            (threads.count, os.environ.get("OMP_NUM_THREADS"), os.environ.get(marker))
        )

    def get_context():
        record_startup_state()
        if fail_context:
            raise RuntimeError("context unavailable")
        return SimpleNamespace(
            Process=lambda **kwargs: SimpleNamespace(
                exitcode=None, start=record_startup_state
            )
        )

    monkeypatch.setattr(engine_utils, "get_mp_context", get_context)
    monkeypatch.setattr(
        engine_utils.numa_utils,
        "configure_subprocess",
        lambda *args, **kwargs: nullcontext(),
    )
    with (
        pytest.raises(RuntimeError, match="context unavailable")
        if fail_context
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
    assert observed == [expected] * (1 if fail_context else local_engine_count + 1)
    assert threads.count == initial_threads
    assert os.environ.get("OMP_NUM_THREADS") == user_threads
    assert marker not in os.environ


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
