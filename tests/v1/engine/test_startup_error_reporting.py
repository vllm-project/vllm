# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import msgspec
import pytest
import zmq

from tests.utils import create_new_process_for_each_test
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.utils import (
    STARTUP_FAILURE,
    STARTUP_HELLO,
    STARTUP_READY,
    CoreEngine,
    EngineZmqAddresses,
    EngineStartupMessage,
    FailedProcessInfo,
    StartupErrorPayload,
    wait_for_engine_startup,
)
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.worker_base import WorkerBase


class FailingStartupWorker(WorkerBase):
    def init_device(self) -> None:
        self.device = None

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        raise RuntimeError("simulated worker startup failure")


class FakeHandshakeSocket:
    def __init__(self, messages):
        self.messages = deque(messages)

    def recv_multipart(self):
        return self.messages.popleft()

    def send_multipart(self, *_args, **_kwargs):
        pass


class FakePoller:
    def __init__(self, events):
        self.events = deque(events)

    def register(self, *_args, **_kwargs):
        pass

    def poll(self, _timeout):
        return self.events.popleft() if self.events else []


def test_wait_for_engine_startup_surfaces_child_failure(monkeypatch):
    handshake_socket = FakeHandshakeSocket(
        [
            (
                (0).to_bytes(2, "little"),
                msgspec.msgpack.encode(
                    EngineStartupMessage(
                        status=STARTUP_FAILURE,
                        local=True,
                        headless=False,
                        error=StartupErrorPayload(
                            error_type="ValueError",
                            message="bad config",
                            traceback="Traceback line 1\nTraceback line 2",
                            source_process="VllmWorker-1",
                            source_rank=1,
                            pid=4321,
                        ),
                    )
                ),
            )
        ]
    )
    monkeypatch.setattr(
        zmq,
        "Poller",
        lambda: FakePoller([[(handshake_socket, zmq.POLLIN)]]),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_engine_startup(
            handshake_socket=handshake_socket,
            addresses=SimpleNamespace(frontend_stats_publish_address=None),
            core_engines=[CoreEngine(index=0, local=True)],
            parallel_config=SimpleNamespace(
                data_parallel_size_local=1,
                data_parallel_hybrid_lb=False,
                data_parallel_external_lb=False,
            ),
            coordinated_dp=False,
            cache_config=SimpleNamespace(num_gpu_blocks=0),
            proc_manager=None,
            coord_process=None,
        )

    message = str(exc_info.value)
    assert "Engine core initialization failed." in message
    assert "Failed core proc(s): VllmWorker-1(pid=4321)" in message
    assert "Source: VllmWorker-1 (rank=1, pid=4321)" in message
    assert "Root cause: ValueError: bad config" in message
    assert "Child traceback:" in message
    assert "Traceback line 1" in message


def test_wait_for_engine_startup_formats_failed_process_summary(monkeypatch):
    handshake_socket = FakeHandshakeSocket([])
    sentinel = object()
    proc_manager = SimpleNamespace(
        sentinels=lambda: [sentinel],
        finished_procs=lambda: [
            FailedProcessInfo(name="EngineCore", pid=1234, exitcode=-9)
        ],
    )
    monkeypatch.setattr(
        zmq,
        "Poller",
        lambda: FakePoller([[(sentinel, zmq.POLLIN)]]),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_engine_startup(
            handshake_socket=handshake_socket,
            addresses=SimpleNamespace(frontend_stats_publish_address=None),
            core_engines=[CoreEngine(index=0, local=True)],
            parallel_config=SimpleNamespace(
                data_parallel_size_local=1,
                data_parallel_hybrid_lb=False,
                data_parallel_external_lb=False,
            ),
            coordinated_dp=False,
            cache_config=SimpleNamespace(num_gpu_blocks=0),
            proc_manager=proc_manager,
            coord_process=None,
        )

    message = str(exc_info.value)
    assert "Engine core initialization failed." in message
    assert "Failed core proc(s): EngineCore(pid=1234, signal=9)" in message


def test_wait_for_engine_startup_dedupes_failed_process_for_error_source(
    monkeypatch,
):
    handshake_socket = FakeHandshakeSocket(
        [
            (
                (0).to_bytes(2, "little"),
                msgspec.msgpack.encode(
                    EngineStartupMessage(
                        status=STARTUP_FAILURE,
                        local=True,
                        headless=False,
                        error=StartupErrorPayload(
                            error_type="RuntimeError",
                            message="bad config",
                            traceback="Traceback line 1\nTraceback line 2",
                            source_process="EngineCore",
                            pid=1234,
                        ),
                    )
                ),
            )
        ]
    )
    proc_manager = SimpleNamespace(
        sentinels=lambda: [],
        finished_procs=lambda: [
            FailedProcessInfo(name="EngineCore", pid=1234, exitcode=-9)
        ],
    )
    monkeypatch.setattr(
        zmq,
        "Poller",
        lambda: FakePoller([[(handshake_socket, zmq.POLLIN)]]),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_engine_startup(
            handshake_socket=handshake_socket,
            addresses=SimpleNamespace(frontend_stats_publish_address=None),
            core_engines=[CoreEngine(index=0, local=True)],
            parallel_config=SimpleNamespace(
                data_parallel_size_local=1,
                data_parallel_hybrid_lb=False,
                data_parallel_external_lb=False,
            ),
            coordinated_dp=False,
            cache_config=SimpleNamespace(num_gpu_blocks=0),
            proc_manager=proc_manager,
            coord_process=None,
        )

    message = str(exc_info.value)
    assert message.count("EngineCore(") == 1
    assert "Failed core proc(s): EngineCore(pid=1234, signal=9)" in message
    assert "Source: EngineCore (pid=1234)" in message
    assert "Root cause: RuntimeError: bad config" in message


def test_wait_for_engine_startup_fails_when_ready_races_with_dead_process(
    monkeypatch,
):
    handshake_socket = FakeHandshakeSocket(
        [
            (
                (0).to_bytes(2, "little"),
                msgspec.msgpack.encode(
                    EngineStartupMessage(
                        status=STARTUP_HELLO,
                        local=True,
                        headless=False,
                    )
                ),
            ),
            (
                (0).to_bytes(2, "little"),
                msgspec.msgpack.encode(
                    EngineStartupMessage(
                        status=STARTUP_READY,
                        local=True,
                        headless=False,
                        num_gpu_blocks=1,
                    )
                ),
            ),
        ]
    )
    sentinel = object()
    proc_manager = SimpleNamespace(
        sentinels=lambda: [sentinel],
        finished_procs=lambda: [
            FailedProcessInfo(name="EngineCore", pid=1234, exitcode=1)
        ],
    )
    monkeypatch.setattr(
        zmq,
        "Poller",
        lambda: FakePoller(
            [
                [(handshake_socket, zmq.POLLIN)],
                [(handshake_socket, zmq.POLLIN), (sentinel, zmq.POLLIN)],
            ]
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_engine_startup(
            handshake_socket=handshake_socket,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            core_engines=[CoreEngine(index=0, local=True)],
            parallel_config=SimpleNamespace(
                data_parallel_size_local=1,
                data_parallel_hybrid_lb=False,
                data_parallel_external_lb=False,
            ),
            coordinated_dp=False,
            cache_config=SimpleNamespace(num_gpu_blocks=0),
            proc_manager=proc_manager,
            coord_process=None,
        )

    message = str(exc_info.value)
    assert "Engine core initialization failed." in message
    assert "Failed core proc(s): EngineCore(pid=1234, exitcode=1)" in message


@create_new_process_for_each_test("spawn")
def test_engine_core_client_surfaces_worker_startup_failure():
    """End-to-end regression: a worker startup exception must propagate
    through WorkerProc -> ready pipe -> MultiprocExecutor -> EngineCoreProc
    -> handshake -> EngineCoreClient.make_client() with the structured
    error format intact.

    Note: VLLM_WORKER_MULTIPROC_METHOD is intentionally not set here.
    The error propagation path is identical for fork and spawn workers,
    and the decorator already controls how the *test* process is spawned.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("V1 multiprocessing startup is only supported on CUDA-alike.")

    model_name = "facebook/opt-125m"
    engine_args = EngineArgs(
        model=model_name,
        distributed_executor_backend="mp",
        worker_cls="tests.v1.engine.test_startup_error_reporting."
        "FailingStartupWorker",
        enforce_eager=True,
    )
    try:
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT
        )
    except OSError:
        pytest.skip(f"{model_name} is unavailable in this environment.")

    executor_class = Executor.get_class(vllm_config)

    with pytest.raises(RuntimeError) as exc_info:
        with set_default_torch_num_threads(1):
            EngineCoreClient.make_client(
                multiprocess_mode=True,
                asyncio_mode=False,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )

    message = str(exc_info.value)
    assert "Engine core initialization failed." in message
    assert "Failed core proc(s): VllmWorker-0(pid=" in message
    assert "Source: VllmWorker-0 (rank=0, pid=" in message
    assert "Root cause: RuntimeError: simulated worker startup failure" in message
    assert "Child traceback:" in message
