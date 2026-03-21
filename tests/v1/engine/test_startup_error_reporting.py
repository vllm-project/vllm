# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import msgspec
import pytest
import zmq

from vllm.v1.engine.utils import (
    STARTUP_FAILURE,
    CoreEngine,
    EngineStartupMessage,
    FailedProcessInfo,
    StartupErrorPayload,
    wait_for_engine_startup,
)


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
