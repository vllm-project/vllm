# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import threading
import time
from queue import Queue

import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine.core import EngineCoreSentinel


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    return {
        "engine_fault_socket_addr": f"tcp://127.0.0.1:{_find_free_port()}",
    }


# Helper to collect exceptions from threads and fail the test if any were raised.
def fail_on_thread_exceptions(thread_excs: Queue) -> None:
    if not thread_excs.empty():
        pytest.fail("Thread raised exception:\n" + "\n".join(thread_excs.queue))


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue,
    busy_loop_active: threading.Event,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        busy_loop_active=busy_loop_active,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        vllm_config=vllm_cfg,
    )


def test_engine_core_sentinel_initialization(addr_dict):
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    sentinel = create_engine_core_sentinel(fault_signal_q, busy_loop_active, addr_dict)

    assert sentinel.engine_index == 0
    assert sentinel.tp_size == 1
    assert sentinel.pp_size == 1
    assert sentinel.engine_running is True

    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict):
    """
    Verify that when a busy loop raises an exception, the
    busy_loop_wrapper puts the exception into fault_signal_q and
    EngineCoreSentinel detects it and forwards a FaultInfo message
    to the client-facing engine fault socket.
    """
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q, busy_loop_active, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to the engine_fault_socket_addr to receive the fault report.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    time.sleep(0.1)
    busy_loop_active.clear()
    fault_signal_q.put(RuntimeError("test exception"))

    # Wait for the sentinel to forward the fault to the engine_fault socket.
    if not engine_fault_receiver.poll(timeout=5000):
        pytest.fail("Timeout waiting for engine fault message from sentinel")

    parts = engine_fault_receiver.recv_multipart()
    assert len(parts) >= 2
    assert "test exception" in parts[-1].decode("utf-8")

    engine_fault_receiver.close()
    sentinel.shutdown()
    ctx.term()
