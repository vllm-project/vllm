# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import time

import pytest
import zmq
from msgspec import msgpack

from vllm.config import (
    DeviceConfig,
    FaultToleranceConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.fault_tolerance import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo

pytestmark = pytest.mark.skip_global_cleanup


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    ports = [_find_free_port() for _ in range(3)]
    return {
        "client_cmd_addr": f"tcp://127.0.0.1:{ports[0]}",
        "worker_cmd_addr": f"tcp://127.0.0.1:{ports[1]}",
        "engine_fault_socket_addr": f"tcp://127.0.0.1:{ports[2]}",
    }


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    vllm_cfg = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        vllm_config=vllm_cfg,
    )


def test_engine_core_sentinel_initialization(addr_dict):
    fault_signal_q: queue.Queue = queue.Queue()

    sentinel = create_engine_core_sentinel(fault_signal_q, addr_dict)

    assert sentinel.engine_index == 0
    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict):
    """
    Verify that when an engine exception is put into fault_signal_q,
    EngineCoreSentinel forwards a FaultInfo message to the
    client-facing engine fault socket.
    """
    fault_signal_q: queue.Queue = queue.Queue()
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to the engine_fault_socket_addr to receive the fault report.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    try:
        time.sleep(0.1)
        fault_signal_q.put(RuntimeError("test exception"))

        # Wait for the sentinel to forward the fault to the engine_fault socket.
        if not engine_fault_receiver.poll(timeout=5000):
            pytest.fail("Timeout waiting for engine fault message from sentinel")

        parts = engine_fault_receiver.recv_multipart()
        assert len(parts) >= 2
        fault_info = msgpack.decode(parts[-1], type=FaultInfo)
        assert fault_info.type == "RuntimeError"
        assert fault_info.engine_id == "0"
        assert fault_info.message == "test exception"
    finally:
        engine_fault_receiver.close(linger=0)
        sentinel.shutdown()
        ctx.term()
