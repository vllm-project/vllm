# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
import time

import pytest
import zmq
from msgspec import msgpack

from vllm.config import ParallelConfig
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
    parallel_config: ParallelConfig,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    return EngineCoreSentinel(
        parallel_config,
        engine_index=0,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
    )


def test_engine_core_sentinel_initialization(addr_dict, mock_parallel_config):
    sentinel = create_engine_core_sentinel(mock_parallel_config, addr_dict)

    assert sentinel.engine_index == 0
    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict, mock_parallel_config):
    """
    Verify that when an engine exception is put into fault_signal_q,
    EngineCoreSentinel forwards a FaultInfo message to the
    client-facing engine fault socket.
    """
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        mock_parallel_config, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to the engine_fault_socket_addr to receive the fault report.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    try:
        time.sleep(0.1)
        sentinel.fault_signal_q.put(RuntimeError("test exception"))
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
