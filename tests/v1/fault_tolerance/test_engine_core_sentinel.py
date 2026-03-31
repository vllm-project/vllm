# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import threading
import time
import traceback
import uuid
from queue import Queue

import pytest
import zmq
from msgspec import msgpack

from vllm.config import (
    DeviceConfig,
    FaultToleranceConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
)

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


# Helper to collect exceptions from threads and fail the test if any were raised.
def fail_on_thread_exceptions(thread_excs: Queue) -> None:
    if not thread_excs.empty():
        pytest.fail("Thread raised exception:\n" + "\n".join(thread_excs.queue))


def create_engine_core_sentinel(
    fault_signal_q: Queue,
    stop_busy_loop: threading.Event,
    addr_dict: dict,
    busy_loop_paused: threading.Event,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    vllm_cfg = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        fault_tolerance_config=FaultToleranceConfig(
            enable_fault_tolerance=True,
            worker_cmd_addr=addr_dict["worker_cmd_addr"],
        ),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        busy_loop_paused=busy_loop_paused,
        stop_busy_loop=stop_busy_loop,
        engine_input_q=Queue(),
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        vllm_config=vllm_cfg,
    )


def test_engine_core_sentinel_initialization(addr_dict):
    fault_signal_q: Queue = Queue()
    stop_busy_loop = threading.Event()
    busy_loop_paused = threading.Event()

    sentinel = create_engine_core_sentinel(
        fault_signal_q,
        stop_busy_loop,
        addr_dict,
        busy_loop_paused=busy_loop_paused,
    )

    assert sentinel.engine_index == 0
    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict):
    """
    Verify that when an engine exception is put into fault_signal_q,
    EngineCoreSentinel forwards a FaultInfo message to the
    client-facing engine fault socket.
    """
    fault_signal_q: Queue = Queue()
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q,
        threading.Event(),
        addr_dict,
        sentinel_identity=sentinel_identity,
        busy_loop_paused=threading.Event(),
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
        assert fault_info.engine_status == EngineStatusType.UNHEALTHY
    finally:
        engine_fault_receiver.close(linger=0)
        sentinel.shutdown()
        ctx.term()


@pytest.mark.parametrize("instruction", ["pause"])
def test_engine_core_sentinel_handles_fault_tolerance_instructions(
    instruction, addr_dict
):
    fault_signal_q: Queue = Queue()
    stop_busy_loop = threading.Event()
    busy_loop_paused = threading.Event()
    thread_excs: Queue = Queue()

    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q=fault_signal_q,
        stop_busy_loop=stop_busy_loop,
        addr_dict=addr_dict,
        sentinel_identity=sentinel_identity,
        busy_loop_paused=busy_loop_paused,
    )

    # Simulate the worker waiting the command and replying with a success response.
    def mock_worker_receiver(cmd_socket):
        try:
            if not cmd_socket.poll(timeout=2000):
                pytest.fail("Timeout waiting for command from engine core sentinel")
            parts = cmd_socket.recv_multipart()
            # DEALER gets [empty_frame, msg_bytes]
            _, msg_bytes = parts

            ft_req = msgpack.decode(msg_bytes, type=FaultToleranceRequest)
            assert ft_req.instruction == instruction
            ft_res = FaultToleranceResult(request_id=ft_req.request_id, success=True)
            cmd_socket.send_multipart([b"", msgpack.encode(ft_res)])
        except Exception:
            thread_excs.put(traceback.format_exc())

    # Simulate sending fault tolerance request from client sentinel
    param = {"timeout": 5}

    # Build a FaultToleranceRequest and send as msgpack
    req_id = str(uuid.uuid4())
    ft_request = FaultToleranceRequest(
        request_id=req_id, instruction=instruction, params=param
    )

    worker_ctx = zmq.Context()
    worker_cmd_socket = worker_ctx.socket(zmq.DEALER)
    worker_cmd_socket.setsockopt(zmq.IDENTITY, b"PP0_TP0")
    worker_cmd_socket.connect(addr_dict["worker_cmd_addr"])
    threading.Thread(
        target=mock_worker_receiver, args=(worker_cmd_socket,), daemon=True
    ).start()
    time.sleep(0.1)

    if instruction == "pause":
        # Simulate that pause is executed by engine core.
        busy_loop_paused.set()

    ft_res = sentinel.handle_fault(ft_request)

    assert ft_res.request_id == req_id
    assert ft_res.success

    time.sleep(0.1)
    fail_on_thread_exceptions(thread_excs)
    worker_cmd_socket.close()
    sentinel.shutdown()
    worker_ctx.term()
