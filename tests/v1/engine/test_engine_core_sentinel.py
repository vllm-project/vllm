# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import threading
import time
import traceback
import uuid
from queue import Queue

import pytest
import zmq
from msgspec import msgpack

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.engine.core import EngineCoreSentinel, EngineLoopPausedError


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
    fault_signal_q: queue.Queue,
    busy_loop_active: threading.Event,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
    cmd_q: queue.Queue | None = None,
):
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )
    if cmd_q is None:
        cmd_q = queue.Queue()

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        cmd_q=cmd_q,
        busy_loop_active=busy_loop_active,
        engine_input_q=queue.Queue(),
        client_cmd_addr=addr_dict["client_cmd_addr"],
        worker_cmd_addr=addr_dict["worker_cmd_addr"],
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
    assert not sentinel.communicator_aborted
    assert sentinel.engine_running is True

    assert sentinel.engine_fault_socket.type == zmq.DEALER
    assert sentinel.upstream_cmd_socket.type == zmq.DEALER
    assert sentinel.downstream_cmd_socket.type == zmq.ROUTER

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


@pytest.mark.parametrize("instruction", ["pause", "retry"])
def test_engine_core_sentinel_handles_fault_tolerance_instructions(
    instruction, addr_dict
):
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()
    thread_excs: Queue = Queue()
    cmd_q: Queue = Queue()

    ctx = zmq.Context()
    client_cmd_socket = ctx.socket(zmq.ROUTER)
    client_cmd_socket.bind(addr_dict["client_cmd_addr"])

    time.sleep(0.1)

    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q,
        busy_loop_active,
        addr_dict,
        sentinel_identity=sentinel_identity,
        cmd_q=cmd_q,
    )

    if instruction == "retry":
        # Simulate that an exception is raised in the busy loop,
        # so that engine is in pause state.
        busy_loop_active.clear()
        fault_signal_q.put(RuntimeError("Pretest exception to trigger pause"))
        time.sleep(0.1)
        assert not busy_loop_active.is_set()

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

    worker_ctx = zmq.Context()
    worker_cmd_socket = worker_ctx.socket(zmq.DEALER)
    worker_cmd_socket.setsockopt(zmq.IDENTITY, b"PP0_TP0")
    worker_cmd_socket.connect(addr_dict["worker_cmd_addr"])
    threading.Thread(
        target=mock_worker_receiver, args=(worker_cmd_socket,), daemon=True
    ).start()
    time.sleep(0.1)

    # Simulate sending fault tolerance request from client sentinel
    param = {"timeout": 3}
    if instruction == "pause":
        param["soft_pause"] = False
    elif instruction == "retry":
        param["new_stateless_dp_group_port"] = 23456
    # Build a FaultToleranceRequest and send as msgpack
    req_id = str(uuid.uuid4())
    ft_request = FaultToleranceRequest(
        request_id=req_id, instruction=instruction, params=param
    )
    client_cmd_socket.send_multipart(
        [sentinel_identity, b"", msgpack.encode(ft_request)]
    )

    if instruction == "pause":
        # Simulate that pause is executed by engine core
        busy_loop_active.clear()
        fault_signal_q.put(EngineLoopPausedError("Simulated pause for testing"))
    elif instruction == "retry":
        cmd_q.get(timeout=2000)
        busy_loop_active.set()

    # Verify the client sentinel receives the response from the engine core sentinel
    if not client_cmd_socket.poll(timeout=2000):
        pytest.fail("Timeout waiting for response from sentinel")

    identity, _, msg_bytes = client_cmd_socket.recv_multipart()
    assert identity == sentinel_identity
    ft_res = msgpack.decode(msg_bytes, type=FaultToleranceResult)
    assert ft_res.request_id == req_id
    assert ft_res.success

    time.sleep(0.1)
    fail_on_thread_exceptions(thread_excs)

    client_cmd_socket.close()
    worker_cmd_socket.close()
    sentinel.shutdown()
    ctx.term()
    worker_ctx.term()
