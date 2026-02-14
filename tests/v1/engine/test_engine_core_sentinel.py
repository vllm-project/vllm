# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import time
import uuid

import pytest
import zmq
from msgspec import msgpack

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.engine.core import (
    EngineCoreSentinel,
    EngineLoopPausedError,
)

# todo: check if this file has been updated with the new changes
CLIENT_CMD_ADDR = "tcp://127.0.0.1:8844"
WORKER_CMD_ADDR = "tcp://127.0.0.1:8845"
ENGINE_FAULT_SOCKET_ADDR = "tcp://127.0.0.1:8846"
SENTINEL_IDENTITY = b"engine_sentinel_0"


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue, busy_loop_active: threading.Event
):
    # Construct a minimal VllmConfig with the required parallel and fault-tolerance
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        cmd_q=queue.Queue(),
        busy_loop_active=busy_loop_active,
        engine_input_q=queue.Queue(),
        client_cmd_addr=CLIENT_CMD_ADDR,
        worker_cmd_addr=WORKER_CMD_ADDR,
        engine_fault_socket_addr=ENGINE_FAULT_SOCKET_ADDR,
        sentinel_identity=SENTINEL_IDENTITY,
        vllm_config=vllm_cfg,
    )


def test_engine_core_sentinel_initialization():
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    sentinel = create_engine_core_sentinel(fault_signal_q, busy_loop_active)

    assert sentinel.engine_index == 0
    assert sentinel.tp_size == 1
    assert sentinel.pp_size == 1
    assert not sentinel.communicator_aborted
    assert sentinel.engine_running is True

    assert sentinel.engine_fault_socket.type == zmq.DEALER
    assert sentinel.upstream_cmd_socket.type == zmq.DEALER
    assert sentinel.downstream_cmd_socket.type == zmq.ROUTER

    sentinel.shutdown()


@pytest.mark.parametrize("instruction", ["pause", "retry"])
def test_run_handle_instruction(instruction):
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    client_socket = make_zmq_socket(
        ctx=zmq.Context(), path=CLIENT_CMD_ADDR, socket_type=zmq.ROUTER, bind=True
    )

    time.sleep(0.1)

    sentinel = create_engine_core_sentinel(fault_signal_q, busy_loop_active)
    time.sleep(0.1)

    ctx = zmq.Context()
    worker_cmd_socket = ctx.socket(zmq.DEALER)
    worker_cmd_socket.setsockopt(zmq.IDENTITY, b"PP0_TP0")
    worker_cmd_socket.connect(WORKER_CMD_ADDR)

    def mock_worker_receiver(cmd_socket):
        time.sleep(0.1)
        if not cmd_socket.poll(timeout=2000):
            pytest.fail("Timeout waiting for command from sentinel")
        parts = cmd_socket.recv_multipart()
        # DEALER receives messages in the form [empty_frame, msg_bytes]
        empty_frame = None
        msg_bytes = None
        if len(parts) == 2:
            empty_frame, msg_bytes = parts
        elif len(parts) == 3:
            # tolerate an extra identity frame if present
            _, empty_frame, msg_bytes = parts
        else:
            pytest.fail(f"Unexpected multipart length: {len(parts)}")

        assert empty_frame == b""
        ft_req = msgpack.decode(msg_bytes, type=FaultToleranceRequest)
        assert ft_req.instruction == instruction

        # Reply with a msgpack-encoded FaultToleranceResult
        ft_res = FaultToleranceResult(request_id=ft_req.request_id, success=True)
        cmd_socket.send_multipart([b"", msgpack.encode(ft_res)])

    param = {"timeout": 3}
    if instruction == "pause":
        param["soft_pause"] = True
    elif instruction == "retry":
        param["new_stateless_dp_group_port"] = 23456

    # Build a FaultToleranceRequest and send as msgpack
    req_id = str(uuid.uuid4())
    ft_request = FaultToleranceRequest(
        request_id=req_id, instruction=instruction, params=param
    )
    client_socket.send_multipart([SENTINEL_IDENTITY, b"", msgpack.encode(ft_request)])

    fault_signal_q.put(EngineLoopPausedError(Exception("test error")))
    if instruction == "retry":
        busy_loop_active.set()

    threading.Thread(
        target=mock_worker_receiver, args=(worker_cmd_socket,), daemon=True
    ).start()

    time.sleep(0.1)
    if not client_socket.poll(timeout=2000):
        pytest.fail("Timeout waiting for response from sentinel")
    identity, _, msg_bytes = client_socket.recv_multipart()

    # The ROUTER identity frame indicates who replied
    assert identity == SENTINEL_IDENTITY
    ft_res = msgpack.decode(msg_bytes, type=FaultToleranceResult)
    assert ft_res.request_id == req_id
    assert ft_res.success

    time.sleep(0.1)

    client_socket.close()
    worker_cmd_socket.close()
    sentinel.shutdown()
    ctx.term()
