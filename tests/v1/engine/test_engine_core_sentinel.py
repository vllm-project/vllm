# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import queue
import threading
import time

import pytest
import zmq

from vllm.config import FaultToleranceConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine.core import (
    EngineCoreSentinel,
    EngineLoopPausedError,
)
from vllm.v1.serial_utils import serialize_method_call

CLIENT_CMD_ADDR = "tcp://127.0.0.1:8844"
WORKER_CMD_ADDR = "tcp://127.0.0.1:8845"
ENGINE_FAULT_SOCKET_ADDR = "tcp://127.0.0.1:8846"
DEALER_SOCKET_IDENTITY = b"engine_sentinel_0"


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue, busy_loop_active: threading.Event
):
    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        cmd_q=queue.Queue(),
        busy_loop_active=busy_loop_active,
        engine_input_q=queue.Queue(),
        client_cmd_addr=CLIENT_CMD_ADDR,
        worker_cmd_addr=WORKER_CMD_ADDR,
        engine_fault_socket_addr=ENGINE_FAULT_SOCKET_ADDR,
        dealer_socket_identity=DEALER_SOCKET_IDENTITY,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
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
        identity, msg = cmd_socket.recv_multipart()
        cmd_dict = json.loads(msg.decode("utf-8"))
        assert cmd_dict["method"] == "pause" if instruction == "pause" else "retry"
        response_dict = {"success": True, "method_uuid": cmd_dict["method_uuid"]}
        cmd_socket.send_multipart([b"", json.dumps(response_dict).encode("utf-8")])

    param = {"timeout": 3}
    if instruction == "pause":
        param["soft_pause"] = True
    elif instruction == "retry":
        param["new_stateless_dp_group_port"] = 23456
    serial_instruction = serialize_method_call(instruction, **param)
    client_socket.send_multipart(
        [DEALER_SOCKET_IDENTITY, b"", serial_instruction.encode("utf-8")]
    )
    fault_signal_q.put(EngineLoopPausedError(Exception("test error")))
    if instruction == "retry":
        busy_loop_active.set()

    threading.Thread(
        target=mock_worker_receiver, args=(worker_cmd_socket,), daemon=True
    ).start()

    time.sleep(0.1)
    if not client_socket.poll(timeout=2000):
        pytest.fail("Timeout waiting for response from sentinel")
    identity, _, msg = client_socket.recv_multipart()
    result_dict = json.loads(msg.decode("utf-8"))
    assert result_dict["sentinel_tag"] == "DP_0"
    assert result_dict["success"]

    time.sleep(0.1)

    client_socket.close()
    worker_cmd_socket.close()
    sentinel.shutdown()
    ctx.term()
