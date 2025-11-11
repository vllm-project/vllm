# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import queue
import threading
import time

import pytest
import zmq

from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine.core import (
    EngineCoreGuard,
    EngineLoopPausedError,
)
from vllm.v1.serial_utils import serialize_method_call

CLIENT_CMD_ADDR = "tcp://127.0.0.1:8844"
WORKER_CMD_ADDR = "tcp://127.0.0.1:8845"
FAULT_REPORT_ADDR = "tcp://127.0.0.1:8846"
GUARD_IDENTITY = b"engine_guard_0"


def create_engine_core_guard(
    fault_signal_q: queue.Queue, busy_loop_active: threading.Event
):
    return EngineCoreGuard(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        cmd_q=queue.Queue(),
        busy_loop_active=busy_loop_active,
        engine_input_q=queue.Queue(),
        client_cmd_addr=CLIENT_CMD_ADDR,
        worker_cmd_addr=WORKER_CMD_ADDR,
        fault_report_addr=FAULT_REPORT_ADDR,
        guard_identity=GUARD_IDENTITY,
        tp_size=1,
        pp_size=1,
        dp_size=1,
    )


def test_engine_core_guard_initialization():
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    guard = create_engine_core_guard(fault_signal_q, busy_loop_active)

    assert guard.engine_index == 0
    assert guard.tp_size == 1
    assert guard.pp_size == 1
    assert not guard.communicator_aborted
    assert guard.engine_running is True
    assert guard.daemon is True

    assert guard.fault_report_socket.type == zmq.DEALER
    assert guard.client_cmd_socket.type == zmq.DEALER
    assert guard.worker_cmd_socket.type == zmq.ROUTER

    guard.shutdown()


@pytest.mark.parametrize("instruction", ["pause", "retry"])
def test_run_handle_instruction(instruction):
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    client_socket = make_zmq_socket(
        ctx=zmq.Context(), path=CLIENT_CMD_ADDR, socket_type=zmq.ROUTER, bind=True
    )

    time.sleep(0.1)

    guard = create_engine_core_guard(fault_signal_q, busy_loop_active)
    time.sleep(0.1)

    ctx = zmq.Context()
    worker_cmd_socket = ctx.socket(zmq.DEALER)
    worker_cmd_socket.setsockopt(zmq.IDENTITY, b"0_0")
    worker_cmd_socket.connect(WORKER_CMD_ADDR)

    def mock_worker_receiver(cmd_socket):
        time.sleep(0.1)
        logging.info("start worker")
        identity, msg = cmd_socket.recv_multipart()
        logging.info(identity)
        cmd_dict = json.loads(msg.decode("utf-8"))
        assert (
            cmd_dict["method"] == "pause_by_signal"
            if instruction == "pause"
            else "retry"
        )
        response_dict = {"success": True, "method_uuid": cmd_dict["method_uuid"]}
        logging.info(identity)
        cmd_socket.send_multipart([b"", json.dumps(response_dict).encode("utf-8")])

    threading.Thread(target=guard.run, daemon=True).start()
    time.sleep(0.1)

    param = {"timeout": 3}
    if instruction == "pause":
        param["soft_pause"] = True
    elif instruction == "retry":
        param["new_stateless_dp_group_port"] = 23456
    serial_instruction = serialize_method_call(instruction, **param)
    client_socket.send_multipart(
        [GUARD_IDENTITY, b"", serial_instruction.encode("utf-8")]
    )
    if instruction == "pause":
        fault_signal_q.put(EngineLoopPausedError(Exception("test error")))
    elif instruction == "retry":
        busy_loop_active.set()

    threading.Thread(target=mock_worker_receiver, args=(worker_cmd_socket,)).start()

    time.sleep(0.1)
    identity, _, msg = client_socket.recv_multipart()
    result_dict = json.loads(msg.decode("utf-8"))
    assert result_dict["engine_index"] == 0
    assert result_dict["success"]

    time.sleep(0.1)

    client_socket.close()
    worker_cmd_socket.close()
    guard.shutdown()
