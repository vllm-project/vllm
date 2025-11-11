# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock

import pytest
import zmq

from vllm.utils.collection_utils import ThreadSafeDict
from vllm.v1.engine.core_client import ClientGuard
from vllm.v1.engine.utils import FaultHandler, FaultInfo

FAULT_RECEIVER_ADDR = "tcp://127.0.0.1:8844"
CMD_ADDR = "tcp://127.0.0.1:8845"
FAULT_PUB_ADDR = "tcp://127.0.0.1:8846"
FAULT_PUB_TOPIC = "vllm_fault"


def create_test_thread_safe_dict(initial_data=None):
    if initial_data is None:
        initial_data = {1: "Healthy"}
    if initial_data is None:
        initial_data = {1: "Healthy"}
    tsd = ThreadSafeDict()
    if initial_data:
        for k, v in initial_data.items():
            tsd[k] = v
    return tsd


def create_client_guard(
    engine_exception_q: asyncio.Queue, engine_status_dict: ThreadSafeDict[int, str]
):
    return ClientGuard(
        fault_receiver_addr=FAULT_RECEIVER_ADDR,
        cmd_addr=CMD_ADDR,
        engine_registry=[b"engine_identity"],
        engine_exception_q=engine_exception_q,
        engine_exception_q_lock=asyncio.Lock(),
        fault_pub_addr=FAULT_PUB_ADDR,
        engine_status_dict=engine_status_dict,
    )


def test_client_guard_initialization():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    assert guard.engine_registry == [b"engine_identity"]
    assert not guard.client_guard_dead
    assert isinstance(guard.fault_handler, FaultHandler)
    assert guard.engine_exception_q is engine_exception_q

    assert guard.fault_receiver_socket.type == zmq.ROUTER
    assert guard.cmd_socket.type == zmq.ROUTER
    assert guard.fault_pub_socket.type == zmq.PUB

    guard.shutdown_guard()


@pytest.mark.asyncio
async def test_handle_fault():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    engine_exception_q.put_nowait(
        FaultInfo(engine_id="1", message="test exception", type="test")
    )

    guard.fault_handler.handle_fault = AsyncMock(return_value=True)

    result = await guard.handle_fault("pause", 5)
    assert result is True
    guard.fault_handler.handle_fault.assert_awaited_once_with("pause", 5)

    guard.shutdown_guard()


def test_fault_receiver():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    def send_test_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"test_sender")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="dead", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode("utf-8")])
        socket.close()
        ctx.term()

    sender_thread = threading.Thread(target=send_test_message)
    sender_thread.start()

    def check_published_message():
        ctx = zmq.Context()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(FAULT_PUB_ADDR)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, FAULT_PUB_TOPIC)

        message = sub_socket.recv_string()
        sub_socket.close()
        ctx.term()

        prefix, data = message.split("|", 1)
        assert prefix == FAULT_PUB_TOPIC
        assert json.loads(data) == {"1": "Dead"}

    check_thread = threading.Thread(target=check_published_message)
    check_thread.start()

    time.sleep(0.1)

    assert not engine_exception_q.empty()
    received_fault = engine_exception_q.get_nowait()
    assert received_fault.engine_id == "1"
    assert received_fault.type == "dead"

    assert engine_status_dict[1] == "Dead"

    guard.shutdown_guard()


def test_fault_receiver_unhealthy():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    def send_unhealthy_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"engine_identity")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="error", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode()])
        socket.close()
        ctx.term()

    threading.Thread(target=send_unhealthy_message).start()
    time.sleep(0.1)

    assert engine_status_dict[1] == "Unhealthy"

    guard.shutdown_guard()


def test_shutdown_guard():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    original_fault_sock = guard.fault_receiver_socket
    original_cmd_sock = guard.cmd_socket
    original_pub_sock = guard.fault_pub_socket
    original_ctx = guard.zmq_ctx

    guard.shutdown_guard()

    assert guard.client_guard_dead is True

    with pytest.raises(zmq.ZMQError):
        original_fault_sock.recv()

    with pytest.raises(zmq.ZMQError):
        original_cmd_sock.recv()

    with pytest.raises(zmq.ZMQError):
        original_pub_sock.send(b"test")

    assert original_ctx.closed


@pytest.mark.asyncio
async def test_handle_fault_async():
    engine_exception_q: asyncio.Queue[FaultInfo] = asyncio.Queue()
    engine_status_dict = create_test_thread_safe_dict({0: "Unhealthy"})
    guard = create_client_guard(engine_exception_q, engine_status_dict)

    time.sleep(0.1)
    ctx = zmq.Context().instance()
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.setsockopt(zmq.IDENTITY, b"engine_identity")
    cmd_socket.connect(CMD_ADDR)

    uuid = None

    def receive_cmd(cmd_socket):
        nonlocal uuid
        time.sleep(0.1)

        identity, msg = cmd_socket.recv_multipart()
        cmd_dict = json.loads(msg.decode("utf-8"))
        assert cmd_dict["method"] == "retry"
        assert cmd_dict["timeout"] == 3
        uuid = cmd_dict["method_uuid"]

    def response_cmd(cmd_socket):
        nonlocal uuid
        while uuid is None:
            time.sleep(0.1)
        execute_result = {"engine_index": 0, "success": True, "method_uuid": uuid}
        cmd_socket.send_multipart([b"", json.dumps(execute_result).encode("utf-8")])

    threading.Thread(target=receive_cmd, args=(cmd_socket,)).start()
    threading.Thread(target=response_cmd, args=(cmd_socket,)).start()

    result = await guard.handle_fault("retry", 3)

    assert result is True
    assert engine_status_dict[0] == "Healthy"

    guard.shutdown_guard()
