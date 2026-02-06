# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import queue
import threading
import time

import pytest
import zmq

from vllm.config import FaultToleranceConfig
from vllm.utils.collection_utils import ThreadSafeDict
from vllm.v1.engine.core_client import ClientSentinel
from vllm.v1.engine.utils import FaultInfo

FAULT_RECEIVER_ADDR = "tcp://127.0.0.1:8844"
CMD_ADDR = "tcp://127.0.0.1:8845"
FAULT_PUB_ADDR = "tcp://127.0.0.1:8846"
FAULT_PUB_TOPIC = "vllm_fault"


def create_test_thread_safe_dict(initial_data=None):
    if initial_data is None:
        initial_data = {1: "Healthy"}

    tsd = ThreadSafeDict()
    if initial_data:
        for k, v in initial_data.items():
            tsd[k] = v
    return tsd


def create_client_sentinel(
    engine_exception_q: queue.Queue, engine_status_dict: ThreadSafeDict[int, str]
):
    return ClientSentinel(
        fault_receiver_addr=FAULT_RECEIVER_ADDR,
        cmd_addr=CMD_ADDR,
        engine_registry={0: b"engine_identity"},
        engine_exception_q=engine_exception_q,
        fault_pub_addr=FAULT_PUB_ADDR,
        engine_status_dict=engine_status_dict,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )


def test_client_sentinel_initialization():
    engine_exception_q: queue.Queue[FaultInfo] = queue.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    sentinel = create_client_sentinel(engine_exception_q, engine_status_dict)

    assert sentinel.engine_registry[0] == b"engine_identity"
    assert not sentinel.sentinel_dead
    assert sentinel.engine_exception_q is engine_exception_q

    assert sentinel.fault_receiver_socket.type == zmq.ROUTER
    assert sentinel.downstream_cmd_socket.type == zmq.ROUTER
    assert sentinel.fault_pub_socket.type == zmq.PUB

    sentinel.shutdown()


def test_fault_receiver():
    engine_exception_q: queue.Queue[FaultInfo] = queue.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    sentinel = create_client_sentinel(engine_exception_q, engine_status_dict)

    def send_test_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"test_sender")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="dead", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode("utf-8")])
        socket.close()
        ctx.term()

    sender_thread = threading.Thread(target=send_test_message, daemon=True)
    sender_thread.start()

    def check_published_message():
        ctx = zmq.Context()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(FAULT_PUB_ADDR)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, FAULT_PUB_TOPIC)

        if not sub_socket.poll(timeout=2000):  # 2-second timeout
            pytest.fail("Timeout waiting for published message")
        message = sub_socket.recv_string()
        sub_socket.close()
        ctx.term()

        prefix, data = message.split("|", 1)
        assert prefix == FAULT_PUB_TOPIC
        assert json.loads(data) == {"1": "Dead"}

    check_thread = threading.Thread(target=check_published_message, daemon=True)
    check_thread.start()

    received_fault = engine_exception_q.get(timeout=1)

    assert received_fault.engine_id == "1"
    assert received_fault.type == "dead"

    assert engine_status_dict[1] == "Dead"

    sentinel.shutdown()


def test_fault_receiver_unhealthy():
    engine_exception_q: queue.Queue[FaultInfo] = queue.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    sentinel = create_client_sentinel(engine_exception_q, engine_status_dict)

    def send_unhealthy_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"engine_identity")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="error", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode()])
        socket.close()
        ctx.term()

    threading.Thread(target=send_unhealthy_message, daemon=True).start()
    time.sleep(0.1)

    assert engine_status_dict[1] == "Unhealthy"

    sentinel.shutdown()


def test_shutdown_sentinel():
    engine_exception_q: queue.Queue[FaultInfo] = queue.Queue()
    engine_status_dict = create_test_thread_safe_dict({1: "Healthy"})
    sentinel = create_client_sentinel(engine_exception_q, engine_status_dict)

    original_fault_sock = sentinel.fault_receiver_socket
    original_cmd_sock = sentinel.downstream_cmd_socket
    original_pub_sock = sentinel.fault_pub_socket
    original_ctx = sentinel.ctx

    sentinel.shutdown()

    assert sentinel.sentinel_dead is True

    with pytest.raises(zmq.ZMQError):
        original_fault_sock.recv()

    with pytest.raises(zmq.ZMQError):
        original_cmd_sock.recv()

    with pytest.raises(zmq.ZMQError):
        original_pub_sock.send(b"test")

    assert original_ctx.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("instruction", ["pause", "retry"])
async def test_handle_fault_async(instruction):
    engine_exception_q: queue.Queue[FaultInfo] = queue.Queue()
    if instruction == "retry":
        engine_status_dict = create_test_thread_safe_dict({0: "Unhealthy"})
    else:
        engine_status_dict = create_test_thread_safe_dict({0: "Healthy"})
    sentinel = create_client_sentinel(engine_exception_q, engine_status_dict)

    time.sleep(0.1)
    ctx = zmq.Context().instance()
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.setsockopt(zmq.IDENTITY, b"engine_identity")
    cmd_socket.connect(CMD_ADDR)
    time.sleep(0.1)

    uuid = None
    uuid_received = threading.Event()

    def receive_cmd(cmd_socket):
        nonlocal uuid
        time.sleep(0.1)

        identity, msg = cmd_socket.recv_multipart()
        cmd_dict = json.loads(msg.decode("utf-8"))
        uuid = cmd_dict["method_uuid"]
        if instruction == "retry":
            assert cmd_dict["method"] == "retry"
        else:
            assert cmd_dict["method"] == "pause"
        assert cmd_dict["timeout"] == 3
        uuid_received.set()

    def response_cmd(cmd_socket):
        nonlocal uuid
        if not uuid_received.wait(timeout=2):
            pytest.fail("Timeout waiting for UUID")
        execute_result = {"sentinel_tag": "DP_0", "success": True, "method_uuid": uuid}
        cmd_socket.send_multipart([b"", json.dumps(execute_result).encode("utf-8")])

    threading.Thread(target=receive_cmd, args=(cmd_socket,), daemon=True).start()
    threading.Thread(target=response_cmd, args=(cmd_socket,), daemon=True).start()

    if instruction == "pause":
        result = await sentinel.handle_fault(instruction, 3, soft_pause=True)
    else:
        result = await sentinel.handle_fault(
            "retry", 3, new_stateless_dp_group_port=None
        )

    assert result is True
    assert engine_status_dict[0] == "Healthy"

    cmd_socket.close()
    ctx.term()
    sentinel.shutdown()
