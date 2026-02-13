# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
import uuid

import msgspec
import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.utils.collection_utils import ThreadSafeDict
from vllm.v1.engine import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.engine.core_client import ClientSentinel
from vllm.v1.engine.exceptions import FaultInfo

# todo: check if this file has been updated with the new changes to
#  ClientSentinel and update tests accordingly

FAULT_RECEIVER_ADDR = "tcp://127.0.0.1:8844"
CMD_ADDR = "tcp://127.0.0.1:8845"
FAULT_PUB_ADDR = "tcp://127.0.0.1:8846"
ENGINE_CORE_CMD_ADDR = "tcp://127.0.0.1:8847"
FAULT_PUB_TOPIC = "vllm_fault"


def create_test_thread_safe_dict(initial_data=None):
    if initial_data is None:
        initial_data = {1: {"status": "Healthy"}}

    tsd = ThreadSafeDict()
    if initial_data:
        for k, v in initial_data.items():
            tsd[k] = v
    return tsd


def create_client_sentinel():
    # Build a minimal VllmConfig that enables fault tolerance and has
    # at least 2 data-parallel ranks so tests can address engine 0 and 1.
    parallel = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_master_ip="127.0.0.1",
    )
    vconfig = VllmConfig(
        parallel_config=parallel,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return ClientSentinel(
        vllm_config=vconfig,
        engine_fault_socket_addr=FAULT_RECEIVER_ADDR,
        client_sentinel_cmd_addr=CMD_ADDR,
        engine_core_sentinel_cmd_addr=ENGINE_CORE_CMD_ADDR,
        engine_core_sentinel_identities={
            0: b"engine_identity",
            1: b"engine_identity_1",
        },
        fault_state_pub_socket_addr=FAULT_PUB_ADDR,
    )


def test_client_sentinel_initialization():
    sentinel = create_client_sentinel()

    # New field name for identities and engine_status_dict shape
    assert sentinel.engine_core_sentinel_identities[0] == b"engine_identity"
    assert not sentinel.sentinel_dead

    assert 0 in sentinel.engine_status_dict
    assert sentinel.engine_status_dict[0]["status"] == "Healthy"

    assert sentinel.fault_receiver_socket.type == zmq.ROUTER
    assert sentinel.downstream_cmd_socket.type == zmq.ROUTER
    assert sentinel.fault_state_pub_socket.type == zmq.PUB

    sentinel.shutdown()


def test_fault_receiver():
    sentinel = create_client_sentinel()

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
        # New published format contains nested dicts: {"1": {"status": "Dead"}}
        parsed = json.loads(data)
        assert parsed.get("1", {}).get("status") == "Dead"

    check_thread = threading.Thread(target=check_published_message, daemon=True)
    check_thread.start()

    # Wait a short time for sentinel to process the fault
    time.sleep(0.2)

    # Verify engine_status_dict updated
    assert sentinel.engine_status_dict[1]["status"] == "Dead"

    sentinel.shutdown()


def test_fault_receiver_unhealthy():
    sentinel = create_client_sentinel()

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
    time.sleep(0.2)

    assert sentinel.engine_status_dict[1]["status"] == "Unhealthy"

    sentinel.shutdown()


def test_shutdown_sentinel():
    sentinel = create_client_sentinel()

    original_fault_sock = sentinel.fault_receiver_socket
    original_cmd_sock = sentinel.downstream_cmd_socket
    original_pub_sock = sentinel.fault_state_pub_socket
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


@pytest.mark.parametrize("instruction", ["pause", "retry"])
def test_handle_fault_roundtrip(instruction):
    """Test the full FaultToleranceRequest -> engine -> FaultToleranceResult flow
    by talking to the sentinel's ROUTER endpoints using msgpack-encoded
    FaultToleranceRequest/Result messages.
    """
    sentinel = create_client_sentinel()

    # Start a DEALER that will act as the downstream engine sentinel and
    # connect it to the engine_core_sentinel_cmd_addr (the sentinel binds there).
    engine_ctx = zmq.Context()
    engine_socket = engine_ctx.socket(zmq.DEALER)
    engine_socket.setsockopt(zmq.IDENTITY, b"engine_identity")
    engine_socket.connect(ENGINE_CORE_CMD_ADDR)

    # Create a client DEALER that will send the FaultToleranceRequest to ClientSentinel
    client_ctx = zmq.Context()
    client_socket = client_ctx.socket(zmq.DEALER)
    client_socket.setsockopt(zmq.IDENTITY, b"test_client")
    client_socket.connect(CMD_ADDR)

    # Prepare the fault tolerance request
    req_id = str(uuid.uuid4())
    params = {"timeout": 3}
    if instruction == "pause":
        params["soft_pause"] = True
    else:
        params["new_stateless_dp_group_port"] = 12568

    ft_req = FaultToleranceRequest(
        request_id=req_id, instruction=instruction, params=params
    )

    # thread: engine receives broadcast from sentinel and replies with success
    def engine_loop():
        # receive broadcast (ROUTER->DEALER framing)
        parts = engine_socket.recv_multipart()
        # last frame is the msgpack-encoded FaultToleranceRequest
        msg_bytes = parts[-1]
        received = msgspec.msgpack.decode(msg_bytes, type=FaultToleranceRequest)
        assert received.instruction in ("pause", "retry")
        # reply with success
        res = FaultToleranceResult(
            request_id=received.request_id, success=True, reason=None
        )
        engine_socket.send_multipart([b"", msgspec.msgpack.encode(res)])

    eng_thread = threading.Thread(target=engine_loop, daemon=True)
    eng_thread.start()

    # Send FT request to sentinel
    client_socket.send_multipart([b"", msgspec.msgpack.encode(ft_req)])

    # Wait for reply from ClientSentinel on client_socket
    parts = client_socket.recv_multipart(flags=0)
    # last frame contains the msgpack-encoded FaultToleranceResult
    msg_bytes = parts[-1]
    ft_res = msgspec.msgpack.decode(msg_bytes, type=FaultToleranceResult)

    assert ft_res.request_id == req_id
    assert ft_res.success is True

    # After a successful retry the sentinel resets states to Healthy
    # Pause doesn't change status from Healthy; ensure key exists and is dict-shaped.
    assert 0 in sentinel.engine_status_dict
    assert isinstance(sentinel.engine_status_dict[0], dict)

    client_socket.close()
    client_ctx.term()
    engine_socket.close()
    engine_ctx.term()

    sentinel.shutdown()
