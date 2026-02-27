# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import threading
import time
import traceback
import uuid
from queue import Queue

import msgspec
import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine import EngineStatusType, FaultToleranceRequest, FaultToleranceResult
from vllm.v1.engine.core_client import ClientSentinel
from vllm.v1.engine.exceptions import FaultInfo


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    ports = [_find_free_port() for _ in range(4)]
    return {
        "fault_receiver_addr": f"tcp://127.0.0.1:{ports[0]}",
        "client_sentinel_request_addr": f"tcp://127.0.0.1:{ports[1]}",
        "fault_pub_addr": f"tcp://127.0.0.1:{ports[2]}",
        "engine_core_cmd_addr": f"tcp://127.0.0.1:{ports[3]}",
    }


# Helper to collect exceptions from threads and fail the test if any were raised.
def fail_on_thread_exceptions(thread_excs: Queue) -> None:
    if not thread_excs.empty():
        pytest.fail("Thread raised exception:\n" + "\n".join(thread_excs.queue))


def send_error_message(
    fault_receiver_addr: str, engine_id: str, fault_type: str, thread_excs: queue.Queue
):
    # send error message to client sentinel, simulating an engine reporting an error
    try:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"test_sender")
        socket.connect(fault_receiver_addr)

        test_fault = FaultInfo(
            engine_id=str(engine_id), type=fault_type, message="test error"
        )
        socket.send_multipart([b"", msgspec.msgpack.encode(test_fault)])
        socket.close()
        ctx.term()
    except Exception:
        thread_excs.put(traceback.format_exc())


def create_client_sentinel(num_engines: int = 2, addr_dict=None):
    # Accept an addr_dict (pytest fixture) to avoid relying on global port state.
    if addr_dict is None:
        # Fallback for callers that don't pass addr_dict (keeps compatibility).
        ports = [_find_free_port() for _ in range(4)]
        addr_dict = {
            "fault_receiver_addr": f"tcp://127.0.0.1:{ports[0]}",
            "client_sentinel_request_addr": f"tcp://127.0.0.1:{ports[1]}",
            "fault_pub_addr": f"tcp://127.0.0.1:{ports[2]}",
            "engine_core_cmd_addr": f"tcp://127.0.0.1:{ports[3]}",
        }

    parallel = ParallelConfig(
        data_parallel_size=num_engines,
        data_parallel_size_local=num_engines,
        data_parallel_master_ip="127.0.0.1",
    )
    vconfig = VllmConfig(
        parallel_config=parallel,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return ClientSentinel(
        vllm_config=vconfig,
        engine_fault_socket_addr=addr_dict["fault_receiver_addr"],
        client_sentinel_request_addr=addr_dict["client_sentinel_request_addr"],
        engine_core_sentinel_cmd_addr=addr_dict["engine_core_cmd_addr"],
        engine_core_sentinel_identities={
            i: f"engine_sentinel_identity_{i}".encode() for i in range(num_engines)
        },
        fault_state_pub_socket_addr=addr_dict["fault_pub_addr"],
    )


def test_client_sentinel_initialization(addr_dict):
    sentinel = create_client_sentinel(addr_dict=addr_dict)

    assert sentinel.engine_core_sentinel_identities[0] == b"engine_sentinel_identity_0"
    assert not sentinel.sentinel_dead
    assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.HEALTHY
    assert sentinel.fault_receiver_socket.type == zmq.ROUTER
    assert sentinel.downstream_cmd_socket.type == zmq.ROUTER
    assert sentinel.fault_state_pub_socket.type == zmq.PUB

    sentinel.shutdown()


def test_client_sentinel_receives_faults_and_publishes_status(addr_dict):
    sentinel = create_client_sentinel(addr_dict=addr_dict)
    thread_excs: Queue = Queue()

    def decode_published_message(sub_socket):
        # decode error messages published by client sentinel
        if not sub_socket.poll(timeout=2000):
            raise TimeoutError("Timeout waiting for published message")
        frames = sub_socket.recv_multipart()
        return frames[0].decode("utf-8"), msgspec.msgpack.decode(frames[-1])

    def check_published_message():
        # listen and verify error messages published from client sentinel
        try:
            ctx = zmq.Context()
            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(addr_dict["fault_pub_addr"])
            sub_socket.setsockopt_string(
                zmq.SUBSCRIBE, sentinel.ft_config.fault_state_pub_topic
            )
            # Consume the first published status update (after the first fault)
            # without validating it here; this test only asserts on the final state
            decode_published_message(sub_socket)
            prefix, status_dict = decode_published_message(sub_socket)
            sub_socket.close()
            ctx.term()
            assert prefix == sentinel.ft_config.fault_state_pub_topic
            assert status_dict[1]["status"] == EngineStatusType.DEAD
            assert status_dict[0]["status"] == EngineStatusType.UNHEALTHY
        except Exception:
            thread_excs.put(traceback.format_exc())

    t1 = threading.Thread(target=check_published_message, daemon=True)
    t1.start()
    time.sleep(0.1)

    threading.Thread(
        target=send_error_message,
        args=(addr_dict["fault_receiver_addr"], "1", "dead", thread_excs),
        daemon=True,
    ).start()
    time.sleep(0.1)
    threading.Thread(
        target=send_error_message,
        args=(addr_dict["fault_receiver_addr"], "0", "RuntimeError", thread_excs),
        daemon=True,
    ).start()

    t1.join()
    fail_on_thread_exceptions(thread_excs)

    # Verify engine_status_dict updated
    assert sentinel.engine_status_dict[1]["status"] == EngineStatusType.DEAD
    assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.UNHEALTHY

    sentinel.shutdown()


def test_shutdown_sentinel(addr_dict):
    sentinel = create_client_sentinel(addr_dict=addr_dict)

    original_fault_sock = sentinel.fault_receiver_socket
    original_cmd_sock = sentinel.downstream_cmd_socket
    original_pub_sock = sentinel.fault_state_pub_socket
    original_ctx = sentinel.ctx

    sentinel.shutdown()

    assert sentinel.sentinel_dead

    with pytest.raises(zmq.ZMQError):
        original_fault_sock.recv()
    with pytest.raises(zmq.ZMQError):
        original_cmd_sock.recv()
    with pytest.raises(zmq.ZMQError):
        original_pub_sock.send(b"test")

    assert original_ctx.closed


@pytest.mark.parametrize("instruction", ["pause", "retry"])
def test_handle_fault_roundtrip(instruction, addr_dict):
    """Test the full FaultToleranceRequest -> engine -> FaultToleranceResult flow
    by talking to the sentinel's ROUTER endpoints using msgpack-encoded
    FaultToleranceRequest/Result messages.
    """
    sentinel = create_client_sentinel(num_engines=1, addr_dict=addr_dict)
    thread_excs: Queue = Queue()

    # Start a DEALER that will act as the downstream engine sentinel.
    engine_ctx = zmq.Context()
    engine_socket = engine_ctx.socket(zmq.DEALER)
    engine_socket.setsockopt(zmq.IDENTITY, b"engine_sentinel_identity_0")
    engine_socket.connect(addr_dict["engine_core_cmd_addr"])

    # Create a client DEALER that will send the FaultToleranceRequest to ClientSentinel
    client_ctx = zmq.Context()
    client_socket = client_ctx.socket(zmq.DEALER)
    client_socket.setsockopt(zmq.IDENTITY, b"test_client")
    client_socket.connect(addr_dict["client_sentinel_request_addr"])

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

    def engine_loop():
        try:
            if not engine_socket.poll(timeout=2000):
                raise TimeoutError(
                    "Timeout waiting for FaultToleranceRequest from client sentinel"
                )
            parts = engine_socket.recv_multipart()
            received = msgspec.msgpack.decode(parts[-1], type=FaultToleranceRequest)
            assert received.instruction in ("pause", "retry")
            # reply with success
            res = FaultToleranceResult(
                request_id=received.request_id, success=True, reason=None
            )
            engine_socket.send_multipart([b"", msgspec.msgpack.encode(res)])
        except Exception:
            thread_excs.put(traceback.format_exc())

    threading.Thread(target=engine_loop, daemon=True).start()
    client_socket.send_multipart([b"", msgspec.msgpack.encode(ft_req)])

    # Wait for reply from ClientSentinel on client_socket
    if not client_socket.poll(timeout=2000):
        pytest.fail("Timeout waiting for FaultToleranceResult from client sentinel.")
    parts = client_socket.recv_multipart()
    ft_res = msgspec.msgpack.decode(parts[-1], type=FaultToleranceResult)
    assert ft_res.request_id == req_id
    assert ft_res.success is True
    if instruction == "pause":
        assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.PAUSED
    elif instruction == "retry":
        assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.HEALTHY

    fail_on_thread_exceptions(thread_excs)
    client_socket.close()
    client_ctx.term()
    engine_socket.close()
    engine_ctx.term()

    sentinel.shutdown()
