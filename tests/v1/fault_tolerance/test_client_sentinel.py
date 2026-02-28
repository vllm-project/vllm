# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import threading
import time
import traceback
from queue import Queue

import msgspec
import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance import ClientSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceZmqAddresses


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ft_addr():
    ports = [_find_free_port() for _ in range(2)]
    return FaultToleranceZmqAddresses(
        fault_state_pub_socket_addr=f"tcp://127.0.0.1:{ports[0]}",
        engine_fault_socket_addr=f"tcp://127.0.0.1:{ports[1]}",
        engine_core_sentinel_identities={0: b"engine_sentinel_identity"},
    )


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


def create_client_sentinel(num_engines: int = 2, ft_addr=None):
    if ft_addr is None:
        ports = [_find_free_port() for _ in range(2)]
        ft_addr = FaultToleranceZmqAddresses(
            fault_state_pub_socket_addr=f"tcp://127.0.0.1:{ports[0]}",
            engine_fault_socket_addr=f"tcp://127.0.0.1:{ports[1]}",
            engine_core_sentinel_identities={
                i: f"engine_sentinel_identity_{i}".encode() for i in range(num_engines)
            },
        )

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
        fault_tolerance_addresses=ft_addr,
        shutdown_callback=lambda: None,
    )


def test_client_sentinel_initialization(ft_addr):
    sentinel = create_client_sentinel(ft_addr=ft_addr)

    assert (
        sentinel.engine_core_sentinel_identities[0]
        == ft_addr.engine_core_sentinel_identities[0]
    )
    assert not sentinel.sentinel_dead
    assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.HEALTHY
    assert sentinel.fault_receiver_socket.type == zmq.ROUTER
    assert sentinel.fault_state_pub_socket.type == zmq.PUB

    sentinel.shutdown()


def test_client_sentinel_receives_faults_and_publishes_status(ft_addr):
    sentinel = create_client_sentinel(ft_addr=ft_addr)
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
            sub_socket.connect(ft_addr.fault_state_pub_socket_addr)
            sub_socket.setsockopt_string(
                zmq.SUBSCRIBE, sentinel.ft_config.fault_state_pub_topic
            )
            # Consume the first published status update (after the first fault)
            # without validating it here; this test only asserts on the final state
            prefix, status_dict = decode_published_message(sub_socket)
            sub_socket.close()
            ctx.term()
            assert prefix == sentinel.ft_config.fault_state_pub_topic
            assert status_dict[1]["status"] == EngineStatusType.DEAD
        except Exception:
            thread_excs.put(traceback.format_exc())

    t1 = threading.Thread(target=check_published_message, daemon=True)
    t1.start()
    time.sleep(0.1)

    threading.Thread(
        target=send_error_message,
        args=(ft_addr.engine_fault_socket_addr, "1", "dead", thread_excs),
        daemon=True,
    ).start()
    time.sleep(0.1)

    t1.join()
    fail_on_thread_exceptions(thread_excs)

    # Verify engine_status_dict updated
    assert sentinel.engine_status_dict[1]["status"] == EngineStatusType.DEAD

    sentinel.shutdown()


def test_shutdown_sentinel(ft_addr):
    sentinel = create_client_sentinel(ft_addr=ft_addr)

    original_fault_sock = sentinel.fault_receiver_socket
    original_pub_sock = sentinel.fault_state_pub_socket

    sentinel.shutdown()

    assert sentinel.sentinel_dead

    with pytest.raises(zmq.ZMQError):
        original_fault_sock.recv()
    with pytest.raises(zmq.ZMQError):
        original_pub_sock.send(b"test")
