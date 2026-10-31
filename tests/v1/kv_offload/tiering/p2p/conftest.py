# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload.tiering.p2p.session import client as client_module


@pytest.fixture
def lookup_clock(monkeypatch):
    """Advance monotonic time for P2P deadlines without sleeping."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock.now)
    return clock


@pytest.fixture
def p2p_control_peer():
    """A control-only peer: delay its handshake or fail fetches on demand."""
    import queue
    import socket
    import threading

    from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
    from vllm.v1.kv_offload.tiering.p2p.control.zmq import ZmqTransport

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    handshake = threading.Event()
    reply_to_fetch = threading.Event()
    fetches: queue.Queue[dict] = queue.Queue()
    ready = threading.Event()
    stop = threading.Event()
    errors = []

    def serve():
        transport = None
        try:
            transport = ZmqTransport(f"127.0.0.1:{port}", "127.0.0.1", port)
            connections: list[ControlConnection] = []
            acknowledged = set()
            ready.set()
            while not stop.is_set():
                connections.extend(transport.poll())
                for conn in connections:
                    if handshake.is_set() and conn.peer_id not in acknowledged:
                        conn.send(
                            {"type": "connect_ack", "peer_id": f"127.0.0.1:{port}"}
                        )
                        acknowledged.add(conn.peer_id)
                    for msg in conn.recv():
                        kind = msg["type"]
                        if kind == "fetch" and msg["keys"]:
                            fetches.put(msg)
                            if not reply_to_fetch.is_set():
                                continue
                            conn.send(
                                {
                                    "type": "transfer_done",
                                    "kv_request_id": msg["kv_request_id"],
                                    "round_seq": msg["round_seq"],
                                    "success": False,
                                }
                            )
                        elif kind == "abort_fetch":
                            conn.send(
                                {
                                    "type": "abort_ack",
                                    "kv_request_id": msg["kv_request_id"],
                                    "round_seq": msg["round_seq"],
                                }
                            )
                stop.wait(0.001)
        except Exception as exc:
            errors.append(exc)
            ready.set()
        finally:
            if transport is not None:
                transport.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(10)
    assert not errors
    try:
        yield port, handshake, reply_to_fetch, fetches
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not errors
