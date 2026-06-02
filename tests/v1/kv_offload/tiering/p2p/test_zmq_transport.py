# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.v1.kv_offload.tiering.p2p.control.zmq."""

from __future__ import annotations

import socket
import time

import pytest

from vllm.v1.kv_offload.tiering.p2p.control.zmq import (
    ZmqConnection,
    ZmqTransport,
    _Sockets,
)


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_mock_connection(peer_id: str = "test:1234") -> ZmqConnection:
    """Create a ZmqConnection with mock sockets for unit testing."""
    from unittest.mock import MagicMock

    sockets = _Sockets(dealer=MagicMock(), monitor=MagicMock())
    return ZmqConnection(peer_id, sockets)


class TestZmqConnection:
    """Tests for ZmqConnection in isolation (no real sockets)."""

    def test_enqueue_and_recv(self):
        """Messages enqueued are returned by recv() in order."""
        conn = _make_mock_connection()

        conn.enqueue({"type": "a"})
        conn.enqueue({"type": "b"})

        msgs = conn.recv()
        assert msgs == [{"type": "a"}, {"type": "b"}]
        # Second recv is empty
        assert conn.recv() == []

    def test_recv_returns_empty_initially(self):
        conn = _make_mock_connection()
        assert conn.recv() == []

    def test_alive_initially_true(self):
        conn = _make_mock_connection()
        assert conn.alive is True

    def test_mark_dead(self):
        conn = _make_mock_connection()
        conn.mark_dead()
        assert conn.alive is False

    def test_send_raises_when_closed(self):
        conn = _make_mock_connection()
        conn.mark_dead()

        with pytest.raises(RuntimeError, match="closed connection"):
            conn.send({"type": "test"})


class TestZmqTransportConnectivity:
    """Integration tests for ZmqTransport with real ZMQ sockets."""

    def test_connect_and_send_message(self):
        """Two transports can connect and exchange messages."""
        port_a = _free_port()
        port_b = _free_port()

        transport_a = ZmqTransport("127.0.0.1:" + str(port_a), "127.0.0.1", port_a)
        transport_b = ZmqTransport("127.0.0.1:" + str(port_b), "127.0.0.1", port_b)

        try:
            # B connects to A
            peer_a_id = f"127.0.0.1:{port_a}"
            conn_b_to_a = transport_b.connect(peer_a_id)

            # Send a message from B to A
            conn_b_to_a.send({"type": "hello", "data": 42})

            # Give ZMQ time to deliver
            time.sleep(0.05)

            # A polls — should get a new inbound connection with the message
            new_conns = transport_a.poll()
            assert len(new_conns) == 1

            conn_a_from_b = new_conns[0]
            assert conn_a_from_b.peer_id == f"127.0.0.1:{port_b}"

            msgs = conn_a_from_b.recv()
            assert len(msgs) == 1
            assert msgs[0] == {"type": "hello", "data": 42}
        finally:
            transport_a.close()
            transport_b.close()

    def test_bidirectional_messaging(self):
        """Both sides can send and receive after connection."""
        port_a = _free_port()
        port_b = _free_port()

        transport_a = ZmqTransport(f"127.0.0.1:{port_a}", "127.0.0.1", port_a)
        transport_b = ZmqTransport(f"127.0.0.1:{port_b}", "127.0.0.1", port_b)

        try:
            # B connects to A
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "from": "b"})

            time.sleep(0.05)

            # A accepts
            new_conns = transport_a.poll()
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            # A sends reply
            conn_a.send({"type": "reply", "from": "a"})

            time.sleep(0.05)

            # B receives via poll (messages go to inbox)
            transport_b.poll()
            msgs = conn_b.recv()
            assert len(msgs) == 1
            assert msgs[0] == {"type": "reply", "from": "a"}
        finally:
            transport_a.close()
            transport_b.close()

    def test_poll_returns_empty_when_no_connections(self):
        port = _free_port()
        transport = ZmqTransport(f"127.0.0.1:{port}", "127.0.0.1", port)
        try:
            assert transport.poll() == []
        finally:
            transport.close()

    def test_multiple_messages(self):
        """Multiple messages are buffered and returned together."""
        port_a = _free_port()
        port_b = _free_port()

        transport_a = ZmqTransport(f"127.0.0.1:{port_a}", "127.0.0.1", port_a)
        transport_b = ZmqTransport(f"127.0.0.1:{port_b}", "127.0.0.1", port_b)

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"seq": 1})
            conn_b.send({"seq": 2})
            conn_b.send({"seq": 3})

            time.sleep(0.05)

            new_conns = transport_a.poll()
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            msgs = conn_a.recv()
            assert len(msgs) == 3
            assert [m["seq"] for m in msgs] == [1, 2, 3]
        finally:
            transport_a.close()
            transport_b.close()

    def test_duplicate_connect_asserts(self):
        """Connecting to the same peer twice raises AssertionError."""
        port_a = _free_port()
        port_b = _free_port()

        transport_b = ZmqTransport(f"127.0.0.1:{port_b}", "127.0.0.1", port_b)
        try:
            transport_b.connect(f"127.0.0.1:{port_a}")
            with pytest.raises(AssertionError, match="already exists"):
                transport_b.connect(f"127.0.0.1:{port_a}")
        finally:
            transport_b.close()

    def test_dead_connection_removed_on_poll(self):
        """Dead connections are cleaned up during poll."""
        port_a = _free_port()
        port_b = _free_port()

        transport_a = ZmqTransport(f"127.0.0.1:{port_a}", "127.0.0.1", port_a)
        transport_b = ZmqTransport(f"127.0.0.1:{port_b}", "127.0.0.1", port_b)

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "hello"})

            time.sleep(0.05)
            new_conns = transport_a.poll()
            assert len(new_conns) == 1

            # Mark connection dead manually
            new_conns[0].mark_dead()

            # Next poll should clean it up
            transport_a.poll()
            assert len(transport_a._connections) == 0
        finally:
            transport_a.close()
            transport_b.close()

    def test_close_is_idempotent(self):
        """Calling close() twice doesn't raise."""
        port = _free_port()
        transport = ZmqTransport(f"127.0.0.1:{port}", "127.0.0.1", port)
        transport.close()
        transport.close()  # should not raise
