# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.v1.kv_offload.tiering.p2p.control.zmq."""

from __future__ import annotations

import socket
import time

import pytest
import zmq

from vllm.v1.kv_offload.base import (
    get_offload_block_hash,
    get_offload_chunk_idx,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.p2p.control.zmq import (
    ZmqConnection,
    ZmqTransport,
    _Sockets,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import FetchMsg


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_transport(host: str = "127.0.0.1", attempts: int = 8):
    """Construct a ZmqTransport on a fresh port, retrying on bind collisions.

    Why: _free_port() releases the probe socket before ZmqTransport binds the
    same port — a parallel test run can steal it in between. Retrying on
    ZMQError/OSError closes that race without a production change.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        port = _free_port()
        try:
            return ZmqTransport(f"{host}:{port}", host, port), port
        except (zmq.ZMQError, OSError) as e:
            last_err = e
    assert last_err is not None
    raise last_err


def _wait_for_inbound(transport: ZmqTransport, deadline: float = 2.0):
    """Poll until at least one new inbound connection is accepted, or fail."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        new = transport.poll()
        if new:
            return new
        time.sleep(0.005)
    raise AssertionError(f"no inbound connection within {deadline}s")


def _wait_for_messages(
    transport: ZmqTransport,
    conn: ZmqConnection,
    n: int,
    deadline: float = 2.0,
) -> list[dict]:
    """Poll until `conn` has received at least `n` messages, then return them."""
    end = time.monotonic() + deadline
    msgs: list[dict] = []
    while time.monotonic() < end:
        transport.poll()
        msgs.extend(conn.recv())
        if len(msgs) >= n:
            return msgs
        time.sleep(0.005)
    raise AssertionError(f"got {len(msgs)}/{n} messages within {deadline}s")


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
        assert list(msgs) == [{"type": "a"}, {"type": "b"}]
        # Second recv is empty
        assert not conn.recv()

    def test_recv_returns_empty_initially(self):
        conn = _make_mock_connection()
        assert not conn.recv()

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
        transport_a, port_a = _make_transport()
        transport_b, port_b = _make_transport()

        try:
            peer_a_id = f"127.0.0.1:{port_a}"
            conn_b_to_a = transport_b.connect(peer_a_id)
            conn_b_to_a.send({"type": "hello", "data": 42})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            conn_a_from_b = new_conns[0]
            assert conn_a_from_b.peer_id == f"127.0.0.1:{port_b}"

            msgs = _wait_for_messages(transport_a, conn_a_from_b, 1)
            assert msgs == [{"type": "hello", "data": 42}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_offload_key_bytes_survive_transport(self):
        """P2P carries OffloadKeys as opaque bytes. A full key with a
        non-zero chunk_idx must round-trip unchanged over the wire and still
        decode to the same (hash, group, chunk) -- so peers must share the
        key format (lockstep deployment)."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            block_hash = bytes(range(8))
            key = make_offload_key(block_hash, group_idx=2, chunk_idx=7)

            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({FetchMsg.KEYS: [key], FetchMsg.BLOCK_INDEXES: [0]})

            new_conns = _wait_for_inbound(transport_a)
            conn_a = new_conns[0]
            msgs = _wait_for_messages(transport_a, conn_a, 1)

            received = msgs[0][FetchMsg.KEYS][0]
            assert received == key
            assert get_offload_block_hash(received) == block_hash
            assert get_offload_group_idx(received) == 2
            assert get_offload_chunk_idx(received) == 7
        finally:
            transport_a.close()
            transport_b.close()

    def test_bidirectional_messaging(self):
        """Both sides can send and receive after connection."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "from": "b"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            conn_a.send({"type": "reply", "from": "a"})

            msgs = _wait_for_messages(transport_b, conn_b, 1)
            assert msgs == [{"type": "reply", "from": "a"}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_poll_returns_empty_when_no_connections(self):
        transport, _ = _make_transport()
        try:
            assert not transport.poll()
        finally:
            transport.close()

    def test_multiple_messages(self):
        """Multiple messages are buffered and returned together."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"seq": 1})
            conn_b.send({"seq": 2})
            conn_b.send({"seq": 3})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            msgs = _wait_for_messages(transport_a, conn_a, 3)
            assert [m["seq"] for m in msgs] == [1, 2, 3]
        finally:
            transport_a.close()
            transport_b.close()

    def test_duplicate_connect_asserts(self):
        """Connecting to the same peer twice raises AssertionError."""
        # port_a is never bound — we just need a syntactically-valid peer id.
        port_a = _free_port()
        transport_b, _ = _make_transport()
        try:
            transport_b.connect(f"127.0.0.1:{port_a}")
            with pytest.raises(AssertionError, match="already exists"):
                transport_b.connect(f"127.0.0.1:{port_a}")
        finally:
            transport_b.close()

    def test_dead_connection_removed_on_poll(self):
        """Dead connections are cleaned up during poll."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "hello"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            # Mark the inbound connection dead manually.
            new_conns[0].mark_dead()

            # Pruning is synchronous within poll().
            transport_a.poll()
            assert len(transport_a._connections) == 0
            assert new_conns[0]._sockets.dealer.closed
        finally:
            transport_a.close()
            transport_b.close()

    def test_close_is_idempotent(self):
        """Calling close() twice doesn't raise."""
        transport, _ = _make_transport()
        transport.close()
        transport.close()  # should not raise


class TestZmqReconnect:
    """Reconnecting to a peer whose connection died (real ZMQ sockets).

    A session marks its connection dead while handling messages, which happens
    after the transport's own sweep has run for that tick — so a dead
    connection stays registered until the next poll(). Reconnecting in that
    window must succeed, and the retired connection must release its sockets.
    Real sockets are required: a mock reports every attribute as closed.
    """

    def test_close_after_mark_dead_releases_sockets(self):
        """close() releases sockets even when mark_dead() ran first.

        mark_dead() must not set the flag close() guards on, or every peer
        disconnect leaks a DEALER and a monitor socket.
        """
        transport, _ = _make_transport()
        try:
            conn = transport.connect(f"127.0.0.1:{_free_port()}")
            dealer, monitor = conn._sockets.dealer, conn._sockets.monitor

            conn.mark_dead()
            assert not conn.alive
            assert not dealer.closed

            conn.close()
            assert dealer.closed
            assert monitor.closed
        finally:
            transport.close()

    def test_connect_retires_dead_connection(self):
        """connect() replaces a registered-but-dead connection."""
        transport, _ = _make_transport()
        try:
            # The peer port is never bound — only the peer id matters here.
            peer_id = f"127.0.0.1:{_free_port()}"
            dead = transport.connect(peer_id)
            dead.mark_dead()

            conn = transport.connect(peer_id)

            assert conn is not dead
            assert conn.alive
            assert transport._connections[peer_id] is conn
            assert dead._sockets.dealer.closed
        finally:
            transport.close()

    def test_repeated_reconnect_to_same_peer(self):
        """A flapping peer stays reconnectable.

        Monitor endpoints are inproc addresses that libzmq releases
        asynchronously, so deriving one from peer_id alone makes each
        reconnect race the previous teardown and fail with EADDRINUSE.
        """
        transport, _ = _make_transport()
        try:
            peer_id = f"127.0.0.1:{_free_port()}"
            for _ in range(10):
                conn = transport.connect(peer_id)
                conn.mark_dead()
                transport.poll()
                assert conn._sockets.dealer.closed

            assert not transport._connections
        finally:
            transport.close()

    def test_inbound_message_survives_dead_registration(self):
        """A reconnecting peer's first message is not dropped.

        poll() must retire connections killed by their session before routing
        traffic, otherwise the message is enqueued into the dead connection and
        discarded when it is swept — and a session announces itself only once.
        Covers only that between-polls window: a peer dying while poll() runs,
        or dying silently until the heartbeat expires, is out of scope.
        """
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "seq": 1})

            inbound = _wait_for_inbound(transport_a)[0]
            _wait_for_messages(transport_a, inbound, 1)

            inbound.mark_dead()
            conn_b.send({"type": "connect", "seq": 2})

            # Wait until the frame is readable on the ROUTER, so the message is
            # known to have arrived rather than merely being slow.
            poller = zmq.Poller()
            poller.register(transport_a._router, zmq.POLLIN)
            assert poller.poll(2000), "message never reached the ROUTER"

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            assert new_conns[0] is not inbound
            msgs = _wait_for_messages(transport_a, new_conns[0], 1)
            assert msgs == [{"type": "connect", "seq": 2}]
        finally:
            transport_a.close()
            transport_b.close()
