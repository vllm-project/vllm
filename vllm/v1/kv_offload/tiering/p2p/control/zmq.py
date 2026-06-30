# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ZMQ-based transport layer for P2P KV cache sharing.

Provides ZmqConnection (per-peer messaging) and ZmqTransport (connection
management). Message-content agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import msgspec
import zmq
import zmq.utils.monitor

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.control.base import (
    ControlConnection,
    ControlTransport,
)

logger = init_logger(__name__)

_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 10000
_HEARTBEAT_TTL_MS = 10000

# Shared sentinels returned when there is nothing to report.
_EMPTY_INBOX: tuple[dict, ...] = ()
_EMPTY_NEW_CONNECTIONS: tuple[ControlConnection, ...] = ()


def _tcp_addr(host: str, port: int | str) -> str:
    return f"tcp://{host}:{port}"


def _apply_heartbeat(sock: zmq.Socket) -> None:
    sock.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
    sock.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)


@dataclass
class _Sockets:
    dealer: zmq.Socket
    monitor: zmq.Socket


class ZmqConnection(ControlConnection):
    """Bidirectional message channel to a single remote peer."""

    def __init__(self, peer_id: str, sockets: _Sockets) -> None:
        super().__init__(peer_id)
        self._sockets = sockets
        self._closed = False
        self._inbox: list[dict] = []

    def send(self, msg: dict) -> None:
        """Send a msgpack-encoded message to this peer."""
        if self._closed:
            raise RuntimeError(
                f"ZmqConnection: send on closed connection to {self.peer_id}"
            )
        data = msgspec.msgpack.encode(msg)
        self._sockets.dealer.send(data)

    def recv(self) -> Sequence[dict]:
        """Drain and return all buffered incoming messages."""
        if not self._inbox:
            return _EMPTY_INBOX
        msgs = self._inbox
        self._inbox = []
        return msgs

    @property
    def alive(self) -> bool:
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("ZmqConnection: closing connection to %s", self.peer_id)
        self._sockets.monitor.close()
        self._sockets.dealer.close()

    def enqueue(self, msg: dict) -> None:
        """Buffer an incoming message."""
        self._inbox.append(msg)

    def mark_dead(self) -> None:
        """Mark connection as disconnected."""
        self._closed = True

    @property
    def monitor_socket(self) -> zmq.Socket:
        """Monitor socket for disconnect detection (used by ZmqTransport)."""
        return self._sockets.monitor


class ZmqTransport(ControlTransport):
    """ZMQ implementation of ControlTransport.

    Manages a ROUTER socket for accepting connections and DEALER sockets
    for outbound connections. Message-content agnostic.
    """

    def __init__(self, local_id: str, host: str, port: int) -> None:
        self._local_id = local_id
        self._closed = False

        self._connections: dict[str, ZmqConnection] = {}
        self._pending_inbound: list[tuple[str, dict]] = []

        self._zmq_ctx = zmq.Context()
        self._router: zmq.Socket = self._zmq_ctx.socket(zmq.ROUTER)
        _apply_heartbeat(self._router)
        bind_addr = _tcp_addr(host, port)
        self._router.bind(bind_addr)
        logger.info("ZmqTransport %s: ROUTER bound on %s", self._local_id, bind_addr)

    # ------------------------------------------------------------------
    # ZmqConnection lifecycle
    # ------------------------------------------------------------------

    def connect(self, peer_id: str) -> ZmqConnection:
        """Open an outbound connection to a remote peer."""
        assert peer_id not in self._connections, (
            f"ZmqConnection to {peer_id} already exists"
        )
        logger.info(
            "ZmqTransport %s: opening OUTBOUND connection to %s",
            self._local_id,
            peer_id,
        )
        return self._open_connection(peer_id, direction="outbound")

    def poll(self) -> Sequence[ControlConnection]:
        """Process all pending I/O. Returns newly accepted connections.

        - Receives messages (buffered in each connection's inbox)
        - Creates connections for new inbound peers (connect msg in inbox)
        - Checks monitors for disconnections
        - Removes and closes dead connections
        """
        self._recv_router()
        self._check_monitors()

        # Create connections for new inbound peers
        new_connections: list[ControlConnection] | None = None
        for sender_id, msg in self._pending_inbound:
            conn = self._connections.get(sender_id)
            if conn is None:
                logger.info(
                    "ZmqTransport %s: accepting INBOUND connection from %s",
                    self._local_id,
                    sender_id,
                )
                conn = self._open_connection(sender_id, direction="inbound")
                if new_connections is None:
                    new_connections = []
                new_connections.append(conn)
            conn.enqueue(msg)
        self._pending_inbound.clear()

        # Remove dead connections
        for pid in [p for p, c in self._connections.items() if not c.alive]:
            self._connections.pop(pid).close()

        return (
            new_connections if new_connections is not None else _EMPTY_NEW_CONNECTIONS
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

        self._router.setsockopt(zmq.LINGER, 0)
        self._router.close()
        self._zmq_ctx.destroy(linger=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_connection(
        self, peer_id: str, direction: str = "outbound"
    ) -> ZmqConnection:
        """Create a DEALER socket + monitor and register the connection."""
        host, port_str = peer_id.rsplit(":", 1)
        dealer_addr = _tcp_addr(host, port_str)

        logger.debug(
            "ZmqTransport %s: creating DEALER for %s peer %s -> %s",
            self._local_id,
            direction,
            peer_id,
            dealer_addr,
        )

        dealer = self._zmq_ctx.socket(zmq.DEALER)
        _apply_heartbeat(dealer)
        dealer.identity = self._local_id.encode()

        safe_id = peer_id.replace(":", "-").replace("/", "-")
        monitor_addr = f"inproc://p2p-monitor-{safe_id}"
        dealer.monitor(monitor_addr, zmq.EVENT_DISCONNECTED)

        monitor_sock = self._zmq_ctx.socket(zmq.PAIR)
        monitor_sock.connect(monitor_addr)

        dealer.connect(dealer_addr)

        sockets = _Sockets(dealer=dealer, monitor=monitor_sock)
        conn = ZmqConnection(peer_id, sockets)
        self._connections[peer_id] = conn
        logger.info(
            "ZmqTransport %s: %s connection established to %s (active connections: %d)",
            self._local_id,
            direction,
            peer_id,
            len(self._connections),
        )
        return conn

    def _recv_router(self) -> None:
        """Non-blocking: receive all pending messages from ROUTER."""
        while True:
            try:
                frames = self._router.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                logger.warning("ZmqTransport %s: recv error: %s", self._local_id, exc)
                break

            if len(frames) != 2:
                logger.warning(
                    "ZmqTransport %s: dropping message with %d frames (expected 2)",
                    self._local_id,
                    len(frames),
                )
                continue

            identity, data = frames
            sender_id = identity.decode()

            logger.debug(
                "ZmqTransport %s: ROUTER recv from %s (%d bytes)",
                self._local_id,
                sender_id,
                len(data),
            )

            try:
                msg = msgspec.msgpack.decode(data)
            except Exception as exc:
                logger.warning(
                    "ZmqTransport %s: failed to decode message from %s: %s",
                    self._local_id,
                    sender_id,
                    exc,
                )
                continue

            conn = self._connections.get(sender_id)

            if conn is not None:
                conn.enqueue(msg)
            else:
                self._pending_inbound.append((sender_id, msg))

    def _check_monitors(self) -> None:
        """Non-blocking: check all monitor sockets for disconnection."""
        for conn in self._connections.values():
            if not conn.alive:
                continue
            try:
                event = zmq.utils.monitor.recv_monitor_message(
                    conn.monitor_socket, zmq.NOBLOCK
                )
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                logger.warning(
                    "ZmqTransport %s: monitor error for peer %s: %s",
                    self._local_id,
                    conn.peer_id,
                    exc,
                )
                continue

            if event["event"] == zmq.EVENT_DISCONNECTED:
                logger.debug(
                    "ZmqTransport %s: peer %s disconnected",
                    self._local_id,
                    conn.peer_id,
                )
                conn.mark_dead()
