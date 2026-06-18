# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ control-plane implementations for ECCPUConnector.

ZmqClientConnection  — one DEALER socket to a single producer peer.
ZmqClientTransport   — pool of ZmqClientConnection objects (consumer side).
ZmqServerTransport   — ROUTER socket + background thread (producer side).
"""

import contextlib
from dataclasses import dataclass, field

import zmq
from zmq.utils.monitor import recv_monitor_message

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.base import (
    ControlConnection,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 4000
_HEARTBEAT_TTL_MS = 8000


# ── Client side ───────────────────────────────────────────────────────────────


@dataclass
class ZmqClientConnection(ControlConnection):
    """One DEALER socket to one producer peer.

    Owns the socket and its optional monitor socket. NIXL state is NOT
    stored here — it belongs in ECSession above this layer.
    """

    dealer: zmq.Socket
    monitor: zmq.Socket | None
    _alive: bool = field(default=True, init=False)

    @property
    def alive(self) -> bool:
        return self._alive

    def send(self, msg: bytes) -> None:
        self.dealer.send_multipart([b"", msg])

    def recv(self) -> list[bytes]:
        msgs: list[bytes] = []
        while True:
            try:
                frames = self.dealer.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                logger.exception("ec: DEALER recv failed")
                break
            if len(frames) != 2 or frames[0] != b"":
                logger.warning(
                    "ec: dropped malformed frame envelope (%d frames)", len(frames)
                )
                continue
            msgs.append(frames[1])
        return msgs

    def mark_dead(self) -> None:
        self._alive = False

    def close(self) -> None:
        if self.monitor is not None:
            with contextlib.suppress(Exception):
                self.dealer.disable_monitor()
            with contextlib.suppress(Exception):
                self.monitor.close(linger=0)
        with contextlib.suppress(Exception):
            self.dealer.close(linger=0)
        self._alive = False


class ZmqClientTransport:
    """Pool of ZmqClientConnection objects — one per producer peer.

    Creates DEALER sockets lazily on first contact. Does not touch NIXL;
    NIXL registration is handled by ECCPUConsumer once an XferAck arrives.
    """

    def __init__(self) -> None:
        self._ctx = zmq.Context()
        self._connections: dict[PeerAddr, ZmqClientConnection] = {}

    def connect(self, addr: PeerAddr) -> ZmqClientConnection:
        """Return the existing connection for addr, or create a new one."""
        existing = self._connections.get(addr)
        if existing is not None:
            return existing

        host, port = addr
        dealer = make_zmq_socket(
            ctx=self._ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.DEALER,
            bind=False,
        )
        dealer.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
        dealer.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
        dealer.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)

        monitor_addr = f"inproc://ec-peer-mon-{host}-{port}"
        dealer.monitor(monitor_addr, zmq.EVENT_DISCONNECTED)
        monitor = self._ctx.socket(zmq.PAIR)
        monitor.connect(monitor_addr)

        conn = ZmqClientConnection(dealer=dealer, monitor=monitor)
        self._connections[addr] = conn
        return conn

    def poll(self) -> dict[PeerAddr, list[bytes]]:
        """Non-blocking drain of messages from every peer's DEALER."""
        msgs: dict[PeerAddr, list[bytes]] = {}
        for addr, conn in list(self._connections.items()):
            conn_msgs = conn.recv()
            msgs[addr] = conn_msgs
        return msgs

    def poll_dead(self) -> list[PeerAddr]:
        """Non-blocking sweep of peer monitors; return addresses that died."""
        dead: list[PeerAddr] = []
        for addr, conn in list(self._connections.items()):
            if conn.monitor is None:
                continue
            try:
                while True:
                    evt = recv_monitor_message(conn.monitor, flags=zmq.NOBLOCK)
                    if evt["event"] == zmq.EVENT_DISCONNECTED:
                        conn.mark_dead()
                        dead.append(addr)
                        break
            except zmq.Again:
                pass
            except Exception:
                logger.warning("ec: monitor poll failed for %s", addr, exc_info=True)
        return dead

    def remove(self, addr: PeerAddr) -> ZmqClientConnection | None:
        """Remove and close the connection for addr, returning it (or None)."""
        conn = self._connections.pop(addr, None)
        if conn is not None:
            conn.close()
        return conn

    def close(self) -> None:
        for conn in list(self._connections.values()):
            conn.close()
        self._connections.clear()
        try:
            self._ctx.destroy(linger=0)
        except Exception:
            logger.debug("ec: zmq context destroy failed", exc_info=True)


# ── Server side ───────────────────────────────────────────────────────────────


class ZmqServerTransport:
    """ROUTER socket for the producer side. Byte-level only — no decoding,
    no callbacks, no thread. ProducerSession owns the background thread and
    drives this transport from it.

    poll(timeout_ms) blocks for up to timeout_ms waiting for messages, then
    drains all that are available. send() routes a reply back to a specific
    peer by ZMQ identity.
    """

    def __init__(self, host: str, port: int) -> None:
        self._ctx = zmq.Context()
        self._router = make_zmq_socket(
            ctx=self._ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self._poller = zmq.Poller()
        self._poller.register(self._router, zmq.POLLIN)

    def poll(self, timeout_ms: int = 0) -> list[tuple[bytes, bytes]]:
        """Wait up to timeout_ms for incoming messages, then drain all pending.

        Returns a list of (identity, payload) pairs.
        """
        if timeout_ms > 0:
            try:
                events = dict(self._poller.poll(timeout=timeout_ms))
            except zmq.ContextTerminated:
                return []
            except Exception:
                logger.exception("ec: router poller failed")
                return []
            if self._router not in events:
                return []

        msgs: list[tuple[bytes, bytes]] = []
        while True:
            try:
                identity, _, payload = self._router.recv_multipart(flags=zmq.NOBLOCK)
                msgs.append((identity, payload))
            except zmq.Again:
                break
            except zmq.ContextTerminated:
                break
            except Exception:
                logger.exception("ec: router recv failed")
                break
        return msgs

    def send(self, identity: bytes, payload: bytes) -> None:
        """Send a reply payload to the peer identified by ZMQ identity."""
        try:
            self._router.send_multipart([identity, b"", payload])
        except Exception:
            logger.exception("ec: router send failed")

    def close(self) -> None:
        try:
            self._router.close(linger=0)
        except Exception:
            logger.debug("ec: router close failed", exc_info=True)
        try:
            self._ctx.destroy(linger=0)
        except Exception:
            logger.debug("ec: zmq context destroy failed", exc_info=True)
