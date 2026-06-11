# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ-backed transport implementations.

Each transport owns the side-channel sockets and the per-peer NIXL agent
registration that shares their lifetime; the per-request transfers themselves
are driven by the producer/consumer delegates against the engine.
"""

import threading
from collections.abc import Callable
from typing import Any

import msgspec
import zmq
from zmq.utils.monitor import recv_monitor_message

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.metadata import (
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import ConsumerPeer, PeerAddr
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

# ZMTP heartbeat for consumer DEALERs: detect dead producers without a ping thread.
# TTL piggybacked in outgoing PINGs lets the producer ROUTER clean up symmetrically.
_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 4000
_HEARTBEAT_TTL_MS = 8000  # 2 × TIMEOUT


class ZmqProducerTransport:
    """ROUTER socket + router thread serving the producer side of the channel.

    The router thread decodes each incoming `XferReq`, hands it to `on_xfer_req`
    for a grant/NACK decision, and sends the returned `XferAck` straight back to
    the requesting identity — the reply is synchronous. Every loop it also calls
    `poll` so the producer can release source pins for completed or timed-out
    reads. The poll timeout is short (5 ms) while reads are pinned, 1000 ms when
    idle.
    """

    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        self._ctx = zmq.Context()
        self._router = make_zmq_socket(
            ctx=self._ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self._stop_event = threading.Event()
        self._router_t: threading.Thread | None = None
        self._on_xfer_req: Callable[[bytes, XferReq], XferAck] | None = None
        self._poll: Callable[[], None] | None = None
        self._has_pending: Callable[[], bool] | None = None
        self._xfer_req_decoder = msgspec.msgpack.Decoder(XferReq)
        self._encoder = msgspec.msgpack.Encoder()

    def start(
        self,
        on_xfer_req: Callable[[bytes, XferReq], XferAck],
        poll: Callable[[], None],
        has_pending: Callable[[], bool],
    ) -> None:
        self._on_xfer_req = on_xfer_req
        self._poll = poll
        self._has_pending = has_pending
        self._router_t = threading.Thread(
            target=self._run, name="ec-nixl-router", daemon=True
        )
        self._router_t.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._router_t is not None:
            self._router_t.join(timeout=5)
        try:
            self._router.close(linger=0)
        except Exception:
            logger.debug("ec: router close failed", exc_info=True)
        try:
            self._ctx.destroy(linger=0)
        except Exception:
            logger.debug("ec: zmq context destroy failed", exc_info=True)

    def _run(self) -> None:
        assert self._on_xfer_req is not None
        assert self._poll is not None
        assert self._has_pending is not None
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while not self._stop_event.is_set():
            timeout_ms = 5 if self._has_pending() else 1000
            try:
                events = dict(poller.poll(timeout=timeout_ms))
            except zmq.ContextTerminated:
                return
            except Exception:
                logger.exception("ec: router poll failed")
                continue
            if self._router in events:
                self._serve_one()
            try:
                self._poll()
            except Exception:
                logger.exception("ec: producer poll failed")

    def _serve_one(self) -> None:
        """Receive one `XferReq` and send back the delegate's `XferAck`."""
        assert self._on_xfer_req is not None
        try:
            identity, _, payload = self._router.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except zmq.ContextTerminated:
            raise
        except Exception:
            logger.exception("ec: router recv failed")
            return
        try:
            req = self._xfer_req_decoder.decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("ec: dropped malformed XferReq")
            return
        # Convert an escaped handler error into a NACK so the consumer fails
        # over to local encode immediately instead of waiting out its timeout.
        try:
            ack = self._on_xfer_req(identity, req)
        except Exception:
            logger.exception(
                "ec: handler failed for XferReq mm_hash=%s; NACKing", req.mm_hash
            )
            ack = XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_INTERNAL)
        try:
            self._router.send_multipart([identity, b"", self._encoder.encode(ack)])
        except Exception:
            logger.exception("ec: failed to send XferAck mm_hash=%s", req.mm_hash)


class ZmqConsumerTransport:
    """Lazy DEALER pool plus the NIXL agent registration for each producer peer.

    A peer's ZMQ DEALER and its registered NIXL agent share a lifetime: the
    DEALER is created on first contact (`ensure_dealer`) so an `XferReq` can be
    sent before any metadata exists; the NIXL agent is added once an `XferAck`
    supplies fresh metadata (`register_source`); both are released together in
    `evict_peer`.
    """

    def __init__(self, engine: Any) -> None:
        self._ctx = zmq.Context()
        self._engine = engine
        self._peer_pool: dict[PeerAddr, ConsumerPeer] = {}
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_ack_decoder = msgspec.msgpack.Decoder(XferAck)

    def ensure_dealer(self, addr: PeerAddr) -> ConsumerPeer:
        """Return the peer for `addr`, creating its DEALER + monitor on first
        contact. Does not touch NIXL — registration happens in
        `register_source` once metadata arrives.
        """
        existing = self._peer_pool.get(addr)
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
        mon = self._ctx.socket(zmq.PAIR)
        mon.connect(monitor_addr)

        peer = ConsumerPeer(zmq_dealer=dealer, zmq_monitor=mon)
        self._peer_pool[addr] = peer
        return peer

    def send_xfer_req(self, peer: ConsumerPeer, req: XferReq) -> None:
        peer.zmq_dealer.send_multipart([b"", self._encoder.encode(req)])

    def register_source(
        self, addr: PeerAddr, metadata: bytes, mem_descriptor: bytes
    ) -> ConsumerPeer:
        """Register the producer as a NIXL read source and prep its read dlist.

        The metadata comes fresh from an `XferAck`, so the rkeys it carries
        always belong to the live producer process. If the peer is already
        registered with the same metadata the existing handle is reused; if the
        metadata differs (the producer restarted at this address) the old agent
        is torn down and replaced.
        """
        peer = self._peer_pool[addr]
        if peer.remote_read_handle is not None and peer.nixl_metadata_bytes == metadata:
            logger.debug(
                "ec: reusing registered source %s:%d agent=%s",
                addr[0],
                addr[1],
                peer.nixl_agent_name,
            )
            return peer
        if peer.remote_read_handle is not None:
            logger.info(
                "ec: producer %s:%d metadata changed; re-registering",
                addr[0],
                addr[1],
            )
            if peer.nixl_agent_name is not None:
                self._engine.remove_remote_agent(peer.nixl_agent_name)
            peer.nixl_agent_name = None
            peer.nixl_metadata_bytes = None
            peer.remote_read_handle = None

        agent_name, read_handle = self._engine.add_remote_source(
            metadata, mem_descriptor
        )
        peer.nixl_agent_name = agent_name
        peer.nixl_metadata_bytes = metadata
        peer.remote_read_handle = read_handle
        logger.debug(
            "ec: registered read source %s:%d agent=%s",
            addr[0],
            addr[1],
            agent_name,
        )
        return peer

    def poll_responses(self) -> list[tuple[PeerAddr, XferAck]]:
        """Non-blocking drain of `XferAck`s from every peer's DEALER."""
        acks: list[tuple[PeerAddr, XferAck]] = []
        for addr, peer in list(self._peer_pool.items()):
            while True:
                try:
                    frames = peer.zmq_dealer.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception:
                    logger.exception("ec: DEALER recv failed")
                    break
                if len(frames) != 2 or frames[0] != b"":
                    logger.warning(
                        "ec: dropped malformed XferAck envelope "
                        "(expected [b'', payload], got %d frames)",
                        len(frames),
                    )
                    continue
                try:
                    acks.append((addr, self._xfer_ack_decoder.decode(frames[1])))
                except (msgspec.DecodeError, msgspec.ValidationError):
                    logger.warning("ec: dropped malformed XferAck")
        return acks

    def poll_dead_peers(self) -> list[PeerAddr]:
        """Non-blocking sweep of peer monitor sockets.

        Returns keys of peers whose DEALER reported EVENT_DISCONNECTED
        (heartbeat timeout fired). Each key appears at most once.
        """
        dead: list[PeerAddr] = []
        for key, peer in list(self._peer_pool.items()):
            if peer.zmq_monitor is None:
                continue
            try:
                while True:
                    evt = recv_monitor_message(peer.zmq_monitor, flags=zmq.NOBLOCK)
                    if evt["event"] == zmq.EVENT_DISCONNECTED:
                        dead.append(key)
                        break
            except zmq.Again:
                pass
            except Exception:
                logger.warning("ec: monitor poll failed for key=%s", key, exc_info=True)
        return dead

    def evict_peer(self, key: PeerAddr) -> str | None:
        """Tear down a peer: close monitor, remove NIXL agent, close DEALER.

        Returns the evicted nixl_agent_name, or None if key was not present.
        """
        entry = self._peer_pool.pop(key, None)
        if entry is None:
            return None
        self._teardown_peer(entry)
        return entry.nixl_agent_name

    def _teardown_peer(self, entry: ConsumerPeer) -> None:
        if entry.zmq_monitor is not None:
            try:
                entry.zmq_dealer.disable_monitor()
            except Exception:
                logger.warning("ec: disable_monitor failed", exc_info=True)
            try:
                entry.zmq_monitor.close(linger=0)
            except Exception:
                logger.warning("ec: close monitor failed", exc_info=True)
        if entry.nixl_agent_name is not None:
            try:
                self._engine.remove_remote_agent(entry.nixl_agent_name)
            except Exception:
                logger.warning(
                    "ec: remove_remote_agent failed for %s",
                    entry.nixl_agent_name,
                    exc_info=True,
                )
        try:
            entry.zmq_dealer.close(linger=0)
        except Exception:
            logger.warning("ec: close DEALER failed", exc_info=True)

    def shutdown(self) -> None:
        """Destroy the ZMQ context. Call only after all sockets are closed."""
        try:
            self._ctx.destroy(linger=0)
        except Exception:
            logger.debug("ec: zmq context destroy failed", exc_info=True)
