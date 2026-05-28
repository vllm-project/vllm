# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ-backed transport implementations."""

import threading
from collections.abc import Callable
from typing import Any

import msgspec
import zmq
from zmq.utils.monitor import recv_monitor_message

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import XferAck, XferReq
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
    """ROUTER socket + router thread for producer-side ZMQ communication.

    The router thread mirrors the original _run_router loop:
    - Polls ROUTER for incoming XferReqs, decoded and forwarded to
      `on_xfer_req` callback.
    - Calls `sweep` each iteration to poll NIXL completions and get ack
      routes, then sends XferAcks back via ROUTER directly.
    - Uses a dynamic poll timeout: 5 ms when transfers are in flight,
      1000 ms when idle.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        host: str,
        port: int,
    ) -> None:
        self._router = make_zmq_socket(
            ctx=ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self._stop_event = threading.Event()
        self._router_t: threading.Thread | None = None
        self._on_xfer_req: Callable[[bytes, XferReq], None] | None = None
        self._sweep: Callable[[], list[tuple[bytes, str, bool]]] | None = None
        self._has_in_flight: Callable[[], bool] | None = None
        self._xfer_req_decoder = msgspec.msgpack.Decoder(XferReq)
        self._encoder = msgspec.msgpack.Encoder()

    def start(
        self,
        on_xfer_req: Callable[[bytes, XferReq], None],
        sweep: Callable[[], list[tuple[bytes, str, bool]]],
        has_in_flight: Callable[[], bool],
    ) -> None:
        self._on_xfer_req = on_xfer_req
        self._sweep = sweep
        self._has_in_flight = has_in_flight
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

    def _run(self) -> None:
        assert self._on_xfer_req is not None
        assert self._sweep is not None
        assert self._has_in_flight is not None
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while not self._stop_event.is_set():
            timeout_ms = 5 if self._has_in_flight() else 1000
            try:
                events = dict(poller.poll(timeout=timeout_ms))
            except zmq.ContextTerminated:
                return
            except Exception:
                logger.exception("ec: router poll failed")
                continue
            if self._router in events:
                try:
                    identity, _, payload = self._router.recv_multipart(
                        flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    pass
                except zmq.ContextTerminated:
                    return
                except Exception:
                    logger.exception("ec: router recv failed")
                else:
                    try:
                        req = self._xfer_req_decoder.decode(payload)
                    except (msgspec.DecodeError, msgspec.ValidationError):
                        logger.warning("ec: dropped malformed XferReq")
                    else:
                        self._on_xfer_req(identity, req)
            try:
                routes = self._sweep()
                self._send_xfer_acks(routes)
            except Exception:
                logger.exception("ec: sweep_completions failed")

    def _send_xfer_acks(self, routes: list[tuple[bytes, str, bool]]) -> None:
        for identity, mm_hash, ok in routes:
            try:
                payload = self._encoder.encode(XferAck(mm_hash=mm_hash, ok=ok))
                self._router.send_multipart([identity, b"", payload])
            except Exception:
                logger.exception(
                    "ec: failed to send XferAck mm_hash=%s ok=%s", mm_hash, ok
                )


class ZmqConsumerTransport:
    """Lazy DEALER pool for consumer-side ZMQ communication."""

    def __init__(self, ctx: zmq.Context, engine: Any) -> None:
        self._ctx = ctx
        self._engine = engine
        self._peer_pool: dict[PeerAddr, ConsumerPeer] = {}
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_ack_decoder = msgspec.msgpack.Decoder(XferAck)

    def get_or_create_peer(self, addr: PeerAddr, metadata: bytes) -> ConsumerPeer:
        host, port = addr
        key = addr
        existing = self._peer_pool.get(key)
        if existing is not None and existing.nixl_metadata_bytes == metadata:
            return existing
        if existing is not None:
            self._teardown_peer(existing)
            self._peer_pool.pop(key, None)

        for stale_key in [k for k in self._peer_pool if k[1] == port and k[0] != host]:
            logger.info(
                "ec: evicting stale peer %s (same port %d, new peer %s)",
                stale_key,
                port,
                host,
            )
            self.evict_peer(stale_key)

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

        agent_name = self._engine.add_remote_agent(metadata)
        entry = ConsumerPeer(
            zmq_dealer=dealer,
            nixl_agent_name=agent_name,
            nixl_metadata_bytes=metadata,
            zmq_monitor=mon,
        )
        self._peer_pool[key] = entry
        return entry

    def send_xfer_req(self, peer: ConsumerPeer, req: XferReq) -> None:
        peer.zmq_dealer.send_multipart([b"", self._encoder.encode(req)])

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
        """Tear down a peer: close monitor, close DEALER, remove NIXL agent.

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

    def drain_acks(self) -> list[XferAck]:
        acks: list[XferAck] = []
        for peer in self._peer_pool.values():
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
                    acks.append(self._xfer_ack_decoder.decode(frames[1]))
                except (msgspec.DecodeError, msgspec.ValidationError):
                    logger.warning("ec: dropped malformed XferAck")
        return acks
