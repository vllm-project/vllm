# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session objects for the ECCPUConnector.

Layer hierarchy:  connector → session → transport (connection)

Two long-lived sessions:

  ProducerSession — owns ZmqServerTransport and a background thread.
      The thread polls the transport for raw bytes, then calls poll() with
      those messages. poll() decodes XferReqs, grants or NACKs them, pins
      source blocks, sends XferAck replies, drains NIXL completion
      notifications, and sweeps expired xfers.

  ConsumerSession — owns one ZmqClientConnection and the NIXL agent
      registration for that peer. The connector calls poll(messages, now)
      once per engine step with the raw bytes collected for this peer.
      poll() decodes XferAcks, dispatches them to ConsumerXfer objects,
      and advances all active xfer state machines.

Internal data classes:

  ProducerXfer — one pinned read grant, keyed by session_id.
  ConsumerXfer — state machine for one NIXL READ, keyed by mm_hash.
"""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq import (
    ZmqClientConnection,
    ZmqClientTransport,
    ZmqServerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.protocol import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
        ECSharedRegion,
    )

logger = init_logger(__name__)

PRODUCER_PIN_LEASE_S = 30.0
_CONSUMER_READ_TIMEOUT_S = 20.0
_CONSUMER_ACK_TIMEOUT_S = 2.0


# ── XferState ─────────────────────────────────────────────────────────────────


class XferState(enum.Enum):
    """Lifecycle states of a single ConsumerXfer."""

    WAITING_ACK = "waiting_ack"  # XferReq sent, awaiting XferAck
    READING = "reading"  # NIXL READ in flight
    DONE = "done"  # READ completed; promote to local cache
    ACK_TIMEOUT = "ack_timeout"  # XferAck never arrived; free blocks + tombstone
    READ_FAILED = "read_failed"  # Unexpected NIXL state; free blocks + tombstone
    QUARANTINED = "quarantined"  # Read timed out, NIXL unabortable; keep blocks
    SETTLED = "settled"  # Quarantined DMA done; safe to free blocks


# ── ProducerXfer ──────────────────────────────────────────────────────────────


@dataclass
class ProducerXfer:
    """One pinned read grant on the producer side.

    Pure data — lifecycle operations (pin/unpin) are performed by ProducerSession.
    Keyed by (consumer_session_id, mm_hash) in ProducerSession._active_xfers.
    """

    mm_hash: str
    block_indices: list[int]
    deadline: float

    def is_expired(self) -> bool:
        return time.monotonic() > self.deadline


# ── ConsumerXfer ──────────────────────────────────────────────────────────────


@dataclass
class ConsumerXfer:
    """State machine for one NIXL READ on the consumer side.

    transfer_handle is the in-flight NIXL handle returned by post_read() —
    used for check_xfer_state() / release_xfer_handle(). The dlist handle
    for the remote peer lives inside DataTransport, not here.
    """

    mm_hash: str
    block_indices: list[int]
    addr: PeerAddr
    deadline: float
    data: DataTransport
    consumer_session_id: str  # session identity; combined with mm_hash for notif_msg

    transfer_handle: Any | None = field(default=None)
    _quarantined: bool = field(default=False, init=False)

    def handle_ack(self, ack: XferAck, agent_name: str) -> bool:
        """Issue the NIXL READ on an OK XferAck. Returns False on any NACK."""
        if ack.status != XferStatus.OK:
            return False
        # notif_msg encodes both session_id and mm_hash so the producer can
        # do a direct (session_id, mm_hash) → ProducerXfer lookup on completion.
        notif_msg = f"{self.consumer_session_id}:{self.mm_hash}".encode()
        self.transfer_handle = self.data.post_read(
            self.block_indices,
            agent_name,
            ack.src_block_indices,
            notif_msg=notif_msg,
        )
        self.deadline = time.monotonic() + _CONSUMER_READ_TIMEOUT_S
        return True

    def poll(self, now: float) -> XferState:
        if self._quarantined:
            return self._poll_quarantined()
        if self.transfer_handle is None:
            return (
                XferState.ACK_TIMEOUT if now > self.deadline else XferState.WAITING_ACK
            )
        return self._poll_reading(now)

    def _poll_reading(self, now: float) -> XferState:
        assert self.transfer_handle is not None
        try:
            state = self.data.check_xfer_state(self.transfer_handle)
        except Exception:
            logger.exception("EC: check_xfer_state failed for mm_hash=%s", self.mm_hash)
            self.data.release_xfer_handle(self.transfer_handle)
            self.transfer_handle = None
            return XferState.READ_FAILED
        if state == "DONE":
            self.data.release_xfer_handle(self.transfer_handle)
            self.transfer_handle = None
            return XferState.DONE
        if state == "PROC":
            if now > self.deadline:
                self._quarantined = True
                logger.warning(
                    "EC: READ for mm_hash=%s from %s:%d timed out; quarantining",
                    self.mm_hash,
                    self.addr[0],
                    self.addr[1],
                )
                return XferState.QUARANTINED
            return XferState.READING
        logger.warning(
            "EC: READ for mm_hash=%s unexpected NIXL state %r", self.mm_hash, state
        )
        self.data.release_xfer_handle(self.transfer_handle)
        self.transfer_handle = None
        return XferState.READ_FAILED

    def _poll_quarantined(self) -> XferState:
        if self.transfer_handle is None:
            return XferState.SETTLED
        try:
            state = self.data.check_xfer_state(self.transfer_handle)
        except Exception:
            logger.exception(
                "EC: check_xfer_state failed for quarantined mm_hash=%s", self.mm_hash
            )
            state = None
        if state == "PROC":
            return XferState.QUARANTINED
        self.data.release_xfer_handle(self.transfer_handle)
        self.transfer_handle = None
        logger.debug(
            "EC: quarantined mm_hash=%s settled (state=%s)", self.mm_hash, state
        )
        return XferState.SETTLED

    def cancel(self) -> None:
        """Cancel a WAITING_ACK xfer (peer down). No tombstone — caller allows retry."""
        assert self.transfer_handle is None, "cancel() requires WAITING_ACK state"

    def release(self) -> None:
        """Force-release the transfer handle. Shutdown only."""
        if self.transfer_handle is not None:
            self.data.release_xfer_handle(self.transfer_handle)
            self.transfer_handle = None


# ── ProducerSession ───────────────────────────────────────────────────────────


class ProducerSession:
    """Long-lived producer session. Owns ZmqServerTransport and its thread.

    The thread polls the transport for raw bytes and calls poll() with them.
    poll() decodes XferReqs, grants or NACKs each one, drains NIXL completion
    notifications, and sweeps expired xfers. No callbacks anywhere.
    """

    def __init__(
        self,
        transport: ZmqServerTransport,
        data: DataTransport,
        region: "ECSharedRegion",
        local_encodings: dict[str, None],
        blocks: dict[str, list[int]],
        lock: threading.Lock,
        compat_hash: str,
    ) -> None:
        self._transport = transport
        self._data = data
        self._region = region
        self._local_encodings = local_encodings
        self._blocks = blocks
        self._lock = lock
        self._compat_hash = compat_hash

        # Router-thread-only state.
        # Key: "{consumer_session_id}:{mm_hash}" — matches the NIXL notif_msg exactly,
        # so completion notification lookup is a direct dict pop with no parsing.
        self._active_xfers: dict[str, ProducerXfer] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._req_decoder = msgspec.msgpack.Decoder(XferReq)
        self._encoder = msgspec.msgpack.Encoder()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ec-nixl-router", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        for xfer in self._active_xfers.values():
            self._region.unpin(xfer.block_indices)  # release any still-held pins
        self._active_xfers.clear()
        self._transport.close()

    def _run(self) -> None:
        """Background thread: polls transport bytes, then drives poll()."""
        while not self._stop.is_set():
            timeout_ms = 5 if self._active_xfers else 1000
            messages = self._transport.poll(timeout_ms=timeout_ms)
            self.poll(messages)

    def poll(self, messages: list[tuple[bytes, bytes]]) -> None:
        """Decode and process inbound XferReqs, drain NIXL notifs, sweep timeouts.

        messages: (identity, payload) pairs from ZmqServerTransport.poll().
        """
        for identity, payload in messages:
            try:
                req = self._req_decoder.decode(payload)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("ec: dropped malformed XferReq")
                continue
            ack = self._grant_or_nack(req)
            self._transport.send(identity, self._encoder.encode(ack))

        self._drain_notifs()
        self._sweep_timeouts()

    def _grant_or_nack(self, req: XferReq) -> XferAck:
        if req.connector_version != EC_CONNECTOR_VERSION:
            logger.warning("EC: incompatible version for mm_hash=%s", req.mm_hash)
            return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_VERSION)
        if req.compatibility_hash != self._compat_hash:
            logger.warning("EC: incompatible compat hash for mm_hash=%s", req.mm_hash)
            return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_INCOMPAT)
        with self._lock:
            if req.mm_hash not in self._local_encodings:
                logger.warning(
                    "EC: mm_hash=%s not in local cache; NACKing", req.mm_hash
                )
                return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_MISSING)
            block_indices = self._blocks[req.mm_hash]
            self._region.pin(block_indices)
        key = f"{req.session_id}:{req.mm_hash}"
        self._active_xfers[key] = ProducerXfer(
            mm_hash=req.mm_hash,
            block_indices=block_indices,
            deadline=time.monotonic() + PRODUCER_PIN_LEASE_S,
        )
        logger.debug("EC: granted mm_hash=%s key=%s", req.mm_hash, key)
        return XferAck(
            mm_hash=req.mm_hash,
            status=XferStatus.OK,
            session_id=req.session_id,  # echo consumer's session_id back
            src_block_indices=block_indices,
            agent_metadata=self._data.get_agent_metadata(),
            mem_descriptor=self._data.get_mem_descriptor(),
        )

    def _drain_notifs(self) -> None:
        try:
            notifs = self._data.get_new_notifs()
        except Exception:
            logger.exception("EC: get_new_notifs failed")
            return
        for msgs in notifs.values():
            for msg in msgs:
                key = msg.decode("utf-8")  # "{session_id}:{mm_hash}"
                xfer = self._active_xfers.pop(key, None)
                if xfer is not None:
                    self._region.unpin(xfer.block_indices)
                    logger.debug("EC: READ done key=%s", key)

    def _sweep_timeouts(self) -> None:
        for key, xfer in list(self._active_xfers.items()):
            if xfer.is_expired():
                logger.warning("EC: grant key=%s expired; releasing pin", key)
                self._region.unpin(xfer.block_indices)
                del self._active_xfers[key]


# ── ConsumerSession ───────────────────────────────────────────────────────────


@dataclass
class ConsumerSessionResults:
    """Results from ConsumerSession.take_results()."""

    completed: set[str]  # promote to local cache
    tombstoned: set[str]  # free blocks + prevent retry this step
    quarantined: set[str]  # timed-out, DMA still running: prevent retry
    # but DO NOT free blocks — session holds them
    cancelled: set[str]  # free blocks, allow retry (peer down)
    settled: list[tuple[str, list[int]]]  # (mm_hash, block_indices) quarantine cleared


class ConsumerSession:
    """Long-lived per-producer session on the consumer side.

    Owns the ZMQ DEALER connection and the NIXL agent registration for one
    producer peer. The connector calls poll(messages, now) once per engine
    step with the raw bytes collected from this peer's DEALER socket.

    poll() decodes XferAcks, dispatches them to the right ConsumerXfer,
    advances all active xfer state machines, and drains quarantined xfers.
    Results accumulate in internal sets; call take_results() to collect them.
    """

    def __init__(
        self,
        addr: PeerAddr,
        zmq_conn: ZmqClientConnection,
        transport: ZmqClientTransport,
        data: DataTransport,
        compat_hash: str,
    ) -> None:
        self._addr = addr
        self._zmq = zmq_conn
        self._transport = transport
        self._data = data
        self._compat_hash = compat_hash
        self._session_id = str(
            uuid.uuid4()
        )  # stable identity for (session_id:mm_hash) keys

        self._nixl_agent_name: str | None = None
        self._nixl_metadata_bytes: bytes | None = None

        self._xfers: dict[str, ConsumerXfer] = {}  # mm_hash → xfer
        self._quarantined: list[ConsumerXfer] = []

        self._completed: set[str] = set()
        self._tombstoned: set[str] = set()
        self._quarantined_set: set[str] = set()
        self._cancelled: set[str] = set()
        self._settled: list[tuple[str, list[int]]] = []

        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(XferAck)

    @property
    def addr(self) -> PeerAddr:
        return self._addr

    def start_xfer(
        self, mm_hash: str, block_indices: list[int], deadline: float
    ) -> None:
        """Create a ConsumerXfer and send the XferReq over the connection."""
        xfer = ConsumerXfer(
            mm_hash=mm_hash,
            block_indices=block_indices,
            addr=self._addr,
            deadline=deadline,
            data=self._data,
            consumer_session_id=self._session_id,
        )
        self._zmq.send(
            self._encoder.encode(
                XferReq(
                    mm_hash=mm_hash,
                    compatibility_hash=self._compat_hash,
                    session_id=self._session_id,
                )
            )
        )
        self._xfers[mm_hash] = xfer
        logger.debug(
            "EC: XferReq sent mm_hash=%s to %s:%d",
            mm_hash,
            self._addr[0],
            self._addr[1],
        )

    def poll(self, messages: list[bytes], now: float) -> None:
        """Decode inbound XferAcks, dispatch to xfers, and advance all state machines.

        messages: raw bytes from ZmqClientTransport.poll() for this peer.
        Results accumulate until take_results() is called.
        """
        for raw in messages:
            try:
                ack = self._decoder.decode(raw)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning(
                    "ec: dropped malformed XferAck from %s:%d",
                    self._addr[0],
                    self._addr[1],
                )
                continue
            self._dispatch_ack(ack)

        for mm_hash, xfer in list(self._xfers.items()):
            state = xfer.poll(now)
            if state == XferState.DONE:
                self._completed.add(mm_hash)
                del self._xfers[mm_hash]
            elif state in (XferState.ACK_TIMEOUT, XferState.READ_FAILED):
                self._tombstoned.add(mm_hash)
                del self._xfers[mm_hash]
            elif state == XferState.QUARANTINED:
                # DMA may still be writing — caller must NOT free blocks.
                self._quarantined.append(xfer)
                del self._xfers[mm_hash]
                self._quarantined_set.add(mm_hash)

        self._drain_quarantine(now)

    def _dispatch_ack(self, ack: XferAck) -> None:
        xfer = self._xfers.get(ack.mm_hash)
        if xfer is None or xfer.transfer_handle is not None:
            return  # unknown mm_hash or duplicate ack

        if ack.status != XferStatus.OK:
            logger.log(
                20 if ack.status == XferStatus.NACK_MISSING else 30,
                "EC: NACK %s from %s:%d for mm_hash=%s",
                ack.status.name,
                self._addr[0],
                self._addr[1],
                ack.mm_hash,
            )
            del self._xfers[ack.mm_hash]
            self._tombstoned.add(ack.mm_hash)
            return

        try:
            agent_name = self._ensure_registered(ack.agent_metadata, ack.mem_descriptor)
        except Exception:
            logger.exception(
                "EC: failed to register peer %s:%d for mm_hash=%s",
                self._addr[0],
                self._addr[1],
                ack.mm_hash,
            )
            del self._xfers[ack.mm_hash]
            self._tombstoned.add(ack.mm_hash)
            return

        if not xfer.handle_ack(ack, agent_name):
            del self._xfers[ack.mm_hash]
            self._tombstoned.add(ack.mm_hash)

    def _ensure_registered(self, metadata: bytes, mem_descriptor: bytes) -> str:
        """Register or re-register the peer in DataTransport. Returns agent_name."""
        if self._nixl_agent_name is not None and self._nixl_metadata_bytes == metadata:
            return self._nixl_agent_name
        if self._nixl_agent_name is not None:
            logger.info(
                "EC: producer %s:%d restarted; re-registering",
                self._addr[0],
                self._addr[1],
            )
            self._data.remove_remote_peer(self._nixl_agent_name)
        agent_name = self._data.add_remote_peer(metadata, mem_descriptor)
        self._nixl_agent_name = agent_name
        self._nixl_metadata_bytes = metadata
        logger.debug(
            "EC: registered peer %s:%d agent=%s",
            self._addr[0],
            self._addr[1],
            agent_name,
        )
        return agent_name

    def _drain_quarantine(self, now: float) -> None:
        still_pending: list[ConsumerXfer] = []
        for xfer in self._quarantined:
            state = xfer.poll(now)
            if state == XferState.SETTLED:
                self._settled.append((xfer.mm_hash, xfer.block_indices))
            else:
                still_pending.append(xfer)
        self._quarantined = still_pending

    def on_peer_down(self) -> None:
        """WAITING_ACK xfers → cancel (retry allowed); READING xfers → quarantine."""
        for mm_hash, xfer in list(self._xfers.items()):
            if xfer.transfer_handle is None:
                xfer.cancel()
                self._cancelled.add(mm_hash)
            else:
                self._quarantined.append(xfer)
                self._tombstoned.add(mm_hash)
        self._xfers.clear()

    def take_results(self) -> ConsumerSessionResults:
        """Return and clear all accumulated results since the last call."""
        results = ConsumerSessionResults(
            completed=self._completed,
            tombstoned=self._tombstoned,
            quarantined=self._quarantined_set,
            cancelled=self._cancelled,
            settled=self._settled,
        )
        self._completed = set()
        self._tombstoned = set()
        self._quarantined_set = set()
        self._cancelled = set()
        self._settled = []
        return results

    def has_in_flight(self) -> bool:
        return bool(self._xfers) or bool(self._quarantined)

    def close(self) -> None:
        """Release all transfer handles, deregister NIXL peer, close connection.

        Connection teardown is routed through the transport so the pooled
        connection is dropped in lock-step with the session — a session and its
        DEALER connection have a 1:1 lifecycle, so a later connect() to the same
        peer builds a fresh socket instead of handing back this dead one.
        """
        for xfer in self._xfers.values():
            xfer.release()
        self._xfers.clear()
        for xfer in self._quarantined:
            xfer.release()
        self._quarantined.clear()
        if self._nixl_agent_name is not None:
            self._data.remove_remote_peer(self._nixl_agent_name)
        self._transport.remove(self._addr)
