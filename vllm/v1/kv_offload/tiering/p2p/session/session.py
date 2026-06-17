# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSession — bidirectional session combining client + server roles.

A single P2PSession per remote peer handles BOTH directions of the P2P
protocol on one ControlConnection: it can request blocks from the peer
(today's "client" role) AND serve blocks to the peer (today's "server"
role). The session owns the connection's recv() queue and dispatches
every message type to the right handler.

Wire protocol is unchanged. Both sides advertise their NIXL metadata
via ConnectMsg when their session is connected; the peer's ConnectMsg
triggers transport.add_remote_peer; ConnectAckMsg confirms the peer
received our ConnectMsg, after which queued outgoing messages are flushed.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    FetchMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 30.0
_ABORT_ACK_TIMEOUT_S = 10.0
_STORE_TIMEOUT_S = 30.0
_CANCEL_DRAIN_TIMEOUT_S = 10.0


@dataclass
class _InboundRequestState:
    """Client-role state for a single load request."""

    job_id: int  # opaque ID assigned by the manager to this load request
    kv_request_id: str
    submitted_at: float
    aborted_at: float | None = None


class LoadResult(NamedTuple):
    """Result from a session poll, client side."""

    job_id: int
    kv_request_id: str
    success: bool


class StoreResult(NamedTuple):
    """Result from a session poll, server side."""

    job_id: int
    success: bool


class _InflightXfer(NamedTuple):
    """Metadata for a single inflight RDMA transfer, keyed by transfer_id."""

    kv_request_id: str
    block_count: int
    # The set of store job IDs that contributed blocks to this transfer.
    job_ids: set[int]


class _MatchResult(NamedTuple):
    """Result of block matching: pairs ready for transfer."""

    local_idxs: list[int]
    remote_idxs: list[int]
    # The set of store job IDs that contributed blocks
    job_ids: set[int]


@dataclass
class _OutboundRequestState:
    """Server-role state for a single peer fetch request.

    The owning ``kv_request_id`` is the dict key in
    ``P2PSession._outbound`` and is not duplicated on the value.
    """

    demand_received: bool = False
    available: dict[OffloadKey, tuple[int, int]] = field(
        default_factory=dict
    )  # key → (job_id, local_block_idx): blocks we have, awaiting demand
    demanded: dict[OffloadKey, int] = field(
        default_factory=dict
    )  # key → remote_block_idx: blocks peer wants, awaiting supply
    remaining: int = 0  # blocks that need to be transferred to client
    finishing: bool = False  # Signal finish request ASAP

    def add_stored_blocks(
        self,
        block_hashes: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: int,
    ) -> _MatchResult:
        """Add locally-stored blocks. Returns matched pairs."""
        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        for block_hash, local_idx in zip(block_hashes, block_ids):
            remote_idx = self.demanded.pop(block_hash, None)
            if remote_idx is not None:
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
            else:
                self.available[block_hash] = (job_id, local_idx)
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids={job_id} if local_idxs else set(),
        )

    def add_fetch_demand(
        self,
        block_hashes: Sequence[OffloadKey],
        block_indexes: Sequence[int],
    ) -> _MatchResult:
        """Register the peer's fetch demand. Returns matched pairs."""
        self.demand_received = True
        self.remaining = len(block_hashes)

        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        job_ids: set[int] = set()
        for block_hash, remote_idx in zip(block_hashes, block_indexes):
            stored_entry = self.available.pop(block_hash, None)
            if stored_entry is not None:
                stored_job_id, local_idx = stored_entry
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
                job_ids.add(stored_job_id)
            else:
                self.demanded[block_hash] = remote_idx
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids=job_ids,
        )


class P2PSession:
    """Bidirectional session — handles both client-role (loads) and
    server-role (stores) traffic toward a single peer.

    Lifecycle:
      - Constructor with conn=None  ⇒ pending. Accepts add_stored_blocks
        but cannot send (used by the prefiller to buffer blocks before
        the decoder connects).
      - Constructor with conn != None ⇒ connected. Sends our own ConnectMsg
        immediately; the peer's ConnectMsg arrives in poll() and is
        dispatched to _on_connect (which calls transport.add_remote_peer
        and replies with ConnectAckMsg). Outgoing sends are queued until
        ConnectAckMsg confirms our metadata reached the peer.
      - attach_connection(conn) on a pending session ⇒ same as above,
        starting from pending.
    """

    def __init__(
        self,
        peer_id: str,
        local_id: str,
        transport: DataTransport,
        local_block_len: int,
        conn: ControlConnection | None = None,
    ) -> None:
        self.peer_id = peer_id
        self._local_id = local_id
        self._transport = transport
        self._local_block_len = local_block_len
        self._conn: ControlConnection | None = None

        # Client-role state
        self._inbound: dict[str, _InboundRequestState] = {}  # kv_request_id -> req
        self._completed_loads: list[LoadResult] = []
        self._send_ready = False  # True after the peer acked our ConnectMsg
        self._queued: list[
            dict
        ] = []  # Msgs waiting to be send on connection establishment

        # Server-role state
        self._outbound: dict[str, _OutboundRequestState] = {}  # kv_request_id -> req
        self._inflight: dict[int, _InflightXfer] = {}  # transfer_id → xfer
        self._store_jobs: dict[int, float] = {}  # job_id → submitted_at
        self._pending_aborts: dict[str, float] = {}  # kv_request_id → start

        if conn is not None:
            self.attach_connection(conn)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        # Pending sessions (awaiting connection) are alive — only a
        # closed real connection counts as dead.
        return self._conn is None or self._conn.alive

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def ready(self) -> bool:
        """True after the peer acked our ConnectMsg (we may send freely)."""
        return self._send_ready

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def attach_connection(self, conn: ControlConnection) -> None:
        """Attach a connection to a pending session and announce ourselves.

        Symmetric: every side advertises its NIXL metadata on connect, so
        whichever peer receives a session first can register the other.
        """
        if self._conn is not None:
            raise ValueError(f"P2PSession {self.peer_id}: already connected")
        self._conn = conn
        self._send_connect()

    # ------------------------------------------------------------------
    # Public API — client role
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
    ) -> None:
        """Send fetch to the peer."""
        logger.debug(
            "P2PSession %s: request_blocks job_id=%d kv_request_id=%s "
            "blocks=%d ready=%s",
            self.peer_id,
            job_id,
            kv_request_id,
            len(block_ids),
            self._send_ready,
        )
        self._inbound[kv_request_id] = _InboundRequestState(
            job_id=job_id,
            kv_request_id=kv_request_id,
            submitted_at=time.monotonic(),
        )
        self._send(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: kv_request_id,
                FetchMsg.BLOCK_HASHES: list(keys),
                FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            }
        )

    def finish_request(self, kv_request_id: str) -> None:
        """Called when the request is finishing locally.

        Cancels any inbound load (client role) and finalizes any
        outbound serving (server role) for this id. Roles that aren't
        active for this id are silent no-ops.
        """
        self._cancel_inbound(kv_request_id)
        self._finish_outbound(kv_request_id)

    def _cancel_inbound(self, kv_request_id: str) -> None:
        """Cancel a pending load request. Sends abort if active."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None and req.aborted_at is None:
            self._send(
                {
                    TYPE_KEY: AbortFetchMsg.TYPE,
                    AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                }
            )

    def _finish_outbound(self, kv_request_id: str) -> None:
        """Mark an outbound request finishing.

        No more submit_store calls will arrive for this id. Any blocks
        the peer demanded but we never stored will never come; tell the
        peer to stop waiting (TransferDoneMsg success=False) instead of
        letting it hit _LOAD_TIMEOUT_S.

        If the decoder hasn't sent fetch yet (no demand received),
        defer — _on_fetch will finalize once demand arrives.

        If inflight transfers exist for this id, defer — the last
        completing transfer in _collect_store_results will fire the
        message.
        """
        req = self._outbound.get(kv_request_id)
        if req is None:
            return
        req.finishing = True
        if not req.demand_received:
            return
        if self._has_inflight_for(kv_request_id):
            return
        del self._outbound[kv_request_id]
        self._send(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: kv_request_id,
                TransferDoneMsg.SUCCESS: False,
            }
        )

    def _has_inflight_for(self, kv_request_id: str) -> bool:
        return any(x.kv_request_id == kv_request_id for x in self._inflight.values())

    # ------------------------------------------------------------------
    # Public API — server role
    # ------------------------------------------------------------------

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
    ) -> None:
        """New blocks stored locally — match against pending fetch demand."""
        self._store_jobs[job_id] = time.monotonic()
        req = self._outbound.setdefault(kv_request_id, _OutboundRequestState())
        result = req.add_stored_blocks(keys, block_ids, job_id)
        if result.local_idxs and req.demand_received:
            self._submit_transfer(kv_request_id, result)

    # ------------------------------------------------------------------
    # Public API — polling and lifecycle
    # ------------------------------------------------------------------

    def poll(self) -> tuple[list[LoadResult], list[StoreResult]]:
        """Process incoming messages, drive transfers, apply timeouts."""
        if self._conn is None:
            # Pending session — store-job timeouts still apply so buffered
            # jobs that never get picked up are surfaced as failures.
            return [], self._timeout_pending_store_jobs()

        for msg in self._conn.recv():
            self._on_message(msg)

        loads = self._collect_load_results()
        stores = self._collect_store_results()
        self._drain_pending_aborts()
        return loads, stores

    def close(self) -> tuple[list[tuple[int, str]], list[int]]:
        """Shut down. Returns (failed_loads, failed_stores).

        failed_loads: list of (job_id, kv_request_id) pairs.
        failed_stores: list of job_ids.
        """
        failed_loads = [
            (req.job_id, req.kv_request_id) for req in self._inbound.values()
        ]
        self._inbound.clear()

        failed_stores = list(self._store_jobs.keys())
        self._store_jobs.clear()
        if self._inflight:
            self._transport.cancel(list(self._inflight.keys()))
        self._inflight.clear()
        self._outbound.clear()
        self._pending_aborts.clear()

        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.close()
            self._conn = None

        return failed_loads, failed_stores

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------

    def _collect_load_results(self) -> list[LoadResult]:
        now = time.monotonic()
        for req_id, req in list(self._inbound.items()):
            if req.aborted_at is None:
                if now - req.submitted_at >= _LOAD_TIMEOUT_S:
                    req.aborted_at = now
                    logger.warning(
                        "P2PSession %s: %s timed out, sending abort",
                        self.peer_id,
                        req_id,
                    )
                    self._send(
                        {
                            TYPE_KEY: AbortFetchMsg.TYPE,
                            AbortFetchMsg.KV_REQUEST_ID: req_id,
                        }
                    )
            else:
                if now - req.aborted_at >= _ABORT_ACK_TIMEOUT_S:
                    self._inbound.pop(req_id)
                    self._completed_loads.append(
                        LoadResult(
                            job_id=req.job_id, kv_request_id=req_id, success=False
                        )
                    )
                    logger.warning(
                        "P2PSession %s: abort_ack timed out for kv_request_id=%s",
                        self.peer_id,
                        req_id,
                    )

        results = self._completed_loads
        self._completed_loads = []
        return results

    def _collect_store_results(self) -> list[StoreResult]:
        results: list[StoreResult] = self._timeout_pending_store_jobs()

        poll_result = self._transport.poll()

        for tid in poll_result.done:
            xfer = self._inflight.pop(tid, None)
            if xfer is None:
                # Bug signal: transport reported a transfer we have no
                # bookkeeping for. Likely a double-completion in the
                # transport or a stale removal in the session. The
                # attached job(s) still live in _store_jobs and will be
                # surfaced as failures by _timeout_pending_store_jobs
                # after _STORE_TIMEOUT_S, but log loudly so the
                # underlying bug is findable.
                logger.error(
                    "P2PSession %s: transport reported done for unknown "
                    "transfer_id=%d; attached job(s) will fail via "
                    "store-timeout instead of completing now",
                    self.peer_id,
                    tid,
                )
                continue
            for job_id in xfer.job_ids:
                self._store_jobs.pop(job_id, None)
                results.append(StoreResult(job_id=job_id, success=True))
            req = self._outbound.get(xfer.kv_request_id)
            if req is not None and req.demand_received:
                req.remaining -= xfer.block_count
                assert req.remaining >= 0, (
                    f"remaining went negative for kv_request_id={xfer.kv_request_id}"
                )
                if req.remaining == 0:
                    del self._outbound[xfer.kv_request_id]
                    self._send(
                        {
                            TYPE_KEY: TransferDoneMsg.TYPE,
                            TransferDoneMsg.KV_REQUEST_ID: xfer.kv_request_id,
                            TransferDoneMsg.SUCCESS: True,
                        }
                    )
                elif req.finishing and not self._has_inflight_for(xfer.kv_request_id):
                    del self._outbound[xfer.kv_request_id]
                    self._send(
                        {
                            TYPE_KEY: TransferDoneMsg.TYPE,
                            TransferDoneMsg.KV_REQUEST_ID: xfer.kv_request_id,
                            TransferDoneMsg.SUCCESS: False,
                        }
                    )

        failed_kv_request_ids: set[str] | None = None
        for tid in poll_result.failed:
            xfer = self._inflight.pop(tid, None)
            if xfer is None:
                # See the matching error log in the done branch above.
                logger.error(
                    "P2PSession %s: transport reported failed for unknown "
                    "transfer_id=%d; attached job(s) will fail via "
                    "store-timeout instead of completing now",
                    self.peer_id,
                    tid,
                )
                continue
            if failed_kv_request_ids is None:
                failed_kv_request_ids = set()
            failed_kv_request_ids.add(xfer.kv_request_id)
            for job_id in xfer.job_ids:
                self._store_jobs.pop(job_id, None)
                results.append(StoreResult(job_id=job_id, success=False))
            req = self._outbound.pop(xfer.kv_request_id, None)
            if req is not None and req.demand_received:
                self._send(
                    {
                        TYPE_KEY: TransferDoneMsg.TYPE,
                        TransferDoneMsg.KV_REQUEST_ID: xfer.kv_request_id,
                        TransferDoneMsg.SUCCESS: False,
                    }
                )

        # Cancel other inflight for the same failed kv_request_ids
        if failed_kv_request_ids:
            ids_to_cancel = [
                tid
                for tid, xfer in self._inflight.items()
                if xfer.kv_request_id in failed_kv_request_ids
            ]
            for tid in ids_to_cancel:
                del self._inflight[tid]
            self._transport.cancel(ids_to_cancel)

        return results

    def _timeout_pending_store_jobs(self) -> list[StoreResult]:
        if not self._store_jobs:
            return []
        deadline = time.monotonic() - _STORE_TIMEOUT_S
        timed_out: list[int] | None = None
        for jid, submitted_at in self._store_jobs.items():
            if submitted_at <= deadline:
                if timed_out is None:
                    timed_out = []
                timed_out.append(jid)
        if timed_out is None:
            return []
        results: list[StoreResult] = []
        for jid in timed_out:
            del self._store_jobs[jid]
            results.append(StoreResult(job_id=jid, success=False))
            logger.warning("P2PSession %s: store job %d timed out", self.peer_id, jid)
        return results

    # ------------------------------------------------------------------
    # Internal message dispatch
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        try:
            self._dispatch_message(msg)
        except Exception as exc:
            logger.warning(
                "P2PSession %s: error handling message %r: %s",
                self.peer_id,
                msg.get(TYPE_KEY) if isinstance(msg, dict) else msg,
                exc,
            )

    def _protocol_error(self, reason: str) -> None:
        """Log a protocol violation and disconnect.

        Best-effort sends ``DisconnectMsg`` so the peer learns why we're
        going away, then marks the connection dead. The manager reaps
        the session on the next poll via ``alive``.
        """
        logger.error(
            "P2PSession %s: protocol error: %s — disconnecting",
            self.peer_id,
            reason,
        )
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.mark_dead()

    def _dispatch_message(self, msg: dict) -> None:
        # Drop messages buffered before disconnect: a poll batch can
        # contain msg-after-DisconnectMsg, and dispatching them would
        # mutate state on a dead session.
        if self._conn is not None and not self._conn.alive:
            return
        msg_type = msg.get(TYPE_KEY) if isinstance(msg, dict) else None
        if msg_type == ConnectMsg.TYPE:
            self._on_connect(msg)
        elif msg_type == ConnectAckMsg.TYPE:
            ConnectAckMsg.validate(msg)
            self._on_connect_ack()
        elif msg_type == FetchMsg.TYPE:
            self._on_fetch(msg)
        elif msg_type == AbortFetchMsg.TYPE:
            self._on_abort_fetch(msg)
        elif msg_type == TransferDoneMsg.TYPE:
            TransferDoneMsg.validate(msg)
            self._on_transfer_done(msg)
        elif msg_type == AbortAckMsg.TYPE:
            AbortAckMsg.validate(msg)
            self._on_abort_ack(msg)
        elif msg_type == DisconnectMsg.TYPE:
            if self._conn is not None:
                self._conn.mark_dead()
        else:
            logger.warning(
                "P2PSession %s: unknown message type %r", self.peer_id, msg_type
            )

    def _on_connect(self, msg: dict) -> None:
        # Validation failures here mean an incompatible or malicious peer.
        # Mark the connection dead so the manager reaps the session;
        # don't call add_remote_peer or send connect_ack.
        if self._send_ready:
            # We've already received connect_ack, so the handshake is
            # complete. A second connect from the peer is a protocol
            # violation — re-registering would corrupt transport state.
            self._protocol_error("duplicate connect after handshake")
            return
        try:
            ConnectMsg.validate(msg)
            if msg[ConnectMsg.BLOCK_LEN] != self._local_block_len:
                raise ValueError(
                    f"block_len mismatch from {self.peer_id}: "
                    f"remote={msg[ConnectMsg.BLOCK_LEN]}, "
                    f"local={self._local_block_len}"
                )
            remote_fp = msg.get(ConnectMsg.CONFIG_FINGERPRINT, "")
            local_fp = self._transport.config_fingerprint
            if local_fp and remote_fp and remote_fp != local_fp:
                raise ValueError(
                    f"config fingerprint mismatch from {self.peer_id}: "
                    f"remote={remote_fp!r}, local={local_fp!r}"
                )
            self._transport.add_remote_peer(
                self.peer_id,
                agent_metadata=msg[ConnectMsg.AGENT_METADATA],
                base_addr=msg[ConnectMsg.BASE_ADDR],
                num_blocks=msg[ConnectMsg.NUM_BLOCKS],
                block_len=msg[ConnectMsg.BLOCK_LEN],
            )
        except ValueError as exc:
            logger.error("P2PSession %s: rejecting peer connect: %s", self.peer_id, exc)
            if self._conn is not None:
                self._conn.mark_dead()
            return

        if self._conn is not None:
            self._conn.send(
                {
                    TYPE_KEY: ConnectAckMsg.TYPE,
                    ConnectAckMsg.PEER_ID: self._local_id,
                }
            )

    def _on_connect_ack(self) -> None:
        if self._queued:
            logger.debug(
                "P2PSession %s: connect_ack received, flushing %d queued msg(s)",
                self.peer_id,
                len(self._queued),
            )
        self._send_ready = True
        for queued in self._queued:
            self._do_send(queued)
        self._queued.clear()

    def _on_fetch(self, msg: dict) -> None:
        FetchMsg.validate(msg)
        kv_request_id = msg[FetchMsg.KV_REQUEST_ID]
        block_hashes = [
            OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
            for bh in msg[FetchMsg.BLOCK_HASHES]
        ]
        block_indexes = msg[FetchMsg.BLOCK_INDEXES]
        logger.debug(
            "P2PSession %s: fetch RECEIVED kv_request_id=%s blocks=%d",
            self.peer_id,
            kv_request_id,
            len(block_hashes),
        )
        existing = self._outbound.get(kv_request_id)
        if existing is not None and existing.demand_received:
            # A second fetch for the same kv_request_id would overwrite
            # `remaining` and leak inflight bookkeeping. Treat as a
            # protocol violation.
            self._protocol_error(f"duplicate fetch for kv_request_id={kv_request_id}")
            return
        req = self._outbound.setdefault(kv_request_id, _OutboundRequestState())
        result = req.add_fetch_demand(block_hashes, block_indexes)
        if result.local_idxs:
            self._submit_transfer(kv_request_id, result)
        # Prefiller-first mode: finish_request may have run before
        # fetch arrived. If so, finalize once we know what was
        # demanded — fully satisfied → success, else early-fail.
        if req.finishing and not self._has_inflight_for(kv_request_id):
            del self._outbound[kv_request_id]
            self._send(
                {
                    TYPE_KEY: TransferDoneMsg.TYPE,
                    TransferDoneMsg.KV_REQUEST_ID: kv_request_id,
                    TransferDoneMsg.SUCCESS: req.remaining == 0,
                }
            )

    def _on_abort_fetch(self, msg: dict) -> None:
        AbortFetchMsg.validate(msg)
        kv_request_id = msg[AbortFetchMsg.KV_REQUEST_ID]
        # Idempotent: receiving AbortFetchMsg again before we've sent the
        # ack just triggers another drain attempt without resetting the
        # deadline.
        self._pending_aborts.setdefault(kv_request_id, time.monotonic())
        self._drain_abort(kv_request_id)

    def _drain_pending_aborts(self) -> None:
        """Re-attempt every parked abort once per poll tick."""
        if not self._pending_aborts:
            return
        for kv_request_id in list(self._pending_aborts):
            self._drain_abort(kv_request_id)

    def _drain_abort(self, kv_request_id: str) -> None:
        """One drain attempt for a pending abort.

        Stops accepting more blocks for ``kv_request_id``, then asks the
        transport to cancel any matching inflight transfers in
        ``mode="wait"``. Sends ``AbortAckMsg`` once nothing remains
        inflight, or after ``_CANCEL_DRAIN_TIMEOUT_S`` falls back to
        ``mode="immediate"`` and acks anyway.
        """
        self._outbound.pop(kv_request_id, None)
        ids = [
            tid
            for tid, xfer in self._inflight.items()
            if xfer.kv_request_id == kv_request_id
        ]
        if not ids:
            self._finalize_abort(kv_request_id)
            return

        started_at = self._pending_aborts.get(kv_request_id)
        expired = (
            started_at is not None
            and time.monotonic() - started_at >= _CANCEL_DRAIN_TIMEOUT_S
        )
        if expired:
            for tid in ids:
                self._inflight.pop(tid, None)
            self._transport.cancel(ids, mode="immediate")
            logger.warning(
                "P2PSession %s: cancel drain timed out for kv_request_id=%s,"
                " force-canceled %d transfers",
                self.peer_id,
                kv_request_id,
                len(ids),
            )
            self._finalize_abort(kv_request_id)
            return

        still = self._transport.cancel(ids, mode="wait")
        # Tids the transport successfully released are gone from its
        # _inflight; mirror that in session bookkeeping so they don't
        # block the drain forever waiting for a poll() event that will
        # never come.
        still_set = set(still)
        for tid in ids:
            if tid not in still_set:
                self._inflight.pop(tid, None)
        if not still:
            self._finalize_abort(kv_request_id)

    def _finalize_abort(self, kv_request_id: str) -> None:
        self._pending_aborts.pop(kv_request_id, None)
        self._send(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: kv_request_id,
            }
        )

    def _on_transfer_done(self, msg: dict) -> None:
        kv_request_id = msg[TransferDoneMsg.KV_REQUEST_ID]
        success = msg[TransferDoneMsg.SUCCESS]
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id, kv_request_id=kv_request_id, success=success
                )
            )
        else:
            logger.warning(
                "P2PSession %s: transfer_done for unknown kv_request_id=%s",
                self.peer_id,
                kv_request_id,
            )

    def _on_abort_ack(self, msg: dict) -> None:
        kv_request_id = msg[AbortAckMsg.KV_REQUEST_ID]
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id, kv_request_id=kv_request_id, success=False
                )
            )
        else:
            logger.warning(
                "P2PSession %s: abort_ack for unknown kv_request_id=%s",
                self.peer_id,
                kv_request_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_transfer(self, kv_request_id: str, result: _MatchResult) -> None:
        logger.debug(
            "P2PSession %s: NIXL write_blocks CALL kv_request_id=%s "
            "local_idxs=%d remote_idxs=%d",
            self.peer_id,
            kv_request_id,
            len(result.local_idxs),
            len(result.remote_idxs),
        )
        transfer_id = self._transport.write_blocks(
            self.peer_id, result.local_idxs, result.remote_idxs
        )
        if transfer_id is not None:
            logger.debug(
                "P2PSession %s: NIXL write_blocks SUBMITTED kv_request_id=%s "
                "transfer_id=%d blocks=%d",
                self.peer_id,
                kv_request_id,
                transfer_id,
                len(result.local_idxs),
            )
            self._inflight[transfer_id] = _InflightXfer(
                kv_request_id=kv_request_id,
                block_count=len(result.local_idxs),
                job_ids=result.job_ids,
            )
        else:
            logger.warning(
                "P2PSession %s: write_blocks failed for %s (%d blocks)",
                self.peer_id,
                kv_request_id,
                len(result.local_idxs),
            )

    def _send_connect(self) -> None:
        """Send our ConnectMsg announcing local NIXL metadata."""
        assert self._conn is not None
        self._conn.send(
            {
                TYPE_KEY: ConnectMsg.TYPE,
                ConnectMsg.PEER_ID: self._local_id,
                ConnectMsg.AGENT_METADATA: self._transport.get_agent_metadata(),
                ConnectMsg.BASE_ADDR: self._transport.base_addr,
                ConnectMsg.NUM_BLOCKS: self._transport.num_blocks,
                ConnectMsg.BLOCK_LEN: self._transport.block_len,
                ConnectMsg.CONFIG_FINGERPRINT: self._transport.config_fingerprint,
            }
        )

    def _send(self, msg: dict) -> None:
        if self._conn is None or not self._send_ready:
            logger.debug(
                "P2PSession %s: queueing %s (ready=%s queue_depth=%d)",
                self.peer_id,
                msg.get(TYPE_KEY),
                self._send_ready,
                len(self._queued) + 1,
            )
            self._queued.append(msg)
            return
        self._do_send(msg)

    def _do_send(self, msg: dict) -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(msg)
            logger.debug("P2PSession %s: sent %s", self.peer_id, msg.get(TYPE_KEY))
        except Exception:
            logger.warning(
                "P2PSession %s: failed to send %s",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
