# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PServerSession — server side, serves blocks to one client peer.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortLookupFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    LookupFetchMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_STORE_TIMEOUT_S = 30.0


class StoreResult(NamedTuple):
    """Result from a server session poll."""

    job_id: int
    success: bool


class _InflightXfer(NamedTuple):
    """Metadata for a single inflight RDMA transfer, keyed by transfer_id."""

    kv_request_id: str
    block_count: int
    job_ids: set[int]


class _MatchResult(NamedTuple):
    """Result of block matching: pairs ready for transfer."""

    local_idxs: list[int]
    remote_idxs: list[int]
    job_ids: set[int]


@dataclass
class _OutboundRequestState:
    """Server-side state for a single client fetch request."""

    kv_request_id: str
    client_id: str | None = None
    available: dict[OffloadKey, tuple[int, int]] = field(
        default_factory=dict
    )  # key → (job_id, local_block_idx): blocks we have, awaiting demand
    demanded: dict[OffloadKey, int] = field(
        default_factory=dict
    )  # key → remote_block_idx: blocks client wants, awaiting supply
    remaining: int = 0

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
        client_id: str,
        block_hashes: Sequence[OffloadKey],
        block_indexes: Sequence[int],
    ) -> _MatchResult:
        """Register client's fetch demand. Returns matched pairs."""
        self.client_id = client_id
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


class P2PServerSession:
    """Serves KV blocks to a single client peer.

    Created from a newly accepted connection (constructor performs the
    connect handshake). Handles lookup_fetch and abort_lookup_fetch
    messages internally. The manager calls add_stored_blocks() and poll().
    """

    def __init__(
        self,
        conn: ControlConnection,
        local_id: str,
        transport: DataTransport,
        local_block_len: int,
    ) -> None:
        """Create a server session from a newly accepted connection.

        Extracts the connect message from the connection's inbox,
        validates block_len, registers the NIXL peer, and sends connect_ack.
        Raises ValueError on validation failure.
        """
        msgs = conn.recv()
        assert len(msgs) == 1, f"Expected 1 connect message, got {len(msgs)}"
        msg = msgs[0]
        assert msg.get(TYPE_KEY) == ConnectMsg.TYPE, (
            f"Expected connect message, got {msg.get(TYPE_KEY)!r}"
        )

        ConnectMsg.validate(msg)

        if msg[ConnectMsg.BLOCK_LEN] != local_block_len:
            raise ValueError(
                f"block_len mismatch from {conn.peer_id}: "
                f"remote={msg[ConnectMsg.BLOCK_LEN]}, local={local_block_len}"
            )

        remote_fp = msg.get(ConnectMsg.CONFIG_FINGERPRINT, "")
        local_fp = transport.config_fingerprint
        if local_fp and remote_fp and remote_fp != local_fp:
            raise ValueError(
                f"config fingerprint mismatch from {conn.peer_id}: "
                f"remote={remote_fp!r}, local={local_fp!r}"
            )

        transport.add_remote_peer(
            conn.peer_id,
            agent_metadata=msg[ConnectMsg.AGENT_METADATA],
            base_addr=msg[ConnectMsg.BASE_ADDR],
            num_blocks=msg[ConnectMsg.NUM_BLOCKS],
            block_len=msg[ConnectMsg.BLOCK_LEN],
        )

        self.peer_id = conn.peer_id
        self._conn = conn
        self._transport = transport

        self._outbound: dict[str, _OutboundRequestState] = {}
        self._inflight: dict[int, _InflightXfer] = {}  # transfer_id → xfer
        self._store_jobs: dict[int, float] = {}  # job_id → submitted_at

        conn.send({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: local_id})

    @property
    def alive(self) -> bool:
        return self._conn.alive

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
    ) -> None:
        """New blocks stored locally — match against pending fetch demand."""
        self._store_jobs[job_id] = time.monotonic()
        req = self._outbound.setdefault(
            kv_request_id, _OutboundRequestState(kv_request_id=kv_request_id)
        )
        result = req.add_stored_blocks(keys, block_ids, job_id)
        if result.local_idxs and req.client_id is not None:
            self._submit_transfer(kv_request_id, result)

    def poll(self) -> list[StoreResult]:
        """Process incoming messages, poll transfers, check timeouts."""
        # Process incoming messages
        for msg in self._conn.recv():
            self._on_message(msg)

        results: list[StoreResult] = []

        # Check store job timeouts
        now = time.monotonic()
        timed_out = [
            jid
            for jid, submitted_at in self._store_jobs.items()
            if now - submitted_at >= _STORE_TIMEOUT_S
        ]
        for jid in timed_out:
            del self._store_jobs[jid]
            results.append(StoreResult(job_id=jid, success=False))
            logger.warning(
                "P2PServerSession %s: store job %d timed out",
                self.peer_id,
                jid,
            )

        # Poll inflight transfers
        poll_result = self._transport.poll()

        for xfer in (self._inflight.pop(tid) for tid in poll_result.done):
            for job_id in xfer.job_ids:
                self._store_jobs.pop(job_id, None)
                results.append(StoreResult(job_id=job_id, success=True))
            req = self._outbound.get(xfer.kv_request_id)
            if req is not None and req.client_id is not None:
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

        failed_kv_request_ids: set[str] = set()
        for xfer in (self._inflight.pop(tid) for tid in poll_result.failed):
            failed_kv_request_ids.add(xfer.kv_request_id)
            for job_id in xfer.job_ids:
                self._store_jobs.pop(job_id, None)
                results.append(StoreResult(job_id=job_id, success=False))
            req = self._outbound.pop(xfer.kv_request_id, None)
            if req is not None and req.client_id is not None:
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

    def close(self) -> Iterable[int]:
        """Shut down. Returns failed job_ids."""
        failed_jobs: list[int] = list(self._store_jobs.keys())
        self._store_jobs.clear()
        self._transport.cancel(list(self._inflight.keys()))
        self._inflight.clear()
        self._outbound.clear()
        self._send({TYPE_KEY: DisconnectMsg.TYPE})
        self._conn.close()
        return failed_jobs

    # ------------------------------------------------------------------
    # Internal message handling
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        try:
            self._dispatch_message(msg)
        except Exception as exc:
            logger.warning(
                "P2PServerSession %s: error handling message %r: %s",
                self.peer_id,
                msg.get(TYPE_KEY) if isinstance(msg, dict) else msg,
                exc,
            )

    def _dispatch_message(self, msg: dict) -> None:
        msg_type = msg.get(TYPE_KEY)
        if msg_type == LookupFetchMsg.TYPE:
            self._on_lookup_fetch(msg)
        elif msg_type == AbortLookupFetchMsg.TYPE:
            self._on_abort_lookup_fetch(msg)
        elif msg_type == DisconnectMsg.TYPE:
            self._conn.mark_dead()
        else:
            logger.warning(
                "P2PServerSession %s: unknown message type %r",
                self.peer_id,
                msg_type,
            )

    def _on_lookup_fetch(self, msg: dict) -> None:
        LookupFetchMsg.validate(msg)

        kv_request_id = msg[LookupFetchMsg.KV_REQUEST_ID]
        block_hashes = [
            OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
            for bh in msg[LookupFetchMsg.BLOCK_HASHES]
        ]
        block_indexes = msg[LookupFetchMsg.BLOCK_INDEXES]

        logger.debug(
            "P2PServerSession %s: lookup_fetch kv_request_id=%s blocks=%d",
            self.peer_id,
            kv_request_id,
            len(block_hashes),
        )

        req = self._outbound.setdefault(
            kv_request_id, _OutboundRequestState(kv_request_id=kv_request_id)
        )
        result = req.add_fetch_demand(self.peer_id, block_hashes, block_indexes)

        if result.local_idxs:
            self._submit_transfer(kv_request_id, result)

    def _on_abort_lookup_fetch(self, msg: dict) -> None:
        AbortLookupFetchMsg.validate(msg)
        kv_request_id = msg[AbortLookupFetchMsg.KV_REQUEST_ID]
        self._outbound.pop(kv_request_id, None)
        ids_to_cancel = (
            tid
            for tid, xfer in self._inflight.items()
            if xfer.kv_request_id == kv_request_id
        )
        for tid in ids_to_cancel:
            del self._inflight[tid]
        self._transport.cancel(ids_to_cancel)
        self._send(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: kv_request_id,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_transfer(self, kv_request_id: str, result) -> None:
        transfer_id = self._transport.write_blocks(
            self.peer_id, result.local_idxs, result.remote_idxs
        )
        if transfer_id is not None:
            self._inflight[transfer_id] = _InflightXfer(
                kv_request_id=kv_request_id,
                block_count=len(result.local_idxs),
                job_ids=result.job_ids,
            )
        else:
            logger.warning(
                "P2PServerSession %s: write_blocks failed for %s (%d blocks)",
                self.peer_id,
                kv_request_id,
                len(result.local_idxs),
            )

    def _send(self, msg: dict) -> None:
        try:
            self._conn.send(msg)
        except Exception:
            logger.warning(
                "P2PServerSession %s: failed to send %s",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
