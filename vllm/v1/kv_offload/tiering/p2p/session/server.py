# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-role state machine for a single peer session.

Owns block matching (supply vs. demand), inflight RDMA transfers, store-job
timeouts, abort-drain, and produces ``StoreResult`` for completed stores.
The session coordinator parses wire messages and dispatches typed
arguments here; this module never touches ``ControlConnection`` directly
— it emits via the ``send`` callback injected by the coordinator (which
gates on ConnectAck).

Protocol violations the role can detect (today: duplicate ``FetchMsg``
for the same ``kv_request_id``) are surfaced as ``ValueError`` so the
coordinator's ``_dispatch_message`` can reuse its existing
``_protocol_error`` path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_STORE_TIMEOUT_S = 30.0
_CANCEL_DRAIN_TIMEOUT_S = 10.0


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
    ``ServerRole._outbound`` and is not duplicated on the value.
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
    # Job IDs that submit_store'd blocks for this request and have not
    # yet emitted a StoreResult. The terminal-finalize helper drains
    # this set; poll-done and poll-failed discard entries as their
    # StoreResults fire.
    pending_job_ids: set[int] = field(default_factory=set)

    def add_stored_blocks(
        self,
        block_hashes: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: int,
    ) -> _MatchResult:
        """Add locally-stored blocks. Returns matched pairs."""
        self.pending_job_ids.add(job_id)
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


class ServerRole:
    """Server-side store/serve state machine for one peer session.

    The coordinator owns the connection and the send-gating; this role
    is given a ``send`` callback, the ``DataTransport``, and the
    ``peer_id`` for transport calls and log messages.
    """

    def __init__(
        self,
        peer_id: str,
        transport: DataTransport,
        send: Callable[[dict], None],
    ) -> None:
        self._peer_id = peer_id
        self._transport = transport
        self._send = send

        self._outbound: dict[str, _OutboundRequestState] = {}
        # transfer_id → xfer. Mutate ONLY via _inflight_add / _inflight_pop
        # so the per-request count below stays in sync.
        self._inflight: dict[int, _InflightXfer] = {}
        # kv_request_id → number of entries in _inflight for that id.
        # Kept in sync with _inflight; entries that hit zero are removed
        # so `kv_request_id in self._inflight_per_req` is an exact
        # "has any inflight transfer" predicate (O(1) replacement for
        # the previous O(N) scan).
        self._inflight_per_req: dict[str, int] = {}
        self._store_jobs: dict[int, float] = {}  # job_id → submitted_at
        self._pending_aborts: dict[str, float] = {}  # kv_request_id → start
        # StoreResults queued by _finalize_outbound for the next poll
        # tick to surface. Mirrors the deferred-result pattern used for
        # load timeouts.
        self._pending_store_results: list[StoreResult] = []

    # ------------------------------------------------------------------
    # Public API
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

    def on_fetch(
        self,
        kv_request_id: str,
        block_hashes: Sequence[OffloadKey],
        block_indexes: Sequence[int],
    ) -> None:
        """Handle a FetchMsg from the peer.

        Raises ``ValueError`` on a duplicate fetch for the same
        ``kv_request_id``; the coordinator's dispatch loop turns that
        into a protocol-error disconnect.
        """
        logger.debug(
            "P2PSession %s: fetch RECEIVED kv_request_id=%s blocks=%d",
            self._peer_id,
            kv_request_id,
            len(block_hashes),
        )
        existing = self._outbound.get(kv_request_id)
        if existing is not None and existing.demand_received:
            # A second fetch for the same kv_request_id would overwrite
            # `remaining` and leak inflight bookkeeping. Treat as a
            # protocol violation.
            raise ValueError(f"duplicate fetch for kv_request_id={kv_request_id}")
        req = self._outbound.setdefault(kv_request_id, _OutboundRequestState())
        result = req.add_fetch_demand(block_hashes, block_indexes)
        if result.local_idxs:
            self._submit_transfer(kv_request_id, result)
        # Prefiller-first mode: finish_request may have run before
        # fetch arrived. If so, finalize once we know what was
        # demanded — fully satisfied → success, else early-fail.
        if req.finishing and not self._has_inflight_for(kv_request_id):
            self._finalize_outbound(kv_request_id)

    def on_abort_fetch(self, kv_request_id: str) -> None:
        """Handle an AbortFetchMsg from the peer."""
        # Abort for an unknown id may be a benign race/duplicate or a
        # real protocol violation; we don't track completed ids, so warn.
        if kv_request_id not in self._outbound and not self._has_inflight_for(
            kv_request_id
        ):
            logger.warning(
                "P2PSession %s: abort_fetch for unknown kv_request_id=%s "
                "(no outbound or inflight state); benign race or stale",
                self._peer_id,
                kv_request_id,
            )
        # Idempotent: receiving AbortFetchMsg again before we've sent the
        # ack just triggers another drain attempt without resetting the
        # deadline.
        self._pending_aborts.setdefault(kv_request_id, time.monotonic())
        self._drain_abort(kv_request_id)

    def finish(self, kv_request_id: str) -> None:
        """Mark an outbound request finishing.

        No more submit_store calls will arrive for this id. Any blocks
        the peer demanded but we never stored will never come; tell the
        peer to stop waiting (TransferDoneMsg success=False) instead of
        letting it hit _LOAD_TIMEOUT_S.

        If the decoder hasn't sent fetch yet (no demand received),
        defer — on_fetch will finalize once demand arrives.

        If inflight transfers exist for this id, defer — the last
        completing transfer in collect_results will fire the message.
        """
        req = self._outbound.get(kv_request_id)
        if req is None:
            return
        req.finishing = True
        if not req.demand_received:
            return
        if self._has_inflight_for(kv_request_id):
            return
        # Remaining > 0 here: if it had hit 0, the poll-done success
        # branch would have already popped _outbound and we'd have
        # returned at `req is None` above. Helper derives success from
        # remaining and emits StoreResult(success=False) for any
        # leftover pending jobs.
        self._finalize_outbound(kv_request_id)

    def collect_results(self) -> list[StoreResult]:
        """Drain timeouts, deferred results, and transport completions."""
        results: list[StoreResult] = self._timeout_pending_store_jobs()

        if self._pending_store_results:
            results.extend(self._pending_store_results)
            self._pending_store_results.clear()

        poll_result = self._transport.poll()

        for tid in poll_result.done:
            xfer = self._inflight_pop(tid)
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
                    self._peer_id,
                    tid,
                )
                continue
            req = self._outbound.get(xfer.kv_request_id)
            for job_id in xfer.job_ids:
                if self._store_jobs.pop(job_id, None) is None:
                    # Already reported (timeout, cancellation, etc.) —
                    # don't double-emit a contradictory success result.
                    continue
                results.append(StoreResult(job_id=job_id, success=True))
                if req is not None:
                    req.pending_job_ids.discard(job_id)
            if req is not None and req.demand_received:
                req.remaining -= xfer.block_count
                assert req.remaining >= 0, (
                    f"remaining went negative for kv_request_id={xfer.kv_request_id}"
                )
                if req.remaining == 0:
                    self._finalize_outbound(xfer.kv_request_id, success=True)
                elif req.finishing and not self._has_inflight_for(xfer.kv_request_id):
                    self._finalize_outbound(xfer.kv_request_id, success=False)

        failed_kv_request_ids: set[str] | None = None
        for tid in poll_result.failed:
            xfer = self._inflight_pop(tid)
            if xfer is None:
                # See the matching error log in the done branch above.
                logger.error(
                    "P2PSession %s: transport reported failed for unknown "
                    "transfer_id=%d; attached job(s) will fail via "
                    "store-timeout instead of completing now",
                    self._peer_id,
                    tid,
                )
                continue
            if failed_kv_request_ids is None:
                failed_kv_request_ids = set()
            failed_kv_request_ids.add(xfer.kv_request_id)
            req_for_xfer = self._outbound.get(xfer.kv_request_id)
            for job_id in xfer.job_ids:
                if self._store_jobs.pop(job_id, None) is None:
                    # Already reported (timeout, cancellation, etc.) —
                    # don't double-emit.
                    continue
                results.append(StoreResult(job_id=job_id, success=False))
                if req_for_xfer is not None:
                    req_for_xfer.pending_job_ids.discard(job_id)
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
                self._inflight_pop(tid)
            self._transport.cancel(ids_to_cancel)

        return results

    def collect_idle_timeouts(self) -> list[StoreResult]:
        """Run only the store-job timeout sweep.

        Used by the coordinator's no-conn poll path: a pending session
        cannot have inflight transfers (no peer registered yet), so we
        skip the transport poll and the deferred-result drain.
        """
        return self._timeout_pending_store_jobs()

    def drain_pending_aborts(self) -> None:
        """Re-attempt every parked abort once per poll tick."""
        if not self._pending_aborts:
            return
        for kv_request_id in list(self._pending_aborts):
            self._drain_abort(kv_request_id)

    def close(self) -> list[int]:
        """Tear down. Cancels inflight, returns failed store job ids."""
        failed_stores = list(self._store_jobs.keys())
        self._store_jobs.clear()
        if self._inflight:
            self._transport.cancel(list(self._inflight.keys()))
        self._inflight.clear()
        self._inflight_per_req.clear()
        self._outbound.clear()
        self._pending_aborts.clear()
        self._pending_store_results.clear()
        return failed_stores

    # ------------------------------------------------------------------
    # Internal — inflight bookkeeping
    # ------------------------------------------------------------------

    def _has_inflight_for(self, kv_request_id: str) -> bool:
        return kv_request_id in self._inflight_per_req

    def _inflight_add(self, tid: int, xfer: _InflightXfer) -> None:
        """Insert an inflight transfer and bump the per-request count."""
        self._inflight[tid] = xfer
        self._inflight_per_req[xfer.kv_request_id] = (
            self._inflight_per_req.get(xfer.kv_request_id, 0) + 1
        )

    def _inflight_pop(self, tid: int) -> _InflightXfer | None:
        """Pop an inflight transfer and decrement the per-request count.

        Removes the per-request entry once the count hits zero so the
        dict stays bounded and `_has_inflight_for` remains exact.
        """
        xfer = self._inflight.pop(tid, None)
        if xfer is None:
            return None
        new_count = self._inflight_per_req.get(xfer.kv_request_id, 0) - 1
        if new_count > 0:
            self._inflight_per_req[xfer.kv_request_id] = new_count
        else:
            self._inflight_per_req.pop(xfer.kv_request_id, None)
        return xfer

    # ------------------------------------------------------------------
    # Internal — finalize / abort drain
    # ------------------------------------------------------------------

    def _finalize_outbound(
        self,
        kv_request_id: str,
        success: bool | None = None,
    ) -> None:
        """Pop the outbound state and emit terminal results.

        Called when no further work will happen for this kv_request_id
        on the server side: either request_finish has fired and there
        are no inflight transfers, or the last inflight just completed
        while finishing.

        If ``success`` is None, derive it from ``req.remaining == 0``.
        The same flag is used for both the peer's TransferDoneMsg and
        the StoreResult(s) emitted for any leftover pending job_ids.
        """
        req = self._outbound.pop(kv_request_id, None)
        if req is None:
            return
        if success is None:
            success = req.remaining == 0
        for job_id in req.pending_job_ids:
            self._store_jobs.pop(job_id, None)
            self._pending_store_results.append(
                StoreResult(job_id=job_id, success=success)
            )
        self._send(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: kv_request_id,
                TransferDoneMsg.SUCCESS: success,
            }
        )

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
                self._inflight_pop(tid)
            self._transport.cancel(ids, mode="immediate")
            logger.warning(
                "P2PSession %s: cancel drain timed out for kv_request_id=%s,"
                " force-canceled %d transfers",
                self._peer_id,
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
                self._inflight_pop(tid)
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

    # ------------------------------------------------------------------
    # Internal — transfers and store-job timeouts
    # ------------------------------------------------------------------

    def _submit_transfer(self, kv_request_id: str, result: _MatchResult) -> None:
        logger.debug(
            "P2PSession %s: NIXL write_blocks CALL kv_request_id=%s "
            "local_idxs=%d remote_idxs=%d",
            self._peer_id,
            kv_request_id,
            len(result.local_idxs),
            len(result.remote_idxs),
        )
        transfer_id = self._transport.write_blocks(
            self._peer_id, result.local_idxs, result.remote_idxs
        )
        if transfer_id is not None:
            logger.debug(
                "P2PSession %s: NIXL write_blocks SUBMITTED kv_request_id=%s "
                "transfer_id=%d blocks=%d",
                self._peer_id,
                kv_request_id,
                transfer_id,
                len(result.local_idxs),
            )
            self._inflight_add(
                transfer_id,
                _InflightXfer(
                    kv_request_id=kv_request_id,
                    block_count=len(result.local_idxs),
                    job_ids=result.job_ids,
                ),
            )
        else:
            logger.warning(
                "P2PSession %s: write_blocks failed for %s (%d blocks)",
                self._peer_id,
                kv_request_id,
                len(result.local_idxs),
            )
            # The matched blocks were popped from req.demanded /
            # req.available, but no inflight will satisfy them, so
            # remaining will never reach 0 on its own. Mark the
            # request as finishing so the existing terminal paths
            # clean up: if other inflight is in flight, the last one
            # to drain will fire _finalize_outbound(success=False)
            # via the elif branch in collect_results. If
            # nothing else is in flight, finalize now so the peer
            # and the local store jobs don't wait for finish_request
            # or for _STORE_TIMEOUT_S / _LOAD_TIMEOUT_S.
            req = self._outbound.get(kv_request_id)
            if req is not None:
                req.finishing = True
                if not self._has_inflight_for(kv_request_id):
                    self._finalize_outbound(kv_request_id, success=False)

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
            logger.warning("P2PSession %s: store job %d timed out", self._peer_id, jid)
        return results
