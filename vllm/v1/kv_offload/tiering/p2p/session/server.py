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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    LookupRespMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId, ParentManager
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_STORE_TIMEOUT_S = 30.0
_CANCEL_DRAIN_TIMEOUT_S = 10.0
# Cap on time the server holds a HIT_PENDING / RETRY key from an inbound
# LookupMsg before falling back to MISS. Long enough that an in-flight
# primary write or a just-started promotion typically completes; short
# enough that the consumer doesn't sit idle on a stuck producer.
_LOOKUP_PENDING_TIMEOUT_S = 5.0


class StoreResult(NamedTuple):
    """Result from a session poll, server side."""

    job_id: int
    success: bool


class _MatchResult(NamedTuple):
    """Result of block matching: pairs ready for transfer."""

    local_idxs: list[int]
    remote_idxs: list[int]
    # The set of store job IDs that contributed blocks
    job_ids: set[int]


@dataclass
class _OutboundRequestState:
    """Server-role state for a single fetch round of a peer request.

    A kv_request_id may run several lookup→fetch rounds; rounds live in
    ``_ServerRequestState.outbound`` keyed by the wire ``round_seq``, so
    terminals touch only their own round.
    """

    # Supply came from inbound lookup pins (symmetric): no late
    # submit_store can arrive, so unmatched fetch demand fails fast. PD
    # rounds park demand for stores instead.
    lookup_supplied: bool = False
    demand_received: bool = False
    available: dict[OffloadKey, tuple[int, int]] = field(
        default_factory=dict
    )  # key → (job_id, local_block_idx): blocks we have, awaiting demand
    demanded: dict[OffloadKey, int] = field(
        default_factory=dict
    )  # key → remote_block_idx: blocks peer wants, awaiting supply
    remaining: int = 0  # blocks that need to be transferred to client
    finishing: bool = False  # Signal finish request ASAP
    inflight: int = 0  # transfers submitted for this round, not yet polled
    # Job IDs that submit_store'd blocks for this round and have not
    # yet emitted a StoreResult. The terminal-finalize helper drains
    # this set; poll-done and poll-failed discard entries as their
    # StoreResults fire.
    pending_job_ids: set[int] = field(default_factory=set)

    def add_stored_blocks(
        self,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: int,
    ) -> _MatchResult:
        """Add locally-stored blocks. Returns matched pairs."""
        self.pending_job_ids.add(job_id)
        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        for key, local_idx in zip(keys, block_ids):
            remote_idx = self.demanded.pop(key, None)
            if remote_idx is not None:
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
            else:
                self.available[key] = (job_id, local_idx)
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids={job_id} if local_idxs else set(),
        )

    def add_fetch_demand(
        self,
        keys: Sequence[OffloadKey],
        block_indexes: Sequence[int],
    ) -> _MatchResult:
        """Register the peer's fetch demand. Returns matched pairs."""
        self.demand_received = True
        self.remaining = len(keys)

        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        job_ids: set[int] = set()
        for key, remote_idx in zip(keys, block_indexes):
            stored_entry = self.available.pop(key, None)
            if stored_entry is not None:
                stored_job_id, local_idx = stored_entry
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
                job_ids.add(stored_job_id)
            else:
                self.demanded[key] = remote_idx
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids=job_ids,
        )


@dataclass
class _InflightXfer:
    """Metadata for a single inflight RDMA transfer, keyed by transfer_id."""

    kv_request_id: str
    block_count: int
    # The set of store job IDs that contributed blocks to this transfer.
    job_ids: set[int]
    # Round this transfer serves and its key in ``st.outbound``;
    # remaining/finalize apply only while the round is still registered.
    # Dummy default for test-seeded entries.
    round: _OutboundRequestState = field(default_factory=_OutboundRequestState)
    round_key: int = 0


@dataclass
class _ActiveLookup:
    """In-flight state for one inbound LookupMsg.

    Aggregates per-key HIT/MISS resolutions and defers the single
    outbound LookupRespMsg until every key has been resolved or the
    ``deadline`` fires (remaining ``pending`` keys then force-MISS).
    Exactly one LookupRespMsg is emitted per LookupMsg, carrying every
    key in the original wire order.
    """

    lookup_id: int
    kv_request_id: str
    ctx: ReqContext
    # Wire round these probes belong to; pins park under it.
    round_seq: int = 0
    # Keys from the inbound LookupMsg, preserved in wire order so
    # the aggregated response goes back in the same order.
    keys: list[OffloadKey] = field(default_factory=list)
    # Per-key resolution: True = HIT, False = MISS. A key is present
    # here once definitively resolved; still-pending keys are only
    # in ``pending``.
    resolved: dict[OffloadKey, bool] = field(default_factory=dict)
    # Keys still awaiting resolution (HIT_PENDING / RETRY from
    # ``parent.lookup``); re-polled by ``_resolve_pending_lookups``.
    pending: set[OffloadKey] = field(default_factory=set)
    # Absolute deadline (``time.monotonic``). Once reached, remaining
    # ``pending`` keys are force-resolved to MISS so the consumer
    # can fall back instead of waiting on a stuck producer.
    deadline: float = 0.0


class _PendingLookup(NamedTuple):
    """A raw inbound LookupMsg awaiting resolution.

    Enqueued by ``on_lookup`` during dispatch and drained by the next
    ``serve_external_requests``. The deadline for any resulting
    HIT_PENDING / RETRY key is measured from ``enqueued_at``.
    """

    keys: list[OffloadKey]
    enqueued_at: float
    round_seq: int = 0


@dataclass
class _ServerRequestState:
    """Per-kv_request_id server-side state.

    Consolidates the outbound serve/transfer state, the symmetric-P2P
    inbound-lookup state, abort bookkeeping, and the inflight-transfer
    count for one kv_request_id. The owning id is the ``_requests`` dict
    key and is not duplicated here. An entry is dropped once every field
    is idle — see ``ServerRole._maybe_prune``.
    """

    # Fetch rounds keyed by wire round_seq. A round is created by its
    # first supply or its fetch and removed at its terminal (finalize /
    # failure / abort).
    outbound: dict[int, _OutboundRequestState] = field(default_factory=dict)
    # Raw inbound LookupMsgs not yet processed against the ParentManager.
    pending_lookups: list[_PendingLookup] = field(default_factory=list)
    # Per-LookupMsg state parked with HIT_PENDING / RETRY keys, keyed by
    # the (globally unique) lookup_id and re-polled each serve.
    lookups: dict[int, _ActiveLookup] = field(default_factory=dict)
    # Transfer ids in ``ServerRole._inflight`` for this id. Kept in sync
    # via _inflight_add / _inflight_pop so a non-empty set is an exact
    # "has any inflight transfer" predicate and the abort drain can
    # enumerate this request's transfers without scanning all of
    # ``_inflight``.
    inflight_tids: set[int] = field(default_factory=set)


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

        # All per-kv_request_id state lives here. Entries are created
        # lazily and dropped by _maybe_prune once every field is idle.
        self._requests: dict[str, _ServerRequestState] = {}
        # kv_request_ids with lookup work (unprocessed pending_lookups or
        # parked lookups) for the next serve to visit — the work-list that
        # keeps serve_external_requests from scanning every request.
        self._serve_pending: set[str] = set()
        # transfer_id → xfer. Mutate ONLY via _inflight_add / _inflight_pop
        # so the per-request inflight_tids stays in sync.
        self._inflight: dict[int, _InflightXfer] = {}
        self._store_jobs: dict[int, float] = {}  # job_id → submitted_at
        # StoreResults queued by _finalize_outbound for the next poll
        # tick to surface. Mirrors the deferred-result pattern used for
        # load timeouts.
        self._pending_store_results: list[StoreResult] = []
        # Synthetic lookup ctxs whose ``on_request_finished`` still needs to
        # fire but which were closed outside a serve window (FetchMsg / local
        # finish popped their parked lookup). Drained via
        # ``parent.on_request_finished`` in ``serve_external_requests``.
        self._finished_lookup_ctxs: list[ReqContext] = []
        self._lookup_id_counter: int = 0
        # Parked aborts awaiting drain, keyed by (kv_request_id, round)
        # with the abort start time.
        self._pending_aborts: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_or_create_request(self, kv_request_id: str) -> _ServerRequestState:
        """Get or create the state entry for a kv_request_id."""
        st = self._requests.get(kv_request_id)
        if st is None:
            st = _ServerRequestState()
            self._requests[kv_request_id] = st
        return st

    def _maybe_prune(self, kv_request_id: str) -> None:
        """Drop the entry once it holds no live state."""
        st = self._requests.get(kv_request_id)
        if (
            st is not None
            and not st.outbound
            and not st.inflight_tids
            and not st.lookups
            and not st.pending_lookups
            and not any(kv == kv_request_id for kv, _ in self._pending_aborts)
        ):
            del self._requests[kv_request_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
        round_seq: int = 0,
        *,
        from_lookup: bool = False,
    ) -> None:
        """New blocks stored locally — match within their fetch round.

        Lookup pins carry the round they were probed under; PD
        submit_store batches share PD's single round 0.
        """
        self._store_jobs[job_id] = time.monotonic()
        st = self._get_or_create_request(kv_request_id)
        rnd = st.outbound.get(round_seq)
        if rnd is None:
            rnd = st.outbound[round_seq] = _OutboundRequestState()
        if from_lookup:
            rnd.lookup_supplied = True
        result = rnd.add_stored_blocks(keys, block_ids, job_id)
        if result.local_idxs and rnd.demand_received:
            self._submit_transfer(kv_request_id, result, rnd, round_seq)

    def on_fetch(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_indexes: Sequence[int],
        round_seq: int = 0,
    ) -> None:
        """Handle a FetchMsg from the peer.

        A non-empty fetch binds and closes its round, leaving lookup
        state alone (the next round's LookupMsg may already be in
        flight). The terminal empty fetch closes the id: parked lookups
        are popped and every remaining round drained. A second fetch for
        a round already holding demand raises ValueError
        (protocol-error disconnect).
        """
        logger.debug(
            "P2PSession %s: fetch RECEIVED kv_request_id=%s round=%s blocks=%d",
            self._peer_id,
            kv_request_id,
            round_seq,
            len(keys),
        )
        st = self._requests.get(kv_request_id)
        existing = st.outbound.get(round_seq) if st is not None else None
        if existing is not None and existing.demand_received:
            raise ValueError(
                f"duplicate fetch for kv_request_id={kv_request_id} round={round_seq}"
            )
        st = self._get_or_create_request(kv_request_id)
        req = st.outbound.get(round_seq)
        if req is None:
            req = st.outbound[round_seq] = _OutboundRequestState()
        result = req.add_fetch_demand(keys, block_indexes)
        if not keys:
            # Terminal empty fetch: close the lookup phase and drain
            # every round with no TransferDoneMsg (nothing waits on it).
            self._finish_inbound_lookups(kv_request_id)
            for key in list(st.outbound):
                self._finalize_outbound(kv_request_id, key, send_done=False)
            return
        if req.lookup_supplied and req.demanded:
            # A symmetric round's supply always precedes its fetch, so
            # unmatched demand is unservable — fail now, not at the load
            # timeout. PD rounds keep parking demand for stores that
            # arrive later.
            logger.warning(
                "P2PSession %s: fetch kv_request_id=%s round=%s demanded %d "
                "blocks but %d have no pinned supply; failing fetch "
                "immediately",
                self._peer_id,
                kv_request_id,
                round_seq,
                len(keys),
                len(req.demanded),
            )
            self._finalize_outbound(kv_request_id, round_seq, success=False)
            return
        if result.local_idxs:
            self._submit_transfer(kv_request_id, result, req, round_seq)
        # Prefiller-first mode: finish_request may have run before
        # fetch arrived. If so, finalize once we know what was
        # demanded — fully satisfied → success, else early-fail.
        if req.finishing and req.inflight == 0:
            self._finalize_outbound(kv_request_id, round_seq)

    def on_abort_fetch(self, kv_request_id: str, round_seq: int = 0) -> None:
        """Handle an AbortFetchMsg from the peer, cancelling one round."""
        # Abort for an unknown id may be a benign race/duplicate or a
        # real protocol violation; we don't track completed ids, so warn.
        st = self._requests.get(kv_request_id)
        if (st is None or not st.outbound) and not self._has_inflight_for(
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
        self._get_or_create_request(kv_request_id)
        self._pending_aborts.setdefault((kv_request_id, round_seq), time.monotonic())
        self._drain_abort(kv_request_id, round_seq)

    def on_lookup(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        round_seq: int = 0,
    ) -> None:
        """Enqueue a LookupMsg from a symmetric-P2P consumer.

        Dispatch runs during ``session.poll()`` where the
        :class:`ParentManager` handle is not available, so this only
        records the raw request. It is resolved — querying the tiering
        manager and emitting the aggregated ``LookupRespMsg`` — by the
        next :meth:`serve_external_requests`, the sole window in which
        parent calls are valid.
        """
        logger.debug(
            "P2P LOOKUP server %s: RECV LookupMsg kv_request_id=%s round=%s keys=%d",
            self._peer_id,
            kv_request_id,
            round_seq,
            len(keys),
        )
        self._get_or_create_request(kv_request_id).pending_lookups.append(
            _PendingLookup(
                keys=list(keys),
                enqueued_at=time.monotonic(),
                round_seq=round_seq,
            )
        )
        self._serve_pending.add(kv_request_id)

    def serve_external_requests(self, parent: ParentManager) -> None:
        """Resolve inbound peer lookups against the tiering manager.

        Called once per scheduler step with a ``parent`` handle valid
        only for this call. Drains newly-enqueued LookupMsgs, re-polls
        any parked HIT_PENDING / RETRY keys, and releases the
        bookkeeping for lookups closed since the last serve.
        """
        for kv_request_id in list(self._serve_pending):
            st = self._requests.get(kv_request_id)
            if st is None:
                self._serve_pending.discard(kv_request_id)
                continue
            if st.pending_lookups:
                pending = st.pending_lookups
                st.pending_lookups = []
                for pl in pending:
                    self._process_inbound_lookup(
                        kv_request_id, pl.keys, pl.enqueued_at, pl.round_seq, parent
                    )
            self._resolve_pending_lookups(kv_request_id, parent)
            st = self._requests.get(kv_request_id)
            if st is None or (not st.pending_lookups and not st.lookups):
                self._serve_pending.discard(kv_request_id)
                self._maybe_prune(kv_request_id)

        if self._finished_lookup_ctxs:
            for ctx in self._finished_lookup_ctxs:
                parent.on_request_finished(ctx)
            self._finished_lookup_ctxs = []

    def _poll_lookup_keys(
        self,
        lookup: _ActiveLookup,
        keys: Iterable[OffloadKey],
        parent: ParentManager,
    ) -> list[OffloadKey]:
        """Poll ``keys`` against the tiering manager and pin any HITs.

        For each key not already definitively resolved, query
        ``parent.lookup`` and record the outcome on ``lookup``: HIT / MISS
        land in ``resolved`` and clear ``pending``; HIT_PENDING / RETRY park
        in ``pending`` for a later serve. Newly-HIT keys are pinned in one
        batch via :meth:`_pin_and_register_hits` and also returned (for
        caller logging).

        Shared by the first-sighting pass (:meth:`_process_inbound_lookup`)
        and the re-poll pass (:meth:`_resolve_pending_lookups`); callers pass
        a de-duplicated ``keys`` collection.
        """
        new_hits: list[OffloadKey] = []
        for h in keys:
            if h in lookup.resolved:
                continue
            result = parent.lookup(h, lookup.ctx)
            if result is LookupResult.HIT:
                new_hits.append(h)
                lookup.resolved[h] = True
                lookup.pending.discard(h)
            elif result is LookupResult.MISS:
                lookup.resolved[h] = False
                lookup.pending.discard(h)
            else:
                lookup.pending.add(h)
        if new_hits:
            self._pin_and_register_hits(lookup, new_hits, parent)
        return new_hits

    def _process_inbound_lookup(
        self,
        kv_request_id: str,
        keys: list[OffloadKey],
        enqueued_at: float,
        round_seq: int,
        parent: ParentManager,
    ) -> None:
        """Resolve one enqueued LookupMsg against ``parent``.

        For each key, query the tiering manager via ``parent.lookup``
        and pin any HITs immediately via ``parent.create_store_job``
        (plumbed into the existing ``add_stored_blocks`` matching path so
        the eventual FetchMsg finds them). HIT_PENDING / RETRY keys park
        in :class:`_ActiveLookup` for re-polling by
        :meth:`_resolve_pending_lookups`.

        The outbound ``LookupRespMsg`` is deferred until every key has
        settled to HIT or MISS (or the batch-level ``deadline`` fires,
        forcing any stragglers to MISS). One LookupRespMsg goes out per
        LookupMsg — carrying every key in wire order — after which
        ``parent.on_request_finished`` fires and the entry is dropped.
        """
        self._lookup_id_counter += 1
        lookup_id = self._lookup_id_counter
        ctx = ReqContext(req_id=f"p2p:{self._peer_id}:{kv_request_id}:lu{lookup_id}")
        lookup = _ActiveLookup(
            lookup_id=lookup_id,
            kv_request_id=kv_request_id,
            ctx=ctx,
            round_seq=round_seq,
            keys=list(keys),
            deadline=enqueued_at + _LOOKUP_PENDING_TIMEOUT_S,
        )

        # Open per-request bookkeeping for this synthetic ctx before the
        # first lookup; released by ``on_request_finished`` once every
        # key has settled.
        parent.on_new_request(ctx)

        # dict.fromkeys de-duplicates keys within the LookupMsg while
        # preserving wire order — each unique key is polled once.
        hit_keys = self._poll_lookup_keys(lookup, dict.fromkeys(lookup.keys), parent)

        logger.debug(
            "P2P LOOKUP server %s: RESOLVED kv_request_id=%s hits=%d misses=%d "
            "pending=%d",
            self._peer_id,
            kv_request_id,
            len(hit_keys),
            sum(1 for v in lookup.resolved.values() if not v),
            len(lookup.pending),
        )

        if lookup.pending:
            self._get_or_create_request(kv_request_id).lookups[lookup_id] = lookup
        else:
            # Every key resolved on first sight — emit the aggregated
            # response now and close the synthetic request.
            self._finalize_lookup(lookup, parent)

    def _pin_and_register_hits(
        self,
        lookup: _ActiveLookup,
        keys: list[OffloadKey],
        parent: ParentManager,
    ) -> None:
        """Pin primary slots for HIT keys and park them as the lookup's
        round supply via ``add_stored_blocks``.

        Caller has already confirmed every key is HIT (single-threaded
        scheduler ⇒ no eviction race), so the JobMetadata returned by
        ``parent.create_store_job`` carries parallel ``keys``/``block_ids``
        of length ``len(keys)``.
        """
        meta = parent.create_store_job(keys, lookup.ctx)
        self.add_stored_blocks(
            lookup.kv_request_id,
            list(meta.keys),
            meta.block_ids.tolist(),
            meta.job_id,
            round_seq=lookup.round_seq,
            from_lookup=True,
        )

    def _resolve_pending_lookups(
        self, kv_request_id: str, parent: ParentManager
    ) -> None:
        """Re-poll a request's deferred LookupMsg keys; finalize when ready.

        Walks every parked :class:`_ActiveLookup` for ``kv_request_id`` and
        re-calls ``parent.lookup`` per still-pending key, moving HIT/MISS
        results into ``resolved``. If ``deadline`` has passed, remaining
        ``pending`` keys are force-resolved to MISS so the consumer
        can fall back instead of waiting on a stuck producer. Newly-HIT
        keys are pinned via ``parent.create_store_job`` in one call per
        affected lookup. Lookups whose ``pending`` empties out get their
        aggregated LookupRespMsg sent by :meth:`_finalize_lookup`.
        """
        st = self._requests.get(kv_request_id)
        if st is None or not st.lookups:
            return
        now = time.monotonic()
        finished_lookups: list[int] = []
        for lookup_id, lookup in st.lookups.items():
            self._poll_lookup_keys(lookup, list(lookup.pending), parent)

            if lookup.pending and now >= lookup.deadline:
                for h in lookup.pending:
                    lookup.resolved[h] = False
                lookup.pending.clear()

            if not lookup.pending:
                finished_lookups.append(lookup_id)

        for lookup_id in finished_lookups:
            lookup = st.lookups.pop(lookup_id)
            self._finalize_lookup(lookup, parent)

    def _finalize_lookup(self, lookup: _ActiveLookup, parent: ParentManager) -> None:
        """Emit the aggregated LookupRespMsg and close the synthetic request.

        Called once per lookup when ``pending`` is empty — either every
        key resolved to HIT or MISS, or the deadline forced remaining
        stragglers to MISS. Preserves the wire order of the inbound
        LookupMsg so the client can zip keys and hits positionally.
        """
        if lookup.keys:
            hits = [lookup.resolved[h] for h in lookup.keys]
            n_hit = sum(1 for v in hits if v)
            logger.debug(
                "P2P LOOKUP server %s: SEND LookupRespMsg kv_request_id=%s "
                "keys=%d hits=%d misses=%d",
                self._peer_id,
                lookup.kv_request_id,
                len(lookup.keys),
                n_hit,
                len(hits) - n_hit,
            )
            self._send(
                {
                    TYPE_KEY: LookupRespMsg.TYPE,
                    LookupRespMsg.KV_REQUEST_ID: lookup.kv_request_id,
                    LookupRespMsg.KEYS: list(lookup.keys),
                    LookupRespMsg.HITS: hits,
                }
            )
        parent.on_request_finished(lookup.ctx)

    def _finish_inbound_lookups(self, kv_request_id: str) -> None:
        """Close the server-side lookup phase for ``kv_request_id``.

        Pops every parked ``_ActiveLookup`` for this id (so
        ``_resolve_pending_lookups`` cannot promote a HIT_PENDING /
        RETRY key into a fresh ``parent.create_store_job`` after this
        point) and queues each ``lookup.ctx`` for
        ``parent.on_request_finished`` (fired by the next
        ``serve_external_requests``, since no parent handle is available
        during dispatch) so the TieringManager can release per-lookup
        bookkeeping. Any still-unprocessed raw LookupMsg for this id is
        dropped — it never got ``on_new_request``, so nothing is owed.
        The aggregated LookupRespMsg is skipped — the client already
        knows the request is over (it just sent a terminal FetchMsg, or
        is finishing locally).

        Called on the two events that mean "no more lookup traffic for
        ``kv_request_id`` is expected on this session": the terminal
        empty FetchMsg from the peer and a local ``finish``. Whichever
        fires second is a no-op.
        """
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        st.pending_lookups.clear()
        for lookup in st.lookups.values():
            self._finished_lookup_ctxs.append(lookup.ctx)
        st.lookups.clear()
        self._serve_pending.discard(kv_request_id)
        self._maybe_prune(kv_request_id)

    def finish(self, kv_request_id: str) -> None:
        """Mark an outbound request finishing.

        No more submit_store calls will arrive for this id. Any blocks
        the peer demanded but we never stored will never come; tell the
        peer to stop waiting (TransferDoneMsg success=False) instead of
        letting it hit _LOAD_TIMEOUT_S.

        Also drops any in-flight lookups for this kv_request_id and
        queues their ctxs for ``parent.on_request_finished`` so the
        TieringManager can release per-request bookkeeping. (For
        symmetric P2P this path is rarely hit since the producer has no
        local request lifecycle for the consumer's id; this cleanup is
        mostly active on the PD side.)

        If the decoder hasn't sent fetch yet (no demand received),
        defer — on_fetch will finalize once demand arrives.

        If inflight transfers exist for this id, defer — the last
        completing transfer in collect_results will fire the message.
        """
        self._finish_inbound_lookups(kv_request_id)

        st = self._requests.get(kv_request_id)
        if st is None:
            return
        for key, req in list(st.outbound.items()):
            req.finishing = True
            if not req.demand_received or req.inflight:
                # No demand yet (prefiller-first): on_fetch finalizes via
                # `finishing`. Inflight: the last completion finalizes.
                continue
            self._finalize_outbound(kv_request_id, key)

    def collect_results(self) -> list[StoreResult]:
        """Drain timeouts, deferred results, and transport completions.

        Inbound LookupMsg resolution (including re-polling HIT_PENDING /
        RETRY keys) is NOT done here — it runs in
        ``serve_external_requests`` where the ParentManager is available.
        """
        results: list[StoreResult] = self._timeout_pending_store_jobs()

        if self._pending_store_results:
            results.extend(self._pending_store_results)
            self._pending_store_results.clear()

        # Scope the poll to this peer: the transport is shared across all peer
        # sessions of the engine, and poll() drains completed handles. An
        # unscoped poll here would consume sibling sessions' completions and
        # report them as "unknown transfer_id", starving those sessions.
        poll_result = self._transport.poll(self._peer_id)

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
            results.extend(self._settle_xfer_jobs(xfer, success=True))
            rnd = xfer.round
            st = self._requests.get(xfer.kv_request_id)
            if st is not None and st.outbound.get(xfer.round_key) is rnd:
                rnd.remaining -= xfer.block_count
                assert rnd.remaining >= 0, (
                    f"remaining went negative for kv_request_id={xfer.kv_request_id}"
                )
                if rnd.remaining == 0:
                    self._finalize_outbound(
                        xfer.kv_request_id, xfer.round_key, success=True
                    )
                elif rnd.finishing and rnd.inflight == 0:
                    self._finalize_outbound(
                        xfer.kv_request_id, xfer.round_key, success=False
                    )
            self._maybe_prune(xfer.kv_request_id)

        failed_rounds: list[tuple[str, _OutboundRequestState]] | None = None
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
            results.extend(self._settle_xfer_jobs(xfer, success=False))
            rnd = xfer.round
            st = self._requests.get(xfer.kv_request_id)
            if st is not None and st.outbound.get(xfer.round_key) is rnd:
                del st.outbound[xfer.round_key]
                if failed_rounds is None:
                    failed_rounds = []
                failed_rounds.append((xfer.kv_request_id, rnd))
                self._send(
                    {
                        TYPE_KEY: TransferDoneMsg.TYPE,
                        TransferDoneMsg.KV_REQUEST_ID: xfer.kv_request_id,
                        TransferDoneMsg.SUCCESS: False,
                        TransferDoneMsg.ROUND_SEQ: xfer.round_key,
                    }
                )
            self._maybe_prune(xfer.kv_request_id)

        # Cancel each failed round's other inflight and fail its
        # remaining store jobs — nothing else will settle them.
        if failed_rounds:
            for kv_request_id, rnd in failed_rounds:
                ids_to_cancel = [
                    tid for tid, x in self._inflight.items() if x.round is rnd
                ]
                for tid in ids_to_cancel:
                    self._inflight_pop(tid)
                if ids_to_cancel:
                    self._transport.cancel(ids_to_cancel)
                results.extend(self._fail_round_jobs(rnd))
                self._maybe_prune(kv_request_id)

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
        for kv_request_id, round_seq in list(self._pending_aborts):
            self._drain_abort(kv_request_id, round_seq)

    def close(self) -> tuple[list[int], list[ReqContext]]:
        """Tear down. Cancels inflight.

        Returns ``(failed_store_job_ids, failed_serves)`` where
        ``failed_serves`` are synthetic lookup ctxs still owing a
        ``parent.on_request_finished``. The session is going away with no
        parent handle in hand, so the manager flushes these in its next
        ``serve_external_requests``.
        """
        failed_stores = list(self._store_jobs.keys())
        self._store_jobs.clear()
        if self._inflight:
            self._transport.cancel(list(self._inflight.keys()))
        self._inflight.clear()
        self._pending_store_results.clear()
        # Surface every synthetic ctx still owing on_request_finished so
        # the manager can release the TieringManager's per-request
        # bookkeeping: parked lookups plus any already queued from a
        # FetchMsg / finish that closed them before this teardown.
        failed_serves = [
            lu.ctx for st in self._requests.values() for lu in st.lookups.values()
        ]
        failed_serves.extend(self._finished_lookup_ctxs)
        self._requests.clear()
        self._serve_pending.clear()
        self._pending_aborts.clear()
        self._finished_lookup_ctxs.clear()
        return failed_stores, failed_serves

    @property
    def has_inflight_transfers(self) -> bool:
        """True if any outbound store transfer is still in flight."""
        return bool(self._inflight)

    # ------------------------------------------------------------------
    # Internal — inflight bookkeeping
    # ------------------------------------------------------------------

    def _has_inflight_for(self, kv_request_id: str) -> bool:
        st = self._requests.get(kv_request_id)
        return st is not None and bool(st.inflight_tids)

    def _inflight_add(self, tid: int, xfer: _InflightXfer) -> None:
        """Insert an inflight transfer and record it on the request."""
        self._inflight[tid] = xfer
        self._get_or_create_request(xfer.kv_request_id).inflight_tids.add(tid)

    def _inflight_pop(self, tid: int) -> _InflightXfer | None:
        """Pop an inflight transfer and drop it from the request's set.

        Callers are responsible for the ``_maybe_prune`` that may follow once
        the request's other state has also cleared.
        """
        xfer = self._inflight.pop(tid, None)
        if xfer is None:
            return None
        xfer.round.inflight -= 1
        assert xfer.round.inflight >= 0
        st = self._requests.get(xfer.kv_request_id)
        if st is not None:
            st.inflight_tids.discard(tid)
        return xfer

    def _settle_xfer_jobs(
        self, xfer: _InflightXfer, success: bool
    ) -> list[StoreResult]:
        """Emit StoreResults for a completed transfer's store jobs.

        Pops each attached job from ``_store_jobs`` and clears it from
        its round's pending set. A job already popped (via timeout,
        cancel, etc.) is skipped so we never double-emit a contradictory
        result.
        """
        results: list[StoreResult] = []
        for job_id in xfer.job_ids:
            if self._store_jobs.pop(job_id, None) is None:
                continue
            results.append(StoreResult(job_id=job_id, success=success))
            xfer.round.pending_job_ids.discard(job_id)
        return results

    # ------------------------------------------------------------------
    # Internal — finalize / abort drain
    # ------------------------------------------------------------------

    def _fail_round_jobs(self, rnd: _OutboundRequestState) -> list[StoreResult]:
        """Fail a terminated round's still-pending store jobs (idempotent)."""
        results: list[StoreResult] = []
        for job_id in rnd.pending_job_ids:
            if self._store_jobs.pop(job_id, None) is None:
                continue
            results.append(StoreResult(job_id=job_id, success=False))
        rnd.pending_job_ids.clear()
        return results

    def _finalize_outbound(
        self,
        kv_request_id: str,
        round_key: int,
        success: bool | None = None,
        send_done: bool = True,
    ) -> None:
        """Pop one round and emit its terminal results.

        If ``success`` is None, derive it from ``req.remaining == 0``.
        ``send_done=False`` skips the TransferDoneMsg (terminal empty
        fetch). Other rounds of the id are untouched.
        """
        st = self._requests[kv_request_id]
        req = st.outbound.pop(round_key)
        if success is None:
            success = req.demand_received and req.remaining == 0
        settled = self._fail_round_jobs(req) if not success else None
        if settled is not None:
            self._pending_store_results.extend(settled)
        else:
            for job_id in req.pending_job_ids:
                if self._store_jobs.pop(job_id, None) is None:
                    continue
                self._pending_store_results.append(
                    StoreResult(job_id=job_id, success=True)
                )
            req.pending_job_ids.clear()
        logger.debug(
            "P2PSession %s: finalize kv_request_id=%s round=%s success=%s "
            "remaining=%d leftover_available=%d send_done=%s",
            self._peer_id,
            kv_request_id,
            round_key,
            success,
            req.remaining,
            len(req.available),
            send_done,
        )
        if send_done and req.demand_received:
            self._send(
                {
                    TYPE_KEY: TransferDoneMsg.TYPE,
                    TransferDoneMsg.KV_REQUEST_ID: kv_request_id,
                    TransferDoneMsg.SUCCESS: success,
                    TransferDoneMsg.ROUND_SEQ: round_key,
                }
            )
        self._maybe_prune(kv_request_id)

    def _drain_abort(self, kv_request_id: str, round_seq: int) -> None:
        """One drain attempt for a pending abort.

        Detaches the aborted round, then asks the transport to cancel its
        inflight transfers in ``mode="wait"``. Sends ``AbortAckMsg`` once
        nothing remains inflight, or after ``_CANCEL_DRAIN_TIMEOUT_S``
        falls back to ``mode="immediate"`` and acks anyway.
        """
        st = self._requests[kv_request_id]
        rnd = st.outbound.pop(round_seq, None)
        if rnd is not None:
            # Its transfers are being cancelled; fail its jobs now
            # instead of leaking them to the store timeout.
            self._pending_store_results.extend(self._fail_round_jobs(rnd))
        ids = [
            tid
            for tid, x in self._inflight.items()
            if x.kv_request_id == kv_request_id and x.round_key == round_seq
        ]
        if not ids:
            self._finalize_abort(kv_request_id, round_seq)
            return

        started_at = self._pending_aborts[(kv_request_id, round_seq)]
        if time.monotonic() - started_at >= _CANCEL_DRAIN_TIMEOUT_S:
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
            self._finalize_abort(kv_request_id, round_seq)
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
            self._finalize_abort(kv_request_id, round_seq)

    def _finalize_abort(self, kv_request_id: str, round_seq: int) -> None:
        self._pending_aborts.pop((kv_request_id, round_seq), None)
        self._send(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: kv_request_id,
                AbortAckMsg.ROUND_SEQ: round_seq,
            }
        )
        self._maybe_prune(kv_request_id)

    # ------------------------------------------------------------------
    # Internal — transfers and store-job timeouts
    # ------------------------------------------------------------------

    def _submit_transfer(
        self,
        kv_request_id: str,
        result: _MatchResult,
        rnd: _OutboundRequestState,
        round_key: int,
    ) -> None:
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
            rnd.inflight += 1
            self._inflight_add(
                transfer_id,
                _InflightXfer(
                    kv_request_id=kv_request_id,
                    block_count=len(result.local_idxs),
                    job_ids=result.job_ids,
                    round=rnd,
                    round_key=round_key,
                ),
            )
        else:
            logger.warning(
                "P2PSession %s: write_blocks failed for %s (%d blocks)",
                self._peer_id,
                kv_request_id,
                len(result.local_idxs),
            )
            # The matched blocks were popped from rnd.demanded /
            # rnd.available, but no inflight will satisfy them, so
            # remaining will never reach 0 on its own. Mark the round
            # as finishing so the existing terminal paths clean up: if
            # other transfers of this round are in flight, the last one
            # to drain will fire _finalize_outbound(success=False) via
            # the elif branch in collect_results. If nothing else is in
            # flight, finalize now so the peer and the local store jobs
            # don't wait for finish_request or for _STORE_TIMEOUT_S /
            # _LOAD_TIMEOUT_S.
            rnd.finishing = True
            st = self._requests.get(kv_request_id)
            if (
                st is not None
                and st.outbound.get(round_key) is rnd
                and rnd.inflight == 0
            ):
                self._finalize_outbound(kv_request_id, round_key, success=False)

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
