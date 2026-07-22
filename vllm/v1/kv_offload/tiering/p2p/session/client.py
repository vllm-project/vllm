# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-role state machine for a single peer session.

Handles outgoing fetch requests, abort-on-timeout, abort-ack timeout,
and produces ``LoadResult`` for completed loads. The session coordinator
parses wire messages and dispatches typed arguments here; this module
never touches ``ControlConnection`` directly — it emits via the ``send``
callback injected by the coordinator (which gates on ConnectAck).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortFetchMsg,
    FetchMsg,
    LookupMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 30.0
_ABORT_ACK_TIMEOUT_S = 10.0


class ClientPhase(enum.Enum):
    """Lifecycle of a request's client-side lookup/fetch signalling.

    Advances monotonically. Only ``finish`` reads it, to decide
    whether a terminal empty FetchMsg is owed to release the peer's
    lookup state (owed only from ``PROBING``: a LookupMsg went out but no
    FetchMsg has since closed the peer's lookup phase).
    """

    REGISTERED = enum.auto()  # hashes in probes/unsent, nothing sent yet
    PROBING = enum.auto()  # LookupMsg flushed, awaiting responses
    FETCH_SENT = enum.auto()  # FetchMsg sent (real or terminal empty)


@dataclass
class _InboundLoadState:
    """Client-role state for a single in-flight load request.

    Lives on ``_ClientRequestState.load`` for the duration of a fetch;
    the owning kv_request_id is the dict key, so it isn't stored here.
    """

    job_id: int  # opaque ID assigned by the manager to this load request
    submitted_at: float
    aborted_at: float | None = None


@dataclass
class _ClientRequestState:
    """Per-kv_request_id client-side state.

    One entry per kv_request_id we're driving. Lookup-phase fields are
    used only by symmetric P2P (``do_p2p_fetch``); PD-only loads leave
    ``probes``/``unsent`` empty and drive just ``phase`` and ``load``. An
    entry is dropped once every field is idle — see ``ClientRole._maybe_prune``.
    """

    # -- Lookup phase (symmetric P2P only; untouched for PD) --
    # Probe outcome per OffloadKey: None while in-flight (registered/sent
    # but unresolved), True/False once a LookupRespMsg lands. There is no
    # timeout — finish (via finish_request) is guaranteed after
    # the request's lookup() calls and clears every probe, so an
    # unanswered probe simply stays None until then.
    probes: dict[OffloadKey, bool | None] = field(default_factory=dict)
    # OffloadKeys registered but not yet flushed onto the wire. Drained and
    # cleared by the next flush_pending_lookups.
    unsent: list[OffloadKey] = field(default_factory=list)

    # Monotonic lookup/fetch signalling phase; see ``ClientPhase``.
    phase: ClientPhase = ClientPhase.REGISTERED
    # Set while a fetch is in flight; cleared on completion/abort/timeout.
    load: _InboundLoadState | None = None


class LoadResult(NamedTuple):
    """Result from a session poll, client side."""

    job_id: int
    kv_request_id: str
    success: bool


class ClientCloseResult(NamedTuple):
    """What the manager must fail when a peer session is torn down.

    ``failed_jobs`` is the ``job_id`` of every load still in flight; each
    becomes a ``JobResult(success=False)``. ``failed_req_ids`` is every
    kv_request_id whose lookup() would otherwise defer forever on the dead
    peer — the in-flight loads plus any request holding an unresolved
    symmetric-P2P probe. ``failed_jobs`` is the subset of ``failed_req_ids``
    that had a load job.
    """

    failed_jobs: list[int]
    failed_req_ids: list[str]


class ClientRole:
    """Client-side load state machine for one peer session.

    The coordinator owns the connection and the send-gating; this role
    is given a ``send`` callback and a ``peer_id`` for log messages and
    is otherwise self-contained.
    """

    def __init__(self, peer_id: str, send: Callable[[dict], None]) -> None:
        self._peer_id = peer_id
        self._send = send
        # All per-kv_request_id state lives here. Entries are created
        # lazily by request_blocks / register_lookup and dropped by
        # _maybe_prune once every field is idle.
        self._requests: dict[str, _ClientRequestState] = {}
        # kv_request_ids with unsent lookup hashes for the next flush to
        # visit — the work-list that keeps flush_pending_lookups from
        # scanning every request each scheduler step. Mirrors the server's
        # _serve_pending. Populated by register_lookup, drained by
        # flush_pending_lookups, and discarded on finish/close.
        self._flush_pending: set[str] = set()
        # kv_request_ids with a fetch in flight (``st.load is not None``) —
        # the work-list collect_results walks for timeouts, and the
        # has_active_loads predicate, instead of scanning every request.
        # Kept in exact sync with ``st.load``: armed in request_blocks,
        # discarded wherever load is cleared, and cleared on close.
        self._active_loads: set[str] = set()
        self._completed_loads: list[LoadResult] = []

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_or_create_request(self, kv_request_id: str) -> _ClientRequestState:
        """Get or create the state entry for a kv_request_id."""
        st = self._requests.get(kv_request_id)
        if st is None:
            st = _ClientRequestState()
            self._requests[kv_request_id] = st
        return st

    def _maybe_prune(self, kv_request_id: str) -> None:
        """Drop the entry once it holds no live load or lookup state.

        The sticky ``phase`` is only read by ``finish``. A probe
        clears when its fetch is issued (``request_blocks``) or when the
        request finishes (``finish``/``close``); in the former case
        ``load`` is set and keeps the entry alive, in the latter the phase
        is no longer needed — so dropping on emptiness never loses a phase
        still in use.
        """
        st = self._requests.get(kv_request_id)
        if st is not None and st.load is None and not st.probes and not st.unsent:
            del self._requests[kv_request_id]

    @property
    def has_active_loads(self) -> bool:
        """True if any kv_request_id has a fetch in flight."""
        return bool(self._active_loads)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
        send_ready: bool,
    ) -> None:
        """Register a load request and send the FetchMsg."""
        logger.debug(
            "P2PSession %s: request_blocks job_id=%d kv_request_id=%s "
            "blocks=%d ready=%s",
            self._peer_id,
            job_id,
            kv_request_id,
            len(block_ids),
            send_ready,
        )
        st = self._get_or_create_request(kv_request_id)
        st.load = _InboundLoadState(
            job_id=job_id,
            submitted_at=time.monotonic(),
        )
        self._active_loads.add(kv_request_id)
        st.phase = ClientPhase.FETCH_SENT
        self._send(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: kv_request_id,
                FetchMsg.BLOCK_HASHES: list(keys),
                FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            }
        )
        # Issuing the fetch ends this request's lookup phase, so drop all
        # probe state. Once the peer serves this fetch both sides unpin, so
        # the producer may evict the block; a stale cached True would
        # otherwise let a re-scheduled lookup() return HIT without
        # re-probing, pointing at a block the producer no longer holds.
        # Clearing forces a fresh LookupMsg on re-schedule so the producer
        # answers from current state. For a symmetric-P2P request (probes
        # populated) every fetched block was a confirmed HIT; a PD-only load
        # never probes, so probes is empty and the clear is a no-op.
        if st.probes:
            assert all(st.probes.get(OffloadKey(key)) is True for key in keys)
        st.probes.clear()

    def finish(self, kv_request_id: str) -> None:
        """Finish a request: abort any in-flight load and release lookup state.

        Called from the session's ``finish_request``. The two branches are
        mutually exclusive: ``load`` is set only by ``request_blocks``, which
        also advances ``phase`` to ``FETCH_SENT``, and nothing moves it back
        to ``PROBING`` — so a fetch in flight never coexists with the
        ``PROBING`` phase.

        - Fetch in flight (``load`` set, phase ``FETCH_SENT``): send an
          AbortFetchMsg unless the load is already aborting, then drop it.
        - Outstanding lookups (``PROBING``): a LookupMsg was flushed but no
          FetchMsg has closed the peer's lookup phase. Every FetchMsg the
          server receives in p2p mode is its "request finished" signal (it
          releases lookup state and fires ``cb.finish_request``); when the
          client's lookups all missed no FetchMsg is otherwise sent, so emit
          a terminal empty one purely to trigger those semantics. In
          ``REGISTERED`` the peer never received a LookupMsg and in
          ``FETCH_SENT`` a FetchMsg already closed the phase, so neither owes
          a terminal FetchMsg.

        Then drop all probe/lookup state and prune the entry.
        """
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        if st.load is not None:
            if st.load.aborted_at is None:
                self._send(
                    {
                        TYPE_KEY: AbortFetchMsg.TYPE,
                        AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                    }
                )
            st.load = None
            self._active_loads.discard(kv_request_id)
        elif st.phase is ClientPhase.PROBING:
            st.phase = ClientPhase.FETCH_SENT
            self._send(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: kv_request_id,
                    FetchMsg.BLOCK_HASHES: [],
                    FetchMsg.BLOCK_INDEXES: [],
                }
            )
        st.probes.clear()
        st.unsent.clear()
        self._flush_pending.discard(kv_request_id)
        self._maybe_prune(kv_request_id)

    def on_transfer_done(self, kv_request_id: str, success: bool) -> None:
        """Handle a TransferDoneMsg from the peer."""
        st = self._requests.get(kv_request_id)
        if st is not None and st.load is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=st.load.job_id,
                    kv_request_id=kv_request_id,
                    success=success,
                )
            )
            st.load = None
            self._active_loads.discard(kv_request_id)
            self._maybe_prune(kv_request_id)
        else:
            # No matching in-flight load: either a duplicate
            # transfer_done from the peer (protocol violation) or a
            # benign race with a local cancel/abort/timeout that
            # already popped the entry. We don't track terminated ids,
            # so we can't tell — log so it's findable.
            logger.warning(
                "P2PSession %s: transfer_done for unknown kv_request_id=%s "
                "(duplicate from peer, or raced with local cancel/timeout)",
                self._peer_id,
                kv_request_id,
            )

    def on_abort_ack(self, kv_request_id: str) -> None:
        """Handle an AbortAckMsg from the peer."""
        st = self._requests.get(kv_request_id)
        if st is not None and st.load is not None:
            logger.warning(
                "P2PSession %s: load request %s (job_id=%d) timed out; "
                "load job completed with failure. If this recurs, ensure "
                "PYTHONHASHSEED is set to the same value on all nodes.",
                self._peer_id,
                kv_request_id,
                st.load.job_id,
            )
            self._completed_loads.append(
                LoadResult(
                    job_id=st.load.job_id,
                    kv_request_id=kv_request_id,
                    success=False,
                )
            )
            st.load = None
            self._active_loads.discard(kv_request_id)
            self._maybe_prune(kv_request_id)
        else:
            # See on_transfer_done: same ambiguity (duplicate ack
            # vs. raced with local cancel/timeout that already popped).
            logger.warning(
                "P2PSession %s: abort_ack for unknown kv_request_id=%s "
                "(duplicate from peer, or raced with local cancel/timeout)",
                self._peer_id,
                kv_request_id,
            )

    # ------------------------------------------------------------------
    # Symmetric-P2P lookup (do_p2p_fetch=true)
    # ------------------------------------------------------------------

    def register_lookup(self, kv_request_id: str, block_hash: bytes) -> bool | None:
        """Register or resolve one (kv_request_id, block_hash) probe.

        Idempotent across scheduler steps:
        - First call: creates a pending entry, returns None.
        - Subsequent calls while in-flight: returns None.
        - Once a LookupRespMsg has resolved the entry: returns the cached
          bool result on every call without popping it.

        A resolved entry is retained until its fetch is issued
        (``request_blocks`` pops it) or the request finishes
        (``finish`` clears all entries for the id). A request's
        block set can be re-probed across steps, so popping on read would
        make a repeat probe of an already-resolved hash look brand-new and
        re-queue it, emitting a redundant LookupMsg for an answer we
        already hold. Keeping the entry until fetch makes repeat probes
        free; clearing it at fetch forces a fresh probe if the request is
        re-scheduled, since the block is unpinned once served.
        """
        st = self._get_or_create_request(kv_request_id)
        key = OffloadKey(block_hash)
        if key in st.probes:
            return st.probes[key]
        st.probes[key] = None
        st.unsent.append(key)
        self._flush_pending.add(kv_request_id)
        logger.debug(
            "P2P LOOKUP client %s: REGISTER kv_request_id=%s hash=%s (unsent=%d)",
            self._peer_id,
            kv_request_id,
            block_hash.hex()[:16],
            len(st.unsent),
        )
        return None

    def flush_pending_lookups(self) -> None:
        """Send a LookupMsg for each kv_request_id with unsent entries.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. A request's block set may be discovered
        across several scheduler steps, so more than one LookupMsg can
        go out per kv_request_id — one per step that registered new
        hashes. register_lookup() de-dups in-flight and already-resolved
        (req_id, hash) pairs, so each LookupMsg carries only the hashes
        first probed in that step. The peer's lookup phase for the id is
        still closed by exactly one FetchMsg, which the client contract
        guarantees is sent after every lookup for the id has resolved
        (see request_blocks / finish). Send-gating is handled by
        the injected ``_send`` callback (queues until ConnectAckMsg if
        needed).

        Only requests that registered new hashes since the last flush are
        visited — the ``_flush_pending`` work-list avoids scanning every
        live request each scheduler step.
        """
        for req_id in self._flush_pending:
            st = self._requests.get(req_id)
            if st is None or not st.unsent:
                continue
            # Record that the peer now holds lookup state for this id so
            # finish knows a terminal empty FetchMsg may be owed.
            # Only promote from REGISTERED: once a fetch has gone out
            # (FETCH_SENT) a later LookupMsg must not regress the phase, as
            # no terminal FetchMsg is owed for an already-fetched request.
            if st.phase is ClientPhase.REGISTERED:
                st.phase = ClientPhase.PROBING
            logger.debug(
                "P2P LOOKUP client %s: SEND LookupMsg kv_request_id=%s hashes=%d",
                self._peer_id,
                req_id,
                len(st.unsent),
            )
            self._send(
                {
                    TYPE_KEY: LookupMsg.TYPE,
                    LookupMsg.KV_REQUEST_ID: req_id,
                    LookupMsg.BLOCK_HASHES: list(st.unsent),
                }
            )
            st.unsent = []
        self._flush_pending.clear()

    def on_lookup_resp(
        self,
        kv_request_id: str,
        block_hashes: Sequence[bytes],
        hits: Sequence[bool],
    ) -> None:
        """Apply per-pair hit/miss results from a peer.

        Pairs that don't match a known entry (already cancelled or
        never asked) are silently dropped — the producer is free to
        split or coalesce responses.
        """
        n_hit = sum(1 for hit in hits if hit)
        logger.debug(
            "P2P LOOKUP client %s: RECV LookupRespMsg kv_request_id=%s "
            "hashes=%d hits=%d misses=%d",
            self._peer_id,
            kv_request_id,
            len(block_hashes),
            n_hit,
            len(hits) - n_hit,
        )
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        for h, hit in zip(block_hashes, hits):
            key = OffloadKey(h)
            if key in st.probes:
                st.probes[key] = hit

    def collect_results(self) -> list[LoadResult]:
        """Walk load timeouts and drain completed loads.

        Active requests past ``_LOAD_TIMEOUT_S`` get an AbortFetchMsg
        sent and enter the aborting phase. Aborting requests past
        ``_ABORT_ACK_TIMEOUT_S`` are surfaced as failed loads.

        Lookups have no timeout: an unanswered probe stays None (RETRY)
        until finish_request clears it — see ``_ClientRequestState.probes``.
        """
        now = time.monotonic()
        to_remove: list[str] = []
        for req_id in self._active_loads:
            st = self._requests.get(req_id)
            load = st.load if st is not None else None
            if load is None:
                continue
            if load.aborted_at is None:
                if now - load.submitted_at >= _LOAD_TIMEOUT_S:
                    load.aborted_at = now
                    logger.warning(
                        "P2PSession %s: %s timed out, sending abort",
                        self._peer_id,
                        req_id,
                    )
                    self._send(
                        {
                            TYPE_KEY: AbortFetchMsg.TYPE,
                            AbortFetchMsg.KV_REQUEST_ID: req_id,
                        }
                    )
            else:
                if now - load.aborted_at >= _ABORT_ACK_TIMEOUT_S:
                    to_remove.append(req_id)
                    self._completed_loads.append(
                        LoadResult(
                            job_id=load.job_id,
                            kv_request_id=req_id,
                            success=False,
                        )
                    )
                    logger.warning(
                        "P2PSession %s: abort_ack timed out for kv_request_id=%s",
                        self._peer_id,
                        req_id,
                    )
        for req_id in to_remove:
            self._requests[req_id].load = None
            self._active_loads.discard(req_id)
            self._maybe_prune(req_id)

        results = self._completed_loads
        self._completed_loads = []
        return results

    def close(self) -> ClientCloseResult:
        """Tear down, reporting work the dead peer can no longer complete.

        A request is failed if it has a load in flight (its job fails) or
        holds an unresolved symmetric-P2P probe (its lookup() would defer
        forever on an answer that can never arrive). See ``ClientCloseResult``.
        """
        failed_jobs = [
            st.load.job_id for st in self._requests.values() if st.load is not None
        ]
        failed_req_ids = [
            req_id
            for req_id, st in self._requests.items()
            if st.load is not None or any(hit is None for hit in st.probes.values())
        ]
        self._requests.clear()
        self._flush_pending.clear()
        self._active_loads.clear()
        self._completed_loads.clear()
        return ClientCloseResult(failed_jobs=failed_jobs, failed_req_ids=failed_req_ids)
