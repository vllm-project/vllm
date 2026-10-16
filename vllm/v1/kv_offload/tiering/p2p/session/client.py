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
_LOOKUP_TIMEOUT_S = 30.0


@dataclass
class _InboundLoadState:
    """Client-role state for a single in-flight load request.

    Lives in ``_ClientRequestState.loads`` keyed by round_seq for the
    duration of a fetch; the owning kv_request_id is the outer dict key.
    """

    job_id: int  # opaque ID assigned by the manager to this load request
    submitted_at: float
    pending_fetch: dict | None  # None once handed to the transport
    aborted_at: float | None = None
    failure_reported: bool = False
    request_finished: bool = False


@dataclass
class _ClientRequestState:
    """Per-kv_request_id client-side state.

    One entry per kv_request_id we're driving. Lookup-phase fields are
    used only by symmetric P2P (``do_p2p_fetch``); PD-only loads leave
    ``probes``/``unsent`` empty and drive just ``phase`` and ``loads``. An
    entry is dropped once every field is idle — see ``ClientRole._maybe_prune``.
    """

    # -- Lookup phase (symmetric P2P only; untouched for PD) --
    # Probe outcome per OffloadKey: None while in-flight (registered/sent
    # but unresolved), True/False once a LookupRespMsg lands. Unanswered
    # probes expire the session so the manager can fall back to local compute.
    probes: dict[OffloadKey, bool | None] = field(default_factory=dict)
    # OffloadKeys registered but not yet flushed onto the wire. Drained and
    # cleared by the next flush_pending_lookups.
    unsent: list[OffloadKey] = field(default_factory=list)
    # Current lookup round. LookupMsgs carry it, each fetch closes it and
    # advances it, so every round's supply/demand/completion is isolated
    # on the wire. PD clients never probe and stay on round 0.
    round_seq: int = 0
    # This id ran the symmetric lookup phase (register_lookup); a fetch
    # with keys then requires every key to be a confirmed probe. PD
    # loads never probe.
    probed: bool = False

    # The peer holds lookup state no FetchMsg has closed: a LookupMsg
    # was flushed since the last fetch. finish owes a terminal empty
    # FetchMsg while set, so the peer releases parked supply.
    peer_lookup_open: bool = False
    # In-flight loads keyed by the round their fetch carried. The
    # scheduler submits loads incrementally as chunks resolve, so several
    # can be in flight at once; TransferDone/AbortAck match by round.
    loads: dict[int, _InboundLoadState] = field(default_factory=dict)


class LoadResult(NamedTuple):
    """Result from a session poll, client side."""

    job_id: int
    kv_request_id: str
    success: bool
    memory_safe: bool = True
    failure_reported: bool = False  # A prior result already failed the request.
    request_finished: bool = False


class ClientCloseResult(NamedTuple):
    """What the manager must fail when a peer session is torn down.

    ``failed_jobs`` contains loads not yet reported as unsafe failures;
    ``memory_safe_jobs`` identifies fetches never handed to the transport.
    ``failed_req_ids`` contains newly failed kv_request_ids whose lookup()
    would otherwise defer forever on the dead peer — unreported loads plus
    any request holding an unresolved
    symmetric-P2P probe. ``failed_jobs`` is the subset of ``failed_req_ids``
    that had a load job.
    """

    failed_jobs: list[int]
    failed_req_ids: list[str]
    memory_safe_jobs: frozenset[int]


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
        # kv_request_ids with unsent lookup keys for the next flush to
        # visit — the work-list that keeps flush_pending_lookups from
        # scanning every request each scheduler step. Mirrors the server's
        # _serve_pending. Populated by register_lookup, drained by
        # flush_pending_lookups, and discarded on finish/close.
        self._flush_pending: set[str] = set()
        # kv_request_ids with at least one fetch in flight or aborting — the
        # work-list collect_results walks for timeouts, and the
        # has_active_loads predicate, instead of scanning every request.
        # Kept in exact sync with ``st.loads``.
        self._active_loads: set[str] = set()
        self._completed_loads: list[LoadResult] = []
        # Per-request progress deadlines. Only newly resolved probes extend
        # them; additional keys and repeated scheduler polls do not.
        self._lookup_deadlines: dict[str, float] = {}

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

        ``peer_lookup_open`` is only read by ``finish``, and every path
        that clears the last probe (fetch / finish / close) also settles
        it, so dropping on emptiness never loses a flag still in use.
        """
        st = self._requests.get(kv_request_id)
        if st is not None and not st.loads and not st.probes and not st.unsent:
            del self._requests[kv_request_id]

    def _on_load_terminal(self, kv_request_id: str, st: _ClientRequestState) -> None:
        """Wind down id-level state once no load remains in flight."""
        if st.loads:
            return
        self._active_loads.discard(kv_request_id)
        self._maybe_prune(kv_request_id)

    @property
    def has_active_loads(self) -> bool:
        """True if any kv_request_id has a fetch in flight."""
        return bool(self._active_loads)

    def get_expired_lookup(self) -> tuple[str, float] | None:
        """Return an expired request and seconds without progress, if any."""
        now = time.monotonic()
        for req_id, deadline in self._lookup_deadlines.items():
            if now >= deadline:
                return req_id, now - deadline + _LOOKUP_TIMEOUT_S
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        send_ready: bool,
    ) -> None:
        """Send the FetchMsg closing the current lookup round.

        The scheduler may submit several loads per kv_request_id as its
        matched prefix resolves incrementally; each fetch carries the
        round it closes so the loads stay independent on the wire.
        """
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
        round_seq = st.round_seq
        st.round_seq += 1
        fetch = {
            TYPE_KEY: FetchMsg.TYPE,
            FetchMsg.KV_REQUEST_ID: kv_request_id,
            FetchMsg.KEYS: list(keys),
            FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            FetchMsg.ROUND_SEQ: round_seq,
        }
        st.loads[round_seq] = _InboundLoadState(
            job_id=job_id,
            submitted_at=time.monotonic(),
            pending_fetch=fetch,
        )
        self._active_loads.add(kv_request_id)
        st.peer_lookup_open = False
        self._send(fetch)
        # Issuing the fetch closes this lookup round, so drop all probe
        # state. Once the peer serves the fetch both sides unpin, so a
        # stale cached True would let a later lookup() return HIT for a
        # block the producer may have evicted; clearing forces a fresh
        # probe under the next round.
        if st.probed and keys:
            assert st.probes, (
                f"symmetric fetch for {kv_request_id} has keys but no probes"
            )
            assert all(st.probes.get(key) is True for key in keys)
        st.probes.clear()
        self._lookup_deadlines.pop(kv_request_id, None)

    def mark_fetch_sent(self, msg: dict) -> bool:
        """Mark a live fetch as possibly sent; reject retired or expired messages."""
        st = self._requests.get(msg[FetchMsg.KV_REQUEST_ID])
        load = st.loads.get(msg[FetchMsg.ROUND_SEQ]) if st is not None else None
        if (
            load is None
            or load.pending_fetch is not msg
            or load.aborted_at is not None
            or time.monotonic() - load.submitted_at >= _LOAD_TIMEOUT_S
        ):
            return False
        load.pending_fetch = None
        return True

    def finish(self, kv_request_id: str) -> None:
        """Finish a request: abort in-flight loads and release lookup state.

        Called from the session's ``finish_request``. Sends an
        AbortFetchMsg per sent load not already aborting, and — independently —
        the terminal empty FetchMsg when the peer still holds lookup
        state no fetch has closed (its "request finished" signal: it
        releases lookup state, drains parked supply, and fires
        ``cb.finish_request``). A later round's supply can be parked
        while an earlier round's load is still in flight, so both can be
        owed at once.

        Then drop all probe/lookup state. Never-sent fetches fail on the next
        poll. Sent loads wait until their abort acknowledgement confirms that
        the remote WRITE has stopped.
        """
        self._completed_loads = [
            result._replace(request_finished=True)
            if result.kv_request_id == kv_request_id
            else result
            for result in self._completed_loads
        ]
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        if st.loads:
            for round_seq, load in st.loads.items():
                load.request_finished = True
                if load.aborted_at is not None:
                    continue
                if load.pending_fetch is not None:
                    load.aborted_at = time.monotonic()
                    continue
                self._send(
                    {
                        TYPE_KEY: AbortFetchMsg.TYPE,
                        AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                        AbortFetchMsg.ROUND_SEQ: round_seq,
                    }
                )
                load.aborted_at = time.monotonic()
        if st.peer_lookup_open:
            st.peer_lookup_open = False
            self._send(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: kv_request_id,
                    FetchMsg.KEYS: [],
                    FetchMsg.BLOCK_INDEXES: [],
                    FetchMsg.ROUND_SEQ: st.round_seq,
                }
            )
        st.probes.clear()
        st.unsent.clear()
        self._lookup_deadlines.pop(kv_request_id, None)
        self._flush_pending.discard(kv_request_id)
        self._maybe_prune(kv_request_id)

    def on_transfer_done(
        self, kv_request_id: str, success: bool, round_seq: int
    ) -> None:
        """Handle a TransferDoneMsg from the peer."""
        st = self._requests.get(kv_request_id)
        load = st.loads.pop(round_seq, None) if st is not None else None
        if st is not None and load is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=load.job_id,
                    kv_request_id=kv_request_id,
                    success=success and not load.failure_reported,
                    failure_reported=load.failure_reported,
                    request_finished=load.request_finished,
                )
            )
            self._on_load_terminal(kv_request_id, st)
        else:
            # No matching in-flight load: either a duplicate
            # transfer_done from the peer (protocol violation) or a
            # benign race with a local cancel/abort/timeout that
            # already popped the entry. We don't track terminated ids,
            # so we can't tell — log so it's findable.
            logger.warning(
                "P2PSession %s: transfer_done for unknown kv_request_id=%s "
                "round=%s (duplicate from peer, or raced with local "
                "cancel/timeout)",
                self._peer_id,
                kv_request_id,
                round_seq,
            )

    def on_abort_ack(self, kv_request_id: str, round_seq: int) -> None:
        """Handle an AbortAckMsg from the peer."""
        st = self._requests.get(kv_request_id)
        load = st.loads.pop(round_seq, None) if st is not None else None
        if st is not None and load is not None:
            logger.warning(
                "P2PSession %s: load request %s (job_id=%d) "
                "completed with failure after abort",
                self._peer_id,
                kv_request_id,
                load.job_id,
            )
            self._completed_loads.append(
                LoadResult(
                    job_id=load.job_id,
                    kv_request_id=kv_request_id,
                    success=False,
                    failure_reported=load.failure_reported,
                    request_finished=load.request_finished,
                )
            )
            self._on_load_terminal(kv_request_id, st)
        else:
            # See on_transfer_done: same ambiguity (duplicate ack
            # vs. raced with local cancel/timeout that already popped).
            logger.warning(
                "P2PSession %s: abort_ack for unknown kv_request_id=%s "
                "round=%s (duplicate from peer, or raced with local "
                "cancel/timeout)",
                self._peer_id,
                kv_request_id,
                round_seq,
            )

    # ------------------------------------------------------------------
    # Symmetric-P2P lookup (do_p2p_fetch=true)
    # ------------------------------------------------------------------

    def register_lookup(self, kv_request_id: str, key: bytes) -> bool | None:
        """Register or resolve one (kv_request_id, key) probe.

        Idempotent across scheduler steps:
        - First call: creates a pending entry, returns None.
        - Subsequent calls while in-flight: returns None.
        - Once a LookupRespMsg has resolved the entry: returns the cached
          bool result on every call without popping it.

        A resolved entry is retained until a fetch closes the round
        (``request_blocks`` clears all probes) or the request finishes
        (``finish`` clears all entries for the id). A request's
        block set can be re-probed across steps, so popping on read would
        make a repeat probe of an already-resolved key look brand-new and
        re-queue it, emitting a redundant LookupMsg for an answer we
        already hold. Keeping the entry until fetch makes repeat probes
        free; clearing at fetch forces a fresh probe under the next
        round, since the block is unpinned once served.
        """
        st = self._get_or_create_request(kv_request_id)
        st.probed = True
        okey = OffloadKey(key)
        if okey in st.probes:
            return st.probes[okey]
        st.probes[okey] = None
        st.unsent.append(okey)
        self._flush_pending.add(kv_request_id)
        logger.debug(
            "P2P LOOKUP client %s: REGISTER kv_request_id=%s key=%s (unsent=%d)",
            self._peer_id,
            kv_request_id,
            key.hex()[:16],
            len(st.unsent),
        )
        return None

    def flush_pending_lookups(self, send_ready: bool) -> None:
        """Send a LookupMsg for each kv_request_id with unsent entries.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. A request's block set may be discovered
        across several scheduler steps, so more than one LookupMsg can
        go out per kv_request_id — one per step that registered new
        keys, all tagged with the current round. register_lookup()
        de-dups in-flight and already-resolved (req_id, key) pairs, so
        each LookupMsg carries only the keys first probed in that step.
        Wait for ConnectAckMsg before sending or starting the response deadline.

        Only requests that registered new keys since the last flush are
        visited — the ``_flush_pending`` work-list avoids scanning every
        live request each scheduler step.
        """
        if not send_ready:
            return
        for req_id in self._flush_pending:
            st = self._requests.get(req_id)
            if st is None or not st.unsent:
                continue
            # The peer now holds lookup state for this id; finish owes a
            # terminal empty FetchMsg until a fetch closes it.
            st.peer_lookup_open = True
            self._lookup_deadlines.setdefault(
                req_id, time.monotonic() + _LOOKUP_TIMEOUT_S
            )
            logger.debug(
                "P2P LOOKUP client %s: SEND LookupMsg kv_request_id=%s keys=%d",
                self._peer_id,
                req_id,
                len(st.unsent),
            )
            self._send(
                {
                    TYPE_KEY: LookupMsg.TYPE,
                    LookupMsg.KV_REQUEST_ID: req_id,
                    LookupMsg.KEYS: list(st.unsent),
                    LookupMsg.ROUND_SEQ: st.round_seq,
                }
            )
            st.unsent = []
        self._flush_pending.clear()

    def on_lookup_resp(
        self,
        kv_request_id: str,
        keys: Sequence[bytes],
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
            "keys=%d hits=%d misses=%d",
            self._peer_id,
            kv_request_id,
            len(keys),
            n_hit,
            len(hits) - n_hit,
        )
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        progressed = False
        for h, hit in zip(keys, hits):
            key = OffloadKey(h)
            if key in st.probes:
                progressed |= st.probes[key] is None
                st.probes[key] = hit
        if all(hit is not None for hit in st.probes.values()):
            self._lookup_deadlines.pop(kv_request_id, None)
        elif progressed:
            self._lookup_deadlines[kv_request_id] = time.monotonic() + _LOOKUP_TIMEOUT_S

    def collect_results(self) -> list[LoadResult]:
        """Walk load timeouts and drain completed loads.

        Sent loads past ``_LOAD_TIMEOUT_S`` get an AbortFetchMsg sent and
        enter the aborting phase. Aborting loads past
        ``_ABORT_ACK_TIMEOUT_S`` report failure but remain tracked until the
        peer confirms completion. Never-sent fetches retire safely on finish
        or load timeout without sending an abort.
        """
        now = time.monotonic()
        unsent_failures: list[tuple[str, int]] = []
        for req_id in self._active_loads:
            st = self._requests[req_id]
            assert st.loads
            for round_seq, load in st.loads.items():
                if load.pending_fetch is not None and (
                    load.aborted_at is not None
                    or now - load.submitted_at >= _LOAD_TIMEOUT_S
                ):
                    self._completed_loads.append(
                        LoadResult(
                            load.job_id,
                            req_id,
                            success=False,
                            request_finished=load.request_finished,
                        )
                    )
                    unsent_failures.append((req_id, round_seq))
                    continue
                if load.aborted_at is None:
                    if now - load.submitted_at >= _LOAD_TIMEOUT_S:
                        load.aborted_at = now
                        logger.warning(
                            "P2PSession %s: %s round=%s timed out, sending abort",
                            self._peer_id,
                            req_id,
                            round_seq,
                        )
                        self._send(
                            {
                                TYPE_KEY: AbortFetchMsg.TYPE,
                                AbortFetchMsg.KV_REQUEST_ID: req_id,
                                AbortFetchMsg.ROUND_SEQ: round_seq,
                            }
                        )
                else:
                    if (
                        not load.failure_reported
                        and now - load.aborted_at >= _ABORT_ACK_TIMEOUT_S
                    ):
                        load.failure_reported = True
                        self._completed_loads.append(
                            LoadResult(
                                job_id=load.job_id,
                                kv_request_id=req_id,
                                success=False,
                                memory_safe=False,
                                request_finished=load.request_finished,
                            )
                        )
                        logger.warning(
                            "P2PSession %s: abort_ack timed out for "
                            "kv_request_id=%s round=%s; quarantining destination slots",
                            self._peer_id,
                            req_id,
                            round_seq,
                        )
        for req_id, round_seq in unsent_failures:
            st = self._requests[req_id]
            del st.loads[round_seq]
            self._on_load_terminal(req_id, st)
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
            load.job_id
            for st in self._requests.values()
            for load in st.loads.values()
            if not load.failure_reported
        ]
        failed_req_ids = [
            req_id
            for req_id, st in self._requests.items()
            if any(
                not load.failure_reported and not load.request_finished
                for load in st.loads.values()
            )
            or any(hit is None for hit in st.probes.values())
        ]
        memory_safe_jobs = frozenset(
            load.job_id
            for st in self._requests.values()
            for load in st.loads.values()
            if load.pending_fetch is not None
        )
        self._requests.clear()
        self._flush_pending.clear()
        self._active_loads.clear()
        self._completed_loads.clear()
        self._lookup_deadlines.clear()
        return ClientCloseResult(
            failed_jobs=failed_jobs,
            failed_req_ids=failed_req_ids,
            memory_safe_jobs=memory_safe_jobs,
        )
