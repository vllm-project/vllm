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


@dataclass
class _InboundLoadState:
    """Client-role state for a single in-flight load request.

    Lives in ``_ClientRequestState.loads`` keyed by round_seq for the
    duration of a fetch; the owning kv_request_id is the outer dict key.
    """

    job_id: int  # opaque ID assigned by the manager to this load request
    submitted_at: float
    aborted_at: float | None = None


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
    # but unresolved), True/False once a LookupRespMsg lands. There is no
    # timeout — finish (via finish_request) is guaranteed after
    # the request's lookup() calls and clears every probe, so an
    # unanswered probe simply stays None until then.
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
        # kv_request_ids with unsent lookup keys for the next flush to
        # visit — the work-list that keeps flush_pending_lookups from
        # scanning every request each scheduler step. Mirrors the server's
        # _serve_pending. Populated by register_lookup, drained by
        # flush_pending_lookups, and discarded on finish/close.
        self._flush_pending: set[str] = set()
        # kv_request_ids with at least one fetch in flight — the work-list
        # collect_results walks for timeouts, and the has_active_loads
        # predicate, instead of scanning every request. Kept in exact sync
        # with ``st.loads``.
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
        st.loads[round_seq] = _InboundLoadState(
            job_id=job_id,
            submitted_at=time.monotonic(),
        )
        self._active_loads.add(kv_request_id)
        st.peer_lookup_open = False
        self._send(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: kv_request_id,
                FetchMsg.KEYS: list(keys),
                FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
                FetchMsg.ROUND_SEQ: round_seq,
            }
        )
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

    def finish(self, kv_request_id: str) -> None:
        """Finish a request: abort in-flight loads and release lookup state.

        Called from the session's ``finish_request``. Sends an
        AbortFetchMsg per load not already aborting, and — independently —
        the terminal empty FetchMsg when the peer still holds lookup
        state no fetch has closed (its "request finished" signal: it
        releases lookup state, drains parked supply, and fires
        ``cb.finish_request``). A later round's supply can be parked
        while an earlier round's load is still in flight, so both can be
        owed at once.

        Then drop all probe/lookup state and prune the entry.
        """
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        if st.loads:
            for round_seq, load in st.loads.items():
                if load.aborted_at is not None:
                    continue
                self._send(
                    {
                        TYPE_KEY: AbortFetchMsg.TYPE,
                        AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                        AbortFetchMsg.ROUND_SEQ: round_seq,
                    }
                )
            st.loads.clear()
            self._active_loads.discard(kv_request_id)
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
                    success=success,
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
                "P2PSession %s: load request %s (job_id=%d) timed out; "
                "load job completed with failure. If this recurs, ensure all "
                "nodes use the same prefix-cache hash seed (matching "
                "PYTHONHASHSEED, if set) and hash algorithm.",
                self._peer_id,
                kv_request_id,
                load.job_id,
            )
            self._completed_loads.append(
                LoadResult(
                    job_id=load.job_id,
                    kv_request_id=kv_request_id,
                    success=False,
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

    def flush_pending_lookups(self) -> None:
        """Send a LookupMsg for each kv_request_id with unsent entries.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. A request's block set may be discovered
        across several scheduler steps, so more than one LookupMsg can
        go out per kv_request_id — one per step that registered new
        keys, all tagged with the current round. register_lookup()
        de-dups in-flight and already-resolved (req_id, key) pairs, so
        each LookupMsg carries only the keys first probed in that step.
        Send-gating is handled by the injected ``_send`` callback
        (queues until ConnectAckMsg if needed).

        Only requests that registered new keys since the last flush are
        visited — the ``_flush_pending`` work-list avoids scanning every
        live request each scheduler step.
        """
        for req_id in self._flush_pending:
            st = self._requests.get(req_id)
            if st is None or not st.unsent:
                continue
            # The peer now holds lookup state for this id; finish owes a
            # terminal empty FetchMsg until a fetch closes it.
            st.peer_lookup_open = True
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
        for h, hit in zip(keys, hits):
            key = OffloadKey(h)
            if key in st.probes:
                st.probes[key] = hit

    def collect_results(self) -> list[LoadResult]:
        """Walk load timeouts and drain completed loads.

        Loads past ``_LOAD_TIMEOUT_S`` get an AbortFetchMsg sent and
        enter the aborting phase. Aborting loads past
        ``_ABORT_ACK_TIMEOUT_S`` are surfaced as failed.

        Lookups have no timeout: an unanswered probe stays None (RETRY)
        until finish_request clears it — see ``_ClientRequestState.probes``.
        """
        now = time.monotonic()
        to_remove: list[tuple[str, int]] = []
        for req_id in self._active_loads:
            st = self._requests[req_id]
            assert st.loads
            for round_seq, load in st.loads.items():
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
                    if now - load.aborted_at >= _ABORT_ACK_TIMEOUT_S:
                        to_remove.append((req_id, round_seq))
                        self._completed_loads.append(
                            LoadResult(
                                job_id=load.job_id,
                                kv_request_id=req_id,
                                success=False,
                            )
                        )
                        logger.warning(
                            "P2PSession %s: abort_ack timed out for "
                            "kv_request_id=%s round=%s",
                            self._peer_id,
                            req_id,
                            round_seq,
                        )
        for req_id, round_seq in to_remove:
            st = self._requests[req_id]
            st.loads.pop(round_seq)
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
            load.job_id for st in self._requests.values() for load in st.loads.values()
        ]
        failed_req_ids = [
            req_id
            for req_id, st in self._requests.items()
            if st.loads or any(hit is None for hit in st.probes.values())
        ]
        self._requests.clear()
        self._flush_pending.clear()
        self._active_loads.clear()
        self._completed_loads.clear()
        return ClientCloseResult(failed_jobs=failed_jobs, failed_req_ids=failed_req_ids)
