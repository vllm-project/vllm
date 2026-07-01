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
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
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
class _InboundRequestState:
    """Client-role state for a single load request."""

    job_id: int  # opaque ID assigned by the manager to this load request
    kv_request_id: str
    submitted_at: float
    aborted_at: float | None = None


@dataclass
class _LookupState:
    """Client-role state for a single (kv_request_id, block_hash) probe.

    submitted_at: when the manager first registered the lookup.
    sent_at: when the LookupMsg carrying this hash was emitted, or
        None while still in the local batch awaiting flush.
    result: True/False once a LookupRespMsg resolved this hash, or
        None while still in-flight.
    """

    submitted_at: float
    sent_at: float | None = None
    result: bool | None = None


class LoadResult(NamedTuple):
    """Result from a session poll, client side."""

    job_id: int
    kv_request_id: str
    success: bool


class ClientRole:
    """Client-side load state machine for one peer session.

    The coordinator owns the connection and the send-gating; this role
    is given a ``send`` callback and a ``peer_id`` for log messages and
    is otherwise self-contained.
    """

    def __init__(self, peer_id: str, send: Callable[[dict], None]) -> None:
        self._peer_id = peer_id
        self._send = send
        self._inbound: dict[str, _InboundRequestState] = {}
        self._completed_loads: list[LoadResult] = []
        # Symmetric-P2P lookup state, keyed by (kv_request_id, block_hash).
        self._lookups: dict[tuple[str, bytes], _LookupState] = {}
        # Fast index of unsent entries, kept in sync with _lookups:
        # (req_id, h) is in _unsent_lookups_by_req[req_id] iff
        # _lookups[(req_id, h)].sent_at is None. Populated in
        # register_lookup, drained in flush_pending_lookups.
        self._unsent_lookups_by_req: dict[str, list[bytes]] = {}
        # Tracks kv_request_ids we have already emitted a LookupMsg for,
        # to enforce the at-most-one-LookupMsg-per-request invariant.
        # Cleared on cancel_lookups / close so an id that is later
        # cancelled and re-used (if that ever happens) is not falsely
        # flagged.
        self._flushed_req_ids: set[str] = set()
        # Tracks kv_request_ids we have already emitted a FetchMsg for.
        # cancel_lookups uses this to decide whether it must send a
        # terminal empty FetchMsg to close the peer's lookup phase — see
        # the ``cancel_lookups`` docstring for the full rationale.
        self._fetch_sent_req_ids: set[str] = set()

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
        self._inbound[kv_request_id] = _InboundRequestState(
            job_id=job_id,
            kv_request_id=kv_request_id,
            submitted_at=time.monotonic(),
        )
        self._fetch_sent_req_ids.add(kv_request_id)
        self._send(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: kv_request_id,
                FetchMsg.BLOCK_HASHES: list(keys),
                FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            }
        )

    def cancel(self, kv_request_id: str) -> None:
        """Cancel a pending load. Sends AbortFetchMsg if still active."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None and req.aborted_at is None:
            self._send(
                {
                    TYPE_KEY: AbortFetchMsg.TYPE,
                    AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                }
            )

    def on_transfer_done(self, kv_request_id: str, success: bool) -> None:
        """Handle a TransferDoneMsg from the peer."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id,
                    kv_request_id=kv_request_id,
                    success=success,
                )
            )
        else:
            # No matching _inbound entry: either a duplicate
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
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id,
                    kv_request_id=kv_request_id,
                    success=False,
                )
            )
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
        - Once a LookupRespMsg has resolved the entry: pops and
          returns the bool result.
        """
        key = (kv_request_id, block_hash)
        state = self._lookups.get(key)
        if state is not None and state.result is not None:
            del self._lookups[key]
            return state.result
        if state is None:
            self._lookups[key] = _LookupState(submitted_at=time.monotonic())
            self._unsent_lookups_by_req.setdefault(kv_request_id, []).append(block_hash)
        return None

    def flush_pending_lookups(self) -> None:
        """Send one LookupMsg per kv_request_id covering all unsent entries.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. Send-gating is handled by the injected
        ``_send`` callback (queues until ConnectAckMsg if needed).
        """
        if not self._unsent_lookups_by_req:
            return
        now = time.monotonic()
        for req_id, hashes in self._unsent_lookups_by_req.items():
            # At most one LookupMsg per kv_request_id per session:
            # all block lookups for a request are registered in a
            # single scheduler step and flushed together here.
            # While in-flight, register_lookup() for the same
            # (req_id, hash) is a no-op, so no new unsent entries
            # can accumulate for a req_id after its first flush.
            assert req_id not in self._flushed_req_ids, (
                f"LookupMsg already sent for kv_request_id={req_id}"
            )
            self._flushed_req_ids.add(req_id)
            self._send(
                {
                    TYPE_KEY: LookupMsg.TYPE,
                    LookupMsg.KV_REQUEST_ID: req_id,
                    LookupMsg.BLOCK_HASHES: list(hashes),
                }
            )
            for h in hashes:
                self._lookups[(req_id, h)].sent_at = now
        self._unsent_lookups_by_req.clear()

    def on_lookup_resp(
        self,
        kv_request_id: str,
        block_hashes: Sequence[bytes],
        hits: Sequence[bool],
    ) -> None:
        """Apply per-pair hit/miss results from a peer.

        Pairs that don't match a known entry (already cancelled, timed
        out, or never asked) are silently dropped — the producer is
        free to split or coalesce responses.
        """
        for h, hit in zip(block_hashes, hits):
            state = self._lookups.get((kv_request_id, h))
            if state is None:
                continue
            state.result = hit

    def cancel_lookups(self, kv_request_id: str) -> None:
        """Drop lookup state and, if needed, close the peer's request.

        Every FetchMsg the server receives in p2p mode is the
        server-side "request finished" signal for its kv_request_id:
        no further ``cb.create_store_job`` will fire, all server-side
        lookup state for the id is released, and ``cb.finish_request``
        fires on the TieringManager. In the happy path the FetchMsg
        carrying blocks is that signal. But if the client's lookups
        all missed (or timed out) no FetchMsg is ever sent, so on the
        finish path we emit an empty FetchMsg purely to trigger those
        semantics on the peer.

        We only send the terminal FetchMsg when a LookupMsg was actually
        flushed (``kv_request_id in _flushed_req_ids``): if the peer
        never received a LookupMsg for this id, it has no state to
        release.
        """
        if (
            kv_request_id in self._flushed_req_ids
            and kv_request_id not in self._fetch_sent_req_ids
        ):
            self._fetch_sent_req_ids.add(kv_request_id)
            self._send(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: kv_request_id,
                    FetchMsg.BLOCK_HASHES: [],
                    FetchMsg.BLOCK_INDEXES: [],
                }
            )
        keys = [k for k in self._lookups if k[0] == kv_request_id]
        for k in keys:
            del self._lookups[k]
        self._unsent_lookups_by_req.pop(kv_request_id, None)
        self._flushed_req_ids.discard(kv_request_id)

    def collect_results(self) -> list[LoadResult]:
        """Walk timeouts and drain completed loads.

        Active requests past ``_LOAD_TIMEOUT_S`` get an AbortFetchMsg
        sent and enter the aborting phase. Aborting requests past
        ``_ABORT_ACK_TIMEOUT_S`` are surfaced as failed loads.
        Lookup entries past ``_LOAD_TIMEOUT_S`` since send are
        resolved as misses (False) so the caller falls back to local
        prefill.
        """
        now = time.monotonic()
        to_remove: list[str] = []
        for req_id, req in self._inbound.items():
            if req.aborted_at is None:
                if now - req.submitted_at >= _LOAD_TIMEOUT_S:
                    req.aborted_at = now
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
                if now - req.aborted_at >= _ABORT_ACK_TIMEOUT_S:
                    to_remove.append(req_id)
                    self._completed_loads.append(
                        LoadResult(
                            job_id=req.job_id,
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
            self._inbound.pop(req_id)

        # Lookup timeout sweep: entries that were sent more than
        # _LOAD_TIMEOUT_S ago without a response resolve to miss so
        # the next register_lookup() returns False and the caller
        # falls back to local prefill.
        for state in self._lookups.values():
            if (
                state.result is None
                and state.sent_at is not None
                and now - state.sent_at >= _LOAD_TIMEOUT_S
            ):
                state.result = False

        results = self._completed_loads
        self._completed_loads = []
        return results

    def close(self) -> list[tuple[int, str]]:
        """Tear down. Returns ``(job_id, kv_request_id)`` for pending loads."""
        failed = [(req.job_id, req.kv_request_id) for req in self._inbound.values()]
        self._inbound.clear()
        self._completed_loads.clear()
        self._lookups.clear()
        self._unsent_lookups_by_req.clear()
        self._flushed_req_ids.clear()
        self._fetch_sent_req_ids.clear()
        return failed
