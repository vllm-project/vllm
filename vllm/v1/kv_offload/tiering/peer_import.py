# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bulk import of a peer's CPU-tier KV cache through the P2P tier.

A starting replica copies the cache of a serving one. The peer lists its keys
most recently used first; batches then move them from the coldest end through
the normal lookup, promotion and fetch path, so the peer's recency order carries
over. Each batch is one internal request with its own `remote_kv_source` id,
and is promoted only once every probe is answered: a fetch drops the round's
unanswered probes. Batches share the session with live requests, and a probe
left unanswered past the lookup timeout ends it.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
    from vllm.v1.kv_offload.tiering.p2p.session.session import KeyPage

logger = init_logger(__name__)

_LIST_PAGE = 16384
# A batch pins its hits on the peer until its fetch, which must land within the
# P2P load timeout, so batches are sized by bytes.
_BATCH_BYTES = 1 << 30
_MAX_BATCH_CHUNKS = 1024
_MAX_BATCHES_IN_FLIGHT = 4
_CONNECT_TIMEOUT_S = 30.0
_CONNECT_RETRY_S = 1.0
_LIST_PAGE_TIMEOUT_S = 30.0


class PeerKeySource(Protocol):
    """What an import needs from its secondary tier (the P2P tier)."""

    @property
    def block_size_bytes(self) -> int: ...

    def peer_port(self, base_port: int) -> int: ...

    def peer_ready(self, peer_id: str, connect: bool = False) -> bool | None: ...

    def peer_down_reason(self, peer_id: str) -> str | None: ...

    def request_peer_keys(
        self, peer_id: str, list_id: str, cursor: int, limit: int
    ) -> bool: ...

    def take_peer_keys(
        self, peer_id: str, list_id: str, abandon: bool = False
    ) -> "KeyPage | None": ...

    def probes_pending(self, peer_id: str, kv_request_id: str) -> bool: ...


@dataclass
class _Batch:
    req_context: ReqContext
    # Probed keys, then after promotion the keys still being fetched.
    keys: list[OffloadKey]
    promoted: bool = False
    done: bool = False


@dataclass
class PeerImport:
    """One import from one peer; advanced once per scheduler step."""

    import_id: str
    host: str
    port: int
    tier: PeerKeySource
    max_chunks: int
    timeout_s: float
    # Chunks in this replica's primary tier.
    capacity: int
    state: str = "connecting"
    error: str | None = None
    # The listing, freed when the import ends; listed counts it.
    keys: list[OffloadKey] = field(default_factory=list)
    listed: int = 0
    imported: int = 0
    present: int = 0
    missed: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _batch_chunks: int = 0
    _session_opened: bool = False
    _next_connect_at: float = 0.0
    _cursor: int = 0
    _listing_done: bool = False
    _page_requested_at: float | None = None
    # Batches are cut from the coldest end of the listing.
    _next_end: int = 0
    _num_batches: int = 0
    _batches: list[_Batch] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Reservations in flight stay within half the tier, or they fail.
        self._batch_chunks = max(
            1,
            min(
                _MAX_BATCH_CHUNKS,
                _BATCH_BYTES // self.tier.block_size_bytes,
                self.capacity // (2 * _MAX_BATCHES_IN_FLIGHT),
            ),
        )

    @property
    def peer_id(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def active(self) -> bool:
        return self.state not in ("done", "failed")

    def status(self) -> dict[str, Any]:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return {
            "import_id": self.import_id,
            "peer": self.peer_id,
            "state": self.state,
            "error": self.error,
            "listed": self.listed,
            "imported": self.imported,
            "already_present": self.present,
            "missed": self.missed,
            "failed": self.failed,
            "elapsed_s": round(end - self.started_at, 3),
        }

    def advance(self, manager: "TieringOffloadingManager") -> None:
        """Move the import forward; called with the manager lock held."""
        if not self.active:
            return
        now = time.monotonic()
        if now - self.started_at > self.timeout_s:
            self._finish(manager, "failed", f"timed out after {self.timeout_s:.0f}s")
            return
        if self.state == "connecting":
            if now < self._next_connect_at:
                return
            # Open the session once; reopening a dead one reconnects every step.
            ready = self.tier.peer_ready(self.peer_id, not self._session_opened)
            self._session_opened = True
            if ready is None:
                if self.tier.peer_down_reason(self.peer_id) is not None:
                    self._fail_on_lost_session(
                        manager, "session to peer closed before the handshake completed"
                    )
                elif now - self.started_at > _CONNECT_TIMEOUT_S:
                    self._finish(manager, "failed", "could not open a session to peer")
                else:  # Session limit, reconnect backoff or a transport error.
                    self._session_opened = False
                    self._next_connect_at = now + _CONNECT_RETRY_S
            elif ready:
                self.state = "importing"
            elif now - self.started_at > _CONNECT_TIMEOUT_S:
                self._finish(manager, "failed", "peer handshake timed out")
            return
        if self.tier.peer_ready(self.peer_id) is None:
            self._fail_on_lost_session(manager, "session to peer lost")
            return
        if manager.promotions_held:  # Lookups miss until a reset is flushed.
            return
        self._pump_listing(manager, now)
        if not self.active:
            return
        self._advance_batches(manager)
        self._start_batches(manager)
        if self._listing_done and self._next_end == 0 and not self._batches:
            self._finish(manager, "done", None)

    def _fail_on_lost_session(
        self, manager: "TieringOffloadingManager", error: str
    ) -> None:
        # The cause (e.g. a config mismatch the peer rejected) is otherwise
        # only in the logs.
        reason = self.tier.peer_down_reason(self.peer_id)
        self._finish(manager, "failed", f"{error}: {reason}" if reason else error)

    def abort(self, manager: "TieringOffloadingManager", reason: str) -> None:
        if self.active:
            self._finish(manager, "failed", reason)

    def _pump_listing(self, manager: "TieringOffloadingManager", now: float) -> None:
        if self._listing_done:
            return
        if self._page_requested_at is None:
            limit = min(_LIST_PAGE, self.max_chunks - len(self.keys))
            if not self.tier.request_peer_keys(
                self.peer_id, self.import_id, self._cursor, limit
            ):
                self._finish(manager, "failed", "session lost while listing keys")
                return
            self._page_requested_at = now
            return
        page = self.tier.take_peer_keys(self.peer_id, self.import_id)
        if page is None:
            if now - self._page_requested_at > _LIST_PAGE_TIMEOUT_S:
                self._finish(manager, "failed", "peer did not answer key listing")
            return
        if page.error is not None:
            self._finish(manager, "failed", f"key listing incomplete: {page.error}")
            return
        keys, next_cursor = page.keys, page.next_cursor
        self._page_requested_at = None
        self.keys.extend(keys[: self.max_chunks - len(self.keys)])
        self.listed = len(self.keys)
        self._cursor = next_cursor
        if next_cursor < 0 or len(self.keys) >= self.max_chunks:
            self._listing_done = True
            self._next_end = len(self.keys)

    def _start_batches(self, manager: "TieringOffloadingManager") -> None:
        if not self._listing_done:
            return
        while len(self._batches) < _MAX_BATCHES_IN_FLIGHT and self._next_end > 0:
            start = max(0, self._next_end - self._batch_chunks)
            keys = self.keys[start : self._next_end]
            self._next_end = start
            kv_request_id = f"{self.import_id}-{self._num_batches}"
            self._num_batches += 1
            ctx = ReqContext(
                req_id=kv_request_id,
                kv_transfer_params={
                    "remote_kv_source": {
                        "kv_request_id": kv_request_id,
                        "remote_host": self.host,
                        "remote_port": self.port,
                    }
                },
            )
            # The lowest position is ranked most recent when the batch finishes.
            for position, key in enumerate(keys):
                ctx.set_offload_key_position(key, position)
            manager.on_new_request(ctx)
            manager.exclude_from_lookup_metrics(ctx)
            probed = []
            for key in keys:
                result = manager.lookup(key, ctx)
                if result is LookupResult.RETRY:
                    probed.append(key)
                elif result is LookupResult.MISS:
                    self.missed += 1
                else:  # Already local, or being written by another request.
                    self.present += 1
            self._batches.append(_Batch(ctx, probed, done=not probed))
        self._finish_batches_in_order(manager)

    def _advance_batches(self, manager: "TieringOffloadingManager") -> None:
        for batch in self._batches:
            if batch.done:
                continue
            ctx = batch.req_context
            if not batch.promoted:
                if self.tier.probes_pending(self.peer_id, ctx.req_id):
                    continue
                in_flight = []
                for key in batch.keys:
                    result = manager.lookup(key, ctx)
                    if result is LookupResult.HIT_PENDING:
                        in_flight.append(key)
                    elif result is LookupResult.HIT:
                        self.present += 1
                    else:  # Gone from the peer, tier full, or session lost.
                        self.missed += 1
                batch.keys = in_flight
                batch.promoted = True  # Submitted at the end of this step.
                continue
            waiting = []
            for key in batch.keys:
                result = manager.primary_tier.lookup(key, ctx)
                if result is LookupResult.HIT:
                    self.imported += 1
                elif result is LookupResult.HIT_PENDING:
                    waiting.append(key)
                else:
                    self.failed += 1
            batch.keys = waiting
            batch.done = not waiting
        self._finish_batches_in_order(manager)

    def _finish_batches_in_order(self, manager: "TieringOffloadingManager") -> None:
        # Finishing makes a batch's keys the most recent: keep start order.
        while self._batches and self._batches[0].done:
            manager.on_request_finished(self._batches.pop(0).req_context)

    def _finish(
        self, manager: "TieringOffloadingManager", state: str, error: str | None
    ) -> None:
        for batch in self._batches:
            manager.on_request_finished(batch.req_context)
        self._batches.clear()
        self.keys = []
        if not self._listing_done:  # Let a late page be dropped, not kept.
            self.tier.take_peer_keys(self.peer_id, self.import_id, abandon=True)
        self.state = state
        self.error = error
        self.finished_at = time.monotonic()
        log = logger.info if state == "done" else logger.warning
        log("KV import %s %s: %s", self.import_id, state, self.status())
