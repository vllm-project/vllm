# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata.

A block's sidecar holds one entry per scope (session) that asked for it. The
block's effective priority is the highest of those entries, so a prefix
shared by several sessions is kept as long as the most demanding one wants
it, and one session lowering or releasing its own hold never drops what
another session still relies on."""

import heapq
import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_hints.retain import RetainDirective


@dataclass(slots=True)
class ScopeHold:
    priority: int
    expiry: float | None


@dataclass(slots=True)
class RetentionMeta:
    """Per-block sidecar: the holds of every scope on the block plus the
    cached effective values the heap orders by."""

    holds: dict[str | None, ScopeHold] = field(default_factory=dict)
    last_freed_time: float = 0.0
    priority: int = 0
    expiry: float | None = None

    @classmethod
    def single(
        cls,
        priority: int,
        expiry: float | None,
        scope: str | None,
        last_freed_time: float = 0.0,
    ) -> "RetentionMeta":
        meta = cls(
            holds={scope: ScopeHold(priority, expiry)}, last_freed_time=last_freed_time
        )
        meta.recompute()
        return meta

    def recompute(self) -> None:
        """Effective priority = max over holds; the block stays protected
        until every hold has lapsed, so the effective expiry is the latest
        one, with None (never) winning."""
        if not self.holds:
            self.priority = 0
            self.expiry = None
            return
        self.priority = max(h.priority for h in self.holds.values())
        expiries = [h.expiry for h in self.holds.values()]
        self.expiry = (
            None
            if any(e is None for e in expiries)
            else max(e for e in expiries if e is not None)
        )

    def purge_lapsed(self, now: float) -> bool:
        """Drop holds whose expiry has passed. Returns whether any was dropped."""
        lapsed = [
            s for s, h in self.holds.items() if h.expiry is not None and h.expiry <= now
        ]
        for s in lapsed:
            del self.holds[s]
        if lapsed:
            self.recompute()
        return bool(lapsed)


class PriorityEvictionQueue:
    # A block shared by very many sessions would otherwise collect a hold per
    # session. Only the maximum matters for eviction, so keep the strongest
    # few; a weaker hold that gets dropped could only have mattered after all
    # the stronger ones lapsed, and then the next hint restores it.
    _MAX_HOLDS_PER_BLOCK = 8

    # Rebuild the heap once stale tuples outnumber live entries by this much.
    _COMPACT_SLACK = 4096

    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()
        # Per-block generation counter. try_insert bumps it and stamps the
        # pushed tuple; pop_lowest skips tuples with a stale generation. Guards
        # against a suspend()+re-insert leaving an outdated tuple that would
        # evict by the old (priority, last_freed_time) and invert the order.
        self._gen: dict[int, int] = {}
        # (expiry, seq, block_id, scope) for every hold with a finite expiry,
        # so release_expired pops only what has lapsed instead of scanning
        # every queued entry on every allocation. A refreshed or dropped hold
        # leaves a stale tuple behind; it is skipped when it surfaces.
        self._expiry_heap: list[tuple[float, int, int, str | None]] = []
        self._seq = itertools.count()

    @property
    def num_blocks(self) -> int:
        return len(self._in_queue)

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._in_queue

    def try_insert(
        self,
        block: KVCacheBlock,
        last_freed_time: float | None = None,
    ) -> bool:
        """Insert the block's sidecar entry into the heap and return True.
        Return False (caller routes to LRU) when there is no entry or every
        hold on it has expired; the expired case drops the sidecar.
        last_freed_time, if given, overrides the stored value so the heap
        tiebreak reflects the most-recent free."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        if meta.purge_lapsed(time.monotonic()) and not meta.holds:
            # Expired-on-free: drop sidecar, route to LRU free list.
            self._meta.pop(block.block_id, None)
            return False
        if last_freed_time is not None:
            meta.last_freed_time = last_freed_time
        self._in_queue.add(block.block_id)
        self._push(block, meta)
        return True

    def _push(self, block: KVCacheBlock, meta: RetentionMeta) -> None:
        """Push the block's current (priority, last_freed_time) and supersede
        any older tuple for it."""
        gen = self._gen.get(block.block_id, 0) + 1
        self._gen[block.block_id] = gen
        heapq.heappush(
            self._heap,
            (meta.priority, meta.last_freed_time, gen, block.block_id, block),
        )
        # Every free of a protected block pushes a tuple and only pop_lowest
        # discards the stale ones. While the LRU list absorbs allocations no pop
        # runs, so a long run with expiring protections piles up millions of
        # stale tuples and the first pop that follows has to skip them all --
        # measured as +11% inter-token latency over a two-hour replay. Rebuild
        # from the live entries once stale ones dominate; amortized O(1).
        if len(self._heap) > 2 * len(self._in_queue) + self._COMPACT_SLACK:
            self._compact()

    def _compact(self) -> None:
        """Drop stale heap tuples (block left the queue, or a newer insert
        superseded it) and re-heapify the live ones."""
        gen = self._gen
        in_queue = self._in_queue
        self._heap = [
            t for t in self._heap if t[3] in in_queue and t[2] == gen.get(t[3])
        ]
        heapq.heapify(self._heap)

    def _track_expiry(self, block_id: int, scope: str | None = None) -> None:
        """Index a hold's expiry so release_expired can find it without a
        scan. With scope None (test helper path) every finite hold of the
        block is indexed. Holds that never expire are not indexed."""
        meta = self._meta[block_id]
        scopes = [scope] if scope in meta.holds else list(meta.holds)
        for s in scopes:
            expiry = meta.holds[s].expiry
            if expiry is None:
                continue
            heapq.heappush(self._expiry_heap, (expiry, next(self._seq), block_id, s))
        if len(self._expiry_heap) > 2 * len(self._meta) + self._COMPACT_SLACK:
            self._expiry_heap = [
                (h.expiry, next(self._seq), bid, s)
                for bid, m in self._meta.items()
                for s, h in m.holds.items()
                if h.expiry is not None
            ]
            heapq.heapify(self._expiry_heap)

    def suspend(self, block: KVCacheBlock) -> None:
        """Drop the block from the eviction-candidate set (_in_queue) only;
        the sidecar (_meta) is KEPT so protection is restored on the next
        free. The stale heap tuple is skipped lazily at pop_lowest. Contrast
        unprotect(), which removes the protection record."""
        self._in_queue.discard(block.block_id)

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop and return the lowest-priority block.

        Skips stale tuples: those whose block_id has left _in_queue (suspend /
        release_expired) and those whose generation is outdated (a newer
        try_insert superseded them). Expired entries are not handled here —
        callers must invoke release_expired() first. Returns None when no
        live entries remain."""
        while self._heap:
            _, _, gen, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            if gen != self._gen.get(block_id):
                # Superseded by a newer insert for the same block_id.
                continue
            self._in_queue.discard(block_id)
            self._meta.pop(block_id, None)
            return block
        return None

    def release_expired(self) -> list[int]:
        """Drop every hold that has lapsed. A queued block left with no hold
        is returned for the caller to route to the LRU free list (expiry =
        "protection released", not "evict now"); one that keeps other holds
        is re-pushed at its new effective priority. Stale heap tuples are
        cleaned up at the next pop_lowest.

        Walks the expiry heap only as far as entries that have lapsed, so an
        allocation with nothing expired costs O(1) rather than a pass over
        every queued entry -- with a full queue and sixteen concurrent
        requests that pass ran once or twice per engine step and showed up as
        +8% inter-token latency. A referenced block's lapsed hold is left for
        try_insert, which drops it when the block is freed."""
        now = time.monotonic()
        drained: list[int] = []
        heap = self._expiry_heap
        while heap and heap[0][0] <= now:
            expiry, _, block_id, scope = heapq.heappop(heap)
            meta = self._meta.get(block_id)
            if meta is None:
                continue
            hold = meta.holds.get(scope)
            if hold is None or hold.expiry != expiry:
                continue  # refreshed, replaced or dropped since it was indexed
            if block_id not in self._in_queue:
                continue  # referenced: try_insert drops it on the next free
            old_priority = meta.priority
            del meta.holds[scope]
            meta.recompute()
            if not meta.holds:
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                drained.append(block_id)
            elif meta.priority != old_priority:
                block = self._queued_block(block_id)
                if block is not None:
                    self._push(block, meta)
        return drained

    def _queued_block(self, block_id: int) -> KVCacheBlock | None:
        """The KVCacheBlock object of a queued block, found through its live
        heap tuple (the queue does not keep a separate id -> block map)."""
        gen = self._gen.get(block_id)
        for t in self._heap:
            if t[3] == block_id and t[2] == gen:
                return t[4]
        return None

    def unprotect(self, block_id: int) -> bool:
        """Permanently drop the block's protection: pop its sidecar (_meta)
        and discard it from _in_queue. Called when the block's hash is reset
        or it is evicted from the prefix cache. Unlike suspend(), the
        protection record does NOT survive.

        Returns:
            Whether the block was queued, i.e. whether it just left the queue
            and now sits in neither free structure until the caller routes it.

        """
        self._meta.pop(block_id, None)
        if block_id not in self._in_queue:
            return False
        self._in_queue.discard(block_id)
        return True

    def clear(self) -> None:
        """Drop all sidecar entries and heap state."""
        self._meta.clear()
        self._heap.clear()
        self._in_queue.clear()
        self._gen.clear()
        self._expiry_heap.clear()

    def apply_directives(
        self,
        blocks: list[KVCacheBlock],
        directives: Sequence[RetainDirective],
        scope: str | None,
        block_size: int,
    ) -> list[int]:
        """Apply the directives for one scope; see apply_directives_ex.
        Returns the block_ids a release took out of this queue."""
        released, _ = self.apply_directives_ex(blocks, directives, scope, block_size)
        return released

    def apply_directives_ex(
        self,
        blocks: list[KVCacheBlock],
        directives: Sequence[RetainDirective],
        scope: str | None,
        block_size: int,
    ) -> tuple[list[int], list[tuple[KVCacheBlock, int]]]:
        """For each full block, find the highest-priority overlapping
        directive and update the caller's own hold on the block:

        - Priority 1-100 sets this scope's hold (priority and expiry) on the
          block. Other scopes' holds are untouched; the block's effective
          priority is the highest hold.
        - Priority 0 drops this scope's hold. Other scopes' holds stay, so a
          block another session still relies on is not released by this one.
        - No matching directive: the block is left alone. Saying nothing about
          a block is not a request to unprotect it -- when silence meant
          release, one turn dropped the protection another turn had just
          placed on a block it was actively reusing.

        A queued block whose effective priority changes is re-pushed, so
        pop_lowest evicts it at the new priority rather than the one it was
        freed with.

        Returns (released, lowered). ``released`` lists the block_ids that lost
        their last hold while queued: they are in neither free structure now,
        and the LRU list belongs to the pool, so the caller has to route them,
        the same way release_expired()'s return value is routed. ``lowered``
        lists (block, new_priority) for every block on which this scope's hold
        went down (0 = dropped), so the pool can cap the scope's holds on the
        block's descendants, whose content is unreachable without it.
        """
        now = time.monotonic()
        released: list[int] = []
        lowered: list[tuple[KVCacheBlock, int]] = []
        for idx, block in enumerate(blocks):
            if block.is_null:
                continue
            token_start = idx * block_size
            token_end = token_start + block_size
            best_priority = -1
            best_duration: float | None = None
            for d in directives:
                if d.end is not None and d.end <= token_start:
                    continue
                if d.start >= token_end:
                    continue
                if d.priority > best_priority:
                    best_priority = d.priority
                    best_duration = d.duration

            if best_priority < 0:
                continue

            meta = self._meta.get(block.block_id)
            had = meta.holds.get(scope) if meta is not None else None
            had_priority = had.priority if had is not None else None
            if best_priority == 0:
                if had is None:
                    continue
                if self._drop_hold(block, scope):
                    released.append(block.block_id)
                lowered.append((block, 0))
                continue

            expiry = now + best_duration if best_duration is not None else None
            self._set_hold(block, scope, best_priority, expiry)
            if had_priority is not None and best_priority < had_priority:
                lowered.append((block, best_priority))

        return released, lowered

    def cap_hold(
        self, block: KVCacheBlock, scope: str | None, priority: int
    ) -> tuple[bool, bool]:
        """Lower this scope's hold on the block to ``priority`` if it is
        higher (0 drops it). Returns (had_hold, released): whether the scope
        held the block at all, and whether the block lost its last hold while
        queued and must be routed to the LRU list by the caller."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False, False
        hold = meta.holds.get(scope)
        if hold is None:
            return False, False
        if hold.priority <= priority:
            return True, False
        if priority <= 0:
            return True, self._drop_hold(block, scope)
        old_priority = meta.priority
        hold.priority = priority
        meta.recompute()
        if meta.priority != old_priority and block.block_id in self._in_queue:
            self._push(block, meta)
        return True, False

    def _set_hold(
        self,
        block: KVCacheBlock,
        scope: str | None,
        priority: int,
        expiry: float | None,
    ) -> None:
        meta = self._meta.get(block.block_id)
        if meta is None:
            meta = RetentionMeta()
            self._meta[block.block_id] = meta
        old_priority = meta.priority if meta.holds else -1
        hold = meta.holds.get(scope)
        if hold is None:
            if len(meta.holds) >= self._MAX_HOLDS_PER_BLOCK:
                weakest = min(meta.holds, key=lambda s: meta.holds[s].priority)
                if meta.holds[weakest].priority >= priority:
                    return  # cannot raise the maximum; not worth a slot
                del meta.holds[weakest]
            meta.holds[scope] = ScopeHold(priority, expiry)
        else:
            hold.priority = priority
            hold.expiry = expiry
        meta.recompute()
        self._track_expiry(block.block_id, scope)
        if meta.priority != old_priority and block.block_id in self._in_queue:
            self._push(block, meta)

    def _drop_hold(self, block: KVCacheBlock, scope: str | None) -> bool:
        """Remove this scope's hold. Returns whether the block lost its last
        hold while queued (it then sits in neither free structure)."""
        meta = self._meta.get(block.block_id)
        if meta is None or scope not in meta.holds:
            return False
        old_priority = meta.priority
        del meta.holds[scope]
        if not meta.holds:
            return self.unprotect(block.block_id)
        meta.recompute()
        if meta.priority != old_priority and block.block_id in self._in_queue:
            self._push(block, meta)
        return False
