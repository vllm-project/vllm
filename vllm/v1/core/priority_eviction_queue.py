# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata."""

import heapq
import time
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_hints.retain import RetainDirective


@dataclass(slots=True)
class RetentionMeta:
    priority: int
    expiry: float | None
    scope: str | None
    last_freed_time: float


def _later_expiry(a: float | None, b: float | None) -> float | None:
    """The later of two expiries, where None means "never expires" and wins."""
    if a is None or b is None:
        return None
    return max(a, b)


class PriorityEvictionQueue:
    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()
        # Per-block generation counter. try_insert bumps it and stamps the
        # pushed tuple; pop_lowest skips tuples with a stale generation. Guards
        # against a suspend()+re-insert leaving an outdated tuple that would
        # evict by the old (priority, last_freed_time) and invert the order.
        self._gen: dict[int, int] = {}
        # (expiry, block_id) for every entry with a finite expiry, so
        # release_expired pops only what has lapsed instead of scanning every
        # queued entry on every allocation. A refreshed or dropped entry leaves
        # a stale tuple behind; it is skipped when it surfaces.
        self._expiry_heap: list[tuple[float, int]] = []

    # Rebuild the heap once stale tuples outnumber live entries by this much.
    _COMPACT_SLACK = 4096

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
        Return False (caller routes to LRU) when there is no entry or it has
        expired; the expired case drops the sidecar. last_freed_time, if given,
        overrides the stored value so the heap tiebreak reflects the most-recent
        free."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        if meta.expiry is not None and meta.expiry <= time.monotonic():
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

    def _track_expiry(self, block_id: int) -> None:
        """Index the entry's expiry so release_expired can find it without a
        scan. Entries that never expire are not indexed."""
        expiry = self._meta[block_id].expiry
        if expiry is None:
            return
        heapq.heappush(self._expiry_heap, (expiry, block_id))
        if len(self._expiry_heap) > 2 * len(self._meta) + self._COMPACT_SLACK:
            meta = self._meta
            self._expiry_heap = [
                (m.expiry, bid) for bid, m in meta.items() if m.expiry is not None
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
        """Release protection from all expired queued entries and return their
        block_ids for the caller to route to the LRU free list (expiry =
        "protection released", not "evict now"). Stale heap tuples are
        cleaned up at the next pop_lowest.

        Walks the expiry heap only as far as entries that have lapsed, so an
        allocation with nothing expired costs O(1) rather than a pass over
        every queued entry -- with a full queue and sixteen concurrent
        requests that pass ran once or twice per engine step and showed up as
        +8% inter-token latency. A referenced block's lapsed entry is left for
        try_insert, which drops it when the block is freed."""
        now = time.monotonic()
        drained: list[int] = []
        heap = self._expiry_heap
        while heap and heap[0][0] <= now:
            expiry, block_id = heapq.heappop(heap)
            meta = self._meta.get(block_id)
            if meta is None or meta.expiry != expiry:
                continue  # refreshed, replaced or dropped since it was indexed
            if block_id in self._in_queue:
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                drained.append(block_id)
        return drained

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
        """For each full block, find the highest-priority overlapping
        directive and update the sidecar entry under these rules:

        - Priority 1-100 protects; priority 0 releases.
        - Escalation (new > current priority): any caller may raise priority
          and takes ownership of the block. The expiry only moves later, so
          raising a block's priority never shortens a hold already in place.
          Only its owner shortens a block's hold, by naming it again.
        - Downgrade or refresh (new <= current priority): only the current
          owner may do this.
        - Release (priority 0): only the current owner may drop the entry, the
          same restriction as a downgrade. A caller that no longer needs a
          block says so; a block nobody renews also expires on its own.
        Returns the block_ids a release took out of this queue. They are in
        neither free structure now, and the LRU list belongs to the pool, so the
        caller has to route them, the same way release_expired()'s return value
        is routed. A released block that was not queued is not returned: a
        referenced one reaches the LRU list when it is freed, and a free one
        whose entry rode along on the LRU list is still linked there.

        A queued block whose priority changes is re-pushed, so pop_lowest
        evicts it at the new priority rather than the one it was freed with.

        - No matching directive: the block is left alone. Saying nothing about
          a block is not a request to unprotect it -- when silence meant
          release, one turn dropped the protection another turn had just
          placed on a block it was actively reusing.
        """
        now = time.monotonic()
        released: list[int] = []
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

            current = self._meta.get(block.block_id)

            if best_priority < 0:
                # No matching directive: leave the block's protection as it is.
                continue

            if best_priority == 0:
                # Explicit release. Restricted to the owner, like a downgrade:
                # otherwise one scope could drop protection another scope is
                # relying on. Nothing to do when the block is unprotected.
                if (
                    current is not None
                    and scope is not None
                    and current.scope == scope
                    and self.unprotect(block.block_id)
                ):
                    released.append(block.block_id)
                continue

            expiry = now + best_duration if best_duration is not None else None
            current_priority = current.priority if current is not None else -1
            if best_priority > current_priority:
                # Escalation: any caller may raise priority and takes ownership.
                # The hold only grows — raising a block's priority must not cut
                # short a longer one someone else is already relying on.
                self._store(
                    block,
                    RetentionMeta(
                        priority=best_priority,
                        expiry=(
                            expiry
                            if current is None
                            else _later_expiry(expiry, current.expiry)
                        ),
                        scope=scope,
                        last_freed_time=current.last_freed_time if current else 0.0,
                    ),
                )
            elif current is not None and best_priority == current_priority:
                # Equal-priority covering reuse (any scope): refresh the expiry so a
                # shared prefix reused across sessions does not lapse. Keep the
                # original owner and priority — a cross-scope caller may only EXTEND
                # the hold, never downgrade or steal ownership.
                self._store(
                    block,
                    RetentionMeta(
                        priority=current.priority,
                        expiry=_later_expiry(expiry, current.expiry),
                        scope=current.scope,
                        last_freed_time=current.last_freed_time,
                    ),
                )
            elif current is not None and scope is not None and current.scope == scope:
                # Same scope: owner may downgrade or refresh.
                self._store(
                    block,
                    RetentionMeta(
                        priority=best_priority,
                        expiry=expiry,
                        scope=scope,
                        last_freed_time=current.last_freed_time,
                    ),
                )
            # Non-owner downgrade: silently ignored.

        return released

    def _store(self, block: KVCacheBlock, meta: RetentionMeta) -> None:
        """Write the block's entry; re-push it if it is queued and its
        priority changed."""
        old = self._meta.get(block.block_id)
        self._meta[block.block_id] = meta
        self._track_expiry(block.block_id)
        if (
            old is not None
            and old.priority != meta.priority
            and block.block_id in self._in_queue
        ):
            self._push(block, meta)
