# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache TTL tracking for offloaded KV-cache blocks.

OffloadTTLTracker applies the same rule as the GPU prefix cache
(see BlockPool): every use of a block renews its TTL, a block never expires
while a request is using it, and once the last request using it is done the
block expires after the largest TTL requested by any of its users.

The tracker only keeps bookkeeping; the caller is responsible for evicting
the keys returned by pop_expired() from the OffloadingManager.
"""

import heapq
import itertools
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey

# Below this many heap items, stale ones are not worth a rebuild.
_MIN_HEAP_REBUILD_SIZE = 64


class TTLEntry:
    """TTL state of a single offloaded block."""

    __slots__ = ("ttl_s", "expires_at", "in_use", "in_heap")

    def __init__(self):
        # Largest TTL of the requests that used this block.
        self.ttl_s: float = 0.0
        # Monotonic time at which the block expires; inf while in use.
        self.expires_at: float = float("inf")
        # Number of live requests/transfers using this block.
        self.in_use: int = 0
        # Whether the expiry heap holds an item for this entry.
        self.in_heap: bool = False


class OffloadTTLTracker:
    def __init__(self):
        # Only blocks currently held by the offloading manager are tracked,
        # so this is bounded by the offloading capacity.
        self._entries: dict[OffloadKey, TTLEntry] = {}
        # Min-heap of (expires_at, seq, key, entry). Holds at most one
        # item per entry: when a popped item's deadline was extended in the
        # meantime, it is pushed back with the new deadline.
        self._heap: list[tuple[float, int, OffloadKey, TTLEntry]] = []
        self._seq = itertools.count()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Clear all tracked entries and the expiry heap."""
        self._entries.clear()
        self._heap.clear()

    def get(self, key: OffloadKey) -> TTLEntry | None:
        """The entry of a tracked block, or None if it is not tracked."""
        return self._entries.get(key)

    def add(self, keys: Iterable[OffloadKey]) -> None:
        """Start tracking blocks that were just stored by the manager."""
        for key in keys:
            if key not in self._entries:
                self._entries[key] = TTLEntry()

    def remove(self, keys: Iterable[OffloadKey]) -> None:
        """Stop tracking blocks the manager evicted on its own."""
        for key in keys:
            self._entries.pop(key, None)
        # Heap items of removed entries stay until their deadline. Rebuild once
        # they dominate, so the heap stays proportional to the tracked blocks.
        if len(self._heap) > 2 * len(self._entries) + _MIN_HEAP_REBUILD_SIZE:
            self._rebuild_heap()

    def acquire(
        self, keys: Iterable[OffloadKey], ttl_s: float | None
    ) -> list[tuple[OffloadKey, TTLEntry]]:
        """Mark tracked blocks as used by a request with the given TTL
        (None means never expire). Untracked blocks are ignored.

        Returns:
            The acquired entries, to be passed to release() when the request
            is done with them.
        """
        if ttl_s is None:
            ttl_s = float("inf")
        acquired = []
        for key in keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            entry.in_use += 1
            if ttl_s > entry.ttl_s:
                entry.ttl_s = ttl_s
            acquired.append((key, entry))
        return acquired

    def release(self, acquired: list[tuple[OffloadKey, TTLEntry]], now: float) -> None:
        """Release entries returned by acquire(). Blocks whose last user is
        released start their TTL now."""
        for key, entry in acquired:
            entry.in_use -= 1
            if entry.in_use > 0 or self._entries.get(key) is not entry:
                # Still in use, or evicted (and possibly re-stored) meanwhile.
                continue
            entry.expires_at = now + entry.ttl_s
            if not entry.in_heap and entry.expires_at != float("inf"):
                self._push(key, entry)

    def pop_expired(self, now: float) -> list[OffloadKey]:
        """Return the blocks whose TTL has elapsed.

        The blocks stay tracked until the caller evicts them via remove(), or
        hands back the ones it could not evict via requeue().
        """
        heap = self._heap
        expired: list[OffloadKey] = []
        while heap and heap[0][0] <= now:
            _, _, key, entry = heapq.heappop(heap)
            entry.in_heap = False
            if self._entries.get(key) is not entry or entry.in_use > 0:
                # Evicted meanwhile, or in use again: release() re-pushes it.
                continue
            if entry.expires_at > now:
                # Deadline was extended after this item was pushed.
                if entry.expires_at != float("inf"):
                    self._push(key, entry)
                continue
            expired.append(key)
        return expired

    def requeue(self, keys: Iterable[OffloadKey], now: float) -> None:
        """Re-arm expired blocks that could not be evicted, so that they are
        returned by pop_expired() again on a later call."""
        for key in keys:
            entry = self._entries.get(key)
            if entry is None or entry.in_use > 0 or entry.in_heap:
                continue
            entry.expires_at = now
            self._push(key, entry)

    def _rebuild_heap(self) -> None:
        self._heap = []
        for key, entry in self._entries.items():
            entry.in_heap = False
            if entry.in_use == 0 and entry.expires_at != float("inf"):
                self._push(key, entry)

    def _push(self, key: OffloadKey, entry: TTLEntry) -> None:
        entry.in_heap = True
        heapq.heappush(self._heap, (entry.expires_at, next(self._seq), key, entry))
