# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EmbeddingCache — named-entry block cache with FIFO eviction.

Manages a fixed pool of block IDs keyed by content identity (mm_hash).
Entries transition through: not-ready → ready (evictable) → pinned.
Eviction targets ready + unpinned entries in FIFO order.

All public methods are thread-safe.
"""

import threading
from collections import OrderedDict


class CacheEntry:
    """A single cache entry. Read `.block_ids` and `.ready` freely;
    mutations only through EmbeddingCache methods."""

    __slots__ = ("block_ids", "_pin_count")

    def __init__(self, block_ids: tuple[int, ...]) -> None:
        self.block_ids = block_ids
        self._pin_count = -1  # not ready

    @property
    def ready(self) -> bool:
        return self._pin_count >= 0

    @property
    def evictable(self) -> bool:
        return self._pin_count == 0

    def mark_ready(self):
        self._pin_count = 0

    def pin(self):
        self._pin_count += 1


class EmbeddingCache:
    """Fixed-size block cache with FIFO eviction.

    Entries are keyed by content identity (e.g. mm_hash). Blocks are
    allocated from a free-set; when space is needed, ready + unpinned
    entries are evicted oldest-first.

    The caller is responsible for deciding *when* to call ``mark_ready``
    (e.g. after enough engine steps have elapsed for the worker to have
    completed the write).
    """

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._lock = threading.Lock()
        # Available block IDs, LIFO stack.
        self._free_blocks: list[int] = list(range(num_blocks))
        # Key → live entry (all states: not-ready, ready, pinned).
        self._entries: dict[str, CacheEntry] = {}
        # Ready + unpinned entries in insertion order (FIFO eviction).
        self._entries_free_list: OrderedDict[str, None] = OrderedDict()
        # Total blocks reclaimable via eviction (fail-fast check).
        self._evictable_block_count: int = 0

    def get(self, key: str) -> CacheEntry | None:
        """Return the entry for *key*, or None if not present."""
        with self._lock:
            return self._entries.get(key)

    def alloc(self, key: str, n_blocks: int) -> CacheEntry | None:
        """Allocate *n_blocks* for *key*, evicting as needed.

        The entry starts not-ready. Returns None if there is not enough
        space even after evicting all evictable entries.
        """
        with self._lock:
            assert key not in self._entries, (
                f"EmbeddingCache: duplicate alloc for {key!r}"
            )
            assert n_blocks <= self._num_blocks, (
                f"EmbeddingCache: {n_blocks} blocks requested but capacity "
                f"is {self._num_blocks}"
            )
            if len(self._free_blocks) + self._evictable_block_count < n_blocks:
                return None
            self._evict_until(n_blocks)
            block_ids = tuple(self._free_blocks.pop() for _ in range(n_blocks))
            entry = CacheEntry(block_ids)
            self._entries[key] = entry
            return entry

    def mark_ready(self, key: str) -> None:
        """Mark an entry as ready (data is CPU-visible)."""
        with self._lock:
            entry = self._entries[key]
            assert entry._pin_count == -1, (
                f"EmbeddingCache: mark_ready on already-ready entry {key!r}"
            )
            entry.mark_ready()
            self._entries_free_list[key] = None
            self._evictable_block_count += len(entry.block_ids)

    def pin(self, key: str) -> None:
        """Pin an entry (prevent eviction)."""
        with self._lock:
            entry = self._entries[key]
            assert entry.ready, f"EmbeddingCache: pin of not-ready entry {key!r}"
            if entry.evictable:
                del self._entries_free_list[key]
                self._evictable_block_count -= len(entry.block_ids)
            entry.pin()

    def unpin(self, key: str) -> None:
        """Unpin an entry. Asserts currently pinned."""
        with self._lock:
            entry = self._entries[key]
            assert entry._pin_count > 0, (
                f"EmbeddingCache: unpin of unpinned entry {key!r}"
            )
            entry._pin_count -= 1
            if entry._pin_count == 0:
                self._entries_free_list[key] = None
                self._evictable_block_count += len(entry.block_ids)

    def _evict_until(self, n_blocks: int) -> None:
        """Evict ready+unpinned entries FIFO until enough space. Lock held."""
        while len(self._free_blocks) < n_blocks:
            key, _ = self._entries_free_list.popitem(last=False)
            entry = self._entries.pop(key)
            self._evictable_block_count -= len(entry.block_ids)
            self._free_blocks.extend(entry.block_ids)
