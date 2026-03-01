# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reuse-frequency gating for CPU KV-cache offload stores.

BlockReuseTracker — O(1) LRU-bounded frequency counter.
FilteredOffloadingManager — OffloadingManager decorator that skips
    storing blocks that have not yet been seen enough times.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)


class BlockReuseTracker:
    """Tracks block-hash reuse frequency to gate CPU offload stores.

    Maintains an LRU-bounded counter mapping block hashes to their observed
    frequency.  Two separate API entry points allow the caller to integrate
    tracking at the right call sites:

    * :meth:`record` — call from ``lookup()`` to note that a block hash
      has been seen.
    * :meth:`check` — call from ``prepare_store()`` to decide whether the
      block has been seen often enough to be worth storing.

    Args:
        max_size: Maximum number of distinct block hashes to track.  When
            the table is full the least-recently-seen entry is evicted.
            Must be >= 1.
        store_threshold: Minimum number of times a block hash must have
            been seen before ``check()`` returns ``True``.
    """

    def __init__(self, max_size: int = 64_000, store_threshold: int = 2):
        if max_size < 1:
            raise ValueError(f"BlockReuseTracker max_size must be >= 1, got {max_size}")
        self.max_size = max_size
        self.store_threshold = store_threshold
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()

    def record(self, block_hash: BlockHash) -> None:
        """Record that *block_hash* has been seen.

        Should be called from :meth:`FilteredOffloadingManager.lookup`
        for each block hash in the lookup sequence.
        """
        if block_hash in self.counts:
            self.counts.move_to_end(block_hash)
            self.counts[block_hash] += 1
        else:
            if len(self.counts) >= self.max_size:
                self.counts.popitem(last=False)  # evict LRU
            self.counts[block_hash] = 1

    def check(self, block_hash: BlockHash) -> bool:
        """Return ``True`` if *block_hash* has been seen ``>= store_threshold``
        times and is therefore worth storing to CPU.

        Should be called from
        :meth:`FilteredOffloadingManager.prepare_store` *before* calling
        the backing manager's ``prepare_store``.
        """
        return self.counts.get(block_hash, 0) >= self.store_threshold


class FilteredOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:

    * ``lookup`` — records each visited block hash in the
      :class:`BlockReuseTracker`.
    * ``prepare_store`` — filters out block hashes that have not yet
      crossed the threshold *before* calling the backing
      ``prepare_store``.

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        store_threshold: A block must be seen this many times before
            it is eligible for offloading.
        max_tracker_size: Maximum entries in the tracker's LRU table.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        self._backing = backing
        self._tracker = BlockReuseTracker(
            max_size=max_tracker_size,
            store_threshold=store_threshold,
        )
        self.stores_skipped: int = 0

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Record each hash, then delegate lookup to backing manager."""
        block_hashes = list(block_hashes)
        for block_hash in block_hashes:
            self._tracker.record(block_hash)
        return self._backing.lookup(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.

        :meth:`check` is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        """
        block_hashes = list(block_hashes)
        eligible = [bh for bh in block_hashes if self._tracker.check(bh)]

        self.stores_skipped += len(block_hashes) - len(eligible)

        # Delegate to the backing manager with only the eligible hashes.
        # Passing an empty list is intentional and safe — both
        # LRUOffloadingManager and ARCOffloadingManager handle it correctly,
        # returning a PrepareStoreOutput with empty lists.
        return self._backing.prepare_store(eligible)

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return self._backing.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        return self._backing.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        return self._backing.complete_load(block_hashes)

    def get_stats(self) -> dict[str, Any]:
        stats = self._backing.get_stats()
        stats["stores_skipped"] = self.stores_skipped
        return stats

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        return self._backing.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
