# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reuse-frequency gating for CPU KV-cache offload stores.

FilterReusedOffloadingManager — OffloadingManager decorator that skips
    storing blocks that have not yet been seen enough times.
"""

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)


class FilterReusedOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:

    * ``lookup`` — records each visited block hash in an internal LRU counter.
    * ``prepare_store`` — filters out block hashes that have not yet
      crossed the threshold *before* calling the backing
      ``prepare_store``.

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        store_threshold: A block must be seen at least this many times in
            ``lookup()`` before it is eligible for offloading.  Must be >= 2
            (a value of 1 would be equivalent to no filtering).
        max_tracker_size: Maximum entries in the internal tracker's LRU table.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        if store_threshold < 2:
            raise ValueError(
                "FilterReusedOffloadingManager store_threshold must be >= 2, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Record each hash, then delegate lookup to backing manager."""
        block_hashes = list(block_hashes)
        for block_hash in block_hashes:
            if block_hash in self.counts:
                self.counts.move_to_end(block_hash)
                self.counts[block_hash] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)  # evict LRU
                self.counts[block_hash] = 1
        return self._backing.lookup(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.

        Filtering is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        """
        block_hashes = list(block_hashes)
        eligible = [
            bh for bh in block_hashes if self.counts.get(bh, 0) >= self.store_threshold
        ]

        # Delegate to the backing manager with only the eligible hashes.
        # Passing an empty list is intentional and safe — CPUOffloadingManager
        # handles it correctly, returning a PrepareStoreOutput with empty lists.
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

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        return self._backing.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
