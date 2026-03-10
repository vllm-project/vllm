# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StoreReusedOffloadingManager: a decorator over any OffloadingManager that
gates GPU->CPU stores on observed block-hash reuse frequency.

Strategy A (P0) from the zero-cost CPU KV cache offloading design:

  When CPU offloading is enabled every evicted GPU block is unconditionally
  written to CPU, even if that block will never be read back (one-shot
  workloads).  This wrapper intercepts ``prepare_store()`` and silently
  suppresses stores for blocks that have not been seen at least
  ``store_threshold`` times, eliminating wasted PCIe bandwidth at zero cost.

Example usage in ``CPUOffloadingSpec.get_manager()``:

    backing = LRUOffloadingManager(backend=backend, enable_events=enable_events)
    return StoreReusedOffloadingManager(backing, store_threshold=2)

When ``store_threshold`` is 1 (or the config key is absent) this wrapper
passes through all calls unchanged, so it is always safe to apply.
"""
from __future__ import annotations

from collections.abc import Iterable

from vllm.v1.kv_offload.abstract import (
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.abstract import LoadStoreSpec  # noqa: F401 (re-exported)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.reuse_tracker import BlockReuseTracker


class StoreReusedOffloadingManager(OffloadingManager):
    """Decorator around an OffloadingManager that filters out stores for
    blocks below a reuse-frequency threshold.

    All methods other than ``prepare_store`` are forwarded straight to the
    wrapped *backing* manager so behaviour is identical for all other
    operations (lookups, loads, events, etc.).

    Args:
        backing: The underlying manager that actually stores/loads/evicts
            blocks (e.g. LRUOffloadingManager or ARCOffloadingManager).
        store_threshold: Minimum number of times a block hash must be
            *seen* before a store is emitted.  Default of 2 means first-time
            blocks are suppressed; from the second occurrence onward they are
            stored.  A value of 1 disables the gate entirely (all stores pass
            through).
        max_tracker_size: Upper bound on the number of tracked block hashes.
            When the tracker is full the least-recently-used entry is evicted
            (LRU policy).  Default 64_000 ≈ 6 MB per scheduler.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ) -> None:
        self._backing = backing
        self._tracker = BlockReuseTracker(
            max_size=max_tracker_size,
            store_threshold=store_threshold,
        )

    # ------------------------------------------------------------------
    # Core overridden method
    # ------------------------------------------------------------------

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """Forward to the backing manager, then filter out first-time blocks.

        Blocks whose hash has been seen fewer than ``store_threshold`` times
        are removed from ``PrepareStoreOutput.block_hashes_to_store``.  If
        all blocks are filtered the empty list is returned inside a valid
        ``PrepareStoreOutput`` (the backing manager has already reserved
        CPU slots; removing a hash from the to-store list simply skips the
        PCIe transfer without releasing the reservation prematurely).
        """
        # Materialise the iterable once so we can reuse it.
        hashes: list[BlockHash] = list(block_hashes)

        output = self._backing.prepare_store(hashes)
        if output is None:
            return None

        # Gate each candidate hash through the reuse tracker.
        filtered = [
            bh
            for bh in output.block_hashes_to_store
            if self._tracker.record_and_check(bh)
        ]
        # Return a new PrepareStoreOutput with only the reuse-worthy hashes;
        # the store_spec and eviction list come from the backing manager.
        return PrepareStoreOutput(
            block_hashes_to_store=filtered,
            store_spec=output.store_spec,
            block_hashes_evicted=output.block_hashes_evicted,
        )

    # ------------------------------------------------------------------
    # Pass-through methods — delegate everything else to the backing manager
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        return self._backing.lookup(block_hashes)

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return self._backing.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self._backing.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        self._backing.complete_load(block_hashes)

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        self._backing.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
