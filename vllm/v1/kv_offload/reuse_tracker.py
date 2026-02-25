# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BlockReuseTracker: gates CPU offload stores on observed block-hash reuse.

When CPU offloading is enabled, blocks are written from GPU to CPU on every
eviction regardless of whether they will ever be re-used.  For one-shot
(non-reuse) workloads this wastes PCIe bandwidth and CPU memory at zero
benefit.

This module provides a simple, O(1) scheduler-side filter: only enqueue a
GPU→CPU store if the block's content hash has been seen at least
``store_threshold`` times across all requests.  First-time blocks are silently
skipped; once a hash crosses the threshold (default: 2) all future evictions
of that hash are stored so they are available for prefix-cache hits.

Memory footprint: ~100 bytes per tracked entry (Python OrderedDict overhead +
32-byte BlockHash).  Default max_size=64 000 → ~6 MB per scheduler process.
In disaggregated deployments with 2–4 prefill schedulers, total tracker memory
stays < 25 MB.
"""
from collections import OrderedDict

from vllm.v1.core.kv_cache_utils import BlockHash


class BlockReuseTracker:
    """Tracks block-hash reuse frequency to gate CPU offload stores.

    Args:
        max_size: Maximum number of distinct hashes to track.  When full,
            the least-recently-used entry is evicted (LRU policy).
        store_threshold: Minimum number of times a hash must be *seen* before
            a store is allowed.  Default 2 means: skip on first occurrence,
            store from the second occurrence onward.
    """

    def __init__(self, max_size: int = 64_000, store_threshold: int = 2) -> None:
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()
        self.max_size = max_size
        self.store_threshold = store_threshold

    def record_and_check(self, block_hash: BlockHash) -> bool:
        """Record a new observation of *block_hash* and return whether to store.

        Returns:
            True  — the hash has been seen >= store_threshold times; caller
                    should enqueue a GPU→CPU store.
            False — the hash is new (or was evicted); caller should skip the
                    store at zero cost.

        Note on hash collisions: vLLM uses content-based hashing (equivalent to
        SHA-256 strength).  The probability of a collision causing a spurious
        skip is negligible and does not affect correctness — the worst case is
        a missed store for one eviction cycle.
        """
        if block_hash in self.counts:
            self.counts.move_to_end(block_hash)
            self.counts[block_hash] += 1
        else:
            if len(self.counts) >= self.max_size:
                self.counts.popitem(last=False)  # evict LRU entry
            self.counts[block_hash] = 1

        return self.counts[block_hash] >= self.store_threshold

    def __len__(self) -> int:
        return len(self.counts)

    def __contains__(self, block_hash: BlockHash) -> bool:
        return block_hash in self.counts
