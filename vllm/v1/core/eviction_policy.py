# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable eviction policies for the GPU KV cache BlockPool."""

from abc import ABC, abstractmethod
from collections import OrderedDict

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


class GPUCachePolicy(ABC):
    """Abstract base class for GPU KV cache block eviction policies.

    Implementations must be O(1) per operation since scheduling is on the
    critical path.
    """

    @abstractmethod
    def insert(self, block: KVCacheBlock) -> None:
        """Register a newly freed block as an eviction candidate."""

    @abstractmethod
    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        """Bulk-insert freed blocks, preserving caller-specified order."""

    @abstractmethod
    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block from the eviction pool (called on prefix-cache hit)."""

    @abstractmethod
    def touch(self, block: KVCacheBlock) -> None:
        """Signal that a block was reused via a prefix-cache hit.

        Called after remove() while the block is outside the pool (ref_cnt > 0).
        Implementations may record promotion metadata (e.g. cold → hot for
        TwoQueue) so that the next insert() places the block correctly.
        """

    @abstractmethod
    def evict_n(self, n: int) -> list[KVCacheBlock]:
        """Select and remove n eviction candidates, returning them."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of blocks currently available for eviction."""


class LRUGPUCachePolicy(GPUCachePolicy):
    """Single-queue LRU policy — identical to vLLM's original behaviour.

    This is a thin wrapper around FreeKVCacheBlockQueue so that BlockPool
    can use the GPUCachePolicy interface without any behaviour change when
    this (the default) policy is selected.
    """

    def __init__(self) -> None:
        self._queue: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])

    def insert(self, block: KVCacheBlock) -> None:
        self._queue.append(block)

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        self._queue.append_n(blocks)

    def remove(self, block: KVCacheBlock) -> None:
        self._queue.remove(block)

    def touch(self, block: KVCacheBlock) -> None:
        # LRU carries no frequency state; nothing to do here.
        pass

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        return self._queue.popleft_n(n)

    def __len__(self) -> int:
        return self._queue.num_free_blocks


class TwoQueueGPUCachePolicy(GPUCachePolicy):
    """Two-queue (2Q) eviction policy for the GPU KV cache.

    Maintains two FIFO queues:
    - Cold queue (A1): blocks freed for the first time, or blocks that
      re-enter after being evicted from the hot queue.
    - Hot queue (Am): blocks that survived at least one prefix-cache hit
      (i.e. were accessed more than once).

    Eviction always drains the cold queue first, then the hot queue.
    This prevents scan pollution: a burst of one-time-use blocks fills
    the cold queue and gets evicted before any hot (frequently-reused)
    prefix blocks are touched.

    Promotion (cold → hot) happens lazily:
      1. touch() records the block's id in _hot_set while the block is
         in use (ref_cnt > 0, outside either queue).
      2. insert() / insert_n() checks _hot_set and routes the block to
         the appropriate queue when ref_cnt drops back to 0.

    Demotion (hot → cold): when the cold queue is exhausted and a hot
    block must be evicted, its id is removed from _hot_set so the next
    time it is freed it goes back to the cold queue.
    """

    def __init__(self) -> None:
        self._cold: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])
        self._hot: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])
        # block_ids currently promoted to (or pending insertion into) the
        # hot queue.
        self._hot_set: set[int] = set()

    def insert(self, block: KVCacheBlock) -> None:
        if block.block_id in self._hot_set:
            self._hot.append(block)
        else:
            self._cold.append(block)

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        cold: list[KVCacheBlock] = []
        hot: list[KVCacheBlock] = []
        for b in blocks:
            if b.block_id in self._hot_set:
                hot.append(b)
            else:
                cold.append(b)
        if cold:
            self._cold.append_n(cold)
        if hot:
            self._hot.append_n(hot)

    def remove(self, block: KVCacheBlock) -> None:
        if block.block_id in self._hot_set:
            self._hot.remove(block)
        else:
            self._cold.remove(block)

    def touch(self, block: KVCacheBlock) -> None:
        # Promote: the next insert() will route this block to the hot queue.
        self._hot_set.add(block.block_id)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        result: list[KVCacheBlock] = []

        # Drain cold queue first.
        from_cold = min(n, self._cold.num_free_blocks)
        if from_cold:
            result.extend(self._cold.popleft_n(from_cold))

        # If cold queue was insufficient, drain from hot queue.
        remaining = n - from_cold
        if remaining:
            victims = self._hot.popleft_n(remaining)
            for b in victims:
                # Demote: next free → cold queue.
                self._hot_set.discard(b.block_id)
            result.extend(victims)

        return result

    def __len__(self) -> int:
        return self._cold.num_free_blocks + self._hot.num_free_blocks


class ARCGPUCachePolicy(GPUCachePolicy):
    """ARC (Adaptive Replacement Cache) eviction policy for the GPU KV cache.

    ARC was designed by Megiddo & Modha (IBM, FAST 2003) and is deployed in
    IBM DS6000/DS8000 storage controllers and ZFS.  It outperforms pure LRU by
    simultaneously tracking both *recency* and *frequency* of accesses and
    self-tuning the balance between them via an adaptive parameter ``p``.

    Data structures
    ---------------
    T1  Recently freed blocks that have not yet had a prefix-cache hit since
        their last eviction.  Managed as a doubly-linked LRU queue
        (FreeKVCacheBlockQueue) for O(1) operations.

    T2  Blocks that survived at least one prefix-cache hit (i.e. they were
        "touched" and are therefore *frequently* accessed).  Same queue type.

    B1  Ghost list — stores the *block hashes* of blocks recently evicted from
        T1.  Contains only metadata (no actual KV data), implemented as an
        OrderedDict for O(1) lookup and FIFO-order trimming.

    B2  Ghost list — stores the block hashes of blocks recently evicted from T2.

    p   Adaptive target for the T1 partition size (float in [0, capacity]).
        Increased when a B1 ghost hit is detected (reward recency); decreased
        when a B2 ghost hit is detected (reward frequency).

    Ghost-hit detection (key insight for GPU KV cache)
    ---------------------------------------------------
    In the traditional ARC the ghost hit is detected at *lookup* time: we ask
    for key X, miss the live cache, but find X in B1/B2.  In the GPU KV cache
    context the equivalent moment is *insert()* time, i.e. when a freed block
    is returned to the pool with its hash still set.

    Lifecycle of a ghost hit:
      1. Block A (hash H) is evicted from T1 → H is added to B1;
         ``_maybe_evict_cached_block`` clears A's hash; A gets new content.
      2. A request for prefix P (which maps to hash H) arrives. Since H is no
         longer in ``cached_block_hash_to_block``, it is a cache miss.  New
         blocks are allocated, prefix P is recomputed, and those blocks are
         cached with hash H (or the equivalent hash for that prefix).
      3. When those blocks are later freed, ``free_blocks()`` calls
         ``insert_n()`` with the blocks still carrying hash H.
      4. ``insert()`` finds H in B1 → **B1 ghost hit**: increase p, route block
         to T2 (this content is worth caching long-term).

    This is semantically equivalent to the original ARC: a B1 ghost hit signals
    that we recently evicted a key that is now being accessed again, meaning the
    recency partition (T1) was too small.

    Eviction rule
    -------------
    When choosing between T1 and T2:
      - If |T1| ≥ max(1, p): evict LRU from T1 (T1 is over its target) → B1
      - Otherwise:            evict LRU from T2                         → B2
    Evicting from T2 demotes the block (removes it from ``_t2_ids``) so it
    re-enters T1 on the next insertion.

    Ghost-list trimming
    -------------------
    |B1| and |B2| are each bounded to ``capacity`` to prevent unbounded memory
    growth (same bound used by the existing CPU ARCCachePolicy).

    Complexity
    ----------
    All operations are O(1) amortised: queue insert/remove, dict lookup, and
    trimming at most one entry per evict_n() call.
    """

    def __init__(self, capacity: int = 0) -> None:
        # Effective number of evictable blocks (num_gpu_blocks - 1 for null).
        self._capacity: int = max(capacity, 1)
        # Adaptive target for T1 size (0 ≤ p ≤ capacity).
        self._p: float = 0.0

        # Live queues — hold actual KVCacheBlock objects.
        self._t1: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])
        self._t2: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])

        # Tracks block_ids currently in T2 (or pending promotion to T2 on the
        # next insert(), while the block is in active use with ref_cnt > 0).
        self._t2_ids: set[int] = set()

        # Ghost lists — store only block hashes (no KV data).
        # OrderedDict preserves insertion order so we can trim the oldest entry.
        self._b1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self._b2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()

    # ------------------------------------------------------------------
    # GPUCachePolicy interface
    # ------------------------------------------------------------------

    def insert(self, block: KVCacheBlock) -> None:
        """Return a freed block to the eviction pool.

        Ghost-hit detection happens here: if the block's hash is in B1/B2,
        we adjust p and route the block directly to T2.
        """
        hash_key = block.block_hash  # None for blocks without a cached hash.

        if hash_key is not None:
            if hash_key in self._b1:
                # B1 ghost hit: we recently evicted this content from T1.
                # Recency partition was too small → grow T1 target.
                n_b1 = max(len(self._b1), 1)
                n_b2 = max(len(self._b2), 1)
                delta = max(1.0, n_b2 / n_b1)
                self._p = min(self._p + delta, float(self._capacity))
                del self._b1[hash_key]
                # Ghost hit means the content is "accessed again" → promote.
                self._t2_ids.add(block.block_id)
            elif hash_key in self._b2:
                # B2 ghost hit: we recently evicted this content from T2.
                # Frequency partition was too small → shrink T1 target.
                n_b1 = max(len(self._b1), 1)
                n_b2 = max(len(self._b2), 1)
                delta = max(1.0, n_b1 / n_b2)
                self._p = max(self._p - delta, 0.0)
                del self._b2[hash_key]
                # Ghost hit → promote.
                self._t2_ids.add(block.block_id)

        if block.block_id in self._t2_ids:
            self._t2.append(block)
        else:
            self._t1.append(block)

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        # Process each block individually so ghost hits are handled correctly.
        for block in blocks:
            self.insert(block)

    def remove(self, block: KVCacheBlock) -> None:
        if block.block_id in self._t2_ids:
            self._t2.remove(block)
        else:
            self._t1.remove(block)

    def touch(self, block: KVCacheBlock) -> None:
        """Mark a prefix-cache hit.  The block will be routed to T2 on the
        next insert() call (when ref_cnt drops back to 0).
        """
        self._t2_ids.add(block.block_id)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        """Evict n blocks using the ARC replacement rule.

        Decision per slot:
          |T1| >= max(1, p)  →  evict LRU of T1, record hash in B1
          otherwise          →  evict LRU of T2, record hash in B2

        After all evictions, trim ghost lists to ``_capacity``.
        """
        result: list[KVCacheBlock] = []
        # Track simulated sizes within this call so multi-block evictions
        # use a consistent view of the queue sizes.
        vt1 = self._t1.num_free_blocks
        vt2 = self._t2.num_free_blocks

        for _ in range(n):
            # Prefer T1 if it meets or exceeds its target, or T2 is empty.
            use_t1 = vt1 > 0 and (vt1 >= max(1, int(self._p)) or vt2 == 0)

            if use_t1:
                block = self._t1.popleft_n(1)[0]
                if block.block_hash is not None:
                    self._b1[block.block_hash] = None
                # Clean up any stale t2_ids entry (defensive).
                self._t2_ids.discard(block.block_id)
                vt1 -= 1
            else:
                block = self._t2.popleft_n(1)[0]
                if block.block_hash is not None:
                    self._b2[block.block_hash] = None
                # Demote: next insert() → T1.
                self._t2_ids.discard(block.block_id)
                vt2 -= 1

            result.append(block)

        # Trim ghost lists: each bounded to _capacity entries.
        self._trim_ghost_lists()
        return result

    def __len__(self) -> int:
        return self._t1.num_free_blocks + self._t2.num_free_blocks

    # ------------------------------------------------------------------
    # Diagnostics / testing helpers
    # ------------------------------------------------------------------

    @property
    def p(self) -> float:
        """Current adaptive T1 target size."""
        return self._p

    @property
    def t1_size(self) -> int:
        return self._t1.num_free_blocks

    @property
    def t2_size(self) -> int:
        return self._t2.num_free_blocks

    @property
    def b1_size(self) -> int:
        return len(self._b1)

    @property
    def b2_size(self) -> int:
        return len(self._b2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_ghost_lists(self) -> None:
        """Evict the oldest ghost entries so neither B1 nor B2 exceeds
        ``_capacity`` entries (same bound as the CPU ARCCachePolicy).
        """
        while len(self._b1) > self._capacity:
            self._b1.popitem(last=False)
        while len(self._b2) > self._capacity:
            self._b2.popitem(last=False)


_GPU_EVICTION_POLICIES: dict[str, type[GPUCachePolicy]] = {
    "lru": LRUGPUCachePolicy,
    "two_queue": TwoQueueGPUCachePolicy,
    "arc": ARCGPUCachePolicy,
}

VALID_GPU_EVICTION_POLICIES = frozenset(_GPU_EVICTION_POLICIES)


def make_gpu_eviction_policy(name: str, capacity: int = 0) -> GPUCachePolicy:
    """Instantiate a GPU eviction policy by name.

    Args:
        name: One of ``"lru"``, ``"two_queue"``, or ``"arc"``.
        capacity: Total number of evictable GPU blocks (``num_gpu_blocks``).
            Required by ``ARCGPUCachePolicy`` for adaptive parameter bounds
            and ghost-list trimming; ignored by the other policies.

    Returns:
        A fresh GPUCachePolicy instance.

    Raises:
        ValueError: If *name* is not a recognised policy.
    """
    cls = _GPU_EVICTION_POLICIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown GPU eviction policy {name!r}. "
            f"Valid options: {sorted(VALID_GPU_EVICTION_POLICIES)}"
        )
    if name == "arc":
        return cls(capacity=capacity)  # type: ignore[call-arg]
    return cls()

