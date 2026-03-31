# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy


class ARCCachePolicy(CachePolicy):
    """
    ARC (Adaptive Replacement Cache) cache policy.

    Data Structures:
        T1: Recent cache containing blocks accessed once.
        T2: Frequent cache containing blocks accessed multiple times.
        B1/B2: Ghost lists tracking recently evicted blocks from T1/T2.
        target_t1_size: Adaptive target size for the T1 partition.

    Algorithm Flow:
        1. Cache lookup (lookup):
           Searches T1 and T2 for block hashes and counts consecutive hits
           until a miss or non-ready block is encountered.

        2. Cache touch (touch) - Adaptive Learning:
           For each block_hash (in reverse order):
           - If in T1: Move to T2 (promotion from recent to frequent).
           - If in T2: Move to MRU position (end of queue).
           - If in B1 ghost list: Increase target_t1_size.
           - If in B2 ghost list: Decrease target_t1_size.

        3. Block eviction (evict) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size >= target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Block insertion (insert):
           New blocks are always inserted into T1 and removed from B1/B2 if
           present. Blocks may later be promoted to T2 during touch operations.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
    """

    def __init__(self, cache_capacity: int):
        self.cache_capacity: int = cache_capacity
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # block_hash -> None (only care about presence)
        self.b1: OrderedDict[BlockHash, None] = OrderedDict()
        self.b2: OrderedDict[BlockHash, None] = OrderedDict()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        return self.t1.get(block_hash) or self.t2.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.t1[block_hash] = block
        self.b1.pop(block_hash, None)
        self.b2.pop(block_hash, None)

    def remove(self, block_hash: BlockHash) -> None:
        if self.t1.pop(block_hash, None) is None:
            self.t2.pop(block_hash, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.t1:
                block = self.t1.pop(block_hash)
                if not block.is_ready:
                    # block was just prepared to be stored, not really touched
                    # twice — keep it in T1 and mark as most recently used
                    self.t1[block_hash] = block
                else:
                    self.t2[block_hash] = block

            elif block_hash in self.t2:
                self.t2.move_to_end(block_hash)

            elif block_hash in self.b1:
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b1.move_to_end(block_hash)

            elif block_hash in self.b2:
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b2.move_to_end(block_hash)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        if n == 0:
            return []

        # Collect candidates atomically: simulate T1 size changes as we select,
        # but do not modify actual data structures until all n are found.
        candidates: list[
            tuple[BlockHash, BlockStatus, bool]
        ] = []  # (hash, block, from_t1)
        already_selected: set[BlockHash] = set()
        virtual_t1_size = len(self.t1)

        for _ in range(n):
            candidate: tuple[BlockHash, BlockStatus, bool] | None = None

            if virtual_t1_size >= int(self.target_t1_size):
                for block_hash, block in self.t1.items():
                    if (
                        block.ref_cnt in (0, -1)
                        and block_hash not in protected
                        and block_hash not in already_selected
                    ):
                        candidate = (block_hash, block, True)
                        virtual_t1_size -= 1
                        break

            if candidate is None:
                for block_hash, block in self.t2.items():
                    if (
                        block.ref_cnt in (0, -1)
                        and block_hash not in protected
                        and block_hash not in already_selected
                    ):
                        candidate = (block_hash, block, False)
                        break
                if candidate is None:
                    return None

            candidates.append(candidate)
            already_selected.add(candidate[0])

        # Apply all evictions now that we know n candidates exist.
        result: list[tuple[BlockHash, BlockStatus]] = []
        for block_hash, block, from_t1 in candidates:
            if from_t1:
                del self.t1[block_hash]
                self.b1[block_hash] = None
            else:
                del self.t2[block_hash]
                self.b2[block_hash] = None
            result.append((block_hash, block))

        # Trim ghost lists to cache_capacity.
        for ghost in (self.b1, self.b2):
            for _ in range(len(ghost) - self.cache_capacity):
                ghost.popitem(last=False)

        return result
