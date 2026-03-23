# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from typing import Literal

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.
    """

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...

    @abstractmethod
    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        """Find block in data structures. Returns None if not present."""

    @abstractmethod
    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists."""

    @abstractmethod
    def remove(self, block_hash: BlockHash) -> None:
        """Remove a block (used to clean up after a failed store)."""

    @abstractmethod
    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        """Mark blocks as recently used."""

    @abstractmethod
    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (block_hash, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """


class LRUCachePolicy(CachePolicy):
    """LRU cache policy backed by a single OrderedDict."""

    def __init__(self, cache_capacity: int):
        # cache_capacity unused by LRU but accepted for a uniform constructor
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        return self.blocks.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.blocks[block_hash] = block

    def remove(self, block_hash: BlockHash) -> None:
        del self.blocks[block_hash]

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.blocks:
                self.blocks.move_to_end(block_hash)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        if n == 0:
            return []
        candidates: list[tuple[BlockHash, BlockStatus]] = []
        for block_hash, block in self.blocks.items():
            if block.ref_cnt == 0 and block_hash not in protected:
                candidates.append((block_hash, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None
        for block_hash, _ in candidates:
            del self.blocks[block_hash]
        return candidates


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
                        block.ref_cnt == 0
                        and block_hash not in protected
                        and block_hash not in already_selected
                    ):
                        candidate = (block_hash, block, True)
                        virtual_t1_size -= 1
                        break

            if candidate is None:
                for block_hash, block in self.t2.items():
                    if (
                        block.ref_cnt == 0
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


_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
    ):
        self.block_size: int = block_size
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        num_fresh = min(
            len(block_hashes), self._num_blocks - self._num_allocated_blocks
        )
        num_reused = len(block_hashes) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        block_hashes: Iterable[BlockHash],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            block = self._policy.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self._policy.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found in cache"
            assert block.is_ready, f"Block {block_hash!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self._policy.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            block = self._policy.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found"
            assert block.ref_cnt > 0, f"Block {block_hash!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_list = list(block_hashes)

        # filter out blocks that are already stored
        block_hashes_to_store = [
            bh for bh in block_hashes_list if self._policy.get(bh) is None
        ]

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_blocks_to_evict = len(block_hashes_to_store) - self._get_num_free_blocks()

        to_evict: list[BlockHash] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(block_hashes_list)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for block_hash, block in evicted:
                self._free_block(block)
                to_evict.append(block_hash)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.block_size,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self._policy.insert(block_hash, block)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        stored_block_hashes: list[BlockHash] = []

        if success:
            for block_hash in block_hashes:
                block = self._policy.get(block_hash)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self._policy.get(block_hash)
                if block is not None and not block.is_ready:
                    self._policy.remove(block_hash)
                    self._free_block(block)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.block_size,
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
