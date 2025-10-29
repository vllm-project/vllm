# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class ARCOffloadingManager(OffloadingManager):
    """
    An OffloadingManager implementing the ARC (Adaptive Replacement Cache)
    eviction policy with a pluggable backend.

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

        3. Block eviction (prepare_store) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size > target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Block insertion (prepare_store):
           New blocks are always inserted into T1 and removed from B1/B2 if
           present. Blocks may later be promoted to T2 during touch operations.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        self.target_t1_size: int = 0
        self.t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # block_hash -> None (only care about presence)
        self.b1: OrderedDict[BlockHash, None] = OrderedDict()
        self.b2: OrderedDict[BlockHash, None] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        self.cache_capacity: int
        if hasattr(self.backend, "num_blocks"):
            self.cache_capacity = self.backend.num_blocks
        else:
            # fallback: use current free blocks as a proxy
            self.cache_capacity = self.backend.get_num_free_blocks()

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found in cache"
            assert block.is_ready, f"Block {block_hash!r} is not ready for reading"

            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.t1:
                block = self.t1.pop(block_hash)
                self.t2[block_hash] = block

            elif block_hash in self.t2:
                self.t2.move_to_end(block_hash)

            elif block_hash in self.b1:
                delta = max(1, len(self.b2) // len(self.b1)) if len(self.b1) > 0 else 1
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )

            elif block_hash in self.b2:
                delta = max(1, len(self.b1) // len(self.b2)) if len(self.b2) > 0 else 1
                self.target_t1_size = max(self.target_t1_size - delta, 0)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found"
            assert block.ref_cnt > 0, f"Block {block_hash!r} ref_cnt is already 0"

            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_to_store = []
        for block_hash in block_hashes:
            if block_hash not in self.t1 and block_hash not in self.t2:
                block_hashes_to_store.append(block_hash)

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict = []
        if num_blocks_to_evict > 0:
            evicted_count = 0

            while evicted_count < num_blocks_to_evict:
                t1_size = len(self.t1)
                t2_size = len(self.t2)
                evict_from_t1 = t1_size > 0 and (
                    t2_size == 0 or t1_size >= self.target_t1_size
                )

                block_to_evict = None
                if evict_from_t1:
                    # try to evict the least recently used (oldest) block from T1
                    for block_hash, block in self.t1.items():
                        if block.ref_cnt == 0:
                            block_to_evict = (block_hash, block)
                            break
                else:
                    # try to evict the least recently used (oldest) block from T2
                    for block_hash, block in self.t2.items():
                        if block.ref_cnt == 0:
                            block_to_evict = (block_hash, block)
                            break

                if block_to_evict:
                    block_hash, block = block_to_evict
                    if evict_from_t1:
                        del self.t1[block_hash]
                        self.b1[block_hash] = None
                    else:
                        del self.t2[block_hash]
                        self.b2[block_hash] = None

                    to_evict.append(block_hash)
                    self.backend.free(block)
                    evicted_count += 1
                else:
                    # cannot evict enough blocks, cache is full of in-use items
                    return None

                while len(self.b1) > self.cache_capacity:
                    self.b1.popitem(last=False)

                while len(self.b2) > self.cache_capacity:
                    self.b2.popitem(last=False)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store), (
            "Backend did not allocate the expected number of blocks"
        )

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.t1[block_hash] = block

            self.b1.pop(block_hash, None)
            self.b2.pop(block_hash, None)

        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored_block_hashes: list[BlockHash] = []

        if success:
            for block_hash in block_hashes:
                block = self.t1.get(block_hash) or self.t2.get(block_hash)

                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.t1.pop(block_hash, None)

                if block is None:
                    block = self.t2.pop(block_hash, None)

                if block is not None and not block.is_ready:
                    self.backend.free(block)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
