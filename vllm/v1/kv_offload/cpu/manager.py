# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Literal

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

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
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
    ):
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
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
