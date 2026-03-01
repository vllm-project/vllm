# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SwapManager: An OffloadingManager that never evicts.

Used by the CPU Swap connector where CPU memory is guaranteed to hold
all KV cache blocks. If there isn't enough CPU space, prepare_store
raises an error instead of evicting.
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
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class SwapManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend that never evicts blocks.

    This manager is designed for CPU swap mode where the CPU is guaranteed
    to have enough memory to hold all KV cache blocks for all layers.
    If the CPU runs out of space, an assertion error is raised rather
    than evicting existing blocks.
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        # Filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes
            if block_hash not in self.blocks
        ]

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        # No eviction: assert we have enough free blocks
        num_free = self.backend.get_num_free_blocks()
        assert num_free >= len(block_hashes_to_store), (
            f"CPU swap manager does not have enough free blocks to store "
            f"{len(block_hashes_to_store)} blocks. "
            f"Only {num_free} free blocks available. "
            f"Ensure cpu_bytes_to_use is large enough to hold all KV cache."
        )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        store_spec = self.backend.get_load_store_spec(
            block_hashes_to_store, blocks
        )

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]

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
