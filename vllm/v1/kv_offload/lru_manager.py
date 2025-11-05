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


class LRUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

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
        # filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash for block_hash in block_hashes if block_hash not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # build list of blocks to evict
        to_evict = []
        if num_blocks_to_evict > 0:
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0:
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                # we could not evict enough blocks
                return None

        # evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

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
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        # build store specs for allocated blocks
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
