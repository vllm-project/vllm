# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from collections.abc import Iterable
from typing import Optional

from vllm.v1.offloading.abstract import (LoadStoreSpec, OffloadingEvent,
                                         OffloadingManager, PrepareStoreOutput)


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    load_store_spec - backend-specific information on how to actually
        read/write the block.
    """
    _fields_ = [("ref_cnt", ctypes.c_int32)]

    def __init__(self):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0

    def get_load_store_spec(self, block_hash: int) -> LoadStoreSpec:
        """
        Get backend-specific information on how to read/write the block.

        Args:
            block_hash: the hash identifying the block.

        Returns:
            A LoadStoreSpec that can be used by a worker to read the block.
        """
        raise NotImplementedError


class Backend(ABC):
    """
    An abstract class for allocating and returning specs for writing
    KV blocks to some backend.
    """

    def __init__(self, block_size: int, medium: str):
        self.block_size = block_size
        self.medium = medium

    @abstractmethod
    def get_num_free_blocks(self):
        """
        Returns the number of current number of blocks that can be allocated.
        """
        pass

    @abstractmethod
    def allocate_blocks(self, block_hashes: list[int]) -> list[BlockStatus]:
        """
        Allocate space for writing blocks.
        This method assumes there is enough space for allocation.
        It is unsafe to use without checking get_num_free_blocks beforehand.

        Args:
            block_hashes: the hashes identifying the blocks to be written.

        Returns:
            A list of BlockStatus for the allocated blocks.
            The ref_cnt of each returned item will be -1, meaning the block
            is not yet ready to be read.
        """
        pass

    @abstractmethod
    def free(self, block: BlockStatus):
        """
        Free a previously allocated block.
        You should only call this function with blocks returned by
        allocate_blocks, and only once per each block.

        Args:
            block: The block to be freed.
        """
        pass


class LRUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    """

    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: OrderedDictType[int, BlockStatus] = OrderedDict()
        self.enable_events = enable_events
        self.events: list[OffloadingEvent] = []

    def lookup(self, block_hashes: list[int]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: list[int]) -> list[LoadStoreSpec]:
        load_specs = []
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            assert block is not None
            assert block.is_ready
            block.ref_cnt += 1
            load_specs.append(block.get_load_store_spec(block_hash))
        return load_specs

    def touch(self, block_hashes: list[int]):
        for block_hash in reversed(block_hashes):
            if not self.blocks.get(block_hash):
                continue
            self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: list[int]):
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            assert block is not None
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(self,
                      block_hashes: list[int]) -> Optional[PrepareStoreOutput]:
        # filter out blocks that are already stored
        block_hashes_to_store: list[int] = []
        for block_hash in block_hashes:
            if self.blocks.get(block_hash) is None:
                block_hashes_to_store.append(block_hash)

        num_blocks_to_evict = max(
            0,
            len(block_hashes_to_store) - self.backend.get_num_free_blocks())

        # build list of blocks to evict
        to_evict = []
        if num_blocks_to_evict > 0:
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0:
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break

        if num_blocks_to_evict > 0:
            # we could not evict enough blocks
            return None

        # evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

        if to_evict and self.enable_events:
            self.events.append(
                OffloadingEvent(block_hashes=to_evict,
                                block_size=self.backend.block_size,
                                medium=self.backend.medium,
                                removed=True))

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        # build store specs for allocated blocks
        store_specs: list[LoadStoreSpec] = []
        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block
            store_specs.append(block.get_load_store_spec(block_hash))

        return PrepareStoreOutput(block_hashes_to_store=block_hashes_to_store,
                                  store_specs=store_specs,
                                  block_hashes_evicted=to_evict)

    def complete_store(self, block_hashes: list[int], success: bool = True):
        stored_block_hashes: list[int] = []
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            assert block is not None
            if not block.is_ready:
                if success:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
                else:
                    del self.blocks[block_hash]

        if stored_block_hashes and self.enable_events:
            self.events.append(
                OffloadingEvent(block_hashes=stored_block_hashes,
                                block_size=self.backend.block_size,
                                medium=self.backend.medium,
                                removed=False))

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self.events
        self.events.clear()
