# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec


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
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
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

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """
        Get backend-specific information on how to read/write blocks.

        Args:
            block_hashes: the list of block hashes identifying the blocks.
            blocks: the list of blocks.

        Returns:
            A LoadStoreSpec that can be used by a worker
            to read/write the blocks.
        """
        raise NotImplementedError
