# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Dict, FrozenSet, List, Optional, Tuple

from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.utils import Device


class CpuGpuBlockAllocator(DeviceAwareBlockAllocator):
    """A block allocator that can allocate blocks on both CPU and GPU memory.

    This class implements the `DeviceAwareBlockAllocator` interface and provides
    functionality for allocating and managing blocks of memory on both CPU and
    GPU devices.

    The `CpuGpuBlockAllocator` maintains separate memory pools for CPU and GPU
    blocks, and allows for allocation, deallocation, forking, and swapping of
    blocks across these memory pools.
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        """Creates a CpuGpuBlockAllocator instance with the specified
        configuration.

        This static method creates and returns a CpuGpuBlockAllocator instance
        based on the provided parameters. It initializes the CPU and GPU block
        allocators with the specified number of blocks, block size, and
        allocator type.

        Args:
            allocator_type (str): The type of block allocator to use for CPU
                and GPU blocks. Currently supported values are "naive" and
                "prefix_caching".
            num_gpu_blocks (int): The number of blocks to allocate for GPU
                memory.
            num_cpu_blocks (int): The number of blocks to allocate for CPU
                memory.
            block_size (int): The size of each block in number of tokens.

        Returns:
            DeviceAwareBlockAllocator: A CpuGpuBlockAllocator instance with the
                specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        reserved_blocks = 0
        block_ids = list(
            range(reserved_blocks, num_gpu_blocks + num_cpu_blocks))
        num_gpu_blocks -= reserved_blocks
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        elif allocator_type == "prefix_caching":
            gpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return CpuGpuBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: BlockAllocator,
                 gpu_block_allocator: BlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        self._allocators = {
            Device.CPU: cpu_block_allocator,
            Device.GPU: gpu_block_allocator,
        }

        self._swap_mapping: Dict[int, int] = {}
        self._null_block: Optional[Block] = None

        self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_or_get_null_block(self) -> Block:
        if self._null_block is None:
            self._null_block = NullBlock(
                self.allocate_mutable_block(None, Device.GPU))
        return self._null_block

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               device: Device,
                               extra_hash: Optional[int] = None) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            Block: The newly allocated mutable block.
        """
        return self._allocators[device].allocate_mutable_block(
            prev_block, extra_hash=extra_hash)

    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            device: Device,
            extra_hash: Optional[int] = None) -> List[Block]:
        """Allocates a new group of immutable blocks with the provided block 
        token IDs on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            block_token_ids (List[int]): The list of block token IDs to be 
                stored in the new blocks.
            device (Device): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            List[Block]: The newly allocated list of immutable blocks 
                containing the provided block token IDs.
        """
        return self._allocators[device].allocate_immutable_blocks(
            prev_block, block_token_ids, extra_hash=extra_hash)

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: Device,
                                 extra_hash: Optional[int] = None) -> Block:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (Device): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """
        return self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, extra_hash=extra_hash)

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        # Null block should never be freed
        if isinstance(block, NullBlock):
            return
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
            memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: A new list of blocks that shares the same memory as the
                original sequence.
        """
        # do not attempt to fork the null block
        assert not isinstance(last_block, NullBlock)
        block_id = last_block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        return allocator.fork(last_block)

    def get_num_free_blocks(self, device: Device) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (Device): The device for which to query the number of free
                blocks. AssertionError is raised if None is passed.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        return self._allocators[device].get_num_free_blocks()

    def get_num_total_blocks(self, device: Device) -> int:
        return self._allocators[device].get_num_total_blocks()

    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain device given the 
        absolute block id.

        Args:
            device (Device): The device for which to query relative block id.
                absolute_id (int): The absolute block id for the block in 
                whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return self._allocators[device].get_physical_block_id(absolute_id)

    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:
        """Execute the swap for the given blocks from source_device
        on to dest_device, save the current swap mapping and append 
        them to the accumulated `self._swap_mapping` for each 
        scheduling move.

        Args:
            blocks: List of blocks to be swapped.
            src_device (Device): Device to swap the 'blocks' from.
            dst_device (Device): Device to swap the 'blocks' to.
        
        Returns:
            Dict[int, int]: Swap mapping from source_device
                on to dest_device.
        """
        src_block_ids = [block.block_id for block in blocks]
        self._allocators[src_device].swap_out(blocks)
        self._allocators[dst_device].swap_in(blocks)
        dst_block_ids = [block.block_id for block in blocks]

        current_swap_mapping: Dict[int, int] = {}
        for src_block_id, dst_block_id in zip(src_block_ids, dst_block_ids):
            if src_block_id is not None and dst_block_id is not None:
                self._swap_mapping[src_block_id] = dst_block_id
                current_swap_mapping[src_block_id] = dst_block_id
        return current_swap_mapping

    def get_num_full_blocks_touched(self, blocks: List[Block],
                                    device: Device) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out the given blocks on to the 'device'.

        Args:
            blocks: List of blocks to be swapped.
            device (Device): Device to swap the 'blocks' on.

        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks on to the 'device'.
                Non full blocks are ignored when deciding the number
                of blocks to touch.
        """
        return self._allocators[device].get_num_full_blocks_touched(blocks)

    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        """Clears the copy-on-write (CoW) state and returns the mapping of
            source to destination block IDs.

        Returns:
            List[Tuple[int, int]]: A list mapping source block IDs to 
                destination block IDs.
        """
        # CoW only supported on GPU
        device = Device.GPU
        return self._allocators[device].clear_copy_on_writes()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_accessed(block_ids, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_computed(block_ids)

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_common_computed_block_ids(
            computed_seq_block_ids)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        assert device in self._allocators
        return self._allocators[device].get_prefix_cache_hit_rate()

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        """Reset prefix cache for specified or all devices."""
        if device:
            return self._allocators[device].reset_prefix_cache()
        success = True
        for allocator in self._allocators.values():
            success = success and allocator.reset_prefix_cache()
        return success

    def get_and_reset_swaps(self) -> List[Tuple[int, int]]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called after every swapping operations for now, and after every
        schedule when BlockManagerV2 become default. Currently not useful.

        Returns:
            List[Tuple[int, int]]: A mapping of source to destination block IDs.
        """
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return list(mapping.items())

    def find_cached_blocks_prefix(
        self,
        block_hashes: List[int],
        device: Device = Device.GPU,
    ) -> List[int]:
        return self._allocators[device].find_cached_blocks_prefix(block_hashes)


class NullBlock(Block):
    """
    Null blocks are used as a placeholders for KV cache blocks that have
    been dropped due to sliding window.
    This implementation just wraps an ordinary block and prevents it from
    being modified. It also allows for testing if a block is NullBlock
    via isinstance().
    """

    def __init__(self, proxy: Block):
        super().__init__()
        self._proxy = proxy

    def append_token_ids(self, token_ids: List[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def block_id(self):
        return self._proxy.block_id

    @block_id.setter
    def block_id(self, value: Optional[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def token_ids(self) -> List[BlockId]:
        return self._proxy.token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for null block")

    @property
    def num_empty_slots(self) -> BlockId:
        return self._proxy.num_empty_slots

    @property
    def is_full(self):
        return self._proxy.is_full

    @property
    def prev_block(self):
        return self._proxy.prev_block

    @property
    def extra_hash(self):
        return None

    @property
    def computed(self):
        return self._proxy.computed

    @computed.setter
    def computed(self, value):
        self._proxy.computed = value

    @property
    def last_accessed(self) -> float:
        return self._proxy.last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._proxy.last_accessed = last_accessed_ts

    @property
    def content_hash(self):
        return self._proxy.content_hash
