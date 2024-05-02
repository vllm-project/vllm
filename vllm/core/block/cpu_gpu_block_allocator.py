from typing import Dict, List, Optional

from vllm.core.block.interfaces import (Block, BlockAllocator,
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
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator = NaiveBlockAllocator(
                create_block=NaiveBlock,
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator = NaiveBlockAllocator(
                create_block=NaiveBlock,
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

        self._block_ids_to_allocator = {}
        self._swap_mapping = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_mutable(self, prev_block: Optional[Block],
                         device: Device) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated mutable block.
        """
        return self._allocators[device].allocate_mutable(prev_block)

    def allocate_immutable(self, prev_block: Optional[Block],
                           token_ids: List[int], device: Device) -> Block:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """
        return self._allocators[device].allocate_immutable(
            prev_block, token_ids)

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        allocator = self._block_ids_to_allocator[block.block_id]
        return allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
            memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: A new list of blocks that shares the same memory as the
                original sequence.
        """
        allocator = self._block_ids_to_allocator[last_block.block_id]
        return allocator.fork(last_block)

    def get_num_free_blocks(self, device: Device) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (Device): The device for which to query the number of free
                blocks.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        return self._allocators[device].get_num_free_blocks()

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

    def swap(self, blocks: List[Block], source_device: Device,
             dest_device: Device) -> dict[int, int]:
        """Execute the swap for the given blocks from source_device
        on to dest_device, save the current swap mapping and append 
        them to the accumulated `self._swap_mapping` for each 
        scheduling move.

        Args:
            blocks: List of blocks to be swapped.
            source_device (Device): Device to swap the 'blocks' from.
            dest_device (Device): Device to swap the 'blocks' to.
        
        Returns:
            dict[int, int]: Swap mapping from source_device
                on to dest_device.
        """
        source_block_ids = [block.block_id for block in blocks]
        self._allocators[source_device].swap_out(blocks)
        self._allocators[dest_device].swap_in(blocks)
        dest_block_ids = [block.block_id for block in blocks]
        # self._swap_mapping = {
        #     src: dest
        #     for src, dest in zip(source_block_ids, dest_block_ids)
        # }
        current_swap_mapping = {}
        for src, dest in zip(source_block_ids, dest_block_ids):
            self._swap_mapping[src] = dest
            current_swap_mapping[src] = dest
        return current_swap_mapping

    def get_num_blocks_touched(self,
                               blocks: List[Block],
                               device: Device,
                               num_lookahead_slots: int = 0) -> int:
        """Returns the number of blocks that will be touched by
        swapping in/out the given blocks on to the 'device'.

        Args:
            blocks: List of blocks to be swapped.
            device (Device): Device to swap the 'blocks' on.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            int: the number of blocks that will be touched by
        swapping in/out the given blocks on to the 'device'.
        """
        return self._allocators[device].get_num_blocks_touched(
            blocks, num_lookahead_slots)

    def clear_copy_on_writes(self) -> Dict[int, List[int]]:
        """Clears the copy-on-write (CoW) state and returns the mapping of
            source to destination block IDs.

        Returns:
            Dict[int, List[int]]: A dictionary mapping source block IDs to lists
                of destination block IDs.
        """
        # CoW only supported on GPU
        device = Device.GPU
        return self._allocators[device].clear_copy_on_writes()

    def mark_blocks_as_computed(self) -> None:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_computed()

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_common_computed_block_ids(
            seq_block_ids)

    def all_block_ids(self) -> frozenset[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def get_and_reset_swaps(self) -> dict[int, int]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called after every swapping operations for now, and after every
        schedule when BlockManagerV2 become default.

        Returns:
            Dict[int, int]: A mapping of source to destination block IDs.
        """
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return mapping
