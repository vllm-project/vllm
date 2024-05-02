from typing import Dict, FrozenSet, List, Optional

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
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
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

    def __init__(
        self,
        cpu_block_allocator: BlockAllocator,
        gpu_block_allocator: BlockAllocator,
    ):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        self._allocators = {
            Device.CPU: cpu_block_allocator,
            Device.GPU: gpu_block_allocator,
        }

        self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
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
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
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
            self, seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_common_computed_block_ids(
            seq_block_ids)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError

    def cow_block_if_not_appendable(self, block: Block) -> Optional[BlockId]:
        raise NotImplementedError
