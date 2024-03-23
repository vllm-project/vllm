from typing import List, Optional, Set, Iterable, Tuple, Dict, Protocol
from abc import ABC, abstractmethod, abstractproperty
from vllm.core.block.interfaces import BlockAllocator, Block
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator

from vllm.utils import Device

class DeviceAwareBlockAllocator:

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ):
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
                num_blocks=num_gpu_blocks,
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
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return DeviceAwareBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(
        self,
        cpu_block_allocator: BlockAllocator,
        gpu_block_allocator: BlockAllocator,
    ):
        assert not (cpu_block_allocator.all_block_ids & gpu_block_allocator.all_block_ids), "cpu and gpu block allocators can't have intersection of block ids"

        self._allocators = {
            Device.CPU: cpu_block_allocator,
            Device.GPU: gpu_block_allocator,
        }
        
        self._block_ids_to_allocator = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_mutable(self, prev_block: Optional[Block], device: Device) -> Block:
        return self._allocators[device].allocate_mutable(prev_block)

    def allocate_immutable(self, prev_block: Optional[Block], token_ids: List[int], device: Device) -> Block:
        return self._allocators[device].allocate_immutable(prev_block, token_ids)
 
    def free(self, block: Block) -> None:
        allocator = self._block_ids_to_allocator[block.physical_block_index]
        return allocator.free(block)

    def get_num_free_blocks(self, device: Device) -> int:
        return self._allocators[device].get_num_free_blocks()

    #@abstractmethod
    #def get_operations(self):
    #    pass
