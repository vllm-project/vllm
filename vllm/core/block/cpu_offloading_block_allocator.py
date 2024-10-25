"""This file implement a block allocator that supports CPU KV cache offloading

The key idea of this implementation is to maintain those allocated blocks 
that didn't hit the cache, and constantly copy them into CPU after each 
scheduler step.

This idea is borrowed from ConServe
(paper link: https://arxiv.org/abs/2410.01228), based on the assumption 
that the CPU-GPU bandwidth is much higher than GPU KV cache generation 
throughput. Thanks Yifan for this idea.

This implementation also allows vLLM to gracefully handle preemption by 
recomputation.
"""
from typing import Dict, FrozenSet, List, Optional, Tuple, Set, Deque
from collections import deque

from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator, \
    NullBlock
from vllm.utils import Device


class CpuOffloadingBlockAllocator(CpuGpuBlockAllocator):
    """A block allocator that supports CPU KV cache offloading

    This class extends the `CpuGpuBlockAllocator` so that the CPU can be used 
    for prefix caching.
    
    It will internally maintain uncached blocks, and trying to copy uncached
    blocks into CPU upon the end of scheduler step (i.e. calling 
    `get_and_reset_swaps`).

    This implementation also allows vLLM to gracefully handle preemption by 
    recomputation.
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        """Initiate CpuOffloadingBlockAllocator. Similar to 
        CpuGpuBlockAllocator.create() but only support prefix caching

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
            DeviceAwareBlockAllocator: A CpuOffloadingBlockAllocator instance with the
                specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        assert num_gpu_blocks < num_cpu_blocks, "CPU offloading block "\
            "allocator requires the allocated CPU memory capacity to be larger"\
            "than GPU memory capacity."
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        assert allocator_type == "prefix_caching", "CpuOffloadingBlock"\
            "Allocator should be only used together with prefix caching."

        # prefix caching block is now the default.
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

        return CpuOffloadingBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: PrefixCachingBlockAllocator,
                 gpu_block_allocator: PrefixCachingBlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        super.__init__(cpu_block_allocator, gpu_block_allocator)
        
        """
        GPU block should only be in one of the following three status:
          uncached: allocated blocks that didn't hit any cache
          cached: allocated blocks that hit the cache, either in GPU or in CPU
          free: the blocks are not allocated by block allocator
        block allocator will automatically track free blocks, and we don't need
        to specially handle cached blocks. So we only track uncached blocks
        """
        self._uncached_blocks: Deque[Block] = deque()

        """
        We probe CPU cache hit by trying to allocate a CPU 
        block and see if it is computed.
        If we hit the CPU cache, we cannot free this CPU block until the end 
        of scheduler step, in order to avoid the CPU cache being overwritten.
        so we track the cpu blocks we allocated, and free it after scheduler
        step (i.e. calling `get_and_reset_swaps`).
        """
        self._allocated_cpu_blocks: Deque[Block] = deque()

    def allocate_mutable_block(self, prev_block: Optional[Block],
                               device: Device) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device == Device.GPU, "Calls to CPU offloading block allocator should always use Device.GPU --- CPU offloading block allocator handles CPU offloading internally."
        # mark this block as uncached
        block = self._allocators[device].allocate_mutable_block(prev_block)
        self._uncached_blocks.append(block)
        return block_id

    def allocate_immutable_blocks(self, prev_block: Optional[Block],
                                  block_token_ids: List[List[int]],
                                  device: Device) -> List[Block]:
        """Allocates a new group of immutable blocks with the provided block 
        token IDs on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            block_token_ids (List[int]): The list of block token IDs to be 
                stored in the new blocks.
            device (Device): The device on which to allocate the new block.

        Returns:
            List[Block]: The newly allocated list of immutable blocks 
                containing the provided block token IDs.
        """

        assert device == Device.GPU, "Calls to CPU offloading block allocator should always use Device.GPU --- CPU offloading block allocator handles CPU offloading internally."

        # repeatedly call allocate_immutable_block
        # because it handles CPU-GPU offloading related logics.
        blocks = []
        for token_ids in block_token_ids:
            prev_block = self.allocate_immutable_block(prev_block=prev_block,
                                                       token_ids=token_ids,
                                                       device=device)
            blocks.append(prev_block)
        return blocks

    def allocate_immutable_block(self, prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: Device) -> Block:
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

        assert device == Device.GPU, "Calls to CPU offloading block allocator"\
            " should always use Device.GPU --- CPU offloading block allocator"\
            " handles CPU offloading internally."

        # allocate a GPU block        
        block = self._allocators[device].allocate_immutable_block(
            prev_block, block_token_ids)
        block_id = block.block_id
        block_computed = self._allocators[device].block_is_computed(block_id)
        
        # deal with prefix caching, three cases in total:
        # 1. cache hit on GPU
        # 2. no cache hit on GPU but cache hit on CPU
        # 3. no cache hit
        if block_computed:
            # cache hit on GPU, no need to put it into uncached blocks
            pass
        else:
            # check if we can hit cache on CPU by trying to allocate CPU block
            cpu_block = self.allocator[Device.CPU].allocate_immutable_block(
                prev_block, block_token_ids)
            cpu_block_id = cpu_block.block_id
            cpu_block_computed = self.allocator[Device.CPU].block_is_computed(
                cpu_block_id
            )
            if cpu_block_computed:
                # CPU cache hit
                # mark the GPU block as computed
                self._allocator[Device.GPU].mark_blocks_as_computed([block_id])
                # copy the CPU cache to GPU
                self._swap_mapping[cpu_block_id] = gpu_block_id
                # and don't free this block until `get_and_reset_swap` is called
                self._allocated_cpu_blocks.append(cpu_block)
            else:
                # No cache hit
                # mark the GPU block as uncached
                self._uncached_blocks.append(block)
                # and free cpu block
                self.allocator[Device.CPU].free(cpu_block)

        return block

    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:
        
            raise NotImplementedError("CPU offloading block allocator only "
                                      "support preemption by recomputation.")

    def get_and_reset_swaps(self, now: float) -> List[Tuple[int, int]]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called right before scheduler step finishes.
        
        This function will do the following things:
            1. Iterate over uncached blocks and see if we can copy it to CPU
            2. Update all allocated CPU block time stamp
            3. Free CPU blocks
            4. Return and clear all swapping status
            
        Args:
            now (float): The time stamp used to update CPU access time, so 
            that CPU evictor can work.
        
        Returns:
            List[Tuple[int, int]]: A mapping of source to destination block IDs.
        """
        
        allocator = self._allocators[Device.GPU]
        cpu_allocator = self._allocator[Device.CPU]
        
        new_uncached_blocks: Deque[Block] = deque()
        
        while self._uncached_blocks:
            block = self._uncached_blocks.pop()
            block_id = block.block_id
            
            # check if this block is freed
            if block_id is None:
                # this block is already freed, no longer need to copy it to CPU
                continue
            
            refcount = allocator._refcounter.get(block_id)
            assert refcount > 0, "A freed block should have block_id None"
            
            # check if this block is computed
            computed = allocator.block_is_computed(block_id)
            if computed: # This block is computed, copy it to CPU
                # allocate a block on CPU
                cpu_block = cpu_allocator.allocate_immutable_block(
                    prev_block = block._prev_block,
                    token_ids = block._token_ids
                )
                self._allocated_cpu_blocks.append(cpu_block)
                
                # copy the GPU block to CPU
                self._swap_mapping[block_id] = cpu_block.block_id
                
                continue
            
            # this block is neither freed nor computed
            # keep marking it as uncached
            new_uncached_blocks.append(block)
            
        # update uncached blocks
        self._uncached_blocks = new_uncached_blocks
                
        # iterate over allocated CPU blocks, update access time and free them
        # need to update access time so that CPU evictor can work
        while self._allocated_cpu_blocks:
            cpu_block = self._allocated_cpu_blocks.pop()
            # update the access time
            cpu_allocator.mark_blocks_as_accessed([cpu_block.block_id], now)
            # free the block
            cpu_allocator.free(cpu_block)
            
        # return the mapping
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return list(mapping.items())
