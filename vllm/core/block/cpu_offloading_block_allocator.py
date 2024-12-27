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
from collections import deque
from dataclasses import dataclass
from typing import (Deque, Dict, Generic, Iterator, List, Optional, Tuple,
                    TypeVar)

from vllm.core.block.cpu_gpu_block_allocator import (CpuGpuBlockAllocator,
                                                     NullBlock)
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.utils import Device

T = TypeVar("T")


@dataclass
class IndexableLinkedListNode(Generic[T]):
    value: Optional[T]
    prev_node: Optional["IndexableLinkedListNode[T]"] = None
    next_node: Optional["IndexableLinkedListNode[T]"] = None


class IndexableLinkedListIterator(Iterator[T], Generic[T]):

    def __init__(self, head: IndexableLinkedListNode[T]):
        self._current = head
        self.skip_head = True

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self._current.next_node is None:
            raise StopIteration

        if self.skip_head:
            self.skip_head = False
            self._current = self._current.next_node
            return self.__next__()

        value = self._current.value
        self._current = self._current.next_node  # type: ignore
        assert value is not None
        return value


class IndexableLinkedList(Generic[T]):
    """
    A double linked list that supports indexing under O(1) time complexity.
    It assumes the index space is limited, and will pre-allocate an array
    to store references.

    This is a helper class for CpuOffloadingBlockAllocator to track uncached 
    blocks.
    """

    def __init__(self, max_index: int):
        self._max_index = max_index
        self._nodes: List[Optional[IndexableLinkedListNode[T]]] = [
            None for _ in range(max_index)
        ]
        self._head = IndexableLinkedListNode[T](None, None,
                                                None)  # type: ignore
        self._tail = IndexableLinkedListNode[T](None, self._head,
                                                None)  # type: ignore
        self._head.next_node = self._tail
        self._num_blocks = 0

    def _insert_before(self, node: IndexableLinkedListNode[T],
                       next_node: IndexableLinkedListNode[T]) -> None:
        prev_node = next_node.prev_node
        assert prev_node is not None
        prev_node.next_node = node
        node.prev_node = prev_node
        node.next_node = next_node
        next_node.prev_node = node
        self._num_blocks += 1

    def __getitem__(self, index: int) -> T:
        assert 0 <= index < self._max_index, "Index out of range"
        node = self._nodes[index]
        assert node is not None, f"Index {index} not found"
        assert node.value is not None
        return node.value

    def __setitem__(self, index: int, value: T):
        assert 0 <= index < self._max_index, "Index out of range"
        node = self._nodes[index]
        if node is None:
            node = IndexableLinkedListNode(value, None, None)
            self._nodes[index] = node
            self._insert_before(node, self._tail)
        else:
            node.value = value

    def __iter__(self):
        return IndexableLinkedListIterator(self._head)

    def remove(self, index: int):
        assert 0 <= index < self._max_index, "Index out of range"
        node = self._nodes[index]
        if node is None:
            return

        prev_node = node.prev_node
        next_node = node.next_node

        # prev and next will be guarded by head and tail, so should not be None
        assert prev_node is not None
        assert next_node is not None
        prev_node.next_node = next_node
        next_node.prev_node = prev_node
        self._nodes[index] = None
        self._num_blocks -= 1

    def __len__(self):
        return self._num_blocks


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

    allocators: Dict[Device, PrefixCachingBlockAllocator]

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
            DeviceAwareBlockAllocator: A CpuOffloadingBlockAllocator instance 
                with the specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        assert num_gpu_blocks < num_cpu_blocks, "CPU offloading block "\
            "allocator requires the allocated CPU memory capacity to be larger"\
            " than GPU memory capacity."
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

        super().__init__(cpu_block_allocator, gpu_block_allocator)
        self._allocators: Dict[Device,
                               PrefixCachingBlockAllocator] = {  # type: ignore
                                   Device.CPU: cpu_block_allocator,
                                   Device.GPU: gpu_block_allocator
                               }
        self.num_gpu_blocks = gpu_block_allocator.get_num_total_blocks()
        self.num_cpu_blocks = cpu_block_allocator.get_num_total_blocks()
        """
        GPU block should only be in one of the following three status:
          uncached: allocated blocks that didn't hit any cache
          cached: allocated blocks that are cached, either in GPU or in CPU
          free: the blocks are not allocated by block allocator
        This implementation aims to transform uncached blocks to cached blocks
        by performing GPU to CPU copy when calling `get_and_reset_swaps`
        
        As block allocator will automatically track free blocks, and we don't 
        need to specially handle cached blocks. So we only track uncached blocks
        """
        self._uncached_blocks: IndexableLinkedList[Block] = \
                IndexableLinkedList(self.num_cpu_blocks + self.num_gpu_blocks)
        """
        We probe CPU cache hit by trying to allocate a CPU 
        block and see if it is computed.
        If we hit the CPU cache, we cannot free this CPU block until the end 
        of scheduler step, in order to avoid the CPU cache being overwritten.
        so we track the cpu blocks we allocated, and free it after scheduler
        step (i.e. calling `get_and_reset_swaps`).
        """
        self._allocated_cpu_blocks: Deque[Block] = deque()

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               device: Device,
                               extra_hash: Optional[int] = None) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device == Device.GPU, "Calls to CPU offloading block allocator "\
            "should always use Device.GPU --- CPU offloading block allocator "\
            "handles CPU offloading internally."\
        # mark this block as uncached

        block = self._allocators[device].allocate_mutable_block(
            prev_block, extra_hash=extra_hash)
        assert block.block_id is not None
        #self._uncached_blocks.append(block)
        self._uncached_blocks[block.block_id] = block
        return block

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

        Returns:
            List[Block]: The newly allocated list of immutable blocks 
                containing the provided block token IDs.
        """
        assert device == Device.GPU, "Calls to CPU offloading block allocator "\
            "should always use Device.GPU --- CPU offloading block allocator"\
            "handles CPU offloading internally."

        # repeatedly call allocate_immutable_block
        # because it handles CPU-GPU offloading related logics.
        blocks = []
        for token_ids in block_token_ids:
            prev_block = self.allocate_immutable_block(prev_block=prev_block,
                                                       token_ids=token_ids,
                                                       device=device,
                                                       extra_hash=extra_hash)
            blocks.append(prev_block)
        return blocks

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

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """

        assert device == Device.GPU, "Calls to CPU offloading block allocator"\
            " should always use Device.GPU --- CPU offloading block allocator"\
            " handles CPU offloading internally."

        # allocate a GPU block
        block = self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, extra_hash=extra_hash)
        block_id = block.block_id
        assert block_id is not None
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
            cpu_block = self._allocators[Device.CPU].allocate_immutable_block(
                prev_block, token_ids, extra_hash=extra_hash)
            cpu_block_id = cpu_block.block_id
            assert cpu_block_id is not None
            cpu_block_computed = self._allocators[
                Device.CPU].block_is_computed(cpu_block_id)
            if cpu_block_computed:
                # CPU cache hit
                # mark the GPU block as computed
                self._allocators[Device.GPU].mark_blocks_as_computed(
                    [block_id])
                # copy the CPU cache to GPU
                self._swap_mapping[cpu_block_id] = block_id
                # and don't free this block until `get_and_reset_swap` is called
                self._allocated_cpu_blocks.append(cpu_block)
            else:
                # No cache hit
                # mark the GPU block as uncached
                self._uncached_blocks[block_id] = block
                # and free cpu block
                self._allocators[Device.CPU].free(cpu_block)

        return block

    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:

        raise NotImplementedError("CPU offloading block allocator only "
                                  "support preemption by recomputation.")

    def _is_gpu_block(self, block_id: int) -> bool:
        return block_id in self._allocators[Device.GPU].all_block_ids

    def _is_gpu_block_unsafe(self, block_id: int) -> bool:
        """Faster version of `_is_gpu_block` that doesn't check the block ID.
        But assumes the that the block IDs are assigned contiguously, with GPU 
        block IDs coming before the CPU block IDs.
        """
        return block_id < self.num_gpu_blocks

    def _get_physical_block_id_unsafe(self, block_id: int) -> int:
        """Returns the physical block ID of the given block ID.

        This function avoids using the `allocator.get_physical_block_id()`
        which is slow (O(NlogN)). Instead, this is based on the assumption
        that the block IDs are assigned contiguously, with GPU block IDs coming
        before CPU block IDs.

        Args:
            block_id (int): The block ID to get the physical block ID of.

        Returns:
            int: The physical block ID of the given block ID.

        Note:
            Please see the implementation of 
            `CpuOffloadingBlockAllocator.create` for how the block IDs are
            assigned.
        """
        if self._is_gpu_block_unsafe(block_id):
            return block_id
        else:
            return block_id - self.num_gpu_blocks

    def get_and_reset_swaps(self,
                            now: float) -> Tuple[List[Tuple[int, int]], ...]:
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
            A tuple of two lists: (blocks_to_swap_out, blocks_to_swap_in).
            Each list is a List[Tuple[int, int]], containing the mapping of 
            source to destination block IDs. The block IDs are physical block
            IDs and it's expected to be used by the cache engine directly.
        """

        allocator = self._allocators[Device.GPU]
        cpu_allocator = self._allocators[Device.CPU]

        #new_uncached_blocks: Deque[Block] = deque()

        should_remove_ids = []
        for block in self._uncached_blocks:
            block_id = block.block_id

            refcount = allocator._refcounter.get(block_id)
            assert refcount > 0, "A freed block should have block_id None"

            # check if this block is computed
            computed = allocator.block_is_computed(block_id)
            if computed:  # This block is computed, copy it to CPU
                # allocate a block on CPU
                cpu_block = cpu_allocator.allocate_immutable_block(
                    prev_block=block.prev_block,
                    token_ids=block.token_ids,
                    extra_hash=block.extra_hash,
                )
                assert cpu_block.block_id is not None
                self._allocated_cpu_blocks.append(cpu_block)

                # mark CPU block as computed
                cpu_allocator.mark_blocks_as_computed([cpu_block.block_id])

                # copy the GPU block to CPU
                assert cpu_block.block_id is not None
                self._swap_mapping[block_id] = cpu_block.block_id

                should_remove_ids.append(block_id)
                continue

        for block_id in should_remove_ids:
            self._uncached_blocks.remove(block_id)

        # iterate over allocated CPU blocks, update access time and free them
        # need to update access time so that CPU evictor can work
        while self._allocated_cpu_blocks:
            cpu_block = self._allocated_cpu_blocks.pop()
            assert cpu_block.block_id is not None
            # update the access time
            cpu_allocator.mark_blocks_as_accessed([cpu_block.block_id], now)
            # free the block
            cpu_allocator.free(cpu_block)

        # populate the swap_out list and swap_in list
        blocks_to_swap_out = []
        blocks_to_swap_in = []
        for src, dst in self._swap_mapping.items():
            # only two possible cases: CPU -> GPU, or GPU -> CPU
            #if src in self._allocators[Device.GPU].all_block_ids:
            if self._is_gpu_block_unsafe(src):
                # swap out
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_out.append((src, dst))
            else:
                # swap in
                src = self._get_physical_block_id_unsafe(src)
                dst = self._get_physical_block_id_unsafe(dst)
                blocks_to_swap_in.append((src, dst))
        self._swap_mapping.clear()
        return blocks_to_swap_out, blocks_to_swap_in

    def will_swap_in_cpu_blocks(self):
        """Check if there are CPU blocks that will be swapped in

        Returns:
            bool: True if there are CPU blocks that will be swapped in, False
                otherwise.
        """
        return bool(self._swap_mapping)

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
        self._uncached_blocks.remove(block_id)
        super().free(block)
