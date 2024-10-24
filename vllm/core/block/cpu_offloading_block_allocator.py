from typing import Dict, FrozenSet, List, Optional, Tuple, Set

from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device


class CpuOffloadingBlockAllocator(CpuGpuBlockAllocator):
    """A block allocator that supports CPU KV cache offloading

    This class extends the `CpuGpuBlockAllocator` so that the CPU can be used for prefix caching.

    The key idea of this implementation is borrowed from ConServe
    (paper link: https://arxiv.org/abs/2410.01228), which constantly copy
    the KV cache of allocated GPU blocks to CPU if those blocks are full, 
    based on the assumption that the PCIe bandwidth is much higher than 
    the KV caches generation throughput.

    This implementation allows vLLM to gracefully handle preemption by 
    doing recomputation at scheduler side, and then prefix cache hit on CPU.
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

        assert allocator_type == "prefix_caching", "CpuOffloadingBlockAllocator should be only used together with prefix caching."

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

    def __init__(self, cpu_block_allocator: BlockAllocator,
                 gpu_block_allocator: BlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        super.__init__(cpu_block_allocator, gpu_block_allocator)

        self._uncopied: Set[int] = set()
        self._copied: Set[int] = set()
        self._free: Set[int] = set(gpu_block_allocator.all_block_ids)
        self._cpu: Set[int] = set(cpu_block_allocator.all_block_ids)

        self._cpu_blocks_allocated: Deque[int] = deque()

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
        block = self._allocators[device].allocate_mutable_block(prev_block)
        # a mutable block needs to be allocated from free blocks
        assert block_id in self._free
        # move block status from free to using
        self._free.remove(block_id)
        self._using.add(block_id)
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

        assert device == Device.GPU, "Calls to CPU offloading block allocator should always use Device.GPU --- CPU offloading block allocator handles CPU offloading internally."

        # allocate a GPU block        
        block = self._allocators[device].allocate_immutable_block(
            prev_block, block_token_ids)
        block_id = block.block_id
        
        # deal with prefix caching, three cases in total:
        # 1. cache hit on GPU
        # 2. no cache hit on GPU but cache hit on CPU
        # 3. no cache hit
        if block.computed:
            # prefix hit on GPU
            # mark the block as copied
            assert block_id in self._free or block_id in self._copied, "Internal implementation error: a caching GPU block block is neither copied nor free. This should never happen."
            if block_id in self._free:
                self._free.remove(block_id)
                self._copied.add(block_id)
        else:
            # The GPU block must be a free block.
            assert block_id in self._free, "Internal implementation error: a non-caching GPU block is not free. This should never happen."
            
            # check if we can hit cache on CPU
            cpu_block = self.allocator[Device.CPU].allocate_immutable_block(prev_block, block_token_ids)
            cpu_block_id = cpu_block.block_id
            if cpu_block.computed:
                # CPU cache hit
                # mark the GPU block as copied, copy the KV cache to CPU, and temporarily hold this cpu block to protect its data
                block.computed = True
                self._free.remove(block_id)
                self._copied.add(block_id)
                self._swap_mapping[cpu_block_id] = gpu_block_id
                self._cpu_blocks_allocated.append(cpu_block_id)
            else:
                # No cache hit
                # mark the GPU block as uncopied, and free cpu block
                self._free.remove(block_id)
                self._uncopied.add(block_id)
                self.allocator[Device.CPU].free(cpu_block)

        return block

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
        allocator = self._block_ids_to_allocator[block_id]\
        allocator.free(block)

    def get_num_free_blocks(self, device: Device) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (Device): The device for which to query the number of free
                blocks. AssertionError is raised if None is passed.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        assert self._allocators[device].get_num_free_blocks() == len(self._free)
        return self._allocators[device].get_num_free_blocks()

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

    def get_computed_block_ids(self, prev_computed_block_ids: List[int],
                               block_ids: List[int],
                               skip_last_block_id: bool) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_computed_block_ids(
            prev_computed_block_ids, block_ids, skip_last_block_id)

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
