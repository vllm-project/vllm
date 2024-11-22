"""A block manager that manages token blocks."""
import math
import time
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.core.block.common import CacheMetricData
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.worker.swap.interface import SwapSpaceManagerBase
from vllm.core.block.radix_cache import RadixCache
logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                 block_position: int) -> None:
        pass

    @abstractmethod
    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_position: int) -> None:
        pass

    @abstractmethod
    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        pass

    @abstractmethod
    def n_rmap(self, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def move_swappable(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_swappable_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def block_in_evictor(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass

    @abstractmethod
    def get_prefix_cache_hit_rate(self) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.current_num_blocks = 0
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

        # Evictor contains the freed pages
        self.evictor: Evictor = make_evictor(eviction_policy)

        # Swapper contains the pages which are still allocated
        # This contains context caching blocks which can be swapped.
        # Ideally the recompute and swap out should also follow
        # adaptive approach
        # Context caching will the be the first step towards it.
        # For now it is also an evictor but every block has some
        # ref count (by some) sequences
        self.swapper: Evictor = make_evictor(eviction_policy)
        # Reverse mapping: Block -> Seq ID and nth block in the sequence
        self.rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}

        self.default_hash_ctr = count()

        self.cache_metric_data = CacheMetricData()

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            # Prioritize the evictor since they
            if len(self.evictor) > 0:
                block = self.evictor.evict()
            else:
                # There must be CCed blocks
                assert len(self.swapper) > 0
                block = self.swapper.evict()
                # Check ref. Should not have CC.
                # No allocated sequence. Otherwise incorrect
                assert block.ref_count == len(self.rmap[block])
                # Also remove it HT to prevent future hit
                del self.cached_blocks[block.block_hash]
                # Block Manager will change the block table

            block.prev_block_hash = block.block_hash
            block.block_hash = block_hash
            block.prev_num_hashed_tokens = block.num_hashed_tokens
            block.num_hashed_tokens = num_hashed_tokens
            return block

        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)

        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks
            block = self.evictor.remove(block_hash)
            assert block.ref_count == 0
            self.cached_blocks[block_hash] = block

        if block_hash in self.cached_blocks:
            self.cache_metric_data.query(hit=True)
            block = self.cached_blocks[block_hash]
            # Remove the block from swapper for CC requests
            if block_hash in self.swapper:
                self.swapper.remove(block_hash)
            block.is_evicted = False
        else:
            self.cache_metric_data.query(hit=False)
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
            block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        is_cc_only_before = block.block_hash in self.swapper
        if is_cc_only_before:
            assert block.block_hash in self.cached_blocks

        block.ref_count -= 1
        # rmap management outside of this func
        is_cc_only_after = block.ref_count == (len(self.rmap[block])
                                               if block in self.rmap else 0)
        # 8 cases cases:
        if block.ref_count == 0:
            if is_cc_only_before:
                # Last CC expired
                self.swapper.remove(block.block_hash)
            self.evictor.add(block)
            del self.cached_blocks[block.block_hash]
        else:
            if not is_cc_only_before and is_cc_only_after:
                self.swapper.add(block)
                assert block.block_hash in self.cached_blocks

    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                 block_position: int) -> None:
        # This rmap will be used when we swap out a mapped CC (only CC)
        self.rmap.setdefault(block, set()).add((seq_id, block_position))

    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_position: int) -> None:
        assert block in self.rmap
        self.rmap[block].remove((seq_id, block_position))
        if self.rmap[block] == set():
            del self.rmap[block]

    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        assert block in self.rmap
        del self.rmap[block]

    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        # Return a ref
        return self.rmap[block] if block in self.rmap else None

    def n_rmap(self, block: PhysicalTokenBlock) -> int:
        return len(self.rmap[block]) if block in self.rmap else 0

    def move_swappable(self, block: PhysicalTokenBlock) -> None:
        # Do nothing for the ref count since the seq still owns it
        if block.ref_count == 0:
            raise ValueError(
                f"Cannot move free block {block} to swappable list."
                "They should go to the evictor!")
        # Block should not reside in the evictor
        assert block.block_hash not in self.evictor
        n_rmaps: int = len(self.rmap[block])
        # Can only have CC requests and Normal requests. Both of which
        # takes refs
        # TODO: Unify them in a general block manager system
        # TODO: take linux mm and implement based on it
        assert n_rmaps <= block.ref_count
        if n_rmaps == block.ref_count and block.block_hash not in self.swapper:
            # Only First time CCed request
            self.swapper.add(block)

        # They should still reside in cached_blocks
        # When they are popped out from the swapper they will be removed
        # from the HT
        assert block.block_hash in self.cached_blocks

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def get_num_swappable_blocks(self) -> int:
        # CC blocks are swappable as long as there are
        # enough space in the next tier
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks + self.swapper.num_blocks)

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def block_in_evictor(self, block_hash: int) -> bool:
        return block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block

    def get_prefix_cache_hit_rate(self) -> float:
        return self.cache_metric_data.get_hit_rate()


class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def get_num_swappable_blocks(self) -> int:
        return len(self.free_blocks)

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def block_in_evictor(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def get_prefix_cache_hit_rate(self) -> float:
        return -1

    def move_swappable(self, block: PhysicalTokenBlock) -> None:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                 block_position: int) -> None:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_position: int) -> None:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def get_rmap(self, block: PhysicalTokenBlock):
        return None

    def n_rmap(self, block: PhysicalTokenBlock) -> int:
        return 0


class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        enable_memory_tiering: bool = False,
        enable_chunked_prefill: bool = False,
        swap_manager: Optional[SwapSpaceManagerBase] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            # Round up to nearest block size to regularize sliding window
            # allocation sizes.
            self.block_sliding_window = math.ceil(sliding_window / block_size)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        self.enable_memory_tiering = enable_memory_tiering
        self.enable_chunked_prefill = enable_chunked_prefill
        # Perform verification, dedup for swap in, modify swap out swap in case
        self.enable_verify = False

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            if self.enable_memory_tiering:
                logger.info("Memory Tiering is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
            
        # self.gpu_radix_cache = RadixCache()
        # self.cpu_radix_cache = RadixCache()
        # self.disk_radix_cache = RadixCache() if self.enable_disk_swap else None
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

        # TODO: Refactor the whole swapping part to modulize it
        # Mapping: seq_id -> BlockTable (CPU):
        # Only used for eviction from HBM to DRAM
        self.evict_block_tables: Dict[int, BlockTable] = {}
        # Mapping: req_id -> BlockTable
        # Note that each SequenceGroup has a unique
        # request ID
        self.cross_block_tables: Dict[str, BlockTable] = {}

        # Mapping: seq_id -> {dev_id -> BlockTable}
        # Device ID info is embedded in the PhysicalTokenBlock now
        # TODO: Organize as dev_id -> seq_id should be better
        # because we have to reduce the invocation of the device
        self.enable_disk_swap: bool = swap_manager is not None
        self.swap_manager: SwapSpaceManagerBase = swap_manager

    def _get_seq_num_required_blocks(
            self, seq: Optional[Sequence],
            is_encoder_decoder: bool) -> Tuple[int, int]:
        # Perform a prefix matching to
        n_prefix_matching_blocks = 0
        # The number of prefix matching blocks which are free
        n_free_prefix_matching_blocks = 0
        if (seq is not None and self.enable_caching
                and self.block_sliding_window is None
                and not is_encoder_decoder):
            # Only enable prefix matching for non sliding window
            # and non encoder decoder case
            num_prompt_blocks = seq.n_blocks
            for logical_idx in range(num_prompt_blocks):
                block_hash = seq.hash_of_block(logical_idx)
                n_prefix_matching_blocks += self.gpu_allocator.contains_block(
                    block_hash)
                n_free_prefix_matching_blocks += \
                    self.gpu_allocator.block_in_evictor(block_hash)

        return 0 if seq is None else seq.n_blocks - \
            n_prefix_matching_blocks, n_free_prefix_matching_blocks

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        is_encoder_decoder = seq_group.is_encoder_decoder()
        self_num_required_blocks, num_free_prefix_matching_blocks = \
            self._get_seq_num_required_blocks(
                seq_group.get_seqs(status=SequenceStatus.WAITING)[0],
                is_encoder_decoder)
        cross_num_required_blocks, _ = self._get_seq_num_required_blocks(
            seq_group.get_encoder_seq(), is_encoder_decoder)
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        if self.block_sliding_window is not None:

            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        num_swappable_gpu_blocks = (
            self.gpu_allocator.get_num_swappable_blocks()
            if self.enable_caching and self.enable_memory_tiering else
            num_free_gpu_blocks)

        num_swap_out_gpu_blocks = max(
            num_required_blocks -
            max(num_free_gpu_blocks - self.watermark_blocks, 0), 0)
        assert num_required_blocks >= num_swap_out_gpu_blocks

        num_free_cpu_blocks = (self.cpu_allocator.get_num_free_blocks()
                               if self.enable_caching
                               and self.enable_memory_tiering else 0)
        num_free_disk_blocks = (
            self.swap_manager.get_num_free_blocks_for_all()
            if self.swap_manager is not None else 0)

        # Have to deduct the matched free blocks from free and swappable
        num_free_gpu_blocks -= num_free_prefix_matching_blocks
        num_swappable_gpu_blocks -= num_free_prefix_matching_blocks
        assert num_free_gpu_blocks >= 0
        assert num_swappable_gpu_blocks >= 0

        logger.debug(
            "SeqGroup %s: %d blocks required, "
            "%d free GPU blocks %d swappable GPU blocks", seq_group.request_id,
            num_required_blocks, num_free_gpu_blocks, num_swappable_gpu_blocks)
        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if (num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks
                or (num_swappable_gpu_blocks - num_required_blocks
                >= self.watermark_blocks and num_free_cpu_blocks +
                num_free_disk_blocks >= num_swap_out_gpu_blocks)):
            # The second condition is a rough estimate
            # which does not consider prefix hit in lower tiers
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _block_in_allocator(self, block_hash: int,
                            num_hashed_tokens: int) -> Tuple[bool, bool, bool]:
        # Check whether the block is in each tier (allocated or in evictor)
        # Now also supports disk
        if self.enable_disk_swap:
            return (self.gpu_allocator.contains_block(block_hash),
                    self.cpu_allocator.contains_block(block_hash),
                    self.swap_manager.contains_block(block_hash,
                                                     num_hashed_tokens))
        return (self.gpu_allocator.contains_block(block_hash),
                self.cpu_allocator.contains_block(block_hash), False)

    def _update_rmap(self,
                     block_from: PhysicalTokenBlock,
                     block_to: PhysicalTokenBlock,
                     rmap_block_from: Optional[Set[Tuple[int, int]]] = None):
        """
        This help function changes the mapping from seq to block_from.
        It also removes the rmap of block_from and update that of block_to
        It also moves the ref count from block_from to block_to
        This function can also take the original rmap of the block_from if 
        it was preserved before
        """
        is_rmap_cleared_for_block_from = rmap_block_from is not None
        # Here assume the rmap of block_to is already preserved
        if rmap_block_from is None:
            if block_from.device == Device.GPU:
                rmap_block_from = self.gpu_allocator.get_rmap(block_from)
                rmap_block_from.copy() if rmap_block_from is not None else None
            elif block_from.device == Device.CPU:
                rmap_block_from = self.cpu_allocator.get_rmap(block_from)
                rmap_block_from.copy() if rmap_block_from is not None else None
            elif block_from.device == Device.SWAP:
                assert self.swap_manager is not None
                rmap_block_from = self.swap_manager.get_rmap(block_from)
                rmap_block_from.copy() if rmap_block_from is not None else None
            else:
                raise ValueError("Unknown device type")
        n_rmap = len(rmap_block_from) if rmap_block_from is not None else 0
        if n_rmap == 0:
            # No rmap
            return

        # Check the rmap of block_to is already cleared or preserved
        if block_to.device == Device.GPU:
            assert self.gpu_allocator.n_rmap(block_to) == 0
        elif block_to.device == Device.CPU:
            assert self.cpu_allocator.n_rmap(block_to) == 0
        elif block_to.device == Device.SWAP:
            assert self.swap_manager.n_rmap(block_to) == 0
        else:
            raise ValueError("Unknown device type")
        if rmap_block_from is not None:
            for seq_id, block_id in rmap_block_from:
                assert self.block_tables[seq_id][block_id] == block_from
                self.block_tables[seq_id][block_id] = block_to
                if block_to.device == Device.GPU:
                    self.gpu_allocator.add_rmap(block_to, seq_id, block_id)
                elif block_to.device == Device.CPU:
                    self.cpu_allocator.add_rmap(block_to, seq_id, block_id)
                elif block_to.device == Device.SWAP:
                    self.swap_manager.add_rmap(block_to, seq_id, block_id)
                else:
                    raise ValueError("Unknown device type")

        if block_from.device == Device.GPU:
            if not is_rmap_cleared_for_block_from:
                self.gpu_allocator.remove_rmap_all(block_from)
            else:
                assert self.gpu_allocator.n_rmap(block_from) == 0
        elif block_from.device == Device.CPU:
            if not is_rmap_cleared_for_block_from:
                self.cpu_allocator.remove_rmap_all(block_from)
            else:
                assert self.cpu_allocator.n_rmap(block_from) == 0
        elif block_from.device == Device.SWAP:
            assert not is_rmap_cleared_for_block_from
            self.swap_manager.remove_rmap_all(block_from)
        else:
            raise ValueError("Unknown device type")

        # Change ref count.
        # Should call free the states cannot be changed here
        block_from.ref_count -= n_rmap

        # Add new rmap and change ref count
        block_to.ref_count += n_rmap

    def _handle_swap_out(
        self, block_table: BlockTable, enable_swap_out_to_cpu: bool,
        enable_swap_out_to_disk: bool,
        block_rmap_from: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]],
        block_rmap_to: Optional[Dict[PhysicalTokenBlock, Set[Tuple[int,
                                                                   int]]]],
        block_table_swap_out_to_cpu: Optional[BlockTable],
        block_table_swap_out_to_disk: Optional[Dict[int, BlockTable]],
        swap_out_mapping_to_cpu: Optional[Dict[PhysicalTokenBlock,
                                               PhysicalTokenBlock]],
        swap_out_mapping_to_disk: Optional[Dict[PhysicalTokenBlock,
                                                PhysicalTokenBlock]]):
        """
        Here block_rmap_from records the rmap of the blocks in the block table
        block_rmap_to record the rmap of the CPU blocks before the state 
        are changed on the fly
        """
        if not enable_swap_out_to_cpu and not enable_swap_out_to_disk:
            return

        # Check arguments
        if enable_swap_out_to_cpu:
            assert block_table_swap_out_to_cpu is not None
            assert swap_out_mapping_to_cpu is not None
        if enable_swap_out_to_disk:
            assert block_table_swap_out_to_disk is not None
            assert swap_out_mapping_to_disk is not None
        if enable_swap_out_to_cpu ^ enable_swap_out_to_disk:
            # No third tier to store the evicted CCed blocks
            assert block_rmap_to is None
        # Prevent the oooxxxooo from happening when the DRAM is not enough
        for block in reversed(block_table):
            # Verifying the ref count. Only swap out CC blocks or free blocks
            n_rmaps = 0
            if block_rmap_from is not None and block in block_rmap_from:
                n_rmaps = len(block_rmap_from[block])
            elif block.device == Device.GPU:
                n_rmaps = self.gpu_allocator.n_rmap(block)
            elif block.device == Device.CPU:
                n_rmaps = self.cpu_allocator.n_rmap(block)
            assert block.ref_count - n_rmaps == 1
            is_in_cpu_allocator = (enable_swap_out_to_cpu
                                   and self.cpu_allocator.contains_block(
                                       block.prev_block_hash))
            is_in_disk = (enable_swap_out_to_disk
                          and self.swap_manager.contains_block(
                              block.prev_block_hash,
                              block.prev_num_hashed_tokens))
            # TODO: Handle the CC case
            if not block.is_evicted or not block.prev_computed:
                continue

            if (enable_swap_out_to_cpu and swap_out_mapping_to_cpu is not None
                    and block in swap_out_mapping_to_cpu):
                # Dedup
                to_block = swap_out_mapping_to_cpu[block]
                to_block.ref_count += 1
                to_block.computed = block.prev_computed
                assert block_table_swap_out_to_cpu is not None
                block_table_swap_out_to_cpu.append(to_block)
            elif (enable_swap_out_to_disk
                  and swap_out_mapping_to_disk is not None
                  and block in swap_out_mapping_to_disk):
                # Dedup
                to_block = swap_out_mapping_to_disk[block]
                to_block.ref_count += 1
                assert to_block.device_id >= Device.SWAP
                dev_id = to_block.device_id - Device.SWAP
                to_block.computed = block.prev_computed
                assert block_table_swap_out_to_disk is not None
                block_table_swap_out_to_disk.setdefault(
                    dev_id, BlockTable()).append(to_block)
            elif enable_swap_out_to_cpu and (
                    self.cpu_allocator.get_num_free_blocks() > 0
                    or is_in_cpu_allocator):
                # new block, evicted block, existing block
                to_block = self.cpu_allocator.allocate(
                    block.prev_block_hash, block.prev_num_hashed_tokens)
                logger.debug("Move GPU block: %d to cpu", block.block_number)
                if not is_in_cpu_allocator:
                    ## CC block check
                    rmap_to_block = self.cpu_allocator.get_rmap(to_block)
                    rmap_to_block = rmap_to_block.copy(
                    ) if rmap_to_block is not None else None
                    if rmap_to_block is not None:
                        # Evict CC block. There must be another tier to keep it.
                        # Otherwise we will have to drop the CC block and
                        # violates the SLO-a
                        assert enable_swap_out_to_disk, \
                            "No lower tier to keep the CC block in CPU."
                        self.cpu_allocator.remove_rmap_all(to_block)
                        assert block_rmap_to is not None
                        block_rmap_to.setdefault(to_block,
                                                 set()).update(rmap_to_block)

                    # new block, evicted block => swap out
                    assert to_block.ref_count == 1
                    assert swap_out_mapping_to_cpu is not None
                    swap_out_mapping_to_cpu[block] = to_block
                    to_block.computed = block.prev_computed
                    if block in block_rmap_from:
                        self._update_rmap(block, to_block,
                                          block_rmap_from[block])
                assert block_table_swap_out_to_cpu is not None
                block_table_swap_out_to_cpu.append(to_block)
            elif enable_swap_out_to_disk and self.swap_manager.can_allocate(
                    block.prev_block_hash, block.prev_num_hashed_tokens):
                # Contains both case of having free space and having the block
                to_block = self.swap_manager.allocate(
                    block.prev_block_hash, block.prev_num_hashed_tokens)
                logger.debug("Move %s block: %d to disk %d, hashed tokens: %d",
                             block.device, block.block_number,
                             to_block.block_number,
                             block.prev_num_hashed_tokens)
                if not is_in_disk:
                    ## CC block check
                    rmap_to_block = self.swap_manager.get_rmap(to_block)
                    rmap_to_block = rmap_to_block.copy(
                    ) if rmap_to_block is not None else None
                    assert rmap_to_block is None, \
                        "No lower tier to keep the CC block in Disk."

                    assert to_block.ref_count == 1
                    assert to_block.device_id >= Device.SWAP
                    assert swap_out_mapping_to_disk is not None
                    swap_out_mapping_to_disk[block] = to_block
                    to_block.computed = block.prev_computed
                    if block in block_rmap_from:
                        self._update_rmap(block, to_block,
                                          block_rmap_from[block])
                dev_id = to_block.device_id - Device.SWAP
                assert block_table_swap_out_to_disk is not None
                block_table_swap_out_to_disk.setdefault(
                    dev_id, BlockTable()).append(to_block)
            else:
                logger.debug("Drop %s block: %d", block.device,
                             block.block_number)
                break

    def _handle_swap_in_cpu_to_gpu(self, gpu_block: PhysicalTokenBlock,
                                   swap_in_cpu_to_gpu: List[Tuple[int, int]],
                                   cpu_block_table: BlockTable):
        cpu_block = self.cpu_allocator.allocate(gpu_block.block_hash,
                                                gpu_block.num_hashed_tokens)
        logger.debug("Move CPU block: %d to GPU %d", cpu_block.block_number,
                     gpu_block.block_number)
        assert cpu_block.computed
        n_rmap_cpu_block = self.cpu_allocator.n_rmap(cpu_block)
        if n_rmap_cpu_block:
            # CC only requests. Verify the ref count
            assert cpu_block.ref_count == n_rmap_cpu_block + 1
            assert cpu_block.computed
            self._update_rmap(cpu_block, gpu_block)
        else:
            # CPU ref count must be one otherwise we hit HBM block
            assert cpu_block.ref_count == 1

        # Hold a reference
        gpu_block.computed = True
        swap_in_cpu_to_gpu.append(
            (cpu_block.block_number, gpu_block.block_number))
        cpu_block_table.append(cpu_block)

    def _handle_swap_in_disk_to_gpu(
            self, gpu_block: PhysicalTokenBlock,
            swap_in_disk_to_gpu: Dict[int, List[Tuple[int, int]]],
            disk_block_table: Dict[int, BlockTable]):
        disk_block = self.swap_manager.allocate(gpu_block.block_hash,
                                                gpu_block.num_hashed_tokens)
        logger.debug("Move disk block: %d to GPU %d", disk_block.block_number,
                     gpu_block.block_number)
        n_rmap_disk_block = self.swap_manager.n_rmap(disk_block)
        if n_rmap_disk_block:
            # CC only requests. Verify the ref count
            assert disk_block.ref_count == n_rmap_disk_block + 1
            assert disk_block.prev_computed
            self._update_rmap(disk_block, gpu_block)
        else:
            # Disk ref count must be one otherwise we hit HBM block
            assert disk_block.computed
            assert disk_block.ref_count == 1
            assert disk_block.device_id >= Device.SWAP

        gpu_block.computed = True
        dev_id = disk_block.device - Device.SWAP
        swap_in_disk_to_gpu.setdefault(dev_id, []).append(
            (disk_block.block_number, gpu_block.block_number))
        disk_block_table.setdefault(dev_id, BlockTable()).append(disk_block)

    # TODO: Refactor at least the return type
    def _allocate_sequence(self, \
                           seq: Optional[Sequence], \
                           ref_count: int, \
                           is_encoder_decoder: bool = True
                           ) -> Tuple[BlockTable,
                                      BlockTable,
                                      Dict[int, BlockTable],
                                      Dict[int, BlockTable],
                                      List[Tuple[int, int]],
                                      List[Tuple[int, int]],
                                      Dict[int, List[Tuple[int, int]]],
                                      Dict[int, List[Tuple[int, int]]],
                                      Dict[int, List[Tuple[int, int]]]]:
        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = seq.n_blocks if seq is not None else 0

        block_table: BlockTable = BlockTable()

        # To record block that needs to be freed later
        gpu_block_table_swap_out: BlockTable = BlockTable()
        cpu_block_table_swap_in: BlockTable = BlockTable()
        cpu_block_table_swap_out: BlockTable = BlockTable()

        # HBM <-> DRAM
        swap_in: List[Tuple[int, int]] = []
        swap_out_mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}

        # HBM <- Disk
        swap_in_from_disk: Dict[int, List[Tuple[int, int]]] = {}
        # DRAM -> Disk
        swap_out_mapping_from_cpu: Dict[PhysicalTokenBlock,
                                        PhysicalTokenBlock] = {}
        # HBM -> Disk
        swap_out_mapping_from_gpu: Dict[PhysicalTokenBlock,
                                        PhysicalTokenBlock] = {}
        # For combined free: dev_id -> BlockTable
        disk_block_table_swap_in: Dict[int, BlockTable] = {}
        disk_block_table_swap_out: Dict[int, BlockTable] = {}

        # Store rmap of the CC GPU blocks
        gpu_block_rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}
        # Store rmap of the CC CPU blocks
        cpu_block_rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}

        is_prev_hit = True
        assert seq is not None

        # # we can put the block hash into the tree here.
        # block_hashes = [seq.hash_of_block[i] for i in range(num_prompt_blocks)]
        # if self.enable_caching:
        #     #TODO only cached on GPU, need to swap between devices
        #     block_table = self.gpu_radix_cache.match_prefix(block_hashes)
        #     if len(block_table.blocks) > 0:
        #         # increase the ref count
        #         for b in block_table.blocks:
        #             b.ref_count += 1
        #         return block_table, BlockTable(), {}, {}, [], [],{}, {}, {}

        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = ref_count
            elif not is_encoder_decoder and self.enable_caching:
                block_hash = seq.hash_of_block(logical_idx)
                num_hashed_tokens = seq.num_hashed_tokens_of_block(logical_idx)
                is_in_allocator = self._block_in_allocator(
                    block_hash, num_hashed_tokens)
                is_hit = True in is_in_allocator
                if self.enable_memory_tiering and is_prev_hit is False:
                    assert is_hit is False, "ooooxxxooo case happened!"
                is_prev_hit = is_hit
                # Either a block existing or a new block or an evicted block
                # Or a swappable CCed block
                block = self.gpu_allocator.allocate(block_hash,
                                                    num_hashed_tokens)

                if self.enable_memory_tiering:
                    # Record the original
                    if not is_in_allocator[0]:
                        n_rmap_gpu_block = self.gpu_allocator.n_rmap(block)
                        if n_rmap_gpu_block > 0:
                            # Evict a CC block
                            assert block.ref_count == n_rmap_gpu_block + 1
                            # Verify the rmap will be replaced
                            assert block.is_evicted and block.prev_computed
                            # Preserve its rmap which is replaced in swap out
                            rmap = self.gpu_allocator.get_rmap(block)
                            rmap = rmap.copy() if rmap is not None else None
                            if rmap is not None:
                                self.gpu_allocator.remove_rmap_all(block)
                                gpu_block_rmap.setdefault(block,
                                                          set()).update(rmap)
                                # Ref count is handled in later swap out

                    # Handle swap in from DRAM
                    if ((not is_in_allocator[0]) and is_in_allocator[1]
                            and is_prev_hit):
                        self._handle_swap_in_cpu_to_gpu(
                            block, swap_in, cpu_block_table_swap_in)

                    # Handle swap in from possible disk
                    if (self.enable_disk_swap and (not is_in_allocator[0])
                            and (not is_in_allocator[1])
                            and is_in_allocator[2]):
                        self._handle_swap_in_disk_to_gpu(
                            block, swap_in_from_disk, disk_block_table_swap_in)

                    # Handle swap out
                    # Skip blocks not computed (e.g. length capped)
                    if block.is_evicted and block.prev_computed:
                        gpu_block_table_swap_out.append(block)
            else:
                block = self.gpu_allocator.allocate()
                # Set the reference counts of the token blocks.
                block.ref_count = ref_count

            # Verify
            if (is_encoder_decoder or not self.enable_caching or
                (self.enable_caching and not self.enable_memory_tiering)):
                assert (len(cpu_block_table_swap_in) == 0
                        and len(cpu_block_table_swap_out)
                        == 0) and (swap_in == []) and (swap_out_mapping == {})

            block_table.append(block)

        if self.enable_memory_tiering and len(gpu_block_table_swap_out) > 0:
            if self.enable_disk_swap:
                self._handle_swap_out(
                    gpu_block_table_swap_out,
                    enable_swap_out_to_cpu=True,
                    enable_swap_out_to_disk=True,
                    block_rmap_from=gpu_block_rmap,
                    block_rmap_to=cpu_block_rmap,
                    block_table_swap_out_to_cpu=cpu_block_table_swap_out,
                    block_table_swap_out_to_disk=disk_block_table_swap_out,
                    swap_out_mapping_to_cpu=swap_out_mapping,
                    swap_out_mapping_to_disk=swap_out_mapping_from_gpu)
            else:
                self._handle_swap_out(
                    gpu_block_table_swap_out,
                    enable_swap_out_to_cpu=True,
                    enable_swap_out_to_disk=False,
                    block_rmap_from=gpu_block_rmap,
                    block_rmap_to=None,
                    block_table_swap_out_to_cpu=cpu_block_table_swap_out,
                    block_table_swap_out_to_disk=None,
                    swap_out_mapping_to_cpu=swap_out_mapping,
                    swap_out_mapping_to_disk=None)

        # CPU Block Table can contain blocks which we only takes a
        # reference in case they are freed or used by other sequences
        cpu_block_table = BlockTable(
            [cpu_block for _, cpu_block in swap_out_mapping.items()])

        # Handle DRAM -> Disk case and prevent oooxxxooo case
        if self.enable_disk_swap and len(cpu_block_table) > 0:
            self._handle_swap_out(
                cpu_block_table,
                enable_swap_out_to_cpu=False,
                enable_swap_out_to_disk=True,
                block_rmap_from=cpu_block_rmap,
                block_rmap_to=None,
                block_table_swap_out_to_cpu=None,
                block_table_swap_out_to_disk=disk_block_table_swap_out,
                swap_out_mapping_to_cpu=None,
                swap_out_mapping_to_disk=swap_out_mapping_from_cpu)

        swap_out_from_cpu: Dict[int, List[Tuple[int, int]]] = {}
        swap_out_from_gpu: Dict[int, List[Tuple[int, int]]] = {}
        for gpu_block, disk_block in swap_out_mapping_from_gpu.items():
            dev_id = disk_block.device_id - Device.SWAP
            swap_out_from_gpu.setdefault(dev_id, []).append(
                (gpu_block.block_number, disk_block.block_number))
        for cpu_block, disk_block in swap_out_mapping_from_cpu.items():
            dev_id = disk_block.device_id - Device.SWAP
            swap_out_from_cpu.setdefault(dev_id, []).append(
                (cpu_block.block_number, disk_block.block_number))

        swap_out = [(gpu_block.block_number, cpu_block.block_number)\
                for gpu_block, cpu_block in swap_out_mapping.items()]
        # self.gpu_radix_cache.insert(block_hashes, block_table.blocks)
        return block_table, cpu_block_table_swap_in + cpu_block_table_swap_out,\
            disk_block_table_swap_in, disk_block_table_swap_out, \
            swap_in, swap_out, swap_in_from_disk, \
            swap_out_from_gpu, swap_out_from_cpu

    def allocate(
        self, seq_group: SequenceGroup
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[
            int, int, int, int]], List[Tuple[int, int, int, int]]]:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        blocks_to_swap_in_from_disk: List[Tuple[int, int, int, int]] = []
        blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        wait_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        seq = wait_seqs[0]
        block_table, cpu_block_table, disk_block_table_swap_in, \
            disk_block_table_swap_out, swap_in, swap_out, \
            swap_in_from_disk, swap_out_from_gpu, swap_out_from_cpu = \
            self._allocate_sequence(seq,
                                    seq_group.num_seqs(),
                                    is_encoder_decoder)
        for dev_id, pairs in swap_in_from_disk.items():
            blocks_to_swap_in_from_disk.extend(
                [pair + (dev_id + Device.SWAP, Device.GPU) for pair in pairs])
        for dev_id, pairs in swap_out_from_gpu.items():
            blocks_to_swap_out_to_disk.extend(
                [pair + (Device.GPU, dev_id + Device.SWAP) for pair in pairs])
        for dev_id, pairs in swap_out_from_cpu.items():
            blocks_to_swap_out_to_disk.extend(
                [pair + (Device.CPU, dev_id + Device.SWAP) for pair in pairs])

        # Assign the self-attention block tables for each sequence.
        for seq in wait_seqs:
            self.block_tables[seq.seq_id] = block_table.copy()
            if self.enable_memory_tiering:
                self.evict_block_tables[seq.seq_id] = cpu_block_table.copy()
                if self.enable_disk_swap:
                    # Update via swap manager
                    self.swap_manager.update_block_tables(
                        seq.seq_id, disk_block_table_swap_in.copy())
                    self.swap_manager.update_block_tables(
                        seq.seq_id, disk_block_table_swap_out.copy())

        # Allocate encoder sequence
        if is_encoder_decoder:
            # A SequenceGroup has only a single encoder sequence (at most),
            # thus allocate with a ref count of 1
            block_table, _, _, _, _, _, _, _, _ = self._allocate_sequence(
                seq_group.get_encoder_seq(), 1, is_encoder_decoder)
            # Assign the cross-attention block table for the SequenceGroup.
            self.cross_block_tables[seq_group.request_id] = block_table

        return swap_in, swap_out, \
            blocks_to_swap_in_from_disk, blocks_to_swap_out_to_disk

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(seq.n_blocks - 1)
        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Sequence,
    ) -> Tuple[PhysicalTokenBlock, Optional[PhysicalTokenBlock],
               Optional[PhysicalTokenBlock]]:
        # Called before a new block is appended.
        # This is in charge of allocating a new physical block (to be appended).

        # None if the last block is not full. Otherwise, we set it to the
        # content hash.
        if not self.enable_caching:
            return self.gpu_allocator.allocate(), None, None
        block_hash: Optional[int] = None
        n_blocks = seq.n_blocks
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash_of_block(n_blocks - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(n_blocks - 1)

        # num_hashed_tokens is used to compute future hashes
        # (e.g. in the hashing function, it is used to ask the sequence for
        # prefix tokens)
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)
        if (self.enable_caching and self.enable_memory_tiering
                and new_block.is_evicted and new_block.prev_computed
                and self.cpu_allocator.get_num_free_blocks() > 0):
            # NOTE: We don't swap in for decode now because decode should be
            # faster than swap in. Also swap out comes with a cost but drop
            # can cause oooxxxooo problem and degrade prefill performance.
            # Another important issue is that here it is possible that the
            # block is dropped because of not enough CPU memory. If decode and
            # prefill is scheduled at the same time like chunked prefill, the
            # dropped block can cause oooxxxooo issue in the future
            cpu_block = self.cpu_allocator.allocate(
                new_block.prev_block_hash, new_block.prev_num_hashed_tokens)
            cpu_block.computed = new_block.prev_computed
            rmap_cpu_block = self.cpu_allocator.get_rmap(cpu_block)
            rmap_cpu_block = rmap_cpu_block.copy(
            ) if rmap_cpu_block is not None else None
            if rmap_cpu_block is not None:
                assert self.enable_disk_swap, \
                    "No lower tier to keep the CC block in CPU."
                self.cpu_allocator.remove_rmap_all(cpu_block)
            self._update_rmap(new_block, cpu_block)
            if (self.enable_disk_swap and cpu_block.is_evicted
                    and cpu_block.prev_computed
                    and self.swap_manager.can_allocate(
                        cpu_block.prev_block_hash,
                        cpu_block.prev_num_hashed_tokens)):
                disk_block = self.swap_manager.allocate(
                    cpu_block.prev_block_hash,
                    cpu_block.prev_num_hashed_tokens)
                disk_block.computed = cpu_block.prev_computed
                rmap_disk_block = self.swap_manager.get_rmap(disk_block)
                rmap_disk_block = rmap_disk_block.copy(
                ) if rmap_disk_block is not None else None
                assert rmap_disk_block is None, \
                    "No lower tier to keep the CC block in Disk."
                self._update_rmap(cpu_block, disk_block, rmap_cpu_block)
                return new_block, cpu_block, disk_block

            return new_block, cpu_block, None

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block, None, None

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[
            int, int]], List[Tuple[int, int, int, int]], List[Tuple[
                int, int, int, int]]]:
        """Allocate a physical slot for a new token.

        Rets:
            cow list: Blocks to copy
            swap in list: Blocks to swap in (unused for now)
            swap out list: Blocks to swap out
        """
        n_blocks = seq.n_blocks
        block_table = self.block_tables[seq.seq_id]
        # If we need to allocate a new physical block
        if len(block_table) < n_blocks:
            # Currently this code only supports adding one physical block
            assert len(block_table) == n_blocks - 1

            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # reuse a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence hash a new logical block.
                # Allocate a new physical block.
                new_block, cpu_block, disk_block = \
                    self._allocate_last_physical_block(seq)
                block_table.append(new_block)
                if cpu_block is not None:
                    self.evict_block_tables.setdefault(
                        seq.seq_id, BlockTable()).append(cpu_block)
                    if disk_block is not None:
                        dev_id = disk_block.device_id - Device.SWAP
                        self.swap_manager.update_block_tables(
                            seq.seq_id, {dev_id: BlockTable([disk_block])})
                        return [], [], [(new_block.block_number,
                                        cpu_block.block_number)], \
                                [], [(cpu_block.block_number,
                                      disk_block.block_number,
                                      Device.CPU,
                                      disk_block.device_id)]
                    return [], [], [(new_block.block_number,
                                     cpu_block.block_number)], [], []
                else:
                    return [], [], [], [], []

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            if self.enable_caching:
                # If the last block is now complete, we may reuse an old block
                # to save memory.
                maybe_new_block = self._maybe_promote_last_block(
                    seq, last_block)
                # maybe_new_block always resides in HBM
                block_table[-1] = maybe_new_block
            return [], [], [], [], []
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            # swap out: Possibly have to evict the block
            new_block, cpu_block, disk_block = \
                self._allocate_last_physical_block(seq)
            if cpu_block is not None:
                self.evict_block_tables.setdefault(
                    seq.seq_id, BlockTable()).append(cpu_block)
                if disk_block is not None:
                    dev_id = disk_block.device_id - Device.SWAP
                    self.swap_manager.update_block_tables(
                        seq.seq_id, {dev_id: BlockTable([disk_block])})

            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            if cpu_block is not None:
                if disk_block is not None:
                    dev_id = disk_block.device_id - Device.SWAP
                    return ([
                        (last_block.block_number, new_block.block_number)
                    ], [], [
                        (new_block.block_number, cpu_block.block_number)
                    ], [], [(cpu_block.block_number, disk_block.block_number,
                             Device.CPU, disk_block.device_id)])
                return ([(last_block.block_number, new_block.block_number)],
                        [], [(new_block.block_number, cpu_block.block_number)
                             ], [], [])
            else:
                return ([(last_block.block_number, new_block.block_number)],
                        [], [], [], [])

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()

        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:

        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        request_id = seq_group.request_id
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        # Cross-attention blocks
        if seq_group.is_encoder_decoder():
            blocks.update(self.cross_block_tables[request_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        if seq_group.is_encoder_decoder():
            num_swapped_seqs += 1
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _swap_in_block_table(
        self,
        block_table: BlockTable,
        swap_in_mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock],
        swap_out_mapping_to_cpu: Dict[PhysicalTokenBlock, PhysicalTokenBlock],
        swap_out_mapping_to_disk: Optional[Dict[PhysicalTokenBlock,
                                                PhysicalTokenBlock]] = None,
    ) -> Tuple[BlockTable, BlockTable, Dict[int, BlockTable]]:
        """
        We specialize the function to make the implementation more clear
        """
        enable_swap_out_to_disk: bool = swap_out_mapping_to_disk is not None

        # Right now, the sequence level swap to disk is not supported yet
        gpu_block_table: BlockTable = BlockTable()
        cpu_block_table: BlockTable = BlockTable()
        disk_block_table: Dict[int, BlockTable] = {}

        # Can be extended to disk next
        gpu_block_table_swap_out: BlockTable = BlockTable()
        cpu_block_table_swap_out: BlockTable = BlockTable()

        # Store the rmap of the CC GPU blocks
        gpu_block_rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}
        cpu_block_rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}
        for from_block in block_table:
            if from_block in swap_in_mapping:
                to_block = swap_in_mapping[from_block]
                to_block.ref_count += 1
            else:
                to_block = self.gpu_allocator.allocate(
                    from_block.block_hash, from_block.num_hashed_tokens)
                swap_in_mapping[from_block] = to_block
                to_block.computed = from_block.computed
                if to_block.is_evicted and to_block.prev_computed:
                    gpu_block_table_swap_out.append(to_block)
                n_rmap_gpu_block = self.gpu_allocator.n_rmap(to_block)
                if n_rmap_gpu_block > 0:
                    # Evict a CC block
                    assert to_block.ref_count == n_rmap_gpu_block + 1
                    # Verify the rmap will be replaced
                    assert to_block.is_evicted and to_block.prev_computed
                    # Preserve its rmap which that be replaced in swap out
                    rmap = self.gpu_allocator.get_rmap(to_block)
                    rmap = rmap.copy() if rmap is not None else None
                    if rmap is not None:
                        self.gpu_allocator.remove_rmap_all(to_block)
                        gpu_block_rmap.setdefault(to_block, set()).update(rmap)
                n_rmap_cpu_block = self.cpu_allocator.n_rmap(from_block)
                if n_rmap_cpu_block:
                    # CC only requests. Verify the ref count
                    assert from_block.ref_count == n_rmap_cpu_block + 1
                    assert from_block.computed
                    self._update_rmap(from_block, to_block)

            gpu_block_table.append(to_block)
            # For delayed free. Only allow sequence level swapping to DRAM now.
            cpu_block_table.append(from_block)

        # Handle swap out from GPU to CPU / Disk. Same as prefill
        if len(gpu_block_table_swap_out) > 0:
            if self.enable_disk_swap:
                self._handle_swap_out(
                    gpu_block_table_swap_out,
                    enable_swap_out_to_cpu=True,
                    enable_swap_out_to_disk=True,
                    block_rmap_from=gpu_block_rmap,
                    block_rmap_to=cpu_block_rmap,
                    block_table_swap_out_to_cpu=cpu_block_table_swap_out,
                    block_table_swap_out_to_disk=disk_block_table,
                    swap_out_mapping_to_cpu=swap_out_mapping_to_cpu,
                    swap_out_mapping_to_disk=swap_out_mapping_to_disk)
            else:
                self._handle_swap_out(
                    gpu_block_table_swap_out,
                    enable_swap_out_to_cpu=True,
                    enable_swap_out_to_disk=False,
                    block_rmap_from=gpu_block_rmap,
                    block_rmap_to=None,
                    block_table_swap_out_to_cpu=cpu_block_table_swap_out,
                    block_table_swap_out_to_disk=None,
                    swap_out_mapping_to_cpu=swap_out_mapping_to_cpu,
                    swap_out_mapping_to_disk=None)
        cpu_block_table.extend(cpu_block_table_swap_out)

        # CPU Block Table can contain blocks which we only takes a
        # reference in case they are freed or used by other sequences
        cpu_evict_block_table = BlockTable(
            [cpu_block for _, cpu_block in swap_out_mapping_to_cpu.items()])

        if enable_swap_out_to_disk and len(cpu_evict_block_table) > 0:
            self._handle_swap_out(
                cpu_evict_block_table,
                enable_swap_out_to_cpu=False,
                enable_swap_out_to_disk=True,
                block_rmap_from=cpu_block_rmap,
                block_rmap_to=None,
                block_table_swap_out_to_cpu=None,
                block_table_swap_out_to_disk=disk_block_table,
                swap_out_mapping_to_cpu=None,
                swap_out_mapping_to_disk=swap_out_mapping_to_disk)
        return gpu_block_table, cpu_block_table, disk_block_table

    def _swap_out_block_table(
        self,
        block_table: BlockTable,
        swap_out_mapping_to_cpu: Dict[PhysicalTokenBlock, PhysicalTokenBlock],
        swap_out_mapping_to_disk: Optional[Dict[PhysicalTokenBlock,
                                                PhysicalTokenBlock]] = None,
    ) -> Tuple[BlockTable, Dict[int, BlockTable]]:
        """
        We specialize the function to make the implementation more clear
        """
        enable_swap_out_to_disk: bool = swap_out_mapping_to_disk is not None

        # Right now, the sequence level swap to disk is not supported yet
        # We only evict the free block or CC blocks to disk
        cpu_block_table: BlockTable = BlockTable()
        disk_block_table: Dict[int, BlockTable] = {}

        # Can be extended to disk next
        cpu_block_table_swap_out: BlockTable = BlockTable()

        # Store the rmap of the CC GPU blocks
        cpu_block_rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}
        for from_block in block_table:
            if from_block in swap_out_mapping_to_cpu:
                to_block = swap_out_mapping_to_cpu[from_block]
                to_block.ref_count += 1
            else:
                to_block = self.cpu_allocator.allocate(
                    from_block.block_hash, from_block.num_hashed_tokens)
                swap_out_mapping_to_cpu[from_block] = to_block
                to_block.computed = from_block.computed
                if to_block.is_evicted and to_block.prev_computed:
                    cpu_block_table_swap_out.append(to_block)
                n_rmap_cpu_block = self.cpu_allocator.n_rmap(to_block)
                if n_rmap_cpu_block > 0:
                    # Evict a CC block
                    assert to_block.ref_count == n_rmap_cpu_block + 1
                    # Verify the rmap will be replaced
                    assert to_block.is_evicted and to_block.prev_computed
                    # Preserve its rmap which that be replaced in swap out
                    rmap = self.cpu_allocator.get_rmap(to_block)
                    rmap = rmap.copy() if rmap is not None else None
                    if rmap is not None:
                        self.cpu_allocator.remove_rmap_all(to_block)
                        cpu_block_rmap.setdefault(to_block, set()).update(rmap)
                n_rmap_gpu_block = self.gpu_allocator.n_rmap(from_block)
                if n_rmap_gpu_block:
                    # CC only requests. Verify the ref count
                    assert from_block.ref_count == n_rmap_gpu_block + 1
                    assert from_block.computed
                    self._update_rmap(from_block, to_block)

            cpu_block_table.append(to_block)
            # In the current scheduler, nothing will be scheduled after swap out
            # So we can safely free them here
            self.gpu_allocator.free(from_block)

        # Handle swap out from GPU to CPU / Disk. Same as prefill
        if enable_swap_out_to_disk and len(cpu_block_table_swap_out) > 0:
            self._handle_swap_out(
                cpu_block_table_swap_out,
                enable_swap_out_to_cpu=False,
                enable_swap_out_to_disk=True,
                block_rmap_from=cpu_block_rmap,
                block_rmap_to=None,
                block_table_swap_out_to_cpu=None,
                block_table_swap_out_to_disk=disk_block_table,
                swap_out_mapping_to_cpu=None,
                swap_out_mapping_to_disk=swap_out_mapping_to_disk)
        return cpu_block_table, disk_block_table

    def _swap_block_table(
        self,
        block_table: BlockTable,
        src_allocator: BlockAllocatorBase,
        dest_allocator: BlockAllocatorBase,
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock],
    ) -> BlockTable:
        new_block_table: BlockTable = BlockTable()

        for from_block in block_table:
            if from_block in mapping:
                to_block = mapping[from_block]
                to_block.ref_count += 1
            else:
                to_block = dest_allocator.allocate(
                    from_block.block_hash, from_block.num_hashed_tokens)
                mapping[from_block] = to_block
                # Update the computed states
                to_block.computed = from_block.computed
            new_block_table.append(to_block)
            # Free the source block swapped in to destination.
            src_allocator.free(from_block)

        return new_block_table

    def swap_in(
        self, seq_group: SequenceGroup
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[
            int, int, int, int]], List[Tuple[int, int, int, int]]]:
        request_id = seq_group.request_id
        # Avoid the case where prefill swaps blocks to
        # freed swap in chunked prefill
        enable_evict = ((not seq_group.is_encoder_decoder())
                        and self.enable_caching and self.enable_memory_tiering)
        # CPU block -> GPU block.
        # dict is efficient in lookup `if cpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        swap_out_mapping_to_cpu: Dict[PhysicalTokenBlock,
                                      PhysicalTokenBlock] = {}
        swap_out_mapping_to_disk: Dict[PhysicalTokenBlock,
                                       PhysicalTokenBlock] = {}

        # Right now the sequence-level swap does not support disk
        blocks_to_swap_in_from_disk: List[Tuple[int, int, int, int]] = []
        blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []

        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if enable_evict:
                self.block_tables[seq.seq_id],  \
                self.evict_block_tables[seq.seq_id], disk_block_table= \
                    self._swap_in_block_table(
                        self.block_tables[seq.seq_id],
                        mapping,
                        swap_out_mapping_to_cpu,
                        swap_out_mapping_to_disk
                        if self.enable_disk_swap else None)
                if self.swap_manager is not None and disk_block_table != {}:
                    self.swap_manager.update_block_tables(
                        seq.seq_id, disk_block_table)
            else:
                self.block_tables[seq.seq_id] = \
                    self._swap_block_table(self.block_tables[seq.seq_id],
                                           self.cpu_allocator,
                                           self.gpu_allocator,
                                           mapping)

        # NOTE: Encoder decoder model does not support prefix caching now
        # so we don't pass enable_evict
        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.cpu_allocator,
                                       self.gpu_allocator,
                                       mapping)
        if self.enable_disk_swap:
            blocks_to_swap_out_to_disk.extend([
                (block.block_number, disk_block.block_number, block.device,
                 disk_block.device_id)
                for block, disk_block in swap_out_mapping_to_disk.items()
            ])

        return ([(cpu_block.block_number, gpu_block.block_number)
                 for cpu_block, gpu_block in mapping.items()],
                [(gpu_block.block_number, cpu_block.block_number)
                 for gpu_block, cpu_block in swap_out_mapping_to_cpu.items()
                 ], blocks_to_swap_in_from_disk, blocks_to_swap_out_to_disk)

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(
        self, seq_group: SequenceGroup
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
        request_id = seq_group.request_id

        # GPU block -> CPU block.
        # dict is efficient in lookup `if gpu_block in mapping`
        # NOTE: swap out frees GPU blocks immediately as nothing will be
        # scheduled after swapped out
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        swap_out_mapping_to_disk: Dict[PhysicalTokenBlock,
                                       PhysicalTokenBlock] = {}
        enable_evict = ((not seq_group.is_encoder_decoder())
                        and self.enable_caching and self.enable_memory_tiering)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            if enable_evict:
                self.block_tables[seq.seq_id], disk_block_table = \
                    self._swap_out_block_table(
                        self.block_tables[seq.seq_id],
                        mapping,
                        swap_out_mapping_to_disk
                        if self.enable_disk_swap else None)
                if self.swap_manager is not None and disk_block_table != {}:
                    self.swap_manager.update_block_tables(
                        seq.seq_id, disk_block_table)
            else:
                self.block_tables[seq.seq_id] = \
                    self._swap_block_table(self.block_tables[seq.seq_id],
                                        self.gpu_allocator,
                                        self.cpu_allocator,
                                        mapping)

        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.gpu_allocator,
                                       self.cpu_allocator,
                                       mapping)
        blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
        if swap_out_mapping_to_disk != {}:
            blocks_to_swap_out_to_disk.extend([
                (block.block_number, disk_block.block_number, block.device,
                 disk_block.device_id)
                for block, disk_block in swap_out_mapping_to_disk.items()
            ])

        return [(gpu_block.block_number, cpu_block.block_number)
                for gpu_block, cpu_block in mapping.items()
                ], blocks_to_swap_out_to_disk

    def _free_block_table(self, block_table: BlockTable) -> None:
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in blocks_to_free:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def _move_block_table_swappable(self, block_table: BlockTable,
                                    seq: Sequence) -> None:
        # Only move the block to swappable without freeing
        # So we need an rmap. we will use rmap walk to change the blocks in
        # the block table
        # TODO: Check the correctness of sliding window as we don't
        # guarantee its correctness
        if self.block_sliding_window:
            raise NotImplementedError("Context Caching with sliding window "
                                      "is not tested yet!")
        # Add blocks to rmap since they still take the ref count and we only
        # keep computed blocks and free the non-computed ones as they won't
        # have hash so these blocks will be anyway discarded in swap out.
        # It is better to free them right now so we don't handle additional
        # case in swap out

        # First free the non-computed ones since they should be at the back
        n_poped = 0
        while len(block_table) > 0 and not block_table[-1].computed:
            # Remove from block table since they don't have rmap
            block = block_table.pop()
            n_poped += 1
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            elif block.device == Device.CPU:
                self.cpu_allocator.free(block)
            else:
                assert self.swap_manager is not None
                raise ValueError("Move block table movable should not be "
                                 "invoked on disk blocks!")

        blocks_to_move = (
            block_table[-(min(self.block_sliding_window - n_poped, 0)):]
            if self.block_sliding_window is not None else block_table)

        # The rest should be computed
        for block_id, block in enumerate(blocks_to_move):
            if self.block_sliding_window:
                block_id += len(block_table) - len(blocks_to_move)
            assert block.computed
            if block.device == Device.GPU:
                # Add rmap
                self.gpu_allocator.add_rmap(block, seq.seq_id, block_id)
                self.gpu_allocator.move_swappable(block)
            elif block.device == Device.CPU:
                self.cpu_allocator.add_rmap(block, seq.seq_id, block_id)
                self.cpu_allocator.move_swappable(block)
                pass
            elif block.device == Device.SWAP:
                # Actually this path should be invoked but anyway
                # we need to keep the rmap when the block is evicted or
                # promoted
                self.swap_manager.add_rmap(block, seq, block_id)
                self.swap_manager.move_swappable(block)
            else:
                raise ValueError("Unknown device!")

    #NOTE: This function can be very slow
    def adjust_swap(
        self,
        new_swap_in: List[Tuple[int, int]],
        new_swap_out: List[Tuple[int, int]],
        old_swap_in: List[Tuple[int, int]],
        old_swap_out: List[Tuple[int, int]],
        old_swap_copy: List[Tuple[int, int]],
    ) -> None:
        if ((not (self.enable_caching and self.enable_memory_tiering))
                or not self.enable_verify):
            old_swap_in.extend(new_swap_in)
            old_swap_out.extend(new_swap_out)
            return
        # Perform in-place change to the old lists
        # Verify first:
        new_swap_in_cpu_blocks = [t[0] for t in new_swap_in]
        new_swap_in_gpu_blocks = [t[1] for t in new_swap_in]
        new_swap_out_cpu_blocks = [t[1] for t in new_swap_out]
        old_swap_in_set = {t[0] for t in old_swap_in}
        old_swap_out_set = {t[1] for t in old_swap_out}
        old_copy_to_set = {t[1] for t in old_swap_copy}

        # Old swap and new swap is joint when they contain the same block
        cpu_only_intersect = old_swap_in_set.intersection(
            set(new_swap_in_cpu_blocks))
        tuple_intersect = set(old_swap_in).intersection(set(new_swap_in))
        assert cpu_only_intersect == {t[0] for t in tuple_intersect}
        # cpu blocks to swap in cannot overlap with any other cpu blocks
        assert old_swap_in_set.isdisjoint(set(new_swap_out_cpu_blocks))

        # gpu blocks copied to should to overlap with those being swapped in
        assert old_copy_to_set.isdisjoint(set(new_swap_in_gpu_blocks))

        # Warning for swap_out and swap_out overlap
        overlap_set = {t[1] for t in new_swap_out if t[1] in old_swap_out_set}
        if len(overlap_set) != 0:
            logger.warning("new swap out overlaps with the old swap out")
            # Remove the old ones and add the new ones:
            old_swap_out = [t for t in old_swap_out if t[1] not in overlap_set]
        old_swap_out += new_swap_out

        # Warning for swap_out and swap_in overlap and change swap_in to copy
        overlap_tuple = [
            t for t in old_swap_out if t[1] in set(new_swap_in_cpu_blocks)
        ]
        overlap_set = {t[1] for t in overlap_tuple}
        if len(overlap_set) != 0:
            logger.warning("new swap in overlaps with the old swap out")
            old_swap_in += [t for t in new_swap_in if t[0] not in overlap_set]
            # Now get the matching tuple: swap_out + swap_in = copy
            overlap_dict: Dict[int, List[Tuple[int, int]]] = {}
            for t in overlap_tuple:
                assert t[1] not in overlap_dict
                overlap_dict[t[1]] = [t]
            overlap_tuple = [t for t in new_swap_in if t[0] in overlap_set]
            for t in overlap_tuple:
                assert t[0] in overlap_dict
                overlap_dict[t[0]].append(t)
            for _, v in overlap_dict.items():
                old_swap_copy.append((v[0][0], v[1][1]))
        else:
            old_swap_in += new_swap_in

        return

    def free_evict(self, seq: Sequence) -> None:
        """
        Late free of the blocks allocated for swap, e.g DRAM and 
        Disk blocks in swap out.
        These blocks are not actually used but we need to give 
        them a unified access time for eviction.
        """
        if seq.seq_id not in self.evict_block_tables:
            return
        access_time = time.time()
        evict_block_table = self.evict_block_tables[seq.seq_id]
        for block in evict_block_table:
            block.last_accessed = access_time
        self._free_block_table(evict_block_table)
        del self.evict_block_tables[seq.seq_id]

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def free_context_cache(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        # Need to remove rmap
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block_id, block in enumerate(blocks_to_free):
            if self.block_sliding_window:
                block_id += len(block_table) - len(blocks_to_free)
            # Context Caching block can scatter anywhere in the system
            if block.device == Device.GPU:
                self.gpu_allocator.remove_rmap(block, seq.seq_id, block_id)
                self.gpu_allocator.free(block)
            elif block.device == Device.CPU:
                self.cpu_allocator.remove_rmap(block, seq.seq_id, block_id)
                self.cpu_allocator.free(block)
            elif block.device == Device.SWAP:
                self.swap_manager.remove_rmap(block, seq.seq_id, block_id)
                self.swap_manager.free(block)
            else:
                raise ValueError("Unknown devices")

        del self.block_tables[seq.seq_id]

    def make_swappable(self, seq: Sequence) -> None:
        """
        This function move the blocks of a sequence to swappable 
        without actually free them 
        """
        assert seq.seq_id in self.block_tables
        block_table = self.block_tables[seq.seq_id]
        self._move_block_table_swappable(block_table, seq)
        # Keep the block table as the ref count still exits
        return

    def free_cross(self, seq_group: SequenceGroup) -> None:
        if seq_group.request_id not in self.cross_block_tables:
            # Already freed or hasn't ben scheduled yet.
            return
        block_table = self.cross_block_tables[seq_group.request_id]
        self._free_block_table(block_table)
        del self.cross_block_tables[seq_group.request_id]

    def reset(self) -> None:
        # Free decoder block tables
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()
        # Free cross-attention block tables
        for block_table in self.cross_block_tables.values():
            self._free_block_table(block_table)
        self.cross_block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        return self.block_tables[seq.seq_id].ids()

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        block_table = self.cross_block_tables[seq_group.request_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.last_accessed = access_time

    def _access_all_cpu_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching and self.enable_memory_tiering:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.evict_block_tables[seq.seq_id]
            for block in block_table:
                print("Set access time: ", block.block_number)
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Sequence, token_chunk_size: int):
        if seq.seq_id not in self.block_tables:
            return

        # When chunked prefill is enabled, the computed full blocks
        # should be calculated based on the number of computed tokens.
        max_computed_tokens = (seq.data.get_num_computed_tokens() +
                               token_chunk_size)
        computed_full_blocks = max_computed_tokens // self.block_size

        block_table = self.block_tables[seq.seq_id]
        if computed_full_blocks == 0:
            return
        for i in reversed(range(computed_full_blocks)):
            #if block_table[i].computed:
            #    break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Sequence) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [
            b.block_number
            for b in takewhile(lambda b: b.computed, block_table[:-1])
        ]

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        if self.enable_caching:
            for seq in seq_group.get_seqs():
                self.compute_full_blocks_in_seq(seq, token_chunk_size)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        if device == Device.GPU:
            return self.gpu_allocator.get_prefix_cache_hit_rate()
        if device == Device.CPU:
            return self.cpu_allocator.get_prefix_cache_hit_rate()
        raise ValueError(f"Invalid device: {device}")
