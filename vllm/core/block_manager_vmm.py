import math
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from collections import deque


logger = init_logger(__name__)

class CacheBufferAllocator:
    def __init__(self, num_cache_buffers: int):
        self.num_cache_buffers = num_cache_buffers
        self.free_buffers = deque(range(num_cache_buffers))

    def allocate(self) -> int:
        buffer_id = self.free_buffers.popleft()
        return buffer_id

    def free(self, buffer_id: int):
        self.free_buffers.append(buffer_id)

    def reset(self):
        self.free_buffers = deque(range(self.num_cache_buffers))
    
    def get_num_free_buffers(self):
        return len(self.free_buffers)
    
    def get_num_total_buffers(self):
        return self.num_cache_buffers
        
        
class BlockSpaceManagerVMM(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        num_cache_buffers: int = 0,
    ) -> None:
    
        if enable_caching or (sliding_window is not None):
            raise NotImplementedError("Prefix Caching or Sliding window is not supported in VMM now.")
        
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        
        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks
        
        self.num_cache_buffers = num_cache_buffers  # == self.scheduler_config.max_num_seqs
        
        self.gpu_allocator = CacheBufferAllocator(num_cache_buffers)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # Mapping from cache buffer ID to the number of allocated blocks.
        self.allocated_block_counts: Dict[int, int] = {}

    def _get_seq_num_required_blocks(self, seq: Sequence) -> int:
        return 0 if seq is None else seq.n_blocks

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = self._get_seq_num_required_blocks(seq)
        
        # If the sequence is not allocated yet, its cache_buffer_id must be -1.
        assert seq.cache_buffer_id == -1
    

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if self.num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            if self.gpu_allocator.get_num_free_buffers() > 0:
                return AllocStatus.OK
            else:
                return AllocStatus.LATER
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        if is_encoder_decoder:
            raise NotImplementedError("Encoder-decoder is not supported in VMM now.")
        
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        cache_buffer_id = self.gpu_allocator.allocate()
        seq.cache_buffer_id = cache_buffer_id
        seq.data.cache_buffer_id = seq.cache_buffer_id
        
        need_blocks_num = self._get_seq_num_required_blocks(seq)
        
        # predict generate content length and pre allocate the blocks
        # need_blocks_num += self.predict_gen_len(seq)
        
        self.num_free_gpu_blocks -= need_blocks_num
        
        # Assign the self-attention block tables for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.allocated_block_counts[seq.seq_id] = need_blocks_num
    
    def predict_gen_len(self, seq: Sequence) -> int:
        # TODO:this function is used to predict the generated content length,
        # which can used to pre allocate the memory handles
        return 1

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerVMM."

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= self.num_free_gpu_blocks

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        
        # If the sequence is allocated, its cache_buffer_id must >= 0.
        assert seq.cache_buffer_id >= 0
        
        logical_blocks_num = seq.n_blocks
        allocate_num = self.allocated_block_counts[seq.seq_id]
        
        # If we need to allocate a new physical block
        if allocate_num < logical_blocks_num:
            # Currently this code only supports adding one physical block
            assert allocate_num == logical_blocks_num - 1

            self.num_free_gpu_blocks -= 1
            self.allocated_block_counts[seq.seq_id] += 1
            return []

        else:
            # the last block is not full, no need to allocate a new block
            return []

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerVMM now.")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerVMM now.")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerVMM now.")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerVMM now.")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerVMM now.")

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.allocated_block_counts:
            # Already freed or haven't been scheduled yet.
            return
        block_num = self.allocated_block_counts[seq.seq_id]
        self.num_free_gpu_blocks += block_num
        del self.allocated_block_counts[seq.seq_id]
        
        self.gpu_allocator.free(seq.cache_buffer_id)
        seq.cache_buffer_id = -1
        seq.data.cache_buffer_id = -1

    def reset(self) -> None:
        # Free decoder block tables
        self.allocated_block_counts.clear()
        self.num_free_gpu_blocks = self.num_total_gpu_blocks
        self.num_free_cpu_blocks = self.num_total_cpu_blocks
        
        self.gpu_allocator.reset()

    def get_block_table(self, seq: Sequence) -> List[int]:
        # logger.warning("block table is not used in BlockSpaceManagerVMM now.")
        return None

    def get_num_free_gpu_blocks(self) -> int:
        return self.num_free_gpu_blocks

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_free_cpu_blocks

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        # logger.warning("Access all blocks in seq is not supported in BlockSpaceManagerVMM now.")
        pass

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        # logger.warning("Common computed block ids is not supported in BlockSpaceManagerVMM now.")
        return None  # type: ignore

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        # logger.warning("Mark blocks as computed is not supported in BlockSpaceManagerVMM now.")
        pass
    
    def get_allocated_block_count(self, seq_id: int):
        return self.allocated_block_counts[seq_id]