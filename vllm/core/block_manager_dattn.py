'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)

  Adopted from https://github.com/vllm-project/vllm/pull/6102/commits
'''
from collections import deque
from typing import Dict, List, Optional, Tuple

from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device, Counter
from collections import deque
import sys

logger = init_logger(__name__)

class CacheAllocator:
    def __init__(self, num_caches: int):
        self.num_caches = num_caches
        self.kv_caches = deque(range(num_caches))

    def allocate(self) -> int:
        cache_id = self.kv_caches.popleft()
        return cache_id

    def free(self, cache_id: int):
        self.kv_caches.append(cache_id)

    def get_total_caches(self):
        return len(self.kv_caches)
        
class BlockSpaceManagerDAttn(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None, # Not supported
        enable_caching: bool = False, # Not supported
        num_caches: int = 0,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        
        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks
        
        self.num_caches = num_caches  # == self.scheduler_config.max_num_seqs

        # use to alloc cache buffer id for seq
        self.gpu_allocator = CacheAllocator(num_caches)

        # Watermark indicates that the least amount of blocks should be free. 
        self.watermark = watermark
        assert watermark >= 0.0

        self.start_free_blocks = 100
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # Mapping from cache buffer ID to the number of allocated blocks.
        self.allocated_blocks: Dict[int, int] = {} # Maintains the state of every used kv_cache 
        self.free_kv_caches: Dict[int, int] = {}
        
        self.cached_free_blocks: int = 0
        self.to_allocate_blocks: Dict[int, int] = {} # Temporary for each step
        self.step_index = 0
        
    def _predict_gen_blocks(self, seq: Sequence) -> int:
        # this function is used to predict the generated content length,
        # at least we will add one more block there.
        return 1

    def _get_seq_num_required_blocks(self, seq: Sequence) -> int:
        return 0 if seq is None else seq.predict_n_blocks

    # This will be invoked in the prefill phase
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # get_seqs will collect a list of sequence with status equalling to SequenceStatus.WAITING
        # then we will get the first sequence in this group 
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        self_num_required_blocks = self._get_seq_num_required_blocks(seq)
        cross_num_required_blocks = self._get_seq_num_required_blocks(
            seq_group.get_encoder_seq())
        
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks + 1
        
        #if seq.cache_id == -1:
        #    print(f"seq_id: {seq.seq_id} while cache_id:{seq.cache_id}", file=sys.stderr)
        # If the sequence is not allocated yet, its cache_id must be -1.
        #assert seq.cache_id != -1


        num_free_gpu_blocks = self.num_free_gpu_blocks + \
            self.cached_free_blocks

        # Ensure that one request should not use more than 90% or 99% of memory
        # This can avoid frequent cache eviction 
        if (self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            # Make sure that we are not holding more than schedule_config.max_num_seqs
            if self.gpu_allocator.get_total_caches() > 0 or len(self.free_kv_caches) > 0:
                return AllocStatus.OK
            else:
                #print(f"req:{seq.seq_id}, AllocStatus.LATER1111", file=sys.stderr)
                return AllocStatus.LATER
        else:
            #print(f"step-{self.step_index}, req:{seq.seq_id}, self.num_free_gpu_blocks-{self.num_free_gpu_blocks}, self.cached_free_blocks-{self.cached_free_blocks}, self AllocStatus.LATER2222. num_free_gpu_blocks-{num_free_gpu_blocks},num_required_blocks-{num_required_blocks}, self.watermark_blocks-{self.watermark_blocks} ", file=sys.stderr)
            return AllocStatus.LATER

    # This function is only invoked by _allocate_and_set_running (invoked by _schedule_prefills)
    # That is, it is allocated when admitting a new request in prefill phase.
    def allocate(self, seq_group: SequenceGroup) -> None:
        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        need_blocks = self._get_seq_num_required_blocks(seq)
        
        cache_id, to_allocate_num, allocated_num = self._allocate_cache(need_blocks)
        seq.cache_id = cache_id
        seq.data.cache_id = cache_id
        self.allocated_blocks[cache_id] = allocated_num
        if to_allocate_num > 0: 
            self.to_allocate_blocks[cache_id] = allocated_num 
        
    def _allocate_cache(self, need_blocks: int) -> Tuple[int, int]:
        to_allocate_num = need_blocks
        cache_id = -1
        allocated_num = need_blocks

        # We prefer to reuse the free_kv_caches at first. 
        if self.cached_free_blocks > 0:
            # Make it block_diff a big number for the better comparison
            block_diff = need_blocks*1000
             
            # find one kv_cache with the smallest difference on the number of blocks
            for id, num_blocks in self.free_kv_caches.items():
                diff = abs(num_blocks - need_blocks)
                # kv_cache : cache_id, blocks 
                if diff < block_diff:
                    cache_id = id
                    block_diff = diff
                    allocated_num = num_blocks

                    if diff == 0:
                        break 
            
            self.cached_free_blocks -= allocated_num

            if allocated_num < need_blocks:
                to_allocate_num = need_blocks - allocated_num
                
                # update the allocated number
                allocated_num = need_blocks
            else:
                to_allocate_num = 0 

            if cache_id == -1: 
                print(f"self.free_kv_caches:{len(self.free_kv_caches)}")
                for id, num_blocks in self.free_kv_caches.items():
                    diff = abs(num_blocks - need_blocks)

                    print(f"id-{id}, diff-{diff}, num_blocks:{num_blocks}, need_blocks:{need_blocks}") 
            # Remove this item from the free_kv_caches
            del self.free_kv_caches[cache_id]
        else:
            assert self.num_free_gpu_blocks >= need_blocks
            cache_id = self.gpu_allocator.allocate()
            
            #print(f"_allocate_buffer new, need_blocks:{need_blocks}, cache_id:{cache_id}", file=sys.stderr)
            self.num_free_gpu_blocks -= need_blocks

        return cache_id, to_allocate_num, allocated_num

    # Invoked by _schedule_running in running phase.  
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)

        return num_seqs < self.num_free_gpu_blocks + self.cached_free_blocks

    # FIXME: there is no handling on num_lookahead_slots, which should be handled.  
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        cache_id = seq.cache_id

        # If the sequence is allocated, its cache_id must >= 0.
        assert cache_id >= 0
        
        logical_blocks_num = seq.predict_n_blocks
        allocated_num = self.allocated_blocks[cache_id]
        
        # If we need to allocate a new physical block
        if allocated_num < logical_blocks_num:
            # Currently this code only supports adding one physical block
            assert allocated_num == logical_blocks_num - 1

            self.num_free_gpu_blocks -= logical_blocks_num - allocated_num 
            self.allocated_blocks[cache_id] = logical_blocks_num
            self.to_allocate_blocks[cache_id] = logical_blocks_num 
            return []

        else:
            # the last block is not full, no need to allocate a new block
            return []

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerDAttn now.")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerDAttn now.")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-in is not supported in BlockSpaceManagerDAttn now.")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerDAttn now.")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap-out is not supported in BlockSpaceManagerDAttn now.")

    """
    Free a sequence. We will append the seq to free_kv_caches. 
    Initially, we did this inside the memory management library. Maybe we should do it here as well. 
    """
    def free(self, seq: Sequence) -> None:
        # Here, we just append free seq to free_kv_caches.
        cache_id = seq.cache_id

        # If no blocks are allocated in the sequence, then this sequence may be deallocated. 
        if cache_id in self.free_kv_caches:
            # Already freed yet, no need to do anything.
            return

        # Get free_blocks of this sequence
        free_blocks = self.allocated_blocks[cache_id]
        self.free_kv_caches[cache_id] = free_blocks
        self.cached_free_blocks += free_blocks
        #print(f"Adding to free_kv_caches: cache_id-{seq.cache_id}, free_blocks:{free_blocks}, cached_free_blocks:{self.cached_free_blocks}", file=sys.stderr)

    def reset(self) -> None:
        # Free decoder block tables
        self.allocated_blocks.clear()
        self.num_free_gpu_blocks = self.num_total_gpu_blocks
        self.num_free_cpu_blocks = self.num_total_cpu_blocks
        
        self.free_kv_caches = {}
        self.to_allocate_blocks = {}

    def get_block_table(self, seq: Sequence) -> List[int]:
        # logger.warning("block table is not used in BlockSpaceManagerDAttn now.")
        return []

    def get_num_free_gpu_blocks(self) -> int:
        return self.num_free_gpu_blocks

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_free_cpu_blocks

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        # logger.warning("Access all blocks in seq is not supported in BlockSpaceManagerDAttn now.")
        pass

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        # logger.warning("Common computed block ids is not supported in BlockSpaceManagerDAttn now.")
        return None  # type: ignore

    def mark_blocks_as_computed(self, seq_group: SequenceGroup, token_chunk_size: int) -> None:
        # logger.warning("Mark blocks as computed is not supported in BlockSpaceManagerDAttn now.")
        pass
    
    def get_allocated_block_count(self, seq_id: int) -> int:
        #print(f"get_allocated_block_count: seq_id: {seq_id}, length:{len(self.allocated_blocks)}")
        return self.allocated_blocks[seq_id]

    def step(self) -> Tuple[Dict[int, int], List[int]]:
        self.step_index += 1

        to_allocate_blocks = self.to_allocate_blocks

        #print(f"self.free_kv_caches has length:{len(self.free_kv_caches)}, cached_free_blocks:{self.cached_free_blocks}, total_available:{self.num_free_gpu_blocks}", file=sys.stderr)
        
        to_free_kv_caches = []

        # Check whether there is a need to free kv caches
        if self.free_kv_caches != None:
            """
            if self.cached_free_blocks > self.start_free_blocks or self.cached_free_blocks > self.num_free_gpu_blocks or self.num_free_gpu_blocks < self.watermark_blocks: 
                to_free_blocks = max(self.start_free_blocks, self.cached_free_blocks-self.num_free_gpu_blocks) 
                if self.num_free_gpu_blocks < self.watermark_blocks:
                    to_free_blocks = self.cached_free_blocks

                for cache_id, num_blocks in self.free_kv_caches.items():
                    to_free_kv_caches.append(cache_id)
                    to_free_blocks -= num_blocks 
                    if to_free_blocks < 0:
                        break
            """
            for cache_id, num_blocks in self.free_kv_caches.items():
                to_free_kv_caches.append(cache_id)


        to_free_blocks = 0
        # Removing all to_free_kv_caches from self.free_kv_caches
        for cache_id in to_free_kv_caches:
            num_blocks = self.free_kv_caches[cache_id]
            self.gpu_allocator.free(cache_id)
            self.cached_free_blocks -= num_blocks
            self.num_free_gpu_blocks += num_blocks
            to_free_blocks += num_blocks
            #print(f"free cache_id:{cache_id}")
            self.free_kv_caches.pop(cache_id)

        # step() is invoked once after _schedule() inside Scheduler::schedule(). It is invoked once for every decode or prefill
        # We actually uses self.free_kv_caches and self.to_allocate_blocks to track all requests 
        # checked by the whole _schedule(). This is a hacky solution but may work correctly.    
        self.to_allocate_blocks = {}

        # we can invoke the reset here
        return to_allocate_blocks, to_free_kv_caches

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return 0