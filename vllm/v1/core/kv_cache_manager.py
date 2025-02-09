# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

from vllm.v1.core.block_pool import BlockPool
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, ReqKVCacheBlocks,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens, intersect_ranges)
from vllm.v1.core.specialized_manager import GroupedManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:
    """
    The KVCacheManager for models with one KV cache type (e.g., Llama) and
    thus one kv cache group (Refer to class `KVCacheConfig` for the meaning of 
    kv cache group).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        # NOTE(Chen): For simplicity, we keep the number of preallocated blocks
        # the same for all kv cache groups, which will result in different
        # preallocated tokens for different groups if their block sizes are
        # different.
        self.num_preallocate_tokens = num_preallocate_tokens
        # NOTE(Chen): For simplicity, we keep the number of preallocated blocks
        # the same for all kv cache groups, which will result in different
        # preallocated tokens for different groups if their block sizes are
        # different.
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            max(g.kv_cache_spec.block_size for g in kv_cache_config.groups))

        self.block_pool = BlockPool(self.num_gpu_blocks, self.enable_caching)

        # Specialized managers for each kv cache group, which handle the
        # different kv cache management logic of different attention layers.
        self.managers = GroupedManager(
            kv_cache_config,
            max_model_len,
            enable_caching,
            self.block_pool,
        )

    @property
    def usage(self) -> float:
        return 1.0 - (self.block_pool.get_num_free_blocks() /
                      self.num_gpu_blocks)

    def get_computed_blocks(self,
                            request: Request) -> Tuple[ReqKVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - The blocks that are computed for the request
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return self.managers.construct(list), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.managers.hash_request_tokens(request)

        prefix_length, computed_blocks = self.managers.get_possible_cached_prefix(
            block_hashes)

        num_computed_tokens = prefix_length[-1].end

        computed_blocks = self.managers.truncate_computed_blocks(
            computed_blocks, num_computed_tokens)
        return computed_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[ReqKVCacheBlocks] = None,
        num_new_computed_tokens: int = 0,
    ) -> Optional[ReqKVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks_all_groups: A list of new computed blocks 
                just hitting the prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or self.managers.construct(
            list)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)

        # We can free blocks that are no longer needed even if we cannot
        # schedule this request due to the limit of free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        blocks_to_free = self.managers.remove_useless_blocks(
            request, num_computed_tokens)
        self.managers.free_blocks(blocks_to_free)

        num_new_blocks = self.managers.get_req_num_new_blocks(
            request, new_computed_blocks, num_computed_tokens, num_tokens)
        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk in self.managers.iter_all(new_computed_blocks)
            if blk.ref_cnt == 0)

        if (total_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Truncate the number of pre-allocated blocks to ensure that we can
        # have at least `num_new_blocks` free blocks for each group.
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.block_pool.get_num_free_blocks() -
             num_evictable_computed_blocks - total_new_blocks) //
            len(self.kv_cache_config.groups))

        new_blocks = self.managers.allocate_slots(request, new_computed_blocks,
                                                  num_new_blocks,
                                                  num_preallocate_blocks,
                                                  num_computed_tokens,
                                                  num_tokens)

        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to None in case a request is freed (aborted) before alloc.
        blocks = self.managers.pop_blocks_of_request(request.request_id)
        if blocks is None:
            # This request is freed before alloc. just return
            return
        else:
            # Reverse the blocks so that the tail blocks can have higher
            # eviction priority.
            self.managers.free_blocks(blocks, need_reverse=True)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """

        return self.block_pool.reset_prefix_cache()

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> List[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step. As of 1/1/2025, the scheduler does not
        allow this case, but it is possible in the future, as we allow more
        flexible scheduling.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            List[int]: The number of common prefix blocks per KV cache group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.managers.get_num_common_prefix_blocks(
            request, num_running_requests)

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.managers.pop_block_hashes_of_request(request.request_id)
