# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv, sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.core.specialized_manager import get_specialized_manager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class HybridKVCacheManager:
    """
    The HybridKVCacheManager for models with multiple KV cache types
     (e.g., Gemma-2) and thus multiple kv cache groups (Refer to class 
     `KVCacheConfig` for the meaning of kv cache groups).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        num_preallocate_tokens: int = 64,
        log_stats: bool = False,
    ) -> None:
        # TODO: adjust the name for item in one group, list of items in all
        # groups, and reduced item for all groups.
        self.kv_cache_config = kv_cache_config
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = [
            cdiv(max_model_len, g.kv_cache_spec.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        # FIXME: make prefix cache stats conditional on log_stats
        self.log_stats = log_stats
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
        # the same for all layers, which will result in different
        # preallocated tokens for different layers if their block sizes are
        # different.
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            max(g.kv_cache_spec.block_size
                for g in kv_cache_config.kv_cache_groups))

        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching)

        self.specialized_managers = [
            get_specialized_manager(
                kv_cache_spec=g.kv_cache_spec,
                block_pool=self.block_pool,
            ) for g in kv_cache_config.kv_cache_groups
        ]

        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[
            str, list[list[KVCacheBlock]]] = defaultdict(
                lambda: [[] for _ in range(self.num_kv_cache_groups)])

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[list[BlockHashType]]] = defaultdict(
                lambda: [[] for _ in range(self.num_kv_cache_groups)])

        # {req_id: The number of cached blocks for each kv cache group}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, list[int]] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    usage = KVCacheManager.usage
    make_prefix_cache_stats = KVCacheManager.make_prefix_cache_stats

    def get_computed_blocks(
            self, request: Request) -> tuple[list[list[KVCacheBlock]], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for each kv cache group.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [[] for _ in range(self.num_kv_cache_groups)], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = [
                hash_request_tokens(self.caching_hash_fn,
                                    g.kv_cache_spec.block_size, request, i)
                for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            ]
            self.req_to_block_hashes[request.request_id] = block_hashes

        self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            # TODO: Fix last block problem
            # if len(block_hashes) * self.block_size == request.num_tokens:
            #     # When prompt length is divisible by the block size and all
            #     # blocks are cached, we need to recompute the last token. This
            #     # have to be achieved by re-computing an entire block because
            #     # allocate_slots() assumes num_computed_tokens is always a
            #     # multiple of the block size. To achieve this, remove the last
            #     # block hash from the block_hashes for find_longest_cache_hit
            #     # This limitation can potentially be removed in the future to
            #     # slightly improve the performance.
            #     last_block_hash = block_hashes.pop()
            # else:
            #     last_block_hash = None

            computed_blocks, num_computed_tokens = self.find_longest_cache_hit(
                request.request_id, block_hashes)

            # if last_block_hash is not None:
            #     # Add back the last block hash if it was removed.
            #     block_hashes.append(last_block_hash)

            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)
            return computed_blocks, num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        num_new_computed_tokens: int = 0,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.
            num_new_computed_tokens: The number of new computed tokens in the
                new_computed_blocks.

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

        new_computed_blocks = new_computed_blocks or [
            [] for _ in range(self.num_kv_cache_groups)
        ]

        req_blocks = self.req_to_blocks[request.request_id]

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        removed_blocks = [
            manager.remove_skipped_blocks(req_blocks,
                                          request.num_computed_tokens)
            for manager in self.specialized_managers
        ]
        self._free_blocks(removed_blocks)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits

        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)

        num_new_blocks: list[int] = []
        for i in range(self.num_kv_cache_groups):
            num_required_blocks_i = cdiv(
                num_computed_tokens + num_tokens,
                self.specialized_managers[i].block_size)
            num_new_blocks.append((num_required_blocks_i - len(req_blocks[i]) -
                                   len(new_computed_blocks[i])))
        total_num_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk_one_layer in new_computed_blocks for blk in blk_one_layer
            if blk.ref_cnt == 0)
        if (total_num_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            for blocks in new_computed_blocks:
                self.block_pool.touch(blocks)
            else:
                assert all(len(blks) == 0 for blks in new_computed_blocks), (
                    "Computed blocks should be empty when "
                    "prefix caching is disabled")
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        for i in range(self.num_kv_cache_groups):
            req_blocks[i].extend(new_computed_blocks[i])

        # Start to handle new blocks
        new_blocks: list[list[KVCacheBlock]] = []
        # Truncate the number of pre-allocated blocks to ensure that we can
        # have at least `num_new_blocks` free blocks for each layer.
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.block_pool.get_num_free_blocks() - total_num_new_blocks) //
            len(self.specialized_managers))

        for i in range(self.num_kv_cache_groups):
            if num_new_blocks[i] <= 0:
                # No new block is needed.
                new_blocks.append([])
            else:
                # Get new blocks from the free block pool considering
                # preallocated blocks.
                num_block_to_allocate = min(
                    num_new_blocks[i] + num_preallocate_blocks,
                    # Should not exceed the maximum number of blocks per request
                    # This is especially because the block table has the shape
                    # [..., max_num_blocks_per_req].
                    # TODO(woosuk): Check and reject requests if
                    # num_prompt_tokens + max_tokens > max_model_len.
                    # Don't need self.block_pool.get_num_free_blocks() as in
                    # KVCacheManager because we already considered it when
                    # calculating num_preallocate_blocks
                    self.max_num_blocks_per_req[i] - len(req_blocks[i]),
                )

                assert num_block_to_allocate > 0
                assert num_block_to_allocate <= \
                    self.block_pool.get_num_free_blocks()

                # Concatenate the computed block IDs and the new block IDs.
                new_blocks_this_layer = self.block_pool.get_new_blocks(
                    num_block_to_allocate)
                new_blocks.append(new_blocks_this_layer)
                req_blocks[i].extend(new_blocks_this_layer)

        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(
            request.request_id,
            [len(blocks) for blocks in new_computed_blocks])
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        for i in range(self.num_kv_cache_groups):
            num_full_blocks_after_append = (
                num_computed_tokens + num_tokens - len(request.spec_token_ids)
            ) // self.specialized_managers[i].block_size

            self.block_pool.cache_full_blocks(
                request=request,
                blocks=req_blocks,
                block_hashes=self.req_to_block_hashes[request.request_id],
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks_after_append,
                block_size=self.specialized_managers[i].block_size,
                hash_fn=self.caching_hash_fn,
                kv_cache_group_id=i,
            )
            num_cached_blocks[i] = num_full_blocks_after_append

        self.num_cached_block[request.request_id] = num_cached_blocks
        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, None)
        if blocks is not None:
            # Reverse the blocks so that the tail blocks can have higher
            # eviction priority.
            self._free_blocks([list(reversed(blks)) for blks in blocks])

        self.num_cached_block.pop(request.request_id, None)

    reset_prefix_cache = KVCacheManager.reset_prefix_cache

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
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
        scheduled in the current step.

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
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = []
        for i in range(self.num_kv_cache_groups):
            num_common_blocks_i = 0
            for block in blocks[i]:
                if block.ref_cnt == num_running_requests:
                    num_common_blocks_i += 1
                else:
                    break
            num_common_blocks.append(num_common_blocks_i)
        return num_common_blocks

    free_block_hashes = KVCacheManager.free_block_hashes

    def find_longest_cache_hit(
        self, request_id: int, block_hashes: list[list[BlockHashType]]
    ) -> tuple[list[list[KVCacheBlock]], int]:
        """Find the longest cache hit for each kv cache group.
        TODO: add more notes
        """
        # TODO: accelerate by make full attention the first layer
        # TODO: add note for the two magic number
        num_computed_tokens = [self.max_model_len + 100] * len(
            self.specialized_managers)
        min_computed_tokens = self.max_model_len

        # Use copy to avoid modifying the original block_hashes
        block_hashes = [block_hash.copy() for block_hash in block_hashes]

        while not max(num_computed_tokens) == min_computed_tokens:
            for i, manager in enumerate(self.specialized_managers):
                if num_computed_tokens[i] > min_computed_tokens:
                    del block_hashes[i][:min_computed_tokens //
                                        manager.block_size]
                    computed_blocks_group_i = manager.find_longest_cache_hit(
                        request_id, block_hashes[i], return_const_list=True)

                    num_computed_tokens[i] = len(computed_blocks_group_i) * \
                        manager.block_size
                    min_computed_tokens = min(min_computed_tokens,
                                              num_computed_tokens[i])

        # Get the non-constlist computed blocks
        computed_blocks = [
            manager.find_longest_cache_hit(request_id,
                                           block_hashes[i],
                                           return_const_list=False)
            for i, manager in enumerate(self.specialized_managers)
        ]

        assert all(
            len(block) * manager.block_size == min_computed_tokens for block,
            manager in zip(computed_blocks, self.specialized_managers))

        return computed_blocks, min_computed_tokens

    def _merge_blocks_by_eviction_order(
            self, blocks: list[list[KVCacheBlock]]) -> list[KVCacheBlock]:
        """
        Merge the blocks of different layers to one list. The returned blocks 
        are sorted by eviction order, with the first block having the highest 
        eviction priority.
        Args:
            blocks: the blocks of each virtual layer, ordered by eviction 
            priority.
        Returns:
            A list of KVCacheBlocks sorted by eviction order.
        """

        if self.enable_caching:
            # NOTE (Chen): A simple strategy that interleaves the blocks of
            # each layer. We can investigate more advanced strategies
            # in the future.
            ordered_blocks = []
            max_len = max(len(blocks_one_layer) for blocks_one_layer in blocks)
            for i in range(max_len):
                for blocks_one_layer in blocks:
                    if i < len(blocks_one_layer):
                        ordered_blocks.append(blocks_one_layer[i])
        else:
            ordered_blocks = []
            for blocks_one_layer in blocks:
                ordered_blocks.extend(blocks_one_layer)

        return ordered_blocks

    def _free_blocks(self, blocks: list[list[KVCacheBlock]]) -> None:
        ordered_blocks = self._merge_blocks_by_eviction_order(blocks)
        self.block_pool.free_blocks(ordered_blocks)
