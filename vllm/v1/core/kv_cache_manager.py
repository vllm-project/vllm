# SPDX-License-Identifier: Apache-2.0
import itertools
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_allocator import HybridMemoryAllocator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        num_preallocate_tokens: int = 64,
        log_stats: bool = False,
    ) -> None:
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        # FIXME: make prefix cache stats conditional on log_stats
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats()

        self.num_preallocate_tokens = num_preallocate_tokens
        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: dict[str, list[list[KVCacheBlock]]] = {}

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_blocks: dict[str, list[int]] = {}

        self.allocator = HybridMemoryAllocator()

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats.
        """
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[list[dict[int, KVCacheBlock]], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], 0

        block_hashes = self.allocator.get_block_hashes(request,
                                                       self.caching_hash_fn)
        self.prefix_cache_stats.requests += 1
        # If the request requires prompt logprobs, we skip prefix caching.
        if request.sampling_params.prompt_logprobs is not None:
            return [], 0

        num_tokens = request.num_tokens
        computed_blocks, num_computed_tokens = (
            self.allocator.find_longest_cache_hit(block_hashes, num_tokens))

        self.prefix_cache_stats.queries += num_tokens
        self.prefix_cache_stats.hits += num_computed_tokens
        return computed_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_input_tokens: int,
        num_draft_tokens: int = 0,
        new_computed_tokens: int = 0,
        new_computed_blocks: Optional[list[list[KVCacheBlock]]] = None,
        num_lookahead_tokens: int = 0,
    ) -> Optional[list[list[KVCacheBlock]]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.

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
        assert num_input_tokens + num_draft_tokens > 0

        if new_computed_blocks is None:
            new_computed_blocks = []
            assert new_computed_tokens == 0
        else:
            assert new_computed_tokens > 0

        req_blocks = self.req_to_blocks[request.request_id]

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        removed_blocks = self.allocator.remove_skipped_blocks(
            req_blocks, request.num_computed_tokens)
        self.block_pool.free_blocks(removed_blocks)

        num_computed_tokens = request.num_computed_tokens + new_computed_tokens
        num_preallocate_tokens = max(self.num_preallocate_tokens,
                                     num_lookahead_tokens)
        # Should not exceed the maximum number of blocks per request.
        # This is especially because the block table has the shape
        # [..., max_num_blocks_per_req].
        total_num_tokens = min(
            num_computed_tokens + num_input_tokens + num_draft_tokens +
            num_preallocate_tokens, self.max_model_len)

        new_blocks = self.allocator.allocate_blocks(
            total_num_tokens,
            num_computed_tokens,
            req_blocks,
            new_computed_blocks,
        )
        if new_blocks is None:
            # Cannot allocate new blocks.
            return None

        # Add the new computed blocks and new blocks to the request.
        # FIXME
        req_blocks.extend(new_computed_blocks)
        req_blocks.extend(new_blocks)
        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_blocks`
        # for a running request.
        num_cached_blocks = self.num_cached_blocks.get(
            request.request_id,
            [len(blocks) for blocks in new_computed_blocks],
        )
        # NOTE(woosuk): Since draft tokens can be rejected, we should not cache
        # any blocks including draft tokens.
        num_cached_blocks = self.allocator.cache_blocks(
            request,
            req_blocks,
            num_computed_tokens,
            num_input_tokens,  # No draft tokens or lookahead tokens
            num_cached_blocks,
            self.caching_hash_fn,
        )
        self.num_cached_blocks[request.request_id] = num_cached_blocks
        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to None in case a request is freed (aborted) before alloc.
        self.num_cached_blocks.pop(request.request_id, None)
        blocks = self.req_to_blocks.pop(request.request_id, None)
        if blocks is None:
            return

        if self.enable_caching:
            ordered_blocks = self.allocator.sort_by_eviction_order(blocks)
        else:
            # When caching is disabled, free the blocks in any order.
            ordered_blocks = itertools.chain.from_iterable(blocks)
        self.block_pool.free_blocks(ordered_blocks)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.block_pool.reset_prefix_cache():
            self.prefix_cache_stats.reset = True
            return True
        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
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
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.allocator.req_to_block_hashes.pop(request.request_id, None)
