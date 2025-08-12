# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.core.specialized_manager import get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    blocks: list[KVCacheBlock]

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(self.blocks + other.blocks)

    @classmethod
    def create_empty(cls) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return cls([])

    def get_block_ids(self) -> list[int]:
        """Converts the KVCacheBlocks instance to a list of block IDs."""
        return [block.block_id for block in self.blocks]


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "KVCacheManager does not support hybrid models with more than 1 "
            "kv cache group")
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching,
                                    enable_kv_cache_events)

        self.single_type_manager = get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            use_eagle=self.use_eagle,
            num_kv_cache_groups=1,
            caching_hash_fn=self.caching_hash_fn,
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """

        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or request.sampling_params.prompt_logprobs is not None):
            return KVCacheBlocks.create_empty(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        if len(block_hashes) * self.block_size == request.num_tokens:
            # When prompt length is divisible by the block size and all
            # blocks are cached, we need to recompute the last token. This
            # have to be achieved by re-computing an entire block because
            # allocate_slots() assumes num_computed_tokens is always a
            # multiple of the block size. To achieve this, remove the last
            # block hash from the block_hashes for find_longest_cache_hit
            # This limitation can potentially be removed in the future to
            # slightly improve the performance.
            last_block_hash = block_hashes.pop()
        else:
            last_block_hash = None

        computed_blocks = (
            self.single_type_manager.find_longest_cache_hit(block_hashes))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)

        if last_block_hash is not None:
            # Add back the last block hash if it was removed.
            # NOTE: Because block_hashes is cached in req_to_block_hashes,
            # we shouldn't modify it directly.
            block_hashes.append(last_block_hash)

        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size
        return KVCacheBlocks(computed_blocks), num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.single_type_manager.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               len(new_computed_block_list) * self.block_size)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)
        num_blocks_to_allocate = (
            self.single_type_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
            ))

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not new_computed_block_list, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.single_type_manager.save_new_computed_blocks(
            request.request_id, new_computed_block_list)

        new_blocks = self.single_type_manager.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        if not self.enable_caching:
            return KVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        self.single_type_manager.cache_blocks(
            request, self.req_to_block_hashes[request.request_id],
            num_computed_tokens + num_new_tokens - len(request.spec_token_ids))

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted 
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.single_type_manager.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

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
        return self.single_type_manager.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()
