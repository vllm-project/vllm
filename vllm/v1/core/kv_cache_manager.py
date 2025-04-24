# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv, sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashType, GroupedKVCacheBlock, KVCacheBlock, hash_request_tokens,
    remove_last_block_hash_for_divisible_prompt_length)
from vllm.v1.core.specialized_manager import SpecializedManager, get_specialized_manager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    blocks: list[list[GroupedKVCacheBlock]]

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        return KVCacheBlocks([
            self_blocks_i + other_blocks_i
            for self_blocks_i, other_blocks_i in zip(self.blocks, other.blocks)
        ])


class KVCacheManager:
    """
    The KVCacheManager for models with multiple KV cache types
     (e.g., Gemma-2) and thus multiple kv cache groups (Refer to class 
     `KVCacheConfig` for the meaning of kv cache groups).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
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
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)

        self.manager_to_group = self.generate_group_manager_map()
        self.num_specialized_managers = len(self.manager_to_group)

        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching,
                                    self.num_specialized_managers,
                                    self.caching_hash_fn)

        self.specialized_managers: list[SpecializedManager] = []
        for i in range(len(self.manager_to_group)):
            group_ids = self.manager_to_group[i]
            kv_cache_spec = kv_cache_config.kv_cache_groups[
                group_ids[0]].kv_cache_spec
            self.specialized_managers.append(
                get_specialized_manager(kv_cache_spec=kv_cache_spec,
                                        block_pool=self.block_pool,
                                        kv_cache_manager_id=i,
                                        num_kv_cache_groups=len(group_ids)))
        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        # block_size -> list[BlockHashType]; TODO update comment
        self.req_to_block_hashes: defaultdict[str, dict[
            int, list[BlockHashType]]] = defaultdict(dict)

        self.all_block_sizes = set(
            g.kv_cache_spec.block_size
            for g in self.kv_cache_config.kv_cache_groups)

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

    def empty_kv_cache_blocks(self) -> KVCacheBlocks:
        return KVCacheBlocks([[]
                              for _ in range(self.num_specialized_managers)])

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
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
            return self.empty_kv_cache_blocks(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if len(block_hashes) == 0:
            block_hashes = {
                block_size:
                hash_request_tokens(self.caching_hash_fn, block_size, request)
                for block_size in self.all_block_sizes
            }
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
        # When the request requires prompt logprobs, we skip prefix caching.
        if request.sampling_params.prompt_logprobs is not None:
            return self.empty_kv_cache_blocks(), 0

        computed_blocks, num_computed_tokens = self.find_longest_cache_hit(
            request, block_hashes)

        if self.log_stats:
            assert self.prefix_cache_stats is not None

            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)
        return KVCacheBlocks(computed_blocks), num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        wrapped_new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_new_computed_tokens: int = 0,
        num_lookahead_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
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

        if wrapped_new_computed_blocks is not None:
            new_computed_blocks = wrapped_new_computed_blocks.blocks
        else:
            new_computed_blocks = [
                [] for _ in range(self.num_specialized_managers)
            ]
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        for i, manager in enumerate(self.specialized_managers):
            manager.remove_skipped_blocks(request.request_id,
                                          request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = (num_computed_tokens + num_tokens +
                                num_lookahead_tokens)

        num_needed_blocks: list[int] = [
            manager.get_num_needed_blocks(request.request_id,
                                          num_tokens_need_slot,
                                          new_computed_blocks[i])
            for manager in self.specialized_managers
        ]
        if (sum(num_needed_blocks) > self.block_pool.get_num_free_blocks()):
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert all(len(blks) == 0 for blks in new_computed_blocks), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        for i in range(self.num_specialized_managers):
            self.specialized_managers[i].req_to_blocks[
                request.request_id].extend(new_computed_blocks[i])

        new_blocks: list[list[GroupedKVCacheBlock]] = []
        # Start to handle new blocks
        for i in range(self.num_specialized_managers):
            new_blocks_i = self.specialized_managers[i].allocate_new_blocks(
                request.request_id, num_tokens_need_slot)
            new_blocks.append(new_blocks_i)

        if not self.enable_caching:
            return KVCacheBlocks(new_blocks)

        for i, manager in enumerate(self.specialized_managers):
            manager.cache_blocks(
                request, new_computed_blocks[i], self.req_to_block_hashes[
                    request.request_id][manager.block_size],
                num_computed_tokens, num_tokens)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        for manager in self.specialized_managers:
            manager.free(request.request_id)

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
        # TODO: implement this
        return [0] * self.num_kv_cache_groups
        # assert request.status == RequestStatus.RUNNING
        # blocks = self.req_to_blocks[request.request_id]
        # num_common_blocks = []
        # for i in range(self.num_kv_cache_groups):
        #     num_common_blocks_i = 0
        #     for block in blocks[i]:
        #         if block.ref_cnt == num_running_requests:
        #             num_common_blocks_i += 1
        #         else:
        #             break
        #     num_common_blocks.append(num_common_blocks_i)
        # return num_common_blocks

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def find_longest_cache_hit(
        self, request: Request, block_hashes_dict: dict[int,
                                                        list[BlockHashType]]
    ) -> tuple[list[list[GroupedKVCacheBlock]], int]:
        """Find the longest cache hit for each kv cache group.
        TODO: add more notes
        """

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token. This
        # have to be achieved by re-computing an entire block because
        # allocate_slots() assumes num_computed_tokens is always a
        # multiple of the block size. To achieve this, remove the last
        # block hash from the block_hashes for find_longest_cache_hit
        # This limitation can potentially be removed in the future to
        # slightly improve the performance.
        with remove_last_block_hash_for_divisible_prompt_length(
                block_hashes_dict, request.num_tokens):
            if self.num_specialized_managers == 1:
                block_size = self.kv_cache_config.kv_cache_groups[
                    0].kv_cache_spec.block_size
                hit_blocks = self.specialized_managers[
                    0].find_longest_cache_hit(block_hashes_dict[block_size])

                return [hit_blocks], len(hit_blocks) * block_size

            # TODO: add note for the two magic number
            num_computed_tokens = [self.max_model_len + 100] * len(
                self.specialized_managers)
            min_computed_tokens = self.max_model_len

            # Use copy to avoid modifying the original block_hashes
            block_hashes = [
                block_hashes_dict[g.kv_cache_spec.block_size].copy()
                for g in self.kv_cache_config.kv_cache_groups
            ]

            computed_blocks: list[Optional[list[GroupedKVCacheBlock]]] = [
                None for _ in range(self.num_specialized_managers)
            ]

            def shrink_length(block_hashes, length):
                del block_hashes[length:]

            while max(num_computed_tokens) != min_computed_tokens:
                for i, manager in enumerate(self.specialized_managers):
                    if num_computed_tokens[i] > min_computed_tokens:
                        shrink_length(
                            block_hashes[i],
                            min_computed_tokens // manager.block_size)
                        computed_blocks_i = (manager.find_longest_cache_hit(
                            block_hashes[i], computed_blocks[i]))

                        num_computed_tokens[i] = len(computed_blocks_i) * \
                            manager.block_size
                        min_computed_tokens = min(min_computed_tokens,
                                                  num_computed_tokens[i])
                        computed_blocks[i] = computed_blocks_i
                        shrink_length(
                            block_hashes[i],
                            num_computed_tokens[i] // manager.block_size)

            assert all(block is not None and len(block) *
                       manager.block_size == min_computed_tokens
                       for block, manager in zip(computed_blocks,
                                                 self.specialized_managers))
            return computed_blocks, min_computed_tokens

    def generate_group_manager_map(self) -> list[list[int]]:
        type_ids = [
            g.kv_cache_spec.type_id
            for g in self.kv_cache_config.kv_cache_groups
        ]
        assert sorted(type_ids) == type_ids, "type_ids must be sorted"
        manager_to_group: list[list[int]] = []
        for i, type_id in enumerate(type_ids):
            if type_id == 0:
                manager_to_group.append([i])
            else:
                if type_id == type_ids[i - 1]:
                    manager_to_group[-1].append(i)
                else:
                    manager_to_group.append([i])
        return manager_to_group

    def to_block_ids(self, kv_cache_blocks: KVCacheBlocks) -> list[list[int]]:
        block_ids: list[list[int]] = [[]
                                      for _ in range(self.num_kv_cache_groups)]
        for blocks_one_manager, group_ids in zip(kv_cache_blocks.blocks,
                                                 self.manager_to_group):
            for blocks in blocks_one_manager:
                for blk, group_id in zip(blocks.blocks, group_ids):
                    block_ids[group_id].append(blk.block_id)
        return block_ids
