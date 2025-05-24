# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from vllm.config import ForestedCascadeConfig
from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         KVCacheBlockPrefixTrie, CommonPrefixGroups)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.request import Request

class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management 
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        forested_cascade_config: ForestedCascadeConfig,
        block_pool: BlockPool,
        use_eagle: bool,
        num_kv_cache_groups: int,
        caching_hash_fn: Callable,
    ) -> None:
        """
        Initializes the SpecializedManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            forested_cascade_config: Defines parameters relevant to forested
            cascade attention.
            block_pool: The block pool.
            use_eagle: Whether to use eagle.
            num_kv_cache_groups: The number of kv cache groups managed by this 
                manager.
            caching_hash_fn: The caching hash function.
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.forested_cascade_config = forested_cascade_config
        self.block_pool = block_pool

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str,
                                        list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request id to the depth of the earliest non-null block
        # allocated for the request.
        self.req_to_depth: dict[str, int] = {}

        # Mapping from block depth to a KVCacheBlockPrefixTrie
        # This is necessary when using forested CascadeAttention
        self.depth_to_prefix_trie: dict[int, KVCacheBlockPrefixTrie] = {}

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.num_kv_cache_groups = num_kv_cache_groups
        self.caching_hash_fn = caching_hash_fn

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = (num_required_blocks - len(new_computed_blocks) -
                          len(self.req_to_blocks[request_id]))
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it will be changed from a free block
        # to a computed block when the request is allocated, so we also count
        # it as needed to be allocated.
        num_evictable_computed_blocks = sum(blk.ref_cnt == 0
                                            for blk in new_computed_blocks)
        return ((num_new_blocks + num_evictable_computed_blocks) *
                self.num_kv_cache_groups)

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_block:
            # A new request.
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:
            # A running request. Should not have new computed blocks.
            assert len(new_computed_blocks) == 0

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks * self.num_kv_cache_groups)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, block_hashes: list[BlockHashType],
                     num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """

        req_id = request.request_id

        num_cached_blocks = self.num_cached_block[req_id]
        num_full_blocks = num_tokens // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[req_id],
            block_hashes=block_hashes,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )

        if req_id not in self.req_to_depth:
            self.req_to_depth[req_id] = 1

        use_forested_prefix = False if self.forested_cascade_config is None \
            else self.forested_cascade_config.use_forested_prefix

        absorption_threshold_ratio = self.forested_cascade_config.absorption_threshold_ratio if (
                self.forested_cascade_config is not None) else float('inf')

        if use_forested_prefix:
            req_depth = self.req_to_depth[req_id]

            new_forested_cascade_trie = self.depth_to_prefix_trie.get(req_depth, None)
            if not new_forested_cascade_trie:
                new_forested_cascade_trie = KVCacheBlockPrefixTrie(
                    self.req_to_depth[request.request_id], absorption_threshold_ratio)
                self.depth_to_prefix_trie[req_depth] = new_forested_cascade_trie
            new_forested_cascade_trie.insert(
                request, self.req_to_blocks[req_id][num_cached_blocks: num_full_blocks])

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        base_block_depth = req_blocks[0].block_depth
        self.depth_to_prefix_trie[base_block_depth].remove(request_id)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    def unschedule_request(self, request_id: str) -> None:
        """
        Sets is_scheduled flag of request to False in its respective
        prefix trie. Used for forested cascade attention.
        """
        depth = self.req_to_depth.get(request_id, 0)
        if depth:
            self.depth_to_prefix_trie[depth].unschedule_request(request_id)

    @abstractmethod
    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        """
        Get the number of common prefix blocks for a request.

        Args:
            request_id: The request ID.
            num_running_requests: The number of currently running requests.

        Returns:
            The number of common prefix blocks.
        """

        raise NotImplementedError

    @abstractmethod
    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int) -> list[KVCacheBlock]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than 
        `max_length`. If no cache hit is found, return an empty list. 
        If eagle is enabled, drop the last matched block to force recompute the 
        last block to get the required hidden states for eagle drafting head. 
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block.
            For example, sliding window manager should return a list like
            [NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)] for block size 4 and 
            sliding window 8. 
        """

        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) \
            -> None:
        """
        Remove the blocks that are no longer needed from `blocks`. The removed
        blocks should be replaced by null_block. Return the removed blocks in
        eviction order, where the first returned block should be evicted first.
        The return statement is necessary to track block depth

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.

        """
        raise NotImplementedError

    def get_common_prefix_groups(self) -> CommonPrefixGroups | None:

        """
        Returns a data structure for retrieving common prefixes from
        different requests
        """

        raise NotImplementedError


class FullAttentionManager(SingleTypeKVCacheManager):

    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int) -> list[KVCacheBlock]:
        computed_blocks: list[KVCacheBlock] = []
        max_num_blocks = max_length // self.block_size
        for i in range(max_num_blocks):
            block_hash = block_hashes[i]
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        if self.use_eagle and len(computed_blocks) > 0:
            computed_blocks.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) \
            -> None:
        # No need to remove blocks for full attention.
        pass

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        blocks = self.req_to_blocks[request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def get_common_prefix_groups(self) -> CommonPrefixGroups | None:
        if self.forested_cascade_config is None:
            return None
        common_prefix_groups = CommonPrefixGroups([], [])
        alloc_method = self.forested_cascade_config.allocate_method
        for depth in self.depth_to_prefix_trie:
            prefix_trie = self.depth_to_prefix_trie[depth]
            groups = prefix_trie.allocate_group(alloc_method)
            common_prefix_groups.extend(groups)
        return common_prefix_groups

class SlidingWindowManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec,
                 forested_cascade_config: ForestedCascadeConfig,
                 block_pool: BlockPool,
                 use_eagle: bool, **kwargs) -> None:
        super().__init__(kv_cache_spec, forested_cascade_config, block_pool,
                         use_eagle, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        self.sliding_window_contiguous_blocks = cdiv(
            (kv_cache_spec.sliding_window - 1), self.block_size)
        if self.use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            self.sliding_window_contiguous_blocks += 1
        self._null_block = block_pool.null_block

    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int) -> list[KVCacheBlock]:
        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // self.block_size
        computed_blocks = [self._null_block] * max_num_blocks
        num_contiguous_blocks = 0

        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := self.block_pool.get_cached_block(
                    block_hashes[i]):
                computed_blocks[i] = cached_block
                num_contiguous_blocks += 1
                if (num_contiguous_blocks
                        >= self.sliding_window_contiguous_blocks):
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    del computed_blocks[i + num_contiguous_blocks:]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            del computed_blocks[num_contiguous_blocks:]
        if self.use_eagle and len(computed_blocks) > 0:
            computed_blocks.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) \
            -> None:
        # Remove the blocks that are no longer be in the sliding window and
        # skipped during the attention computation.
        first_useful_token = num_computed_tokens - self.sliding_window + 1
        first_useful_block_depth = first_useful_token // self.block_size
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        for i in range(first_useful_block_depth - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block

        use_forested_prefix = False if self.forested_cascade_config is None \
            else self.forested_cascade_config.use_forested_prefix

        old_req_depth = -1
        if removed_blocks:
            #TODO: Verify block_depth is not fixed at 0.
            # These blocks should have been fully cached.
            old_req_depth = removed_blocks[-1].block_depth
            new_req_depth = removed_blocks[0].block_depth + 1
            self.req_to_depth[request_id] = new_req_depth

        if removed_blocks and use_forested_prefix:
            old_forested_cascade_trie = self.depth_to_prefix_trie.get(old_req_depth, None)
            if old_forested_cascade_trie:
                old_forested_cascade_trie.remove(request_id)

        self.block_pool.free_blocks(removed_blocks)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return 
        0 here for correctness. Need to support cascade attention + sliding 
        window in the future.
        """
        return 0

    def get_common_prefix_groups(self) -> CommonPrefixGroups | None:
        pass


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
}


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
