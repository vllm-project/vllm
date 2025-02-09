from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import math
from typing import Callable, DefaultDict, Dict, Iterator, List, Optional, Tuple, Type, TypeVar

from vllm.core.block.common import BlockPool
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         PrefixLength, PrefixLengthRange,
                                         ReqKVCacheBlocks, hash_request_tokens,
                                         intersect_ranges)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, SlidingWindowSpec)
from vllm.v1.request import Request
from vllm.v1.utils import ConstantList

T = TypeVar("T")


class SpecializedManager(ABC):
    """
    An abstract base class for specialized managers that handle the kv
    cache management logic of different attention layers.
    """
    block_size: int
    max_num_blocks_per_req: int

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        max_model_len: int,
        enable_caching: bool,
        kv_cache_group_id: int,
        block_pool: BlockPool,
    ) -> None:
        """
        Initializes the SpecializedManager.

        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.

        Returns:
            None
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.max_num_blocks_per_req = cdiv(max_model_len, self.block_size)
        self.enable_caching = enable_caching
        self.kv_cache_group_id = kv_cache_group_id

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: DefaultDict[str,
                                        List[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: DefaultDict[
            str, List[BlockHashType]] = defaultdict(list)

    def hash_request_tokens(self, request: Request) -> List[BlockHashType]:
        """
        Hash the tokens of a request to block hashes.

        Args:
            request: The request to hash.

        Returns:
            List[BlockHashType]: The block hashes of the request.
        """
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.block_size, request,
                                               self.kv_cache_group_id)
            self.req_to_block_hashes[request.request_id] = block_hashes
        return block_hashes

    def truncate_computed_blocks(
            self, computed_blocks: List[KVCacheBlock],
            num_computed_tokens: int) -> List[KVCacheBlock]:
        # Truncate the computed blocks to the number of computed tokens.
        # E.g., group 0 has 3 computed blocks, and group 1 has 4 computed
        # blocks with the same block size, we truncate both groups to 3 blocks.
        computed_blocks = computed_blocks[:num_computed_tokens //
                                          self.block_size]
        return computed_blocks

    def get_req_num_new_blocks(self, request, new_computed_blocks,
                               num_computed_tokens, num_tokens):
        req_blocks = self.req_to_blocks[request.request_id]
        new_computed_blocks = new_computed_blocks if new_computed_blocks is not None else []
        return self.get_num_new_blocks(
            num_computed_tokens, num_tokens,
            len(req_blocks) + len(new_computed_blocks))

    def allocate_slots(
        self,
        request: Request,
        new_computed_blocks: Optional[List[KVCacheBlock]],
        num_new_blocks: int,
        num_preallocate_blocks: int,
        num_computed_tokens: int,
        num_tokens: int,
    ):
        if new_computed_blocks is None:
            new_computed_blocks = []
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert len(new_computed_blocks) == 0, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks = self.req_to_blocks[request.request_id]
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks
        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + num_preallocate_blocks,
                # Should not exceed the maximum number of blocks per request
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                # TODO(woosuk): Check and reject requests if
                # num_prompt_tokens + max_tokens > max_model_len.
                self.max_num_blocks_per_req - len(req_blocks),
            )

            assert num_new_blocks >= 0

            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        # NOTE(rickyx): We are assuming the `num_tokens` are actual
        # tokens rather than lookahead slots (e.g. for speculative decoding).
        # TODO(rickyx): When supporting speculative decoding, we will need to
        # differentiate between them so that we can know how many blocks are
        # full after appending the actual tokens.
        num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size
        num_computed_full_blocks = num_computed_tokens // self.block_size

        new_full_blocks = req_blocks[num_computed_full_blocks:num_full_blocks]
        if new_full_blocks:
            self.block_pool.cache_full_blocks(
                request=request,
                block_hashes=self.req_to_block_hashes[request.request_id],
                block_size=self.block_size,
                blk_start_idx=num_computed_full_blocks,
                # The new full blocks are the full blocks that are not
                # computed.
                full_blocks=new_full_blocks,
                prev_block=(req_blocks[num_computed_full_blocks - 1]
                            if num_computed_full_blocks > 0 else None),
                kv_cache_group_id=self.kv_cache_group_id,
            )

        return new_blocks

    def get_num_common_prefix_blocks(self, request: Request,
                                     num_running_requests: int) -> int:
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    @abstractmethod
    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[PrefixLength, List[KVCacheBlock]]:
        """
        Get the possible cached prefixes of a request based on its block hashes.
        If no cached prefixes are found, returns a tuple with a prefix length 
        range of [0, 0] and an empty list of blocks.

        Args:
            block_hashes: The block hashes of the request.

        Returns:
            A tuple containing:
                - A list of all possible cached prefix lengths.
                - The computed blocks that are cached.
        """

        raise NotImplementedError

    @abstractmethod
    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        """
        Calculate the number of new blocks needed by this manager.

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            num_append_tokens: The number of tokens that need to be appended.
            num_allocated_blocks: The number of blocks that have already been 
            allocated.

        Returns:
            int: The number of new blocks needed.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_useless_blocks(self, request: Request,
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        """
        Update the `block_table` in place to remove blocks that are no longer 
        needed. Replace the removed blocks with null_block and returns the 
        removed blocks. 
        The removed blocks should be in the order of the
        priority to be evicted, where the first block should have the highest
        priority.
        
        Args:
            block_table: The block table to be updated.
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            List[KVCacheBlock]: The removed blocks.
        """
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        computed_blocks: List[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        return [PrefixLengthRange(0,
                                  len(computed_blocks) * self.block_size)
                ], computed_blocks

    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        num_required_blocks = cdiv(num_computed_tokens + num_append_tokens,
                                   self.block_size)
        num_new_blocks = num_required_blocks - num_allocated_blocks
        return num_new_blocks

    def remove_useless_blocks(self, request: Request,
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        return []


class SlidingWindowManager(FullAttentionManager):

    def __init__(
        self,
        kv_cache_spec: SlidingWindowSpec,
        max_model_len: int,
        enable_caching: bool,
        kv_cache_group_id: int,
        block_pool: BlockPool,
    ) -> None:
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            block_pool=block_pool,
        )
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.get_null_block()

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        # TODO: check the hit every num_block_sliding_window blocks, to optimize
        # the time complexity from O(num_block) to
        # O(num_block / num_block_sliding_window) + O(num_computed_block),
        # which is good for low cache hit rate scenarios.
        start = 0
        ranges = []
        computed_blocks: List[KVCacheBlock] = []

        dummy_block_hash = BlockHashType(-1, (), -1)
        # Add a dummy block hash to support the case that the last block is
        # cached.
        for i, block_hash in enumerate(chain(block_hashes,
                                             [dummy_block_hash])):
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                if start == 0:
                    # All tokens between [0, i * block_size] are cached.
                    # All of them are possible cached prefix.
                    ranges.append(PrefixLengthRange(0, i * self.block_size))
                elif (i - start) * self.block_size >= self.sliding_window:
                    # All tokens between [start * block_size,
                    # i * block_size)] are cached. These tokens except the
                    # first `self.sliding_window - 1` ones are possible cached
                    # prefix.
                    first_cached_token = start * self.block_size
                    # should be first_cached_token + self.sliding_window - 1 + 1
                    # +1 is for converting the token index to the prefix length.
                    first_possible_length = first_cached_token + \
                        self.sliding_window
                    ranges.append(
                        PrefixLengthRange(first_possible_length,
                                          i * self.block_size))
                computed_blocks.append(self._null_block)
                start = i + 1
        computed_blocks = computed_blocks[:-1]  # remove the dummy block
        return ranges, computed_blocks

    def remove_useless_blocks(self, request: Request,
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        # Remove the blocks that are no longer be in the sliding window.
        last_useful_token = num_computed_tokens - self.sliding_window
        last_useful_block = last_useful_token // self.block_size

        block_table = self.req_to_blocks[request.request_id]
        removed_blocks: List[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if block_table[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also be null blocks.
                break
            removed_blocks.append(block_table[i])
            block_table[i] = self._null_block
        return removed_blocks


spec_manager_map: Dict[Type[KVCacheSpec], Type[SpecializedManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def transpose_output(outputs: List[Tuple[T]]) -> Tuple[List[T]]:
    return tuple(map(list, zip(*outputs)))


class GroupedManager:

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 enable_caching: bool, block_pool: BlockPool) -> None:
        self.enable_caching = enable_caching
        self.managers: List[SpecializedManager] = []
        self.kv_cache_config = kv_cache_config
        for i, g in enumerate(kv_cache_config.groups):
            manager_class = spec_manager_map[type(g.kv_cache_spec)]
            manager = manager_class(g.kv_cache_spec, max_model_len,
                                    enable_caching, i, block_pool)
            self.managers.append(manager)
        self.block_pool = block_pool

    # Simple broadcast functions
    # TODO: a better way to handle the broadcast functions
    def hash_request_tokens(self,
                            request: Request) -> List[List[BlockHashType]]:
        return [
            manager.hash_request_tokens(request) for manager in self.managers
        ]

    def get_possible_cached_prefix(self,
                                   block_hashes: List[List[BlockHashType]]):
        outputs = [
            manager.get_possible_cached_prefix(block_hashes[i])
            for i, manager in enumerate(self.managers)
        ]

        return transpose_output(outputs)

    def truncate_computed_blocks(self, computed_blocks: ReqKVCacheBlocks,
                                 num_computed_tokens: int) -> ReqKVCacheBlocks:
        return [
            manager.truncate_computed_blocks(computed_blocks[i],
                                             num_computed_tokens)
            for i, manager in enumerate(self.managers)
        ]

    def get_req_num_new_blocks(self, request: Request,
                               new_computed_blocks: ReqKVCacheBlocks,
                               num_computed_tokens: int, num_tokens: int):
        return [
            manager.get_req_num_new_blocks(request, new_computed_blocks[i],
                                           num_computed_tokens, num_tokens)
            for i, manager in enumerate(self.managers)
        ]

    def allocate_slots(self, request: Request,
                       new_computed_blocks: Optional[ReqKVCacheBlocks],
                       num_new_blocks: int, num_preallocate_blocks: int,
                       num_computed_tokens: int, num_tokens: int):
        return [
            manager.allocate_slots(
                request, new_computed_blocks[i] if new_computed_blocks
                is not None else None, num_new_blocks[i],
                num_preallocate_blocks, num_computed_tokens, num_tokens)
            for i, manager in enumerate(self.managers)
        ]

    def get_num_common_prefix_blocks(self, request: Request,
                                     num_running_requests: int):
        return [
            manager.get_num_common_prefix_blocks(request, num_running_requests)
            for manager in self.managers
        ]

    def remove_useless_blocks(self, request: Request,
                              num_computed_tokens: int) -> None:
        """
        Frees memory blocks that are not needed. E.g., sliding window 
        layer with window size 2 and block size 1, we have req_blocks as 
        [[1, 2, 3]], this function will free block 1 and change the req_blocks
        to [[-1, 2, 3]] (-1 refers to null block)

        Args:
            req_blocks: The KV cache blocks of one request.
            num_computed_tokens: The number of computed tokens.
        """
        return [
            manager.remove_useless_blocks(request, num_computed_tokens)
            for i, manager in enumerate(self.managers)
        ]

    def pop_blocks_of_request(
            self, request_id: int) -> Optional[List[List[KVCacheBlock]]]:
        blocks = [
            manager.req_to_blocks.pop(request_id, None)
            for manager in self.managers
        ]
        if all(blks is None for blks in blocks):
            return None
        assert all(blks is not None for blks in blocks)
        return blocks

    def pop_block_hashes_of_request(self, request_id: int) -> None:
        for manager in self.managers:
            manager.req_to_block_hashes.pop(request_id, None)

    def construct(self, lambda_fn: Callable[[], T]) -> List[T]:
        return [lambda_fn() for _ in range(len(self.managers))]

    # Complex functions
    def get_common_computed_tokens(self,
                                   prefix_length: List[PrefixLength]) -> int:
        """
        Find the longest prefix that is cached by all KV cache groups. Returns 
        the number of tokens in that prefix.

        Args:
            prefix_length (List[PrefixLength]): The valid cached prefix lengths 
            of each KV cache group.
    
        Returns:
            The number of tokens in the common prefix.
        """
        if len(self.kv_cache_config.groups
               ) == 1:  # TODO: split num_group=1 case
            return prefix_length[0][-1].end

        intersection = intersect_ranges(prefix_length)

        # Since incomplete blocks are not eligible for sharing,
        # `num_computed_tokens` should be a multiple of `block_size` of
        # all managers, so we take the least common multiple (LCM) of them
        alignment = math.lcm(
            *[manager.block_size for manager in self.managers])

        # Get the longest common prefix that is aligned with the block size.
        num_computed_tokens = 0
        for range_ in intersection:
            aligned_end = cdiv(range_.end, alignment) * alignment
            if aligned_end >= range_.start:
                num_computed_tokens = aligned_end
                break

        return num_computed_tokens

    def iter_all(self, x: List[List[T]]) -> Iterator[T]:
        return chain.from_iterable(x)

    def _sort_blocks_by_eviction_order(
            self, blocks: List[List[KVCacheBlock]],
            need_reverse: bool) -> List[KVCacheBlock]:
        """
        Merge the blocks of different groups to one list. The returned blocks 
        are sorted by eviction order, with the first block having the highest 
        eviction priority.

        Args:
            blocks: the blocks of each kv cache group, ordered by eviction 
            priority.

        Returns:
            A list of KVCacheBlocks sorted by eviction order.
        """
        if need_reverse:
            blocks = [
                list(reversed(blocks_of_group)) for blocks_of_group in blocks
            ]

        # TODO: seperate group_size = 1
        if len(blocks) == 1:
            # Fast path for single kv cache group models.
            return blocks[0]

        if self.enable_caching:
            # NOTE (Chen): A simple strategy that interleaves the blocks of
            # different KV cache groups. We can investigate more advanced
            # strategies in the future.
            ordered_blocks = []
            max_len = max(len(blocks_of_group) for blocks_of_group in blocks)
            for i in range(max_len):
                for blocks_of_group in blocks:
                    if i < len(blocks_of_group):
                        ordered_blocks.append(blocks_of_group[i])
        else:
            ordered_blocks = []
            for blocks_of_group in blocks:
                ordered_blocks.extend(blocks_of_group)

        return ordered_blocks

    def free_blocks(self,
                    blocks_to_free: List[List[KVCacheBlock]],
                    need_reverse: bool = False) -> None:
        ordered_blocks = self._sort_blocks_by_eviction_order(
            blocks_to_free, need_reverse)
        self.block_pool.free_blocks(ordered_blocks)
