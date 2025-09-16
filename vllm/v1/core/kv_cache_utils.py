# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import os
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, NewType, Optional, Union

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, cdiv, sha256_cbor
from vllm.v1.kv_cache_interface import (ChunkedLocalAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        KVCacheTensor, SlidingWindowSpec)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from ``bytes`` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# ``BlockHashWithGroupId`` combines a ``BlockHash`` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the ``BlockHash`` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of ``bytes`` and ``int`` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash = Union[bytes, int]


def make_block_hash_with_group_id(block_hash: BlockHash,
                                  group_id: int) -> BlockHashWithGroupId:
    """Pack a ``BlockHash`` and group id into a ``BlockHashWithGroupId``.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash +
                                group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the ``BlockHash`` from a ``BlockHashWithGroupId``."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a ``BlockHashWithGroupId``."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn is sha256_cbor:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using sha256_cbor as the hash function."
            "Consider setting PYTHONHASHSEED to a fixed value for "
            "reproducibility.")

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


class PrefixCachingMetrics:
    """Metrics for prefix caching with a hit rate of the max recent N requests.

    Args:
        max_recent_requests: The number of the max recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000):
        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue: deque[tuple[int, int, int]] = deque()

    def observe(self, stats: PrefixCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `max_recent_requests` requests, the oldest set
        of requests are removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats if the number of requests exceeds.
        if self.aggregated_requests > self.max_recent_requests:
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: Optional[BlockHashWithGroupId] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> Optional[BlockHashWithGroupId]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = (self.prev_free_block.block_id
                         if self.prev_free_block else None)
        next_block_id = (self.next_free_block.block_id
                         if self.next_free_block else None)
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, "
                f"_block_hash={self._block_hash!r}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if (self.fake_free_list_head.next_free_block
                is self.fake_free_list_tail
                or self.fake_free_list_head.next_free_block is None):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list.")
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError("Invalid block found in popleft() "
                               "which doesn't have a valid next_free_block")

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist")
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist")
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist")
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA ID.
    # Request with provided cache salt need to include the salt.
    return bool(request.mm_features) or (request.lora_request
                                         is not None) or (request.cache_salt
                                                          is not None)


def _gen_mm_extra_hash_keys(request: Request, start_token_idx: int,
                            end_token_idx: int,
                            start_mm_idx: int) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_feature.identifier)

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[int]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA id of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_int_id]


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> tuple[Optional[tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: list[int] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = [request.cache_salt] if (
        start_token_idx == 0 and request.cache_salt) else []

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys + cache_salt_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
        hash_function: Callable[[Any], bytes],
        parent_block_hash: Optional[BlockHash],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[tuple[Any, ...]] = None) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)))


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (request.block_hashes[-1]
                                 if request.block_hashes else None)
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx)

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(caching_hash_fn,
                                           prev_block_hash_value, block_tokens,
                                           extra_keys)

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def max_memory_usage_bytes(vllm_config: VllmConfig,
                           kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(
        spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(vllm_config: VllmConfig,
                           kv_cache_spec: dict[str, KVCacheSpec],
                           available_memory: int) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config,
                                               kv_cache_spec.values())
        return memory_needed <= available_memory

    # Binary search for the maximum model length
    current_max = vllm_config.model_config.max_model_len
    left, right = 1, current_max

    # If even the smallest model length doesn't fit, return 0
    if not fits_in_memory(left):
        return 0

    # Binary search for the maximum model length that fits
    result = 1
    while left <= right:
        mid = (left + right) // 2
        if fits_in_memory(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: dict[str, KVCacheSpec],
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if not kv_cache_spec:
        return

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())

    if needed_memory > available_memory:
        # Estimate the maximum model length that can fit in the available memory
        estimated_max_len = estimate_max_model_len(vllm_config, kv_cache_spec,
                                                   available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}.")

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/GiB_bytes:.2f} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/GiB_bytes:.2f} GiB). "
            f"{estimated_msg} "
            f"Try increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def create_kv_cache_group_specs(
        kv_cache_spec: dict[str, KVCacheSpec],
        grouped_layer_names: list[list[str]]) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
    return kv_cache_groups


def is_kv_cache_type_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups)
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config,
        (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups))
    memory_per_block = kv_cache_config.kv_cache_groups[
        0].kv_cache_spec.page_size_bytes * num_layer_per_group
    num_block_per_request = cdiv(max_memory_usage_per_request,
                                 memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def get_num_blocks(vllm_config: VllmConfig, num_layers: int,
                   available_memory: int, page_size: int) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override
    return num_blocks


def get_uniform_page_size(kv_cache_spec: dict[str, KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = set(layer.page_size_bytes for layer in kv_cache_spec.values())
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_type(
        kv_cache_specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs,
                                       [list(kv_cache_specs.keys())])


def is_kv_cache_page_size_uniform(
        kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def is_kv_cache_type_attention_free(
        kv_cache_spec: dict[str, KVCacheSpec]) -> bool:

    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
        kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple 
    attention types but still with a uniform page size (physical memory per 
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times. 
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the 
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer 
    in the group.
    For example:
    1. A model only uses full attention. The pattern is 
    (num_hidden_layers * full), so there is only one group and the block table 
    is shared by all layers. It is already handled by 
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window 
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so 
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups. 
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use 
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV 
    cache group, but within each KV cache group, all layers must share the same 
    block size.
    3. Physical memory per token per layer: This property is decided by model 
    config. Currently we only support models that have the same physical memory 
    per token per layer for all layers. Can be relaxed with a simple extension, 
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers. 
    Can be relaxed with a simple extension, but still need to keep physical 
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when 
    `--disable-hybrid-kv-cache-manager` is true, the single group for full 
    attention layers may also include attention layers using sliding window or 
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is 
    general to an arbitrary number of attention types. But 
    `find_longest_cache_hit` only supports one attention type or two 
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to 
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and 
    number of layers per group are the same now, we can ensure that physical 
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    group_size = min([len(layers) for layers in same_type_layers.values()])
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(vllm_config: VllmConfig,
                                    kv_cache_groups: list[KVCacheGroupSpec],
                                    kv_cache_specs: dict[str, KVCacheSpec],
                                    available_memory: int) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        kv_cache_specs: The KV cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    # We will have group_size memory pools, each is shared by one layer from
    # each group. As layers of different groups have different block table,
    # they will use different parts of the shared Tensor.
    # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
    # (sw.1, padding) will be: (group_size = 2)
    # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
    # full.1, sw.2: share another Tensor with size=available_memory//2
    group_size = max(len(group.layer_names) for group in kv_cache_groups)

    page_size = get_uniform_page_size(kv_cache_specs)
    assert group_size > 0, "group_size must be greater than 0"
    num_blocks = get_num_blocks(vllm_config, group_size, available_memory,
                                page_size)
    per_memory_pool_size = page_size * num_blocks
    kv_cache_tensors = []
    for i in range(group_size):
        shared_by = []
        for j in range(len(kv_cache_groups)):
            if i < len(kv_cache_groups[j].layer_names):
                shared_by.append(kv_cache_groups[j].layer_names[i])
        kv_cache_tensors.append(
            KVCacheTensor(size=per_memory_pool_size, shared_by=shared_by))

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_groups])

    # Print the KV cache size and maximum concurrency.
    num_tokens = num_blocks // len(kv_cache_groups) * min_block_size
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        num_tokens *= vllm_config.parallel_config.decode_context_parallel_size
        logger.info(
            "Multiplying the GPU KV cache size by the dcp_world_size %d.",
            vllm_config.parallel_config.decode_context_parallel_size)
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config)
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)
    return kv_cache_config


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_type_uniform(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved.")

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values())
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values())
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec)
        for spec in kv_cache_spec.values())
    if has_full_attention and (has_sliding_window
                               or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    use_mla=spec.use_mla,
                    sliding_window=spec.sliding_window,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    use_mla=spec.use_mla,
                    attention_chunk_size=spec.attention_chunk_size,
                )

    if not is_kv_cache_type_uniform(kv_cache_spec):
        raise ValueError("Hybrid KV cache manager is disabled but failed to "
                         "convert the KV cache specs to one unified type.")


def get_kv_cache_groups(
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []
    elif is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_type(kv_cache_spec)
    elif is_kv_cache_page_size_uniform(kv_cache_spec):
        # Model contains multiple attention types, but KV cache of all layers
        # have the same physical memory per block per layer. Split the layers
        # into groups with the same number of layers, and thus same total page
        # size.
        return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)

    raise NotImplementedError


def get_kv_cache_configs(vllm_config: VllmConfig,
                         kv_cache_specs: list[dict[str, KVCacheSpec]],
                         available_memory: list[int]) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model. 
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
    3. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    4. Change the num_blocks of each worker to the smallest among all workers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each 
        worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Check if the available memory is enough for each worker.
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
            kv_cache_specs, available_memory):
        check_enough_kv_cache_memory(vllm_config, kv_cache_spec_one_worker,
                                     available_memory_one_worker)

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet.")
    global_kv_cache_groups = get_kv_cache_groups(vllm_config,
                                                 merged_kv_cache_specs)

    kv_cache_configs: list[KVCacheConfig] = []
    for kv_cache_spec_one_worker, available_memory_one_worker in zip(
            kv_cache_specs, available_memory):
        kv_cache_groups_one_worker: list[KVCacheGroupSpec] = []
        for group in global_kv_cache_groups:
            group_layer_names_one_worker = [
                layer_name for layer_name in group.layer_names
                if layer_name in kv_cache_spec_one_worker
            ]
            kv_cache_groups_one_worker.append(
                KVCacheGroupSpec(group_layer_names_one_worker,
                                 group.kv_cache_spec))
        assert sum(
            len(group.layer_names) for group in
            kv_cache_groups_one_worker) == len(kv_cache_spec_one_worker), (
                "Some layers are not assigned to any group.")
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(vllm_config,
                                            kv_cache_groups_one_worker,
                                            kv_cache_spec_one_worker,
                                            available_memory_one_worker))

    # Change the num_blocks of each rank to the smallest among all ranks. We
    # do not need to shrink the tensor size because it is valid to only use the
    # first `num_blocks` blocks of the tensor.
    min_num_blocks = min(kv_cache_config.num_blocks
                         for kv_cache_config in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.num_blocks = min_num_blocks

    return kv_cache_configs
