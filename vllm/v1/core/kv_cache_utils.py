# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import os
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, cdiv, sha256
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        KVCacheTensor, SlidingWindowSpec)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHash(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


class BlockHashWithGroupId(NamedTuple):
    # The hash value for the contents (e.g., token_ids) of a block without group
    # ID. The value is the same for blocks representing the same tokens but for
    # different groups.
    block_hash: BlockHash
    # The KV cache group ID.
    group_id: int

    def get_hash_value(self) -> int:
        return self.block_hash.hash_value


# The hash seed for the first block of the prefix block sequence.
#
# Even if the hash function is the builtin hash(), we use sha256 to generate
# the initial hash to simplify the code. This is not performance critical
# as it is done one per process.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed.
# This aligns with the behavior of Python's hash() function, which also uses
# a random seed if PYTHONHASHSEED is not set.
NONE_HASH = int.from_bytes(os.urandom(32), byteorder="big") if os.getenv(
    "PYTHONHASHSEED") is None else sha256(os.getenv("PYTHONHASHSEED"))


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

        When there are more than `interval` requests, the oldest set of
        requests are removed from the metrics.

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
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashWithGroupId] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

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
                f"_block_hash={self._block_hash}, "
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

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: Optional[KVCacheBlock] = blocks[0]
        self.free_list_tail: Optional[KVCacheBlock] = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
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
    return bool(request.mm_positions) or (request.lora_request
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

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return extra_keys, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    if mm_positions[-1].offset + mm_positions[-1].length < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
        offset = mm_positions[curr_mm_idx].offset
        length = mm_positions[curr_mm_idx].length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_hashes[curr_mm_idx])

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
        hash_function: Callable,
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[tuple[Any, ...]] = None) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
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
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)


def hash_request_tokens(hash_function: Any, block_size: int,
                        request: Request) -> list[BlockHash]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for block-hash computation.
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


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
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


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


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: dict[str, KVCacheSpec],
                                      available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """

    page_size = get_uniform_page_size(kv_cache_spec)
    num_blocks = get_num_blocks(vllm_config, len(kv_cache_spec),
                                available_memory, page_size)

    per_layer_size = page_size * num_blocks
    # All layers have the same KV cache spec, so we create one kv cache group
    # for all layers.
    grouped_layer_names = [list(kv_cache_spec.keys())]

    # Each layer uses a separate Tensor to store its KV cache.
    kv_cache_tensors = [
        KVCacheTensor(size=per_layer_size, shared_by=[layer_name])
        for layer_name in kv_cache_spec
    ]

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names),
    )

    num_tokens = num_blocks * vllm_config.cache_config.block_size
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config)
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)
    return kv_cache_config


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


def _get_kv_cache_config_uniform_page_size(
        vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec],
        available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for hybrid models with multiple 
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
        vllm_config: The global VllmConfig
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.
    Returns:
        The generated KVCacheConfig
    """
    # Group all layers by type_id.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[str, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec.type_id].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.1), (sw.2, padding).
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
        for i in range(0, len(layers), group_size):
            grouped_layers.append(layers[i:i + group_size])
    kv_cache_groups = create_kv_cache_group_specs(kv_cache_spec,
                                                  grouped_layers)

    # Determine how model runners should initialize the KV cache tensors.
    # We will have group_size memory pools, each is shared by one layer from
    # each group. As layers of different groups have different block table,
    # they will use different parts of the shared Tensor.
    # The memory layout in the example will be:
    # full.0, sw.0, sw.2: share a Tensor with size=available_memory//2
    # full.1, sw.1: share another Tensor with size=available_memory//2
    page_size = get_uniform_page_size(kv_cache_spec)
    num_blocks = get_num_blocks(vllm_config, group_size, available_memory,
                                page_size)
    per_memory_pool_size = page_size * num_blocks
    kv_cache_tensors = []
    for i in range(group_size):
        shared_by = []
        for j in range(len(kv_cache_groups)):
            if i < len(grouped_layers[j]):
                shared_by.append(grouped_layers[j][i])
        kv_cache_tensors.append(
            KVCacheTensor(size=per_memory_pool_size, shared_by=shared_by))

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    # Print the KV cache size and maximum concurrency.
    num_tokens = num_blocks // len(
        grouped_layers) * vllm_config.cache_config.block_size
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

    def is_hybrid(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
        type_ids = set(layer_spec.type_id
                       for layer_spec in kv_cache_spec.values())
        return len(type_ids) > 1

    if not is_hybrid(kv_cache_spec):
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
    if has_full_attention and has_sliding_window:
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

    if is_hybrid(kv_cache_spec):
        raise ValueError("Hybrid KV cache manager is disabled but failed to "
                         "convert the KV cache specs to one unified type.")


def get_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)
    elif is_kv_cache_page_size_uniform(kv_cache_spec):
        # Model contains multiple attention types, but KV cache of all layers
        # have the same physical memory per block per layer. Split the layers
        # into groups with the same number of layers, and thus same total page
        # size.
        return _get_kv_cache_config_uniform_page_size(vllm_config,
                                                      kv_cache_spec,
                                                      available_memory)

    raise NotImplementedError


def unify_kv_cache_configs(kv_cache_configs: list[KVCacheConfig]):
    """
    Make the KV cache configurations for each worker consistent, so that all
    workers can be controlled by the same KVCacheManager.
    This function verifies that the layer group of each worker are the same,
    and changes the num_blocks of each worker to the smallest among all workers.

    Args:
        kv_cache_configs: The KV cache configurations for each worker. Will be
            in-place modified to make them consistent.
    """

    # Sort the kv cache groups by the type_id of their KV cache spec.
    # This can avoid the inconsistency caused by the order of groups.
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_cache_groups.sort(
            key=lambda x: x.kv_cache_spec.type_id)

    # Verify that the groups of each rank are the same.
    for kv_cache_config in kv_cache_configs[1:]:
        for group_rank_0, group_rank_i in zip(
                kv_cache_configs[0].kv_cache_groups,
                kv_cache_config.kv_cache_groups):
            assert group_rank_0.kv_cache_spec == group_rank_i.kv_cache_spec

    # Change the num_blocks of each rank to the smallest among all ranks. We
    # do not need to shrink the tensor size because it is valid to only use the
    # first `num_blocks` blocks of the tensor.
    min_num_blocks = min(kv_cache_config.num_blocks
                         for kv_cache_config in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.num_blocks = min_num_blocks

    return kv_cache_configs
