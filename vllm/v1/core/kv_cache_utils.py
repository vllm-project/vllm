# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NewType, TypeAlias, cast, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
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
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


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
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

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
                "prev_free_block of fake_free_list_tail should always exist"
            )
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
            "prev_free_block of fake_free_list_tail should always exist"
        )
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
                "next_free_block of fake_free_list_head should always exist"
            )
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
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
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
    if last_pos.offset + last_pos.length <= start_token_idx:
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
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

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


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

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
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
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
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp * pcp``. Multiple groups: LCM of every
      group's block size — context parallelism is not supported here.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.hash_block_size`` override if set, else the GCD of group
      block sizes; every group's block size must be divisible by it. Returns
      the scheduler block size (i.e. disables finer hashing) if block hashing
      is inactive or a mamba group's block size diverges from the cache
      block size (mamba_cache_mode != "align").
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:  # Single group: block_size * dcp * pcp
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        raise ValueError(
            "Hybrid KV cache groups with multiple block sizes do not "
            "support context parallelism (dcp_world_size/pcp_world_size > 1)."
        )

    group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # Mamba groups with block_size != cache_config.block_size
    # (mamba_cache_mode != "align") break divisibility; back off to the
    # scheduler block size.
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size

    requested = cache_config.hash_block_size
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid hash_block_size={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by hash_block_size. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size


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

        if start_token_idx + block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

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
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
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
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
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
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def _pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    if all(
        isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in kv_cache_groups
    ):
        # DeepseekV4: shared layout sized by the largest per-page-size bucket.
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        layer_tuple_page_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).get_num_layer_tuples()
            for g in kv_cache_groups
        )
        return layer_tuple_page_bytes * num_layer_tuples
    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
    return page_size * group_size


def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
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
    return may_override_num_blocks(vllm_config, num_blocks)


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
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
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
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


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            b[specs[name].page_size_bytes].append(name)
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values())

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(
                KVCacheTensor(size=ps * num_blocks, shared_by=shared_by)
            )

    return num_blocks, kv_cache_tensors


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
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
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden sizes. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # DeepseekV4: UniformTypeKVCacheSpecs but multiple groups.
        # Delegate to the DeepseekV4-specific allocator.
        num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4(
            vllm_config, kv_cache_groups, available_memory
        )
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_swa_mla = any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    )

    uniform_block_size: int | None = None
    if has_swa_mla:
        # For DeepseekV4, block sizes can be different for different KV cache groups.
        # E.g., Full MLA: 256; SWA MLA: 64; C4 partial states: 4, C128 states: 8.
        assert has_full_attention
        any_full_spec = next(
            iter(
                spec
                for spec in kv_cache_spec.values()
                if isinstance(spec, FullAttentionSpec)
            )
        )
        uniform_block_size = any_full_spec.block_size

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowMLASpec):
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=uniform_block_size
                    if uniform_block_size is not None
                    else spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    page_size_padded=spec.page_size_padded,
                    cache_dtype_str=spec.cache_dtype_str,
                    alignment=spec.alignment,
                    compress_ratio=spec.compress_ratio,
                    model_version=spec.model_version,
                )
            elif isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    head_size_v=spec.head_size_v,
                    dtype=spec.dtype,
                    kv_quant_mode=spec.kv_quant_mode,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None

    mla_specs: dict[str, KVCacheSpec] = {}
    grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = defaultdict(
        dict
    )
    # NOTE: Here we group SWA layers by (block_size, sliding_window), which separates
    # SWA layers, C4I+C4A layers, and C128A layers into three different groups. It can
    # be fragile with only block_size and sliding_window as keys, but fine for now.
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec

    assert len(mla_specs) > 0
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform_spec is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [mla_uniform_spec, *swa_uniform_specs]


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]

    swa_mla_specs = grouped_specs[1:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, *swa_mla_groups]


def _annotate_eagle_groups_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    # Detection uses the merged MLA spec's model_version.
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return
    # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
    # group contains it.
    # FIXME(yifan): avoid/generalize this hacky check.
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
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

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups

    # Pull HiddenStateCacheSpec layers out before the general multi-group
    # path so they don't affect page-size unification or grouping.
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """
    Log resolved KV cache configuration.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_config: The resolved KV cache configuration
    """
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )

    # GPU KV cache size in tokens = max_concurrency * max_model_len: the total
    # tokens of context the pool can hold at peak utilization. Sourcing this
    # from the concurrency calculation handles hybrid layouts correctly: SWA /
    # chunked-local groups have a per-request block count that's capped by
    # their window, so a naive `num_blocks // num_groups * block_size` formula
    # underestimates capacity for these models. DCP/PCP sharding is already
    # accounted for in each spec's `max_memory_usage_bytes`.
    num_tokens = int(max_concurrency * max_model_len)

    logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{max_model_len:,}",
        max_concurrency,
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs.
        # They must already be page_size aligned and share a common padded
        # layer-tuple layout. Even groups with fewer actual tuples still reserve
        # the global number of tuple slots in the shared tensor layout.
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )

        total_max_mem_usage_bytes = 0
        for group in kv_cache_groups:
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes
        return total_max_mem_usage_bytes

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
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
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

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
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency, by concatenating consecutive
    hashes at `hash_block_size` to form each hash at `target_block_size`.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    concatenating two 16-size hashes yields one 32-size hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | AB   | CD    |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        base = idx * self.scale_factor
        end = base + self.scale_factor
        return BlockHash(b"".join(self.block_hashes[base:end]))


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
