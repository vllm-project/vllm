"""KV-Cache Utilities."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. But please note that 
    hash collisions can still theoretically occur, albeit with an extremely 
    low probability.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: Tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashType]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashType):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None


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

    def __init__(self, blocks: List[KVCacheBlock]) -> None:
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

    def get_all_free_blocks(self) -> List[KVCacheBlock]:
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


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> Tuple[Optional[Tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).
    For multi-modal inputs, the extra keys are (mm_hash, start_offset) that
    indicate a mm input contained in the block and its starting offset in
    the block tokens.
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return None, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    if mm_positions[-1]["offset"] + mm_positions[-1][
            "length"] < start_token_idx:
        return None, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    extra_keys = []
    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
        offset = mm_positions[curr_mm_idx]["offset"]
        length = mm_positions[curr_mm_idx]["length"]
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
    return tuple(extra_keys), curr_mm_idx


def hash_block_tokens(
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[Tuple[Any, ...]] = None) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    TODO: Support arbitrary metadata so that we could support more
    features such as LoRA adapter.

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
    return BlockHashType(hash((parent_block_hash, *curr_block_token_ids)),
                         tuple(curr_block_token_ids), extra_keys)


def hash_request_tokens(block_size: int,
                        request: Request) -> List[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids
    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match.")

    # TODO: Extend this to support other features such as LoRA.
    need_extra_keys = bool(mm_positions)
    extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        # Add extra keys if the block is a multi-modal block.
        if need_extra_keys:
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(parent_block_hash_value,
                                       block_token_ids, extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: KVCacheSpec,
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at 
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
        needed_memory += layer_spec.bytes_for_tokens(max_model_len)

    if needed_memory > available_memory:
        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/1024/1024/1024:.2f} GB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/1024/1024/1024:.2f} GB). Try "
            f"increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def is_kv_cache_type_uniform(kv_cache_spec: KVCacheSpec) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The KVCacheSpec of the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: KVCacheSpec,
                                      available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    logger.info("# GPU blocks: %d", num_blocks)

    per_layer_size = page_size * num_blocks

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        tensors={
            layer_name: KVCacheTensor(size=per_layer_size)
            for layer_name in kv_cache_spec
        },
        groups=[[layer_name for layer_name in kv_cache_spec]],
        kv_cache_spec=kv_cache_spec)
    return kv_cache_config


def get_kv_cache_config(vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec,
                        available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model
    TODO: support hybrid models with more than one type of KV cache.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for most models.
        # Allocate the same amount of memory for each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)
    else:
        raise NotImplementedError
