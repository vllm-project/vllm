"""KV-Cache Utilities."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block and the token IDs in the block.
    The reason we keep a tuple of token IDs is to make sure no hash
    collision happens when the hash value is the same.
    """
    hash_value: int
    token_ids: Tuple[int, ...]


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


def hash_block_tokens(parent_block_hash: Optional[int],
                      curr_block_token_ids: Sequence[int]) -> BlockHashType:
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

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    return BlockHashType(hash((parent_block_hash, *curr_block_token_ids)),
                         tuple(curr_block_token_ids))


def hash_request_tokens(block_size: int,
                        token_ids: Sequence[int]) -> List[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        token_ids: A sequence of token ids in the request.

    Returns:
        The list of computed hash values.
    """
    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break
        block_hash = hash_block_tokens(parent_block_hash_value,
                                       block_token_ids)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret
