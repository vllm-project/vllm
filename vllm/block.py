"""Token blocks."""
from typing import List, Optional

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1


class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
        previous_block: Optional["LogicalTokenBlock"],
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        self.token_ids = [_BLANK_TOKEN_ID] * block_size
        self.num_tokens = 0

        self._previous_block = previous_block
        self._cached_hash = None

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, token_ids: List[int]) -> None:
        assert len(token_ids) <= self.get_num_empty_slots()
        curr_idx = self.num_tokens
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens]

    def get_last_token_id(self) -> int:
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]

    def maybe_get_content_hash(self) -> Optional[int]:
        """Return the content-based hash of the current block, or None if it is
        not yet defined.

        For the content-based hash to be defined, the current block must be
        full.
        """

        # If the hash is already computed, return it.
        if self._cached_hash is not None:
            return self._cached_hash

        # We cannot compute a hash for the current block because it is not full.
        if not self.is_full():
            return None

        is_first_block = self._previous_block is None
        prev_block_hash = (None if is_first_block else self._previous_block.maybe_get_content_hash(
        ))

        # Previous block exists but does not yet have a hash.
        # Return no hash in this case.
        if prev_block_hash is None and not is_first_block:
            return None

        self._cached_hash = LogicalTokenBlock.get_content_hash(
            is_first_block,
            prev_block_hash,
            cur_block_token_ids=self.token_ids)
        return self._cached_hash

    @staticmethod
    def get_content_hash(is_first_block: bool, prev_block_hash: Optional[int],
                         cur_block_token_ids: List[int]) -> int:
        """Computes a hash value corresponding to the contents of a block and
        the contents of the preceding block(s). The hash value is used for
        prefix caching.

        NOTE: Content-based hashing does not support LoRA.

        Parameters:
        - is_first_block (bool): A flag indicating if the block is the first in
            the sequence.
        - prev_block_hash (Optional[int]): The hash of the previous block. None
            if this is the first block.
        - cur_block_token_ids (List[int]): A list of token ids in the current
            block. The current block is assumed to be full.

        Returns:
        - int: The computed hash value for the block.
        """
        assert (prev_block_hash is None) == is_first_block
        return hash((is_first_block, prev_block_hash, *cur_block_token_ids))

class BlockMapping:
    pass

    def create_from_sequence(sequence):
        """Create a block mapping from the sequence.

        """
        pass



"""
BlockTable
    create_from_sequence(sequence) # for allocation of single sequence
    clone_from_blocktable(block table) # for allocation of SequenceGroup
    create_from_fork(block table, new sequence)

    append_slots(...)
        - need to identify missing logical->physical mapping
        - for each, need to fulfill
        - for any blocks that are sealed, need to maybe promote
        - for any blocks that are modified, need to check CoW.

    get_physical_blocks(...) # used by can_swap_out
        - return all physical blocks

    swap_in(...)
        - for each block in CPU, allocate a GPU block (use content hash!)
        - free the CPU block
        - (bad) if a block already has a destination, increment refcount

    swap_out(...)
        - same as swap_in but reversed

    free(...)
        - for each unique block, free it in the corresponding allocator.

    access_all_blocks_in_seq(...)
        - ??? unsure of design
        - need to update access time of all physical blocks

    compute_last_full_block_in_seq(...)
        - ??? unsure of design
        - mark the last full block as computed=True

    get_all_block_ids_till_computed
        - ??? unsure of design

LogicalBlock
    get_content_hash(...)
        - get last one, calculate hash

Allocator
    allocate(logical_block)
        - get a physical block for logical block
        - if logical block has a hash, then we can check existing content.
        - if existing content is found, increment the refcount.
        - otherwise, get a hashless block. potentially, remove one from cache.


missing things:
* freelists / 


"""

class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]
