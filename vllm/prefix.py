from typing import Any, List, Sequence, Tuple, Optional
from collections import OrderedDict

from vllm.block import BlockTable


class Prefix:
    """Data and states associated with a prefix of prompt tokens for multiple
    sequence groups.

    NOTE: This feature is experimental and may be replaced with automatic
        prefix caching in the future.

    Args:
        token_ids: The token ids of the prefix.
        block_size: The block size of the executed model.
    """

    def __init__(self, token_ids: Sequence[int], block_size: int) -> None:
        self.token_ids = tuple(token_ids)
        self.block_size = block_size
        self.length = len(token_ids)
        self.hash = hash(token_ids)
        assert self.length % block_size == 0
        self.block_table: Optional[BlockTable] = None
        self.computed = False

        # Contains a reference count of the number of sequences that share this
        # prefix, regardless of whether they are swapped out or not.
        # Must not be initialized to 1 at creation time because a prefix might be created
        # and thrown away, or sequences sharing this prefix might never be allocated.
        self.seq_ref_count = 0
        self.expired = False

    @property
    def allocated(self) -> bool:
        return self.block_table is not None

    def get_num_blocks(self) -> int:
        return self.length // self.block_size

    def get_block_numbers(self) -> List[int]:
        return [block.block_number for block in self.block_table]

    def get_length(self) -> int:
        return self.length

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Prefix):
            return False
        return self.hash == other.hash

    def set_block_table(self, block_table: BlockTable) -> None:
        self.block_table = block_table.copy()


class PrefixPool:
    """Manages all the prompt prefixes. If the max_capacity argument is not None,
    the pool will act as a LRU cache and remove the least recently used prefix once
    the capacity is reached.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
        max_capacity_in_blocks: The maximum number of blocks that can be used for prefixes in the pool
            at any given time. The default value is 0, which effectively means there is no prefix cache.
            If the value is float('inf'), is means that the capacity of the pool is unbounded (not recommended).
    """

    def __init__(self,
                 block_size: int,
                 max_capacity_in_blocks: int | float = 0) -> None:
        self.prefixes: OrderedDict[int, Prefix] = OrderedDict()
        self.block_size = block_size
        self._current_block_usage = 0

        self.max_allowed_prefix_length = float('inf')
        if max_capacity_in_blocks < float('inf'):
            assert max_capacity_in_blocks >= 0, "max_capacity must be non-negative."
            self.max_allowed_prefix_length = self.block_size * max_capacity_in_blocks

        self.max_capacity_in_blocks = max_capacity_in_blocks

        self._candidates_to_deallocate: List[Prefix] = []

    def __len__(self) -> int:
        """
        Returns the number of prefixes in the pool.
        """
        return len(self.prefixes)

    @property
    def current_block_usage(self) -> int:
        """
        Returns the number of blocks currently used by the pool.
        """
        return self._current_block_usage

    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        new_length = min(new_length, self.max_allowed_prefix_length)
        return tuple(token_ids[:new_length])

    def add_or_get_prefix(self,
                          token_ids: Sequence[int],
                          lora_int_id: int = 0) -> Optional[Prefix]:
        """
        Arguments:
        - token_ids: The token ids of the prefix to add to the pool.
        - lora_int_id: The lora_int_id of the request, which will be used to hash the prefix too.
            If the lora_int_id is not given, defaults to 0.
        
        Adds a prefix to the pool if it does not already exist. If it does exist,
        it returns the existing prefix. If the pool is at max capacity, it removes
        the least recently used prefix before adding the new prefix.

        There are two situations when None is returned:
        1. If the length of the token_ids of the prefix is less than the block_size, no 
        prefix is created and None is returned.
        2. If the max_capacity of the pool is 0, then no prefix is created and None is returned.

        There is also two situations where the prefix is shortened to fit block boundaries:
        1. If the length of the token_ids of the prefix is not a multiple of the block_size.
        2. If the number of blocks needed to allocate the prefix exceeds the max_capacity of the pool,
        the prefix is shortened to fit the max_capacity. Notice that this second occurence happens once
        we have already attempted all other recourses to be able to allocate the prefix on its entirity, such
        as evicting older prefixes from the pool. 
        """
        if self.max_capacity_in_blocks == 0:
            # Prefix cache is disabled.
            return None

        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            # Prefix is empty.
            return None

        # Check first if prefix exists, moving it to the end of the OrderedDict.
        # so that the LRU policy is maintained. Return the existing prefix.
        prefix = Prefix(token_ids, self.block_size)
        prefix_hash = hash((prefix, lora_int_id))
        if prefix_hash in self.prefixes:
            prefix = self.prefixes[prefix_hash]
            self.prefixes.move_to_end(prefix_hash)
            return prefix

        # Prefix does not exist. Add created prefix to the pool and return it.
        # Always, before adding anything to the pool, check the capacity constraints and
        # remove the least recently used prefix if capacity constraints are violated.
        prefix_num_blocks = prefix.get_num_blocks()

        while self._current_block_usage > 0 and prefix_num_blocks > self.max_capacity_in_blocks - self._current_block_usage:
            _, candidate_prefix = self.prefixes.popitem(last=False)
            self._candidates_to_deallocate.append(candidate_prefix)
            self._current_block_usage -= candidate_prefix.get_num_blocks()

        self.prefixes[prefix_hash] = prefix
        self._current_block_usage += prefix_num_blocks
        return prefix

    def get_prefixes_to_deallocate(self) -> List[Prefix]:
        """
        Returns a list of prefixes that are ready to be deallocated.
        For a prefix to be deallocated, it must fulfill the following two conditions:
        1. It must have been allocated already.
        2. It must have a seq_ref_count of 0.

        Condition number 1 is not evident, but is necessary because of the following rare situation:
        1. Prefix A is created, added to the pool, and assigned to a sequence group S.
            Sequence group S becomes part of the sequence groups waiting to be allocated
        2. At some point in the future, while sequence group S is still waiting to be allocated,
            prefix A is removed from the pool because of capacity constraints.
        3. If we remove prefix A from the self._candidates_to_free list at this point,
            we will end up with a memory leak because of the following situation:
            3.1. Sequence group S eventually gets allocated, altogether with prefix A,
                which is no longer in any data structure in the pool.
            3.2 Prefix A memory will never be removed from the GPU, even if its seq_ref_count
                reaches 0 in the future, because it is not in the pool anymore and that 
                means that the .get_prefixes_to_free() function will not return it.
        """
        indexes_to_remove = [
            i for i, prefix in enumerate(self._candidates_to_deallocate)
            if prefix.seq_ref_count == 0 and prefix.allocated
        ]

        # Mark the prefix as expired, so that if a sequence group still in the
        # waiting list that shares this prefix tries to allocate it as a prefix,
        # it will fail.
        for i in indexes_to_remove:
            prefix = self._candidates_to_deallocate[i]
            prefix.expired = True

        # Popping needs to happen with the indexes_to_remove list in reverse order
        # so that we don't get Index out of range errors
        return [
            self._candidates_to_deallocate.pop(i)
            for i in indexes_to_remove[::-1]
        ]
