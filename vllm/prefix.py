from typing import Any, Dict, List, Sequence, Tuple, Optional
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

    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int
    ) -> None:
        self.token_ids = tuple(token_ids)
        self.block_size = block_size
        self.length = len(token_ids)
        self.hash = hash(token_ids)
        assert self.length % block_size == 0
        self.block_table: Optional[BlockTable] = None
        self.computed = False

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
        max_capacity: The maximum number of prefixes allowed in the pool at any given time. 
            The default value is None, which means there is no limit. It can only take positive
            values.
    """

    def __init__(
        self,
        block_size: int,
        max_capacity: Optional[int] = None
    ) -> None:
        self.prefixes: OrderedDict[int, Prefix] = OrderedDict()
        self.block_size = block_size
        
        if max_capacity is not None:
            # NOTE(to remove after consultation): I have also been thinking if we need to 
            # assert that max_capacity must be greater than or equal to the max allowed 
            # batch size. My own analysis has led me to believe that this is not necessary
            # because even if a prefix is removed from the pool before it even gets computed,
            # this will not stop the prefix from being computed. It only means that the prefix
            # will not stay allocated for next cycles of computation and future requests
            # with the prefix will have to recompute it.
            assert max_capacity > 0, "max_capacity must be positive."
        
        self.max_capacity = max_capacity

    def __len__(self) -> int:
        return len(self.prefixes)
    
    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        return tuple(token_ids[:new_length])

    def add_or_get_prefix(self, token_ids: Sequence[int]) -> Optional[Prefix]:
        """
        Adds a prefix to the pool if it does not already exist. If it does exist,
        it returns the existing prefix. If the pool is at max capacity, it removes
        the least recently used prefix before adding the new prefix.

        Notice that if the length of token_ids is less than the block_size, no 
        prefix is created and None is returned.
        """
        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            # Prefix is empty.
            return None
        
        # Check first if prefix exists, moving it to the end of the OrderedDict.
        # so that the LRU policy is maintained. Return the existing prefix.
        prefix = Prefix(token_ids, self.block_size)
        prefix_hash = hash(prefix)
        if prefix_hash in self.prefixes:
            prefix = self.prefixes[prefix_hash]
            self.prefixes.move_to_end(prefix_hash)
            return prefix
        
        # Prefix does not exist. Add created prefix to the pool and return it.
        # Always, before adding anything to the pool, checking the capacity constraints.
        if len(self.prefixes) == self.max_capacity:
            _ = self.prefixes.popitem(last=False)
        self.prefixes[prefix_hash] = prefix
        return prefix
