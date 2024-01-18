from typing import Dict, List, Sequence, Tuple, Optional

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
        block_size: int,
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

    def set_block_table(self, block_table: BlockTable) -> None:
        self.block_table = block_table.copy()


class PrefixPool:
    """Manages all the prompt prefixes.

    NOTE: This feature is experimental and may be replaced with automatic
        prefix caching in the future.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
    """

    def __init__(
        self,
        block_size: int,
    ) -> None:
        # TODO(zhuohan): Add a capacity limit to the prefix pool.
        self.prefixes: Dict[int, Prefix] = {}
        self.block_size = block_size

    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        return tuple(token_ids[:new_length])

    def add_or_get_prefix(self, token_ids: Sequence[int]) -> Optional[Prefix]:
        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            # Prefix is empty.
            return None
        prefix = Prefix(token_ids, self.block_size)
        prefix_hash = hash(prefix)
        if prefix_hash not in self.prefixes:
            self.prefixes[prefix_hash] = prefix
        return self.prefixes[prefix_hash]
