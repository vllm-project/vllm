from typing import List, Optional


class Prefix:
    """Data and states associated with a prefix of prompt tokens for multiple sequence groups.

    Args:
        prefix_id: The id of the prefix in the prefix pool.
        token_ids: The token ids of the prefix.
        block_size: The block size of the executed model.
    
    Attributes:
        on_gpu: True if the prefix will be on GPU before the execution of the model.
        on_cpu: True if the prefix is on CPU.
        swap_to_gpu: True when the prefix will be computed during the execution of the model.
    """

    def __init__(
        self,
        prefix_id: int,
        token_ids: List[int],
        block_size: int,
    ) -> None:
        self.prefix_id = prefix_id
        self.token_ids = token_ids
        self.block_size = block_size
        self.length = len(token_ids)
        assert self.length % block_size == 0
        self.on_gpu = False
        self.on_cpu = False
        self.block_table: Optional[List[int]] = None
        # a lock to prevent multiple sequence from calculating the same prefix
        self.swap_to_gpu = False

    def get_num_blocks(self) -> int:
        return self.length // self.block_size

    def get_block_numbers(self) -> List[int]:
        return [block.block_number for block in self.block_table]

    def match(self, tokens: List[int]) -> bool:
        return tokens[:self.length] == self.token_ids

    # whether the prefix is on GPU or not
    def get_status(self) -> bool:
        return self.on_gpu

    def get_length(self) -> int:
        return self.length


class PrefixPool:
    """Manages all the prompt prefixes.

    Args:
        block_size: The block size of the executed model.
    
    Attributes:
        prefixes: A list of all the prefixes.
        prefixes_hash: Mapping from the hash of the prefix to the prefix id.
        block_size: The block size of the executed model.
    """

    def __init__(
        self,
        block_size: int,
    ) -> None:
        self.prefixes = []
        self.prefixes_hash = {}
        self.block_size = block_size

    def add_prefix(self, token_ids: List[int]) -> Prefix:
        # generate prefix_id
        prefix_id = len(self.prefixes)
        # create a new prefix
        prefix = Prefix(prefix_id, token_ids, self.block_size)
        self.prefixes.append(prefix)
        prefix_hash = hash(tuple(prefix.token_ids))
        self.prefixes_hash[prefix_hash] = prefix.prefix_id
        return prefix

    # use this first, if we already know from the application which part of the tokens are prefix.
    def fixed_search(self, prefix_hash: int) -> Optional[Prefix]:
        if prefix_hash not in self.prefixes_hash:
            return None
        prefix_id = self.prefixes_hash[prefix_hash]
        return self.prefixes[prefix_id]
