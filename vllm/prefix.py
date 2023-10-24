"""Prefix and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union

from vllm.block import LogicalTokenBlock

class Prefix:
    """Stores the data, status, and block information of a prefix.

    Args:
        prefix_id: The ID of the prefix.
        prompt: The prompt of the prefix.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the prefix. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        prefix_id: int,
        prefix_string: str,
        prefix_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.prefix_id = prefix_id
        self.prefix_string = prefix_string
        self.block_size = block_size

        self.token_ids = prefix_token_ids

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prefix_token_ids)

    def get_prefix_offset(self):
        return self.logical_token_blocks[-1].get_num_empty_slots()

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def get_len(self) -> int:
        return len(self.token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids

    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

class PrefixMetaData:
    def __init__(
            self,
            prefix_data: Dict[int, Prefix],
            block_tables: Dict[int, List[int]]) -> None:
        self.prefix_data = prefix_data
        self.block_tables = block_tables
