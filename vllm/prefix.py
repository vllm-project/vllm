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
        prefix_strings: str,
        prefix_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.prefix_id = prefix_id
        self.prompt = prefix_strings
        self.block_size = block_size

        self.token_ids = prefix_token_ids

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prefix_token_ids)

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

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

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return len(self.data)

    def get_token_ids(self) -> List[int]:
        return self.data

    def get_last_token_id(self) -> int:
        return self.data[-1]

class PrefixMetaData:
    def __init__(
            self,
            prefix_data: Dict[int, Prefix],
            block_tables: Dict[int, List[int]]) -> None:
        self.prefix_data = prefix_data
        self.block_tables = block_tables
