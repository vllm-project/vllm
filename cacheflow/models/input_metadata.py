from typing import List

import torch


class InputMetadata:

    def __init__(
        self,
        seq_ids: List[int],
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        # FIXME: Rename
        max_context_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        self.seq_ids = seq_ids
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        self.num_prompts = len(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        if block_tables.numel() > 0:
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        assert self.num_generation_tokens == block_tables.shape[0]
        assert self.num_prompts + self.num_generation_tokens == len(seq_ids)

    def __repr__(self) -> str:
        return (f'InputMetadata('
                f'seq_ids={self.seq_ids}, '
                f'num_prompts={self.num_prompts}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'max_context_len={self.max_context_len})')
