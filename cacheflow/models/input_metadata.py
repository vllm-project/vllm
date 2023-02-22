from typing import List

import torch


class InputMetadata:

    def __init__(
        self,
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        self.prompt_lens = prompt_lens
        self.prompt_block_table = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        self.num_prompts = len(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]
        self.max_num_blocks_per_seq = block_tables.shape[1]
        assert self.num_generation_tokens == block_tables.shape[0]
