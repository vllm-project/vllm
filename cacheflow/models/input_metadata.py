from typing import List, Dict, Tuple

import torch

from cacheflow.sampling_params import SamplingParams


class InputMetadata:

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_logprobs: Dict[int, float],                         # Seq id -> cumulative logprobs.
        prompt_lens: List[int],
        cumulative_prompt_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_logprobs = seq_logprobs
        self.prompt_lens = prompt_lens
        self.cumulative_prompt_lens = cumulative_prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.max_prompt_len = max(prompt_lens) if prompt_lens else 0
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        if block_tables.numel() > 0:
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        assert block_tables.shape[0] == self.num_generation_tokens
        assert context_lens.shape[0] == self.num_generation_tokens

    def __repr__(self) -> str:
        return (f'InputMetadata('
                f'num_prompts={self.num_prompts}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'max_prompt_len={self.max_prompt_len}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'max_context_len={self.max_context_len}), '
                f'prompt_lens={self.prompt_lens}, '
                f'cumulative_prompt_lens={self.cumulative_prompt_lens}, '
                f'slot_mapping={self.slot_mapping}, '
                f'context_lens={self.context_lens}, '
                f'block_tables={self.block_tables})')
