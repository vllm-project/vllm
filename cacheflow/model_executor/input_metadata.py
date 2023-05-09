from typing import List, Dict, Tuple

import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from cacheflow.sampling_params import SamplingParams


class InputMetadata:

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_logprobs: Dict[int, float],                         # Seq id -> cumulative logprobs.
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_logprobs = seq_logprobs
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        self.attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
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
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}), '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}), '
                f'slot_mapping={self.slot_mapping}')
