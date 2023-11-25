from typing import Dict, List, Optional, Tuple

import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData


class SamplingMetadata:
    """Metadata for sampling."""

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Dict[SamplingType, torch.Tensor],
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices

        self.max_prompt_len = max(prompt_lens) if prompt_lens else 0
        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = self.num_prompts * self.max_prompt_len

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (
            f'InputMetadata('
            f'num_prompt_tokens={self.num_prompt_tokens}, '
            f'num_prompts={self.num_prompts}, '
            f'prompt_lens={self.prompt_lens}, '
            f'num_generation_tokens={self.num_generation_tokens}, '
            f'context_lens={self.context_lens}, '
            f'max_context_len={self.max_context_len}), '
            f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
            f'block_tables={self.block_tables}, '
            f'selected_token_indices={self.selected_token_indices}, '
            f'categorized_sample_indices={self.categorized_sample_indices}, '
            f'slot_mapping={self.slot_mapping})')
