from typing import Dict, List, Tuple

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

        self.num_prompts = len(prompt_lens)
