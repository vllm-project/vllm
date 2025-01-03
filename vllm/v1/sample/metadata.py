from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch


@dataclass
class LogitsProcessMetadata:

    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    no_top_p: bool
    no_top_k: bool
    no_penalties: bool


@dataclass
class SamplingMetadata:

    no_penalties: bool
    all_greedy: bool
    all_random: bool
    logits_process_metadata: LogitsProcessMetadata
    generators: Dict[int, torch.Generator]
    max_num_logprobs: int
    prompt_token_ids: Optional[torch.Tensor]
    output_token_ids: List[List[int]]
    min_tokens: List[int]
    stop_token_ids: List[Set[int]]


@dataclass
class PromptLogprobsMetadata:

    prompt_indices: torch.Tensor
    logits_process_metadata: LogitsProcessMetadata
