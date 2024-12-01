from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor
    top_k: torch.Tensor
    no_top_p: bool
    no_top_k: bool

    generators: Dict[int, torch.Generator]

    max_num_logprobs: int

    output_token_ids: List[List[int]]
    prompt_token_ids: List[List[int]]
    frequency_penalties: List[float]
    presence_penalties: List[float]
    repetition_penalties: List[float]
    min_tokens: List[int]
    stop_token_ids: List[List[int]]
