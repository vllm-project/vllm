from dataclasses import dataclass
from typing import Dict

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

    min_tokens: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
