from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class LogitsProcessMetadata:

    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    no_top_p: bool
    no_top_k: bool


@dataclass
class SamplingMetadata:

    all_greedy: bool
    all_random: bool
    logits_process_metadata: LogitsProcessMetadata
    generators: Dict[int, torch.Generator]
    max_num_logprobs: int


@dataclass
class PromptLogprobsMetadata:

    prompt_indices: torch.Tensor
    logits_process_metadata: LogitsProcessMetadata
