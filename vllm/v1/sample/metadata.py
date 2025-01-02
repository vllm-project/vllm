from dataclasses import dataclass
from typing import Dict, List

import numpy
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

    sample_indicies: torch.Tensor
    all_greedy: bool
    all_random: bool
    logits_process_metadata: LogitsProcessMetadata
    generators: Dict[int, torch.Generator]
    max_num_logprobs: int


@dataclass
class PromptLogprobsMetadata:
    
    req_ids: List[str]
    masks: List[int]
    logits_process_metadatas: List[LogitsProcessMetadata]
    num_prompt_logprobs: List[int]
