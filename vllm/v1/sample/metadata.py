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

    # Indicies in the batch needing 
    sample_indicies: torch.Tensor
    all_greedy: bool
    all_random: bool
    logits_process_metadata: LogitsProcessMetadata
    generators: Dict[int, torch.Generator]
    max_num_logprobs: int


@dataclass
class PromptLogprobsMetadata:

    # Mask of the indices needed for prompt logprobs.
    prompt_logprobs_mask: numpy.ndarray[bool]

    # Note: req_ids must be in order of the requests
    # in prompt_indicies.
    req_ids: List[str]
    prompt_lens: Dict[str, int]

    logits_process_metadata: LogitsProcessMetadata
    max_num_logprobs: int
