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

    # req_id -> mask of indices each prompt logprob
    logits_masks: Dict[str, numpy.ndarray[bool]]

    # Logits process metadata for all elts of the batch
    logits_process_metadata: LogitsProcessMetadata
