# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool
    rejection_sampling: bool
    spec_token_ids: List[List[int]]

    top_p: torch.Tensor
    top_k: torch.Tensor
    no_top_p: bool
    no_top_k: bool
    min_p: torch.Tensor
    no_min_p: bool

    generators: Dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: List[List[int]]
    min_tokens: List[int]
    stop_token_ids: List[Set[int]]

    logit_bias: List[Optional[Dict[int, float]]]
