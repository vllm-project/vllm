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

    # These two parameters are for allowed_token_ids.
    # `has_allowed_token_ids`` is a bool list, the length is the max batch size.
    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    has_allowed_token_ids: List[bool]
    allowed_token_ids_mask: torch.Tensor
