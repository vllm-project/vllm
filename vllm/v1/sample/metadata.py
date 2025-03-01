# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class SamplingMetadata:

    temperature: Optional[torch.Tensor]
    all_greedy: bool
    all_random: bool

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]
    min_p: Optional[torch.Tensor]

    generators: Dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: List[List[int]]

    # req_index -> (min_tokens, stop_token_ids)
    min_tokens: Dict[int, Tuple[int, Set[int]]]

    logit_bias: List[Optional[Dict[int, float]]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]
