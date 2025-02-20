# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from vllm.v1.sample.logits_processor import LogitsProcessor


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool

    # None when there are no speculated tokens.
    spec_token_ids: Optional[List[List[int]]]

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]

    generators: Dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: List[List[int]]

    logits_procs: List[LogitsProcessor]
    nongreedy_logits_procs: List[LogitsProcessor]
