# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch

from vllm.logits_process import LogitsProcessor


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

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: List[List[int]]
    min_tokens: List[int]
    stop_token_ids: List[Set[int]]

    logits_processors: List[Optional[List[LogitsProcessor]]]
    prompt_token_ids_cpu: List[torch.Tensor]
