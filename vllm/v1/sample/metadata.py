from dataclasses import dataclass
from typing import Dict, Optional

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
    max_num_prompt_logprobs: int

    num_query_tokens: Optional[torch.Tensor] = None
    maybe_sample_logits_indices: Optional[torch.Tensor] = None
    prompt_logits_mask: Optional[torch.Tensor] = None
