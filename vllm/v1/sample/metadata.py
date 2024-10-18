from dataclasses import dataclass
from typing import List, Optional

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

    generators: List[Optional[torch.Generator]]
    no_generator: bool

    max_num_logprobs: int
