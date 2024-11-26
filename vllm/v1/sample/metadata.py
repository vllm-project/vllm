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

    # Max number of sample or prompt logprobs
    # (respectiely) at the batch level
    max_num_logprobs: int
    max_num_prompt_logprobs: int

    # Attributes which support logprob computation
    query_start_loc: Optional[torch.Tensor]
    num_query_tokens: Optional[torch.Tensor]
    num_input_tokens: int
    partial_req_index: int  # >0 if there is a partial request, -1 o/w
