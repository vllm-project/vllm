from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplerOutput:

    sampled_token_ids: torch.Tensor

    logprob_token_ids: Optional[torch.Tensor]
    logprobs: Optional[torch.Tensor]

    prompt_logprob_token_ids: Optional[torch.Tensor]
    prompt_logprobs: Optional[torch.Tensor]

    model_forward_time: float
    model_execute_time: float
