from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: torch.Tensor

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]

    # [num_prompt_tokens, max_num_prompt_logprobs + 1]
    prompt_logprob_token_ids: Optional[torch.Tensor]
    # [num_prompt_tokens, max_num_prompt_logprobs + 1]
    prompt_logprobs: Optional[torch.Tensor]


@dataclass
class ModelRunnerOutput:

    # [num_reqs]
    req_ids: List[str]
    # req_id -> index
    req_id_to_index: Dict[str, int]

    # [num_reqs]
    sampled_token_ids_cpu: torch.Tensor

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids_cpu: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs_cpu: Optional[torch.Tensor]

    # [num_reqs, max_num_prompt_logprobs]
    prompt_logprob_token_ids_cpu: Optional[torch.Tensor]
    # [num_reqs, max_num_prompt_logprobs]
    prompt_logprobs_cpu: Optional[torch.Tensor]
