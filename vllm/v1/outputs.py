from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy.typing as npt
import torch


@dataclass
class PromptLogprobsOutput:

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use List instead.
@dataclass
class ModelRunnerOutput:

    # [num_reqs]
    req_ids: List[str]
    # req_id -> index
    req_id_to_index: Dict[str, int]

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    batch_logprob_token_ids_cpu: Optional[npt.NDArray]
    # [num_reqs, max_num_logprobs + 1]
    batch_logprobs_cpu: Optional[npt.NDArray]

    # [num_reqs, max_num_prompt_logprobs]
    batch_prompt_logprob_token_ids_cpu: Optional[npt.NDArray]
    # [num_reqs, max_num_prompt_logprobs]
    batch_prompt_logprobs_cpu: Optional[npt.NDArray]
