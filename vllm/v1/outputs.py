# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import torch


class LogprobsTensors(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Optional[LogprobsTensors]


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
    logprob_token_ids: Optional[List[List[int]]]
    logprobs: Optional[List[List[float]]]
    # [num_reqs]
    sampled_token_ranks: Optional[List[int]]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: Dict[str, LogprobsTensors]
