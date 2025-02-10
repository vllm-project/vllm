# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import torch


class LogprobsLists(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: List[List[int]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: List[List[float]]
    # [num_reqs]
    sampled_token_ranks: List[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )


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
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: Dict[str, LogprobsTensors]
