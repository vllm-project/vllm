from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs]
    logprob_token_ids: List[int]
    logprobs: List[int]


@dataclass
class PromptLogprobsOutput:

    # req_id -> [max_num_prompt_logprobs]
    logprob_token_ids: Dict[str, List[int]]
    logprobs: Dict[str, List[float]]


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

    # [num_reqs, max_num_logprobs]
    logprob_token_ids: List[List[int]]
    logprobs: List[List[float]]

    # req_id -> [max_num_prompt_logprobs]
    prompt_logprob_token_ids: Dict[str, List[int]]
    prompt_logprobs: Dict[str, List[float]]
