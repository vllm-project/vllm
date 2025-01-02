from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs]
    logprob_token_ids: List[List[int]]
    logprobs: List[List[int]]


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

    # req_id -> (prompt_logprobs_token_ids, prompt_logprobs)
    # [num_reqs, max_num_prompt_logprobs]
    prompt_logprobs: Dict[str, Tuple[List[List[int], List[List[float]]]]]
