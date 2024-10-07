from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from vllm.request import Request


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: torch.Tensor

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]

    # TODO: Support prompt logprobs.
    prompt_logprob_token_ids: Optional[torch.Tensor]
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


@dataclass
class CompletionOutput:

    index: int
    text: str
    token_ids: List[int]
    logprobs: Optional[Dict[int, float]]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None

    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool

    @classmethod
    def from_request(cls, request: Request) -> "RequestOutput":
        # TODO: Support `n` > 1.
        completion_output = CompletionOutput(
            index=0,
            text="",
            token_ids=request.output_token_ids,
            logprobs=None,
            finish_reason=request.get_finished_reason(),
            stop_reason=request.stop_reason,
        )
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            outputs=[completion_output],
            finished=request.is_finished(),
        )
