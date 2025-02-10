# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any, Dict, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams


class ParentRequestState:
    request_id: str
    sampling_params: SamplingParams
    request_output: Optional[RequestOutput] = None

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params

    def get_child_sampling_params(
        self,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> SamplingParams:
        sampling_params = copy(self.sampling_params)
        if kwargs is not None:
            for kw in kwargs:
                setattr(sampling_params, kw, kwargs[kw])
        return sampling_params

    def add_output(
        self,
        child_req_output: RequestOutput,
    ) -> None:
        if self.request_output is None:
            # Save the first request output; reinstate
            # original request ID; metrics are not
            # supported for parallel sampling
            child_req_output.request_id = self.request_id
            child_req_output.metrics = None
            self.request_output = child_req_output
        else:
            # Add completion to the request output
            new_completion = child_req_output.outputs[0]
            new_completion.index = self.num_completions
            self.request_output.outputs.append(new_completion)

    def get_warmup_request_id(self) -> str:
        return "w_" + self.request_id

    def get_child_request_id(
        self,
        index: int,
    ) -> str:
        return str(index) + "_" + self.request_id

    @property
    def num_completions(self) -> int:
        assert self.request_output is not None
        return len(self.request_output.outputs)

    @property
    def n(self) -> int:
        return self.sampling_params.n

    @property
    def logprobs(self) -> Optional[int]:
        return self.sampling_params.logprobs

    @property
    def prompt_logprobs(self) -> Optional[int]:
        return self.sampling_params.prompt_logprobs

    @property
    def output_kind(self) -> RequestOutputKind:
        return self.sampling_params.output_kind


class ParallelSamplingOutputProcessor:

    def __init__(
        self,
        parent_state: ParentRequestState,
    ) -> None:
        self.parent_state = parent_state

    def process_output(
        self,
        child_req_output: RequestOutput,
        index: int,
    ) -> Optional[RequestOutput]:
        if self.parent_state.output_kind == RequestOutputKind.FINAL_ONLY:
            # stream=false: accumulate child completions
            self.parent_state.add_output(child_req_output)
            if self.parent_state.num_completions == self.parent_state.n:
                # Return accumulated request output after obtaining
                # all completions
                return self.parent_state.request_output
        else:
            # stream=true: return child completions immediately
            child_req_output.request_id = self.parent_state.request_id
            child_req_output.outputs[0].index = index
            return child_req_output

        return None
