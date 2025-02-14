# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams


class ParentRequestState:
    """Info and state for parallel sampling request.
    
    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    When stream mode is disabled, then `self.request_output`
    aggregates completions.
    """

    request_id: str
    sampling_params: SamplingParams
    request_output: Optional[RequestOutput] = None

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params

    def get_child_sampling_params(
        self,
        seed: Optional[int],
    ) -> SamplingParams:
        sampling_params = copy(self.sampling_params)
        sampling_params.n = 1
        sampling_params.seed = seed
        return sampling_params

    def add_output(
        self,
        child_req_output: RequestOutput,
        index: int,
    ) -> None:
        """Aggregate a parallel sampling child
        request output.
        
        Non-stream-mode (`output_kind == FINAL_ONLY`) 
        only. Inject correct parent request ID and
        completion index.

        Args:
          child_req_output: a single request output
                            from a parallel sampling
                            child request.   
          index: index within `n` child    
        """
        if self.request_output is None:
            # Save the first request output; reinstate
            # original request ID; metrics are not
            # supported for parallel sampling
            child_req_output.request_id = self.request_id
            child_req_output.metrics = None
            self.request_output = child_req_output
        else:
            # Aggregate additional completion into request
            # output
            new_completion = child_req_output.outputs[0]
            new_completion.index = index
            # Note: will be sorted by index later
            self.request_output.outputs.append(new_completion)

    def get_parent_request_output(self) -> RequestOutput:
        """Invariant: parent completion outputs sorted by index"""
        assert self.request_output is not None
        self.request_output.outputs = sorted(self.request_output.outputs,
                                             key=lambda x: x.index)
        return self.request_output

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
    def output_kind(self) -> RequestOutputKind:
        return self.sampling_params.output_kind


class ParallelSamplingOutputProcessor:
    """For parallel sampling requests,
    filter and transform child request
    outputs."""

    def __init__(
        self,
        parent_state: ParentRequestState,
    ) -> None:
        """Store parent request state."""
        self.parent_state = parent_state

    def process_output(
        self,
        child_req_output: RequestOutput,
        index: int,
    ) -> Optional[RequestOutput]:
        """Filter, aggregate and transform parallel sampling
        child request outputs.

        If the parent request has `stream=false`
        (`output_kind == FINAL_ONLY`), each child will also have
        `output_kind == FINAL_ONLY`. All child request outputs
        must be aggregated into a single request output, with
        multiple completions. This request output is only returned
        once `n` completions are aggregated.

        If the parent request has `stream=true`
        (`output_kind == DELTA`), each child will also have
        `output_kind == DELTA`. All child request outputs
        must be streamed directly to the caller.

        Args:
          child_req_output: a single child request output
          index: index within `n` child requests

        Returns:
          `None`, unless a processed request output is ready to
          send back to the caller.
        """
        if self.parent_state.output_kind != RequestOutputKind.FINAL_ONLY:
            # stream=true: return child completions immediately
            child_req_output.request_id = self.parent_state.request_id
            child_req_output.outputs[0].index = index
            return child_req_output

        # stream=false: aggregate child completions
        self.parent_state.add_output(child_req_output, index)
        if self.parent_state.num_completions == self.parent_state.n:
            # Return aggregated request output after obtaining
            # all completions
            return self.parent_state.get_parent_request_output()
        return None
