# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import AsyncGenerator, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams


class ParallelSamplingRequestManager:
    """Info, state & processing for parallel sampling request.
    
    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    Transform child request outputs into parent request
    outputs.
    When stream mode is disabled, then `self.request_output`
    aggregates child request completions.
    """

    request_id: str
    sampling_params: SamplingParams
    cached_child_sampling_params: Optional[SamplingParams]
    request_output: Optional[RequestOutput] = None

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.cached_child_sampling_params = None

    def get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then 
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        seed = self.sampling_params.seed
        if seed is None and self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        c_sampling_params = copy(self.sampling_params)
        c_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = c_sampling_params
        else:
            # Each child gets a clone with a unique seed
            c_sampling_params.seed = seed + index
        return c_sampling_params

    def _add_output(
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
        new_completion = child_req_output.outputs[0]
        new_completion.index = index
        if self.request_output is None:
            # Save the first request output; reinstate
            # original request ID; metrics are not
            # supported for parallel sampling
            child_req_output.request_id = self.request_id
            child_req_output.metrics = None
            self.request_output = child_req_output
        else:
            # Aggregate additional completion into request output
            # Note: will be sorted by index later
            self.request_output.outputs.append(new_completion)

    def _get_parent_request_output(self) -> RequestOutput:
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

    def _process_output(
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
        if self.output_kind != RequestOutputKind.FINAL_ONLY:
            # stream=true: return child completions immediately
            child_req_output.request_id = self.request_id
            child_req_output.outputs[0].index = index
            return child_req_output

        # stream=false: aggregate child completions
        self._add_output(child_req_output, index)
        if self.num_completions == self.n:
            # Return aggregated request output after obtaining
            # all completions
            return self._get_parent_request_output()
        return None

    async def parallel_sampling_child_gen(
        self,
        child_gen: AsyncGenerator[RequestOutput, None],
        index: int,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Output generator for a single parallel sampling
        child request.

        Each parallel sampling request triggers at
        least two child requests. This generator
        yields zero or more request outputs to
        return to the caller, as they become
        available.

        Args:
          child_gen: generator for child request
                     outputs.
          index: index within the `n` child requests

        Returns:
          Yields zero or more request outputs to return
          to the caller.
        """
        async for out in child_gen:
            if req_out := self._process_output(out, index):
                yield req_out

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
