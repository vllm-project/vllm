# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Optional, Protocol, Union

from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import merge_async_iterators


class AsyncGenerateMethodType(Protocol):

    def __call__(self,
                 prompt: PromptType,
                 sampling_params: SamplingParams,
                 request_id: str,
                 lora_request: Optional[LoRARequest] = None,
                 trace_headers: Optional[Mapping[str, str]] = None,
                 prompt_adapter_request: Optional[PromptAdapterRequest] = None,
                 priority: int = 0) -> AsyncGenerator[RequestOutput, None]:
        ...


class SyncAddRequestMethodType(Protocol):

    def __call__(self,
                 request_id: str,
                 prompt: PromptType,
                 params: Union[SamplingParams, PoolingParams],
                 arrival_time: Optional[float] = None,
                 lora_request: Optional[LoRARequest] = None,
                 trace_headers: Optional[Mapping[str, str]] = None,
                 prompt_adapter_request: Optional[PromptAdapterRequest] = None,
                 priority: int = 0) -> None:
        ...


class ParallelSamplingRequest:
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
    request_output: Optional[RequestOutput]
    num_finished_completions: int

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.cached_child_sampling_params = None
        self.request_output = None
        self.num_finished_completions = 0

    def _get_child_sampling_params(
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
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

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
        self.num_finished_completions += 1
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

    def _get_final_request_output(self) -> RequestOutput:
        """Invariant: parent completion outputs sorted by index"""
        assert self.request_output is not None
        self.request_output.finished = True
        self.request_output.outputs = sorted(self.request_output.outputs,
                                             key=lambda x: x.index)
        return self.request_output

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        
        Args:
          index: index within `n` child requests.
        
        Returns:
          (request ID, sampling_params) tuple
        """
        return (f"{index}_{self.request_id}",
                self._get_child_sampling_params(index))

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
        if self.output_kind != RequestOutputKind.FINAL_ONLY:
            # stream=true: return child completions immediately
            child_req_output.request_id = self.request_id
            child_req_output.outputs[0].index = index
            if child_req_output.finished:
                # Parent request is complete if all child requests are
                # complete.
                self.num_finished_completions += 1
                child_req_output.finished = (
                    self.num_finished_completions == self.n)
            return child_req_output

        # stream=false: aggregate child completions
        self._add_output(child_req_output, index)
        if self.num_finished_completions == self.n:
            # Return aggregated request output after obtaining
            # all completions
            return self._get_final_request_output()
        return None

    async def wrap_child_async_generator(
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
            if req_out := self.process_output(out, index):
                yield req_out

    @property
    def n(self) -> int:
        return self.sampling_params.n

    @property
    def output_kind(self) -> RequestOutputKind:
        return self.sampling_params.output_kind


class SyncParallelSamplingManager:

    def __init__(self):
        # Parent req ID -> parent request manager
        self.parent_reqs: dict[str, ParallelSamplingRequest] = {}
        # Child req ID -> (child req index, parent req ID)
        self.child_reqs: dict[str, tuple[int, str]] = {}

    def _register_parent_request(self, req: ParallelSamplingRequest) -> None:
        """Register parallel sampling parent request."""
        self.parent_reqs[req.request_id] = req

    def _register_child_request(self, req_id: str, child_req_id: str,
                                index: int) -> None:
        """Register parallel sampling child request with parent.
        
        Args:
          req_id: parent request ID
          child_req_id: child request ID
          index: child request index within `n` child requests
        """
        self.child_reqs[child_req_id] = (index, req_id)

    def get_num_unfinished_requests(self, num_core_reqs: int) -> int:
        """Get the number of unfinished requests, correcting for parallel
           sampling.
        
        Args:
          num_core_reqs: The number of unfinished requests in the engine core.
        
        Returns:
          Number of unfinished requests, where each parallel sampling req 
          counts as 1
        """
        return num_core_reqs + len(self.parent_reqs) - len(self.child_reqs)

    def add_request_parallel_sampling(
        self,
        add_request: SyncAddRequestMethodType,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        """Add sync parallel sampling request."""
        req = ParallelSamplingRequest(request_id, params)
        self._register_parent_request(req)
        # Add n child requests with unique request IDs & random seeds and n=1
        for idx in range(req.n):
            child_req_id, child_params = req.get_child_info(idx)
            self._register_child_request(request_id, child_req_id, idx)
            add_request(request_id=child_req_id,
                        prompt=prompt,
                        params=child_params,
                        arrival_time=arrival_time,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=priority)  # type: ignore

    def step(
        self,
        outputs: list[RequestOutput],
    ) -> list[RequestOutput]:
        """Build parallel sampling request outputs.
        
        Extract child request outputs, aggregate them
        into parent request output, and return parent
        output when complete.

        Do not modify `n=1` requests.

        Args:
          outputs: step request outputs. Mix of child request
                   outputs & `n=1` request outputs.

        Return:
          List of parallel sampling parent request outputs &
          unmodified `n=1` request outputs passed-thru from input.
        """
        if not (self.parent_reqs and outputs):
            # Return unmodified
            return outputs
        agg_outputs = []
        for output in outputs:
            req_id = output.request_id
            if child_req_entry := self.child_reqs.get(req_id, None):
                # For each parallel sampling child request output:
                (index, parent_req_id) = child_req_entry
                req = self.parent_reqs[parent_req_id]
                # Update parallel sampling request
                if out := req.process_output(output, index):
                    # Return parent request output if complete;
                    # cleanup parent request bookkeeping.
                    agg_outputs.append(out)
                    del self.parent_reqs[parent_req_id]
                # Cleanup child request bookkeeping.
                del self.child_reqs[req_id]
            else:
                # Not a parallel sampling request output
                agg_outputs.append(output)
        return agg_outputs


async def generate_parallel_sampling_async(
    generate: AsyncGenerateMethodType,
    prompt: PromptType,
    sampling_params: SamplingParams,
    request_id: str,
    lora_request: Optional[LoRARequest] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    priority: int = 0,
) -> AsyncGenerator[RequestOutput, None]:
    """Generate completions for async parallel sampling requests."""
    parent_req = ParallelSamplingRequest(request_id, sampling_params)

    # Aggregate generators for n child requests
    gens: list[AsyncGenerator[RequestOutput, None]] = []
    for idx in range(parent_req.n):
        child_req_id, child_params = parent_req.get_child_info(idx)
        child_gen = generate(
            prompt=prompt,
            sampling_params=child_params,
            request_id=child_req_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )  # type: ignore
        gen = parent_req.wrap_child_async_generator(child_gen, idx)
        gens.append(gen)

    # Merge generators
    async for _, out in merge_async_iterators(*gens):
        yield out
