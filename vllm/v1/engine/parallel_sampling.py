# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Callable, Optional, Union

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import IterationStats


class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    sampling_params: SamplingParams

    # To track the completion of child requests
    child_requests: set[str]

    # To aggregate child completions when not streaming
    output_aggregator: Optional[RequestOutput]

    # To find the max number of generated tokens across all children
    max_num_generation_tokens: int

    # To efficiently obtain child sampling params
    cached_child_sampling_params: Optional[SamplingParams]

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = None
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    @classmethod
    def from_params(
        cls,
        request_id: str,
        params: Union[SamplingParams, PoolingParams],
    ) -> Optional['ParentRequest']:
        if not isinstance(params, SamplingParams) or params.n == 1:
            return None
        return cls(request_id, params)

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

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        
        Args:
          index: index within `n` child requests.
        
        Returns:
          (request ID, sampling_params) tuple
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return (child_req_id, self._get_child_sampling_params(index))

    def finish_child_request(self, req_id: str):
        self.child_requests.remove(req_id)

    @property
    def n(self) -> int:
        return self.sampling_params.n

    def make_request_output(
        self,
        final_only: bool,
        completion_output: CompletionOutput,
        new_request_output: Callable[[str], RequestOutput],
    ) -> Optional[RequestOutput]:
        # Use an existing RequestOutput if we're aggregating
        request_output = self.output_aggregator

        # Make new RequestOutput otherwise
        if request_output is None:
            request_output = new_request_output(self.request_id)

        # Add a new completion
        request_output.outputs.append(completion_output)

        # If not streaming, aggregate until all child requests complete
        if final_only and len(request_output.outputs) != self.n:
            self.output_aggregator = request_output
            return None

        # We're done aggregating
        self.output_aggregator = None

        # Parent completion output list must be sorted by index
        request_output.outputs = sorted(request_output.outputs,
                                        key=lambda x: x.index)
        return request_output

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        self.max_num_generation_tokens = max(num_generation_tokens,
                                             self.max_num_generation_tokens)
        return self.max_num_generation_tokens

    @staticmethod
    def observe_finished_request(parent_req: Optional['ParentRequest'],
                                 iteration_stats: IterationStats,
                                 num_generation_tokens: int):

        n_param = parent_req.n if parent_req is not None else 1

        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens)

        # Child requests finished, we can now record to iteration stats
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(
                num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)
