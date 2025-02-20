# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, Tuple, Type, Union

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


class ParentRequest(ABC):
    """A base class to allow for requests that may have children.

    The parallel sub-class implements the `n` property and `get_child_info()`
    method to fan out the request with multiple children. The singular
    sub-class is just a thin wrapper of the request ID and params.
    """

    request_id: str
    params: Union[SamplingParams, PoolingParams]

    def __init__(self, request_id: str, params: Union[SamplingParams,
                                                      PoolingParams]) -> None:
        self.request_id = request_id
        self.params = params

    @property
    @abstractmethod
    def n(self) -> int:
        return 1

    @abstractmethod
    def get_child_info(
            self,
            index: int) -> Tuple[str, Union[SamplingParams, PoolingParams]]:
        pass

    @staticmethod
    def from_params(
            request_id: str, params: Union[SamplingParams,
                                           PoolingParams]) -> 'ParentRequest':
        cls: Type[ParentRequest] = SingularSamplingRequest
        if (isinstance(params, SamplingParams) and params.n is not None
                and params.n > 1):
            cls = ParallelSamplingRequest
        return cls(request_id, params)


class SingularSamplingRequest(ParentRequest):
    """A request with no fan-out child requests."""

    @property
    def n(self) -> int:
        return 1

    def get_child_info(
            self,
            index: int) -> Tuple[str, Union[SamplingParams, PoolingParams]]:
        return (self.request_id, self.params)


class ParallelSamplingRequest(ParentRequest):
    """Info, state & processing for parallel sampling request.
    
    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    cached_child_sampling_params: Optional[SamplingParams]

    def __init__(self, request_id: str,
                 sampling_params: SamplingParams) -> None:
        super().__init__(request_id, sampling_params)

        self.cached_child_sampling_params = None

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
        assert isinstance(self.params, SamplingParams)
        seed = self.params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> Tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        
        Args:
          index: index within `n` child requests.
        
        Returns:
          (request ID, sampling_params) tuple
        """
        return (f"{index}_{self.request_id}",
                self._get_child_sampling_params(index))

    @property
    def n(self) -> int:
        assert isinstance(self.params, SamplingParams)
        return self.params.n
