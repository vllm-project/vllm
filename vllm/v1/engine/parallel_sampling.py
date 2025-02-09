# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any, Dict, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams


class ParentRequestState:
    sampling_params: SamplingParams
    request_output: Optional[RequestOutput] = None

    def get_child_sampling_params(
        self,
        kwargs: Dict[str, Any] = {},
    ) -> SamplingParams:
        sampling_params = copy(self.sampling_params)
        for kw in kwargs:
            setattr(sampling_params, kw, kwargs[kw])
        return sampling_params

    def add_output(
        self,
        child_req_output: RequestOutput,
    ) -> None:
        if self.output_kind != RequestOutputKind.DELTA:
            pass

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
