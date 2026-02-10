# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Generic, TypeVar

from vllm.config import VllmConfig
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()

        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        if callable(parse_request := getattr(self, "parse_request", None)):
            warnings.warn(
                "`parse_request` has been renamed to `parse_data`. "
                "Please update your IO Processor Plugin to use the new name. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return parse_request(data)  # type: ignore

        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        if callable(
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore

        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        if callable(
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore

        return params or PoolingParams()

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to vLLM.
        # Let's sort them by id before post_processing
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, **kwargs)
