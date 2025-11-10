# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from collections.abc import AsyncGenerator, Sequence
from enum import Enum
from typing import Any, Generic, TypeVar

from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessorPluginType(Enum):
    INPUT_OUTPUT = 1
    INPUT_ONLY = 2
    OUTPUT_ONLY = 3


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    plugin_type = IOProcessorPluginType.INPUT_OUTPUT

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        if self.plugin_type != IOProcessorPluginType.OUTPUT_ONLY:
            raise NotImplementedError(
                ".pre_process() is not implemented for this IO plugin"
            )
        raise AssertionError(
            ".pre_process should not be invoked for output only plugins"
        )

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        if self.plugin_type != IOProcessorPluginType.INPUT_ONLY:
            raise NotImplementedError(
                ".post_process() is not implemented for this IO plugin"
            )
        raise AssertionError(
            ".post_process() should not be invoked for input only plugins"
        )

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
        return self.post_process(collected_output, request_id, **kwargs)

    def parse_request(
        self,
        request: Any,
        has_preprocess_partial: bool = False,
    ) -> IOProcessorInput:
        if self.plugin_type != IOProcessorPluginType.OUTPUT_ONLY:
            raise NotImplementedError(
                ".parse_request() is not implemented for this IO plugin"
            )
        raise AssertionError(
            ".parse_request() should not be invoked for output only plugins"
        )

    def validate_or_generate_params(
        self, params: SamplingParams | PoolingParams | None = None
    ) -> SamplingParams | PoolingParams:
        return params or PoolingParams()

    def output_to_response(
        self, plugin_output: IOProcessorOutput
    ) -> IOProcessorResponse:
        if self.plugin_type != IOProcessorPluginType.INPUT_ONLY:
            raise NotImplementedError(
                ".output_to_response() is not implemented for this IO plugin"
            )
        raise AssertionError(
            ".output_to_response() should not be invoked for input only plugins"
        )

    def get_modified_lora_request(self, engine_prompts, lora_request):
        # FIXME - clean this up, unify sync/async
        return lora_request
