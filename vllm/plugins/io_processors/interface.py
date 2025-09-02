# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Generic, Optional, TypeVar, Union

from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput

IOProcessorInput = TypeVar('IOProcessorInput')
IOProcessorOutput = TypeVar('IOProcessorOutput')


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(self,
                     model_output: Sequence[PoolingRequestOutput],
                     request_id: Optional[str] = None,
                     **kwargs) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to vLLM.
        # Let's sort them by id before post_processing
        sorted_output = sorted([(i, item) async for i, item in model_output],
                               key=lambda output: output[0])
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> IOProcessorInput:
        raise NotImplementedError

    @abstractmethod
    def output_to_response(
            self, plugin_output: IOProcessorOutput) -> IOProcessorResponse:
        raise NotImplementedError
