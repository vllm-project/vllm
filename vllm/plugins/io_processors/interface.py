# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorPluginResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput


class IOProcessor(ABC):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompt: Any,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: Any,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(self,
                     model_out: Sequence[Optional[PoolingRequestOutput]],
                     request_id: Optional[str] = None,
                     **kwargs) -> Any:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_out: Sequence[Optional[PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        return self.post_process(model_out, request_id, **kwargs)

    @abstractmethod
    def parse_request(self, request: Any) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def plugin_out_to_response(self,
                               plugin_out: Any) -> IOProcessorPluginResponse:
        raise NotImplementedError
