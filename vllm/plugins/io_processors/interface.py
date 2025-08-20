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
        ...

    @abstractmethod
    async def pre_process_async(
        self,
        prompt: Any,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        ...

    @abstractmethod
    def post_process(self,
                     model_out: Sequence[Optional[PoolingRequestOutput]],
                     request_id: Optional[str] = None,
                     **kwargs) -> Any:
        ...

    @abstractmethod
    async def post_process_async(
        self,
        model_out: Sequence[Optional[PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        ...

    @abstractmethod
    def parse_request(self, request: Any) -> Optional[Any]:
        ...

    @abstractmethod
    def plugin_out_to_response(self,
                               plugin_out: Any) -> IOProcessorPluginResponse:
        ...
