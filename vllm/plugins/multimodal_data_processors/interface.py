# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

from vllm.config import VllmConfig
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.multimodal_data_processors.types import (
    MultiModalRequestOutput)


class MultimodalDataProcessor(ABC):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        ...

    @abstractmethod
    async def pre_process_async(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        ...

    @abstractmethod
    def post_process(self,
                     model_out: Sequence[Optional[PoolingRequestOutput]],
                     request_id: Optional[str] = None,
                     **kwargs) -> Sequence[MultiModalRequestOutput]:
        ...

    @abstractmethod
    async def post_process_async(
        self,
        model_out: Sequence[Optional[PoolingRequestOutput]],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Sequence[MultiModalRequestOutput]:
        ...
