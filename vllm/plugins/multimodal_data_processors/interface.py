# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from vllm.config import VllmConfig
from vllm.inputs.data import MultiModalPromptType, PromptType
from vllm.outputs import MultiModalRequestOutput, PoolingRequestOutput


class MultimodalDataProcessor(ABC):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    @abstractmethod
    def pre_process(
        self,
        prompt: MultiModalPromptType,
        request_id: Optional[str] = None,
    ) -> Sequence[PromptType]:
        ...

    @abstractmethod
    def post_process(
        self,
        model_out: Sequence[PoolingRequestOutput],
        request_id: Optional[str] = None,
    ) -> MultiModalRequestOutput:
        ...
