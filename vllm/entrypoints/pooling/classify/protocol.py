# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any, TypeAlias

from pydantic import Field

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizationParams
from vllm.utils import random_uuid


class ClassificationCompletionRequest(PoolingBasicRequestMixin, CompletionRequestMixin):
    # --8<-- [start:classification-extra-params]
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )
    # --8<-- [end:classification-extra-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizationParams:
        return TokenizationParams.from_config(
            model_config,
            max_length=model_config.max_model_len,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
        )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class ClassificationChatRequest(PoolingBasicRequestMixin, ChatRequestMixin):
    # --8<-- [start:chat-classification-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )
    # --8<-- [end:chat-classification-extra-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizationParams:
        return TokenizationParams.from_config(
            model_config,
            max_length=model_config.max_model_len,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
        )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


ClassificationRequest: TypeAlias = (
    ClassificationCompletionRequest | ClassificationChatRequest
)


class ClassificationData(OpenAIBaseModel):
    index: int
    label: str | None
    probs: list[float]
    num_classes: int


class ClassificationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"classify-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ClassificationData]
    usage: UsageInfo
