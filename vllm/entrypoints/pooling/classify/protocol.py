# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any, TypeAlias

from pydantic import Field

from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


class ClassificationCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, ClassifyRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )


class ClassificationChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, ClassifyRequestMixin
):
    # --8<-- [start:chat-classification-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
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
