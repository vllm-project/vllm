# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import Field

from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


class EmbeddingCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, EmbedRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        pooler_config = model_config.pooler_config

        if pooler_config:
            if pooler_config.enable_chunked_processing:
                max_length = None
            else:
                max_length = pooler_config.max_embed_len
        else:
            max_length = model_config.max_model_len

        return TokenizeParams.from_config(
            model_config,
            max_length=max_length,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
        )


class EmbeddingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin
):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        pooler_config = model_config.pooler_config

        if pooler_config:
            if pooler_config.enable_chunked_processing:
                max_length = None
            else:
                max_length = pooler_config.max_embed_len
        else:
            max_length = model_config.max_model_len

        return TokenizeParams.from_config(
            model_config,
            max_length=max_length,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
        )


EmbeddingRequest: TypeAlias = EmbeddingCompletionRequest | EmbeddingChatRequest


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class EmbeddingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None = None
    media_type: str = "application/octet-stream"
