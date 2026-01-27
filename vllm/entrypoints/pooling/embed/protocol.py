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
        max_total_tokens = model_config.max_model_len
        pooler_config = model_config.pooler_config
        encoder_config = model_config.encoder_config or {}

        if pooler_config:
            if pooler_config.enable_chunked_processing:
                max_embed_len = max_total_tokens
            else:
                max_embed_len = pooler_config.max_embed_len or max_total_tokens
        else:
            max_embed_len = max_total_tokens

        return TokenizeParams(
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_total_tokens - max_embed_len,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
            max_output_tokens_param="max_model_len - max_embed_len",
        )


class EmbeddingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin
):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        max_total_tokens = model_config.max_model_len
        pooler_config = model_config.pooler_config
        encoder_config = model_config.encoder_config or {}

        if pooler_config:
            if pooler_config.enable_chunked_processing:
                max_embed_len = max_total_tokens
            else:
                max_embed_len = pooler_config.max_embed_len or max_total_tokens
        else:
            max_embed_len = max_total_tokens

        return TokenizeParams(
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_total_tokens - max_embed_len,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
            max_output_tokens_param="max_model_len - max_embed_len",
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
