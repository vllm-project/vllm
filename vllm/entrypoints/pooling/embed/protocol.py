# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import (
    Field,
)

from vllm import PoolingParams
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


def _get_max_total_output_tokens(
    model_config: ModelConfig,
) -> tuple[int | None, int]:
    max_total_tokens = model_config.max_model_len
    pooler_config = model_config.pooler_config

    if pooler_config is None:
        return max_total_tokens, 0

    if pooler_config.enable_chunked_processing:
        return None, 0

    max_embed_len = pooler_config.max_embed_len or max_total_tokens
    max_output_tokens = max_total_tokens - max_embed_len
    return max_total_tokens, max_output_tokens


class EmbeddingCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, EmbedRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        (
            max_total_tokens,
            max_output_tokens,
        ) = _get_max_total_output_tokens(model_config)

        return TokenizeParams(
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_output_tokens,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
            max_output_tokens_param="max_model_len - max_embed_len",
        )


class EmbeddingChatRequest(PoolingBasicRequestMixin, ChatRequestMixin):
    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None

    # --8<-- [start:chat-embedding-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    def to_pooling_params(self):
        return PoolingParams(
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
