# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import TypeAlias

from pydantic import (
    Field,
)

from vllm import PoolingParams
from vllm.entrypoints.openai.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    EmbeddingRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.utils import random_uuid


class EmbeddingCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, EmbeddingRequestMixin
):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings

    def to_pooling_params(self):
        return PoolingParams(
            dimensions=self.dimensions,
            normalize=self.normalize,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
        )


class EmbeddingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbeddingRequestMixin
):
    def to_pooling_params(self):
        return PoolingParams(
            dimensions=self.dimensions,
            normalize=self.normalize,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
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
