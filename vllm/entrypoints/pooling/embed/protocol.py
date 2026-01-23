# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import (
    Field,
)

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
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    pass


class EmbeddingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin
):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
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
