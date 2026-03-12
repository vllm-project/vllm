# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    CompletionRequestMixin,
    EmbedRequestMixin,
)


class SparseEmbeddingCompletionRequestMixin(CompletionRequestMixin, EmbedRequestMixin):
    return_tokens: bool | None = Field(
        default=None,
        description="Whether to return dict shows the mapping of token_id to text."
        "`None` or False means not return.",
    )

    def to_embed_requests(self) -> list[EmbedRequestMixin]:
        if isinstance(self.input, list):
            return [self] * len(self.input)
        return [self]


class SparseEmbeddingTokenWeight(BaseModel):
    token_id: int
    weight: float
    token: str | None


class SparseEmbeddingResponseData(BaseModel):
    index: int
    object: str = "sparse&dense"
    sparse_embedding: list[SparseEmbeddingTokenWeight]
    dense_embedding: list[float]


class SparseEmbeddingResponse(BaseModel):
    data: list[SparseEmbeddingResponseData]
    usage: UsageInfo
