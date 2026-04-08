# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    CompletionRequestMixin,
    EmbedRequestMixin,
)

EmbedTask = Literal[
    "sparse",
    "dense",
    "dense&sparse",
]

EMBED_TASKS: tuple[EmbedTask, ...] = get_args(EmbedTask)


class SparseEmbeddingCompletionRequestMixin(CompletionRequestMixin, EmbedRequestMixin):
    return_tokens: bool | None = Field(
        default=None,
        description="Whether to return dict shows the mapping of token_id to text."
        "`None` or False means not return.",
    )
    embed_task: EmbedTask = Field(
        default="dense&sparse",
        description="embed task, can be one of 'sparse', 'dense' , 'dense&sparse', "
        "default to 'dense&sparse'",
    )

    def to_embed_requests_offline(self) -> list[EmbedRequestMixin]:
        if isinstance(self.input, list):
            return [self] * len(self.input)
        return [self]

    def to_embed_requests_online(self) -> list[EmbedRequestMixin]:
        return [self]


class SparseEmbeddingTokenWeight(BaseModel):
    token_id: int
    weight: float
    token: str | None


class SparseEmbeddingResponseData(BaseModel):
    index: int
    object: str = "dense&sparse"
    sparse_embedding: list[SparseEmbeddingTokenWeight] | None
    dense_embedding: list[float] | None


class SparseEmbeddingResponse(BaseModel):
    data: list[SparseEmbeddingResponseData]
    usage: UsageInfo
