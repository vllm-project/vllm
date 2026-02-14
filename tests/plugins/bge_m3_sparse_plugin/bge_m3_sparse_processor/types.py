# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import CompletionRequestMixin


class SparseEmbeddingCompletionRequestMixin(CompletionRequestMixin):
    input: list[int] | list[list[int]] | str | list[str]
    return_token_id_texts_map: bool | None = Field(
        default=None,
        description="Whether to return dict shows the mapping of token_id to text."
        "`None` or False means not return.",
    )


class SparseEmbeddingTokenWeight(BaseModel):
    token_id: int
    weight: float
    token: str | None


class SparseEmbeddingResponseData(BaseModel):
    index: int
    object: str = "sparse-embedding"
    sparse_embedding: list[SparseEmbeddingTokenWeight]


class SparseEmbeddingResponse(BaseModel):
    data: list[SparseEmbeddingResponseData]
    usage: UsageInfo
