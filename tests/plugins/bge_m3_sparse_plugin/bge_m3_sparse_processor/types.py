# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Annotated

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import CompletionRequestMixin


class SparseEmbeddingCompletionRequestMixin(CompletionRequestMixin):
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None


class SparseEmbeddingResponseData(BaseModel):
    index: int
    object: str = "sparse-embedding"
    sparse_embedding: dict[int, float]


class SparseEmbeddingResponse(BaseModel):
    request_id: str | None
    data: list[SparseEmbeddingResponseData]
    usage: UsageInfo
