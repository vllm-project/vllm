# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import CompletionRequestMixin

InputType = Literal["query", "document"]
INPUT_TYPES: tuple[InputType, ...] = get_args(InputType)
QUERY_MAXLEN = 32


class ColBERTEmbeddingCompletionRequestMixin(CompletionRequestMixin):
    input_type: InputType = Field(
        description="Whether to encode the input as a ColBERT 'query' "
        f"(query marker + [mask] expansion to {QUERY_MAXLEN} tokens) or as a "
        "'document' (document marker only). Required.",
    )


class ColBERTEmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    input_type: InputType
    embedding: list[list[float]]


class ColBERTEmbeddingResponse(BaseModel):
    data: list[ColBERTEmbeddingResponseData]
    usage: UsageInfo
