# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.outputs import LateChunkingMetadata
from vllm.pooling_params import LateChunkingParams, PoolingParams
from vllm.utils import random_uuid

from ..base.protocol import (
    CompletionRequestMixin,
    EmbedRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
    PoolingBasicRequestMixin,
    reject_removed_pooling_parameters,
)


class LateChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(gt=0, le=2**63 - 1, strict=True)


class TokenEmbeddingRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    """Full-document late chunking on a token embedding server."""

    input: str | list[str]
    late_chunking_params: LateChunkingConfig
    encoding_format: Literal["float"] = "float"
    dimensions: None = None

    @model_validator(mode="before")
    @classmethod
    def check_late_chunking(cls, data):
        data = reject_removed_pooling_parameters(data)
        if isinstance(data, dict) and data.get("late_chunking_params") is None:
            raise ValueError(
                "late_chunking_params is required here; use /pooling with "
                "task='token_embed' for token embeddings."
            )
        return data

    def to_pooling_params(self):
        return PoolingParams(
            task="token_embed",
            use_activation=self.use_activation,
            late_chunking_params=LateChunkingParams(
                chunk_size=self.late_chunking_params.chunk_size
            ),
        )


class TokenEmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[list[float]]
    late_chunking: LateChunkingMetadata


class TokenEmbeddingResponse(OpenAIBaseModel):
    """vLLM extension returning one matrix of chunk vectors per document."""

    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    data: list[TokenEmbeddingResponseData]
    usage: UsageInfo
