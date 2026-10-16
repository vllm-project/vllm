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
    """Token embeddings with optional full-document late chunking."""

    late_chunking_params: LateChunkingConfig | None = None
    encoding_format: Literal["float"] = "float"
    dimensions: None = None

    @model_validator(mode="before")
    @classmethod
    def check_removed_parameters(cls, data):
        return reject_removed_pooling_parameters(data)

    def to_pooling_params(self):
        return PoolingParams(
            task="token_embed",
            use_activation=self.use_activation,
            late_chunking_params=(
                LateChunkingParams(chunk_size=self.late_chunking_params.chunk_size)
                if self.late_chunking_params is not None
                else None
            ),
        )


class TokenEmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[list[float]]
    late_chunking: LateChunkingMetadata | None = None


class TokenEmbeddingResponse(OpenAIBaseModel):
    """vLLM extension returning one matrix of token or chunk vectors per input."""

    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    data: list[TokenEmbeddingResponseData]
    usage: UsageInfo
