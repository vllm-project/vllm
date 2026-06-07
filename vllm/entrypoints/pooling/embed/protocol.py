# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedding API protocol models for OpenAI and Cohere formats.

OpenAI: https://platform.openai.com/docs/api-reference/embeddings
Cohere: https://docs.cohere.com/reference/embed
"""

import builtins
import struct
import time
from collections.abc import Sequence
from typing import Literal, TypeAlias

import pybase64 as base64
from pydantic import BaseModel, Field

from vllm import PoolingParams
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid

from ..base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    EmbeddingTokenizeParamsMixin,
    EmbedRequestMixin,
    PoolingBasicRequestMixin,
)


class EmbeddingCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    EmbeddingTokenizeParamsMixin,
):
    def to_pooling_params(self):
        return PoolingParams(
            task="embed",
            dimensions=self.dimensions,
            use_activation=self.use_activation,
        )


class EmbeddingChatRequest(
    PoolingBasicRequestMixin,
    ChatRequestMixin,
    EmbedRequestMixin,
    EmbeddingTokenizeParamsMixin,
):
    def to_pooling_params(self):
        return PoolingParams(
            task="embed",
            dimensions=self.dimensions,
            use_activation=self.use_activation,
        )


EmbeddingRequest: TypeAlias = EmbeddingCompletionRequest | EmbeddingChatRequest


# ---------------------------------------------------------------------------
# OpenAI /v1/embeddings — response models
# ---------------------------------------------------------------------------


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class EmbeddingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None = None
    media_type: str = "application/octet-stream"


# ---------------------------------------------------------------------------
# Cohere /v2/embed — request models
# ---------------------------------------------------------------------------

CohereEmbeddingType = Literal[
    "float",
    "binary",
    "ubinary",
    "base64",
]
CohereTruncate = Literal["NONE", "START", "END"]


class CohereEmbedContent(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, str] | None = None


class CohereEmbedInput(BaseModel):
    content: list[CohereEmbedContent]


class CohereEmbedRequest(BaseModel):
    model: str | None = None
    input_type: str | None = None
    texts: list[str] | None = None
    images: list[str] | None = None
    inputs: list[CohereEmbedInput] | None = None
    output_dimension: int | None = None
    embedding_types: list[CohereEmbeddingType] | None = None
    truncate: CohereTruncate = "END"
    max_tokens: int | None = None
    priority: int = 0


# ---------------------------------------------------------------------------
# Cohere /v2/embed — response models
# ---------------------------------------------------------------------------


class CohereApiVersion(BaseModel):
    version: str = "2"


class CohereBilledUnits(BaseModel):
    input_tokens: int | None = None
    image_tokens: int | None = None


class CohereMeta(BaseModel):
    api_version: CohereApiVersion = Field(default_factory=CohereApiVersion)
    billed_units: CohereBilledUnits | None = None


class CohereEmbedByTypeEmbeddings(BaseModel):
    # The field name ``float`` shadows the builtin type, so the annotation
    # must use ``builtins.float`` to avoid a self-referential type error.
    float: list[list[builtins.float]] | None = None
    binary: list[list[int]] | None = None
    ubinary: list[list[int]] | None = None
    base64: list[str] | None = None


class CohereEmbedResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    embeddings: CohereEmbedByTypeEmbeddings
    texts: list[str] | None = None
    meta: CohereMeta | None = None
    response_type: Literal["embeddings_by_type"] = "embeddings_by_type"


# ---------------------------------------------------------------------------
# Cohere embedding type conversion helpers
# ---------------------------------------------------------------------------

_UNSIGNED_TO_SIGNED_DIFF = 1 << 7  # 128


def _pack_binary_embeddings(
    float_embeddings: list[list[float]],
    signed: bool,
) -> list[list[int]]:
    """Bit-pack float embeddings: positive -> 1, negative -> 0.

    Each bit is shifted left by ``7 - idx%8``, and every 8 bits are packed
    into one byte.
    """
    result: list[list[int]] = []
    for embedding in float_embeddings:
        dim = len(embedding)
        if dim % 8 != 0:
            raise ValueError(
                "Embedding dimension must be a multiple of 8 for binary "
                f"embedding types, but got {dim}."
            )
        packed_len = dim // 8
        packed: list[int] = []
        byte_val = 0
        for idx, value in enumerate(embedding):
            bit = 1 if value >= 0 else 0
            byte_val += bit << (7 - idx % 8)
            if (idx + 1) % 8 == 0:
                if signed:
                    byte_val -= _UNSIGNED_TO_SIGNED_DIFF
                packed.append(byte_val)
                byte_val = 0
        assert len(packed) == packed_len
        result.append(packed)
    return result


def _encode_base64_embeddings(
    float_embeddings: list[list[float]],
) -> list[str]:
    """Encode float embeddings as base64 (little-endian float32)."""
    result: list[str] = []
    for embedding in float_embeddings:
        buf = struct.pack(f"<{len(embedding)}f", *embedding)
        result.append(base64.b64encode(buf).decode("utf-8"))
    return result


def build_typed_embeddings(
    float_embeddings: list[list[float]],
    embedding_types: Sequence[str],
) -> CohereEmbedByTypeEmbeddings:
    """Convert float embeddings to all requested Cohere embedding types."""
    result = CohereEmbedByTypeEmbeddings()

    for emb_type in embedding_types:
        if emb_type == "float":
            result.float = float_embeddings
        elif emb_type == "binary":
            result.binary = _pack_binary_embeddings(float_embeddings, signed=True)
        elif emb_type == "ubinary":
            result.ubinary = _pack_binary_embeddings(float_embeddings, signed=False)
        elif emb_type == "base64":
            result.base64 = _encode_base64_embeddings(float_embeddings)

    return result
