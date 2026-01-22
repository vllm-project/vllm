# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import (
    Field,
)

from vllm import PoolingParams
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    CompletionRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.utils import random_uuid
from vllm.utils.serial_utils import EmbedDType, EncodingFormat, Endianness


class EmbeddingCompletionRequest(PoolingBasicRequestMixin, CompletionRequestMixin):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings

    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None

    # --8<-- [start:embedding-extra-params]
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )
    # --8<-- [end:embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            dimensions=self.dimensions,
            use_activation=self.normalize,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
        )


class EmbeddingChatRequest(PoolingBasicRequestMixin, ChatRequestMixin):
    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None

    # --8<-- [start:chat-embedding-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )
    # --8<-- [end:chat-embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=self.normalize,
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
