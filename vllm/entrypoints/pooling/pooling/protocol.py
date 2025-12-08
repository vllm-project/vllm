# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Generic, TypeAlias, TypeVar

from pydantic import (
    Field,
)

from vllm import PoolingParams
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.openai.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from vllm.tasks import PoolingTask
from vllm.utils import random_uuid
from vllm.utils.serial_utils import EmbedDType, EncodingFormat, Endianness


class PoolingCompletionRequest(EmbeddingCompletionRequest):
    task: PoolingTask | None = None
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )
    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )
    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "If it is a classify or token_classify task, the default is True; "
        "for other tasks, this value should be None.",
    )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
            use_activation=get_use_activation(self),
        )


class PoolingChatRequest(EmbeddingChatRequest):
    task: PoolingTask | None = None
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )
    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )
    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "If it is a classify or token_classify task, the default is True; "
        "for other tasks, this value should be None.",
    )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
            use_activation=get_use_activation(self),
        )


T = TypeVar("T")


class IOProcessorRequest(OpenAIBaseModel, Generic[T]):
    model: str | None = None

    priority: int = Field(default=0)
    """
    The priority of the request (lower means earlier handling;
    default: 0). Any priority other than 0 will raise an error
    if the served model does not use priority scheduling.
    """
    data: T

    task: PoolingTask = "plugin"
    encoding_format: EncodingFormat = "float"
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

    def to_pooling_params(self):
        return PoolingParams()


class IOProcessorResponse(OpenAIBaseModel, Generic[T]):
    request_id: str | None = None
    """
    The request_id associated with this response
    """
    created_at: int = Field(default_factory=lambda: int(time.time()))

    data: T
    """
    When using plugins IOProcessor plugins, the actual output is generated
    by the plugin itself. Hence, we use a generic type for the response data
    """


PoolingRequest: TypeAlias = (
    PoolingCompletionRequest | PoolingChatRequest | IOProcessorRequest
)


class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str = "pooling"
    data: list[list[float]] | list[float] | str


class PoolingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"pool-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo


class PoolingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None = None
    media_type: str = "application/octet-stream"
