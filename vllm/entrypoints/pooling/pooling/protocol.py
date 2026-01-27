# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic import Field

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    EncodingRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams
from vllm.tasks import PoolingTask
from vllm.utils import random_uuid


class PoolingCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
):
    task: PoolingTask | None = None

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )


class PoolingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin, ClassifyRequestMixin
):
    task: PoolingTask | None = None

    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )


T = TypeVar("T")


class IOProcessorRequest(PoolingBasicRequestMixin, EncodingRequestMixin, Generic[T]):
    data: T
    task: PoolingTask = "plugin"

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
