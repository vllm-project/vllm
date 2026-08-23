# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Generic, TypeAlias, TypeVar

from pydantic import BeforeValidator, Field, model_validator

from vllm import PoolingParams, envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams
from vllm.tasks import PoolingTask
from vllm.utils import random_uuid

from ..base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    EncodingRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
    PoolingBasicRequestMixin,
    reject_removed_pooling_parameters,
)


class PoolingCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    task: PoolingTask | None = None

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
        )


class PoolingChatRequest(
    PoolingBasicRequestMixin,
    ChatRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    task: PoolingTask | None = None

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
        )


T = TypeVar("T")


class IOProcessorRequest(PoolingBasicRequestMixin, EncodingRequestMixin, Generic[T]):
    data: T
    task: PoolingTask = "plugin"

    @model_validator(mode="before")
    @classmethod
    def validate_data_list_length(cls, data):
        if not isinstance(data, dict):
            return data
        max_prompts = envs.VLLM_MAX_COMPLETION_PROMPTS
        data_val = data.get("data")
        if isinstance(data_val, list) and len(data_val) > max_prompts:
            raise VLLMValidationError(
                f"data list length {len(data_val)} exceeds the maximum "
                f"allowed count of {max_prompts}. To increase this limit, "
                "set the VLLM_MAX_COMPLETION_PROMPTS environment variable.",
                parameter="data",
            )
        return data

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return self._build_pooling_tok_params(
            model_config,
            add_special_tokens=not model_config.is_encoder_decoder,
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
        )

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
        )


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


PoolingRequest: TypeAlias = Annotated[
    PoolingCompletionRequest | PoolingChatRequest | IOProcessorRequest,
    BeforeValidator(reject_removed_pooling_parameters),
]


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
