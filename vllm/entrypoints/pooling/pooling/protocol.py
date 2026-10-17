# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Generic, TypeAlias, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.outputs import LateChunkingMetadata
from vllm.pooling_params import LateChunkingParams
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


class LateChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(gt=0, le=2**63 - 1, strict=True)


class PoolingCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    task: PoolingTask | None = None
    late_chunking_params: LateChunkingConfig | None = None

    @model_validator(mode="after")
    def check_late_chunking(self):
        if self.late_chunking_params is not None:
            if self.task not in (None, "token_embed"):
                raise ValueError("Late chunking requires token_embed")
            if self.encoding_format != "float":
                raise ValueError("Late chunking requires encoding_format='float'")
        return self

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
            late_chunking_params=(
                LateChunkingParams(chunk_size=self.late_chunking_params.chunk_size)
                if self.late_chunking_params is not None
                else None
            ),
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
    late_chunking: LateChunkingMetadata | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


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
