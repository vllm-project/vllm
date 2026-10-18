# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Generic, TypeAlias, TypeVar

from pydantic import BaseModel, BeforeValidator, Field, field_validator

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
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


class DataLocation(BaseModel):
    locator: str


class DataRef(BaseModel):
    object_id: str

    location: DataLocation


class MetadataRef(BaseModel):
    """A reference to per-request metadata stored by a connector.
    Analogous to :class:`DataRef` for data payloads: carries the
    ``request_id`` and a ``location`` whose ``locator`` encodes the
    remote address (e.g. ``tcp://host:port`` for P2P, ``host:port``
    for gRPC) from which the metadata can be fetched.
    """

    request_id: str

    location: DataLocation


class DataObject(BaseModel, Generic[T]):
    data: T


class GeneratePoolingRequest(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    token_ids: list[int] = [1]
    """The token ids to generate text from. Default: [1]"""

    @field_validator("token_ids")
    @classmethod
    def validate_token_ids(cls, v: list[int]) -> list[int]:
        if any(t < 0 for t in v):
            raise ValueError("token_ids must not contain negative values")
        return v

    data_ref: DataRef | None = None

    metadata_ref: MetadataRef | None = None

    data: DataObject | None = None

    metadata: dict | None = None

    model: str | None = None

    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    priority: int = Field(
        default=0,
        ge=-(2**63),
        le=2**63 - 1,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )

    def to_pooling_params(self):
        return PoolingParams(
            task="plugin",
        )


class GeneratePoolingResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    prompt_token_ids: list[int] = [1]
    """The token ids to generate text from. Default: [1]"""

    output_data_ref: DataRef | None = None

    output_metadata_ref: MetadataRef | None = None

    output_data: DataObject | None = None

    output_metadata: dict | None = None

    num_cached_tokens: int = 0
    """The number of tokens with prefix cache hit."""


class PostProcessPoolingRequest(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )

    model: str | None = None

    responses: list[GeneratePoolingResponse]

    def to_pooling_params(self):
        return PoolingParams(
            task="plugin",
        )
