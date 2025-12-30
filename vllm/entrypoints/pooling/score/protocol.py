# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
)

from vllm import PoolingParams
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.openai.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.score_utils import ScoreContentPartParam, ScoreMultiModalParam
from vllm.utils import random_uuid


class ScoreRequest(OpenAIBaseModel):
    model: str | None = None
    text_1: list[str] | str | ScoreMultiModalParam
    text_2: list[str] | str | ScoreMultiModalParam
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:score-extra-params]

    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )

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
        "Default is True.",
    )
    # --8<-- [end:score-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class RerankRequest(OpenAIBaseModel):
    model: str | None = None
    query: str | ScoreMultiModalParam
    documents: list[str] | ScoreMultiModalParam
    top_n: int = Field(default_factory=lambda: 0)
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:rerank-extra-params]

    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )

    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )

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
        "Default is True.",
    )
    # --8<-- [end:rerank-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class RerankDocument(BaseModel):
    text: str | None = None
    multi_modal: ScoreContentPartParam | None = None


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]


class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float


class ScoreResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo
