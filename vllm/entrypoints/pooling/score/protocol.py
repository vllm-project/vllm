# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Required, TypeAlias, TypedDict

from pydantic import (
    BaseModel,
    Field,
)

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
)
from vllm.entrypoints.openai.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ClassificationRequestMixin,
    MMProcessorRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.utils import random_uuid

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam | ChatCompletionContentPartImageEmbedsParam
)


class ScoreMultiModalParam(TypedDict, total=False):
    """
    A specialized parameter type for scoring multimodal content

    The reasons why don't reuse `CustomChatCompletionMessageParam` directly:
    1. Score tasks don't need the 'role' field (user/assistant/system) that's required in chat completions
    2. Including chat-specific fields would confuse users about their purpose in scoring
    3. This is a more focused interface that only exposes what's needed for scoring
    """  # noqa: E501

    content: Required[list[ScoreContentPartParam]]
    """The multimodal contents"""


class ScoreRequest(
    PoolingBasicRequestMixin, ClassificationRequestMixin, MMProcessorRequestMixin
):
    text_1: list[str] | str | ScoreMultiModalParam
    text_2: list[str] | str | ScoreMultiModalParam


class RerankRequest(
    PoolingBasicRequestMixin, ClassificationRequestMixin, MMProcessorRequestMixin
):
    query: str | ScoreMultiModalParam
    documents: list[str] | ScoreMultiModalParam
    top_n: int = Field(default_factory=lambda: 0)


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
