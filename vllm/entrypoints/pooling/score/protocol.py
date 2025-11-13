# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias

from pydantic import (
    BaseModel,
)
from typing_extensions import Required, TypedDict

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
)
from vllm.entrypoints.openai.protocol_base import (
    ClassifyRequestMixin,
    MM_ProcessorRequestMixin,
    OpenAIBaseModel,
    PriorityRequestMixin,
)

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
    ClassifyRequestMixin, PriorityRequestMixin, MM_ProcessorRequestMixin
):
    model: str | None = None

    text_1: list[str] | str | ScoreMultiModalParam | None = None
    text_2: list[str] | str | ScoreMultiModalParam | None = None

    data_1: list[str] | str | ScoreMultiModalParam | None = None
    data_2: list[str] | str | ScoreMultiModalParam | None = None

    def __post_init__(self):
        self.data_1 = self.data_1 if self.data_1 is not None else self.text_1
        self.data_2 = self.data_2 if self.data_2 is not None else self.text_2


class RerankRequest(
    ClassifyRequestMixin, PriorityRequestMixin, MM_ProcessorRequestMixin
):
    model: str | None = None
    query: str | ScoreMultiModalParam
    documents: list[str] | ScoreMultiModalParam
    top_n: int = 0


class RerankDocument(BaseModel):
    text: str | None = None
    multi_modal: ScoreContentPartParam | None = None


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]
