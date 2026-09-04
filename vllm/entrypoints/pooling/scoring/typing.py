# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TypeAlias

from typing_extensions import Required, TypedDict

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoParam,
)

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartTextParam
    | ChatCompletionContentPartVideoParam
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


# Raw input data with content key in ScoreMultiModalParam.
ScoreInput = str | ScoreMultiModalParam
# Score data without content key.
ScoreData = str | list[ScoreContentPartParam]


@dataclass
class ScoringData:
    data_1: list[ScoreData]
    data_2: list[ScoreData]
