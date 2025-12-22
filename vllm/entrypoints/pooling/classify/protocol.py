# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TypeAlias

from pydantic import (
    Field,
)

from vllm.entrypoints.openai.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    ClassificationRequestMixin,
    CompletionRequestMixin,
    MMProcessorRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.utils import random_uuid


class ClassificationCompletionRequest(
    PoolingBasicRequestMixin, ClassificationRequestMixin, CompletionRequestMixin
):
    pass


class ClassificationChatRequest(
    PoolingBasicRequestMixin,
    ClassificationRequestMixin,
    ChatRequestMixin,
    MMProcessorRequestMixin,
):
    pass


ClassificationRequest: TypeAlias = (
    ClassificationCompletionRequest | ClassificationChatRequest
)


class ClassificationData(OpenAIBaseModel):
    index: int
    label: str | None
    probs: list[float]
    num_classes: int


class ClassificationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"classify-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ClassificationData]
    usage: UsageInfo
