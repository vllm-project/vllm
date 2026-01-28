# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any, TypeAlias

from pydantic import (
    Field,
)

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.utils import random_uuid


class ClassificationCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, ClassifyRequestMixin
):
    pass


class ClassificationChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, ClassifyRequestMixin
):
    # --8<-- [start:chat-classification-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )


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
