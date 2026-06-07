# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TypeAlias

from pydantic import Field

from vllm import PoolingParams
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.logger import init_logger
from vllm.utils import random_uuid

from ..base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
    PoolingBasicRequestMixin,
)

logger = init_logger(__name__)


class ClassificationCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    def to_pooling_params(self):
        return PoolingParams(
            task="classify",
            use_activation=self.use_activation,
        )


class ClassificationChatRequest(
    PoolingBasicRequestMixin,
    ChatRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    def to_pooling_params(self):
        return PoolingParams(
            task="classify",
            use_activation=self.use_activation,
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
