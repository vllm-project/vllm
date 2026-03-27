# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingRequestOutput, PromptType
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreRequest,
    ScoreResponse,
)
from vllm.entrypoints.pooling.scoring.typing import ScoringOfflineInputs
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest

PoolingCompletionLikeRequest: TypeAlias = (
    EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | PoolingCompletionRequest
)

PoolingChatLikeRequest: TypeAlias = (
    EmbeddingChatRequest | ClassificationChatRequest | PoolingChatRequest
)

AnyPoolingRequest: TypeAlias = (
    PoolingCompletionLikeRequest
    | PoolingChatLikeRequest
    | IOProcessorRequest
    | RerankRequest
    | ScoreRequest
    | CohereEmbedRequest
)

AnyPoolingResponse: TypeAlias = (
    ClassificationResponse
    | EmbeddingResponse
    | EmbeddingBytesResponse
    | PoolingResponse
    | ScoreResponse
)

PoolingRequestT = TypeVar("PoolingRequestT", bound=AnyPoolingRequest)


@dataclass(kw_only=True)
class PoolingServeContext(Generic[PoolingRequestT]):
    request: PoolingRequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None

    engine_inputs: list[EngineInput] | None = None
    prompt_request_ids: list[str] | None = None
    intermediates: Any | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class OfflineInputsContext:
    prompts: PromptType | Sequence[PromptType] | ScoringOfflineInputs
    tokenization_kwargs: dict[str, Any] | None = None
    intermediates: Any | None = None


@dataclass
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]
    intermediates: Any | None = None
