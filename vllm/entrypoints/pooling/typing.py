# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
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
    PoolingBytesResponse,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.scoring.protocol import ScoringRequest, ScoringResponse
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.inputs import DataPrompt, EngineInput
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
    | ScoringRequest
    | CohereEmbedRequest
)

AnyPoolingResponse: TypeAlias = (
    ClassificationResponse
    | EmbeddingResponse
    | EmbeddingBytesResponse
    | PoolingResponse
    | PoolingBytesResponse
    | ScoringResponse
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
    pooling_params: PoolingParams | list[PoolingParams] | None = None
    engine_inputs: Sequence[EngineInput] | None = None
    prompt_request_ids: list[str] | None = None
    intermediates: Any | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ## for bi-encoder & late-interaction
    n_queries: int | None = None

    ## for IOProcessorResponse
    response: Any | None = None

    ## for flash-late-interaction
    query_final_res_batch: list[PoolingRequestOutput] | None = None


@dataclass
class OfflineInputsContext:
    prompts: PromptType | Sequence[PromptType] | DataPrompt | ScoringData
    pooling_params: PoolingParams | Sequence[PoolingParams]
    tokenization_kwargs: dict[str, Any] | None = None
    chat_template: str | None = None

    ## for bi-encoder & late-interaction
    n_queries: int | None = None


@dataclass
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]

    ## for bi-encoder & late-interaction
    n_queries: int | None = None
