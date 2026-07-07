# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest

from .classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from .embed.protocol import (
    CohereEmbedRequest,
    EmbeddingBytesResponse,
    EmbeddingChatInputRequest,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
)
from .pooling.protocol import (
    IOProcessorRequest,
    PoolingBytesResponse,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from .scoring.protocol import ScoringRequest, ScoringResponse

PoolingCompletionLikeRequest: TypeAlias = (
    EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | PoolingCompletionRequest
)

PoolingChatLikeRequest: TypeAlias = (
    EmbeddingChatRequest
    | EmbeddingChatInputRequest
    | ClassificationChatRequest
    | PoolingChatRequest
)

AnyPoolingRequest: TypeAlias = (
    EmbeddingRequest
    | PoolingCompletionLikeRequest
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

PromptGenerator: TypeAlias = Generator[tuple[dict[str, Any], dict[str, Any]]]
PromptFactory: TypeAlias = Callable[[], PromptGenerator]


@dataclass(kw_only=True)
class ChunkedEmbeddingMetadata:
    prompt_index: int
    chunk_index: int


@dataclass(kw_only=True)
class PoolingServeContext(Generic[PoolingRequestT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: PoolingRequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    pooling_params: PoolingParams | list[PoolingParams]
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: Sequence[EngineInput] | None = None
    prompt_request_ids: list[str] | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    ## for Long Text Embedding with Chunked Processing
    original_engine_inputs: Sequence[EngineInput] | None = None
    chunked_embedding_metadata: list[ChunkedEmbeddingMetadata] | None = None

    ## for bi-encoder & late-interaction
    n_queries: int | None = None

    ## for IOProcessorResponse
    response: Any | None = None

    ## for flash-late-interaction
    query_final_res_batch: list[PoolingRequestOutput] | None = None


@dataclass
class OfflineInputsContext:
    prompts: Sequence[PromptType]
    pooling_params: Sequence[PoolingParams]
    tokenization_kwargs: dict[str, Any]
    seq_lora_requests: Sequence[LoRARequest | None]
    priorities: Sequence[int]

    # for scoring
    chat_template: str | None = None

    ## for bi-encoder & late-interaction
    n_queries: int | None = None


@dataclass
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]

    ## for bi-encoder & late-interaction
    n_queries: int | None = None
