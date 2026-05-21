# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.inputs import DataPrompt, EngineInput
from vllm.lora.request import LoRARequest
from vllm.tracing.otel import Tracer

from .classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from .embed.protocol import (
    CohereEmbedRequest,
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
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
from .scoring.typing import ScoringData

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: PoolingRequestT
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

    # for Observability
    trace_headers: Mapping[str, str] | None = None
    entrypoint_tracer: Tracer | None = None
    request_span_context: Any | None = None
    entrypoint_span_links: Any | None = None
    # time_ns() - monotonic_ns()
    time_offset: int = 0
    # timestamp time_ns
    arrival_time: int = 0
    preprocessing_finished: int = 0
    engine_call_finished: int = 0
    postprocessing_finished: int = 0

    ## for Long Text Embedding with Chunked Processing
    original_engine_inputs: Sequence[EngineInput] | None = None

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
