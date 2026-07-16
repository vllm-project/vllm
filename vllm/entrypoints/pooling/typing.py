# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypedDict, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.inputs import DataPrompt, EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers import ChatParams, TokenizeParams
from vllm.renderers.inputs import DictPrompt

from ...tasks import PoolingTask
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
from .scoring.typing import ScoreData, ScoringData

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
    pooling_task: PoolingTask
    tokenization_kwargs: dict[str, Any] | None
    lora_request: Sequence[LoRARequest | None] | None
    priorities: Sequence[int] | None


@dataclass
class OfflineEncodeInputsContext(OfflineInputsContext):
    prompts: PromptType | Sequence[PromptType]
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


@dataclass
class OfflineScoringInputsContext(OfflineInputsContext):
    scoring_data: ScoringData
    chat_template: str | None
    pooling_params: PoolingParams


@dataclass
class OfflinePluginInputsContext(OfflineInputsContext):
    prompts: DataPrompt
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


ALLOfflineInputsContext: TypeAlias = (
    OfflineEncodeInputsContext
    | OfflineScoringInputsContext
    | OfflinePluginInputsContext
)


@dataclass
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]

    ## for bi-encoder & late-interaction
    n_queries: int | None = None


class RenderParams(TypedDict):
    tok_params: TokenizeParams
    prompt_extras: dict[str, Any] | None
    skip_mm_cache: bool

    params: PoolingParams
    lora_requests: LoRARequest | None
    priorities: int


class EncodeCMPLRenderParams(RenderParams):
    prompts: DictPrompt


class EncodeChatRenderParams(RenderParams):
    conversations: list["ChatCompletionMessageParam"]
    chat_params: ChatParams


class ScoringRenderParams(RenderParams):
    data_1: ScoreData
    data_2: ScoreData
    chat_template: str | None


class PoolingEngineInput(TypedDict):
    prompts: EngineInput
    params: PoolingParams
    lora_requests: LoRARequest | None
    priorities: int


RequestGenerator: TypeAlias = Generator[
    EncodeCMPLRenderParams | EncodeChatRenderParams | ScoringRenderParams
]
RequestFactory: TypeAlias = Callable[[], RequestGenerator]
