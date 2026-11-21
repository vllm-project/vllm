# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, NotRequired, TypeAlias, TypedDict, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.exceptions import VLLMValidationError
from vllm.inputs import DataPrompt, EngineInput
from vllm.lora.request import LoRARequest
from vllm.outputs import LateChunkingMetadata
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.renderers.inputs import DictPrompt
from vllm.tasks import PoolingTask

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
from .scoring.types import ScoreData, ScoringData
from .token_embed.protocol import TokenEmbeddingRequest, TokenEmbeddingResponse

PoolingCompletionLikeRequest: TypeAlias = (
    EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | PoolingCompletionRequest
    | TokenEmbeddingRequest
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
    | TokenEmbeddingResponse
)

PoolingRequestT = TypeVar("PoolingRequestT", bound=AnyPoolingRequest)


@dataclass(kw_only=True, slots=True)
class ChunkedEmbeddingMetadata:
    prompt_index: int
    chunk_index: int


@dataclass(slots=True)
class LateChunkingSlot:
    """Per-document source ranges shared by frontend rendering and output contexts."""

    metadata: LateChunkingMetadata | None = None


@dataclass(kw_only=True, slots=True)
class PoolingServeContext(Generic[PoolingRequestT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: PoolingRequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    pooling_params: PoolingParams
    lora_request: LoRARequest | None
    priorities: int | Sequence[int] | None
    prompt_extras: dict[str, Any] | None

    created_time: int = field(default_factory=lambda: int(time.time()))
    engine_inputs: Sequence["PoolingEngineInput"] | None = None
    prompt_request_ids: list[str] | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)
    late_chunking: list[LateChunkingSlot] = field(default_factory=list)

    ## for Long Text Embedding with Chunked Processing
    original_engine_inputs: Sequence["PoolingEngineInput"] | None = None
    chunked_embedding_metadata: list[ChunkedEmbeddingMetadata] | None = None

    ## for bi-encoder & late-interaction
    n_queries: int | None = None

    ## for IOProcessorResponse
    response: Any | None = None

    ## for flash-late-interaction
    query_final_res_batch: list[PoolingRequestOutput] | None = None
    late_interaction_query_keys: list[str] | None = None
    late_interaction_doc_keys: list[str] | None = None


@dataclass(kw_only=True, slots=True)
class OfflineInputsContext:
    pooling_task: PoolingTask
    lora_request: Sequence[LoRARequest | None] | None = None
    priorities: int | Sequence[int] | None = None
    late_chunking: list[LateChunkingSlot | None] = field(default_factory=list)

    # tok_params:
    # Tri-state: `None` means "not specified" and falls back to the per-path default
    add_special_tokens: bool | None = None
    tokenization_kwargs: dict[str, Any] | None = None

    # chat_params:
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    # Tri-state: `None` means "not specified" and falls back to default
    add_generation_prompt: bool | None = None
    continue_final_message: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    # mm_params:
    skip_mm_cache: bool = False
    cache_salt: str | None = None
    media_io_kwargs: dict[str, dict[str, Any]] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None

    def build_tok_params(
        self,
        default_tok_params: TokenizeParams,
    ) -> TokenizeParams:
        # Priority:
        # explicit `add_special_tokens` > `tokenization_kwargs` > per-path default.
        tok_kwargs: dict[str, Any] = dict(self.tokenization_kwargs or {})
        if self.add_special_tokens is not None:
            tok_kwargs["add_special_tokens"] = self.add_special_tokens
        if not tok_kwargs:
            return default_tok_params
        return default_tok_params.with_kwargs(**tok_kwargs)

    def build_chat_params(
        self,
        params: PoolingParams,
        default_template: str | None,
        chat_template_kwargs: dict[str, Any] | None = None,
        default_template_content_format: ChatTemplateContentFormatOption = "auto",
    ) -> ChatParams:
        chat_template = self.chat_template or default_template
        given_format = (
            self.chat_template_content_format
            if self.chat_template_content_format != "auto"
            else default_template_content_format
        )
        merged_chat_template_kwargs = self.build_chat_template_kwargs(
            params, chat_template_kwargs
        )
        return ChatParams(
            chat_template=chat_template,
            chat_template_content_format=given_format,
            chat_template_kwargs=merged_chat_template_kwargs,
            media_io_kwargs=self.media_io_kwargs,
        )

    def build_chat_template_kwargs(
        self,
        params: PoolingParams,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Priority: explicit `add_generation_prompt` / `continue_final_message`
        #         > `self.chat_template_kwargs`
        #         > `params.extra_kwargs["chat_template_kwargs"]`
        #         > `chat_template_kwargs` (e.g., saved ST kwargs)
        #         > default (`add_generation_prompt=False`,
        #                    `continue_final_message=False`)
        default_flags: dict[str, Any] = {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }
        extra_kwargs: dict[str, Any] = dict(params.extra_kwargs or {})

        # Merge base defaults, passed-in kwargs (such as saved ST kwargs),
        # `params.extra_kwargs`, and context-level `self.chat_template_kwargs`.
        merged = dict(
            merge_kwargs(
                merge_kwargs(
                    merge_kwargs(default_flags, chat_template_kwargs),
                    extra_kwargs.get("chat_template_kwargs"),
                ),
                self.chat_template_kwargs,
            )
            or {}
        )

        # Explicit tri-state fields take highest precedence when not None.
        if self.add_generation_prompt is not None:
            merged["add_generation_prompt"] = self.add_generation_prompt
        if self.continue_final_message is not None:
            merged["continue_final_message"] = self.continue_final_message

        if merged.get("add_generation_prompt") and merged.get("continue_final_message"):
            raise VLLMValidationError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return merged

    def build_engine_prompt_extras(self, params: PoolingParams):
        # Priority: explicit mm_processor_kwargs / cache_salt
        #         > params.extra_kwargs["chat_template_kwargs"]
        prompt_extras: dict[str, Any] = dict(params.extra_kwargs or {})
        merged = {
            k: v
            for k in ("mm_processor_kwargs", "cache_salt")
            if (v := prompt_extras.get(k)) is not None
        }
        if self.mm_processor_kwargs is not None:
            merged["mm_processor_kwargs"] = self.mm_processor_kwargs

        if self.cache_salt is not None:
            merged["cache_salt"] = self.cache_salt

        return merged


@dataclass(slots=True)
class OfflineEncodeInputsContext(OfflineInputsContext):
    prompts: PromptType | Sequence[PromptType]
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


@dataclass(slots=True)
class OfflineScoringInputsContext(OfflineInputsContext):
    scoring_data: ScoringData
    pooling_params: PoolingParams


@dataclass(slots=True)
class OfflinePluginInputsContext(OfflineInputsContext):
    prompts: DataPrompt
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


AnyOfflineInputsContext: TypeAlias = (
    OfflineEncodeInputsContext
    | OfflineScoringInputsContext
    | OfflinePluginInputsContext
)


@dataclass(slots=True)
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]

    ## for bi-encoder & late-interaction
    n_queries: int | None = None
    late_chunking: list[LateChunkingSlot | None] = field(default_factory=list)


class RenderParams(TypedDict):
    tok_params: TokenizeParams
    prompt_extras: dict[str, Any] | None

    params: PoolingParams
    lora_requests: LoRARequest | None
    priorities: int
    late_chunking: NotRequired[LateChunkingSlot]

    # mm_kwargs:
    skip_mm_cache: bool


class EncodeCMPLRenderParams(RenderParams):
    prompts: DictPrompt


class EncodeChatRenderParams(RenderParams):
    conversations: list["ChatCompletionMessageParam"]
    chat_params: ChatParams


class ScoringRenderParams(RenderParams):
    data_1: ScoreData
    data_2: ScoreData
    chat_params: ChatParams
    max_tokens_per_query: int
    max_tokens_per_doc: int


AnyRenderParam: TypeAlias = (
    EncodeCMPLRenderParams | EncodeChatRenderParams | ScoringRenderParams
)
RequestGenerator: TypeAlias = Generator[AnyRenderParam]
RequestFactory: TypeAlias = Callable[[], RequestGenerator]


class PoolingEngineInput(TypedDict):
    prompts: EngineInput
    params: PoolingParams
    lora_requests: LoRARequest | None
    priorities: int
