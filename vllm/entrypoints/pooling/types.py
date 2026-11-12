# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, NotRequired, TypeAlias, TypedDict, TypeGuard, TypeVar

from fastapi import Request
from pydantic import ConfigDict

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.exceptions import VLLMValidationError
from vllm.inputs import DataPrompt, EngineInput
from vllm.lora.request import LoRARequest
from vllm.outputs import LateChunkingMetadata
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.chat_utils.types import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    OpenAIHarmonyMessage,
)
from vllm.renderers.inputs import DictPrompt
from vllm.tasks import PoolingTask

from ..generate.base.protocol import validate_cache_salt
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


ChatType: TypeAlias = list[ChatCompletionMessageParam]
PoolingInput: TypeAlias = PromptType | ChatType


@dataclass(kw_only=True)
class ChunkedEmbeddingMetadata:
    prompt_index: int
    chunk_index: int


@dataclass
class LateChunkingSlot:
    """Per-document source ranges shared by frontend rendering and output contexts."""

    metadata: LateChunkingMetadata | None = None


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class OfflineInputsContext:
    pooling_task: PoolingTask
    tokenization_kwargs: dict[str, Any] | None
    lora_request: Sequence[LoRARequest | None] | None
    priorities: int | Sequence[int] | None
    late_chunking: list[LateChunkingSlot | None] = field(default_factory=list)

    # Mirror ChatRequestOptionsMixin.
    add_generation_prompt: bool = False
    continue_final_message: bool = False
    add_special_tokens: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template_content_format: ChatTemplateContentFormatOption
    media_io_kwargs: dict[str, dict[str, Any]] | None = None

    # Mirror PoolingBasicRequestMixin.
    mm_processor_kwargs: dict[str, Any] | None = None
    cache_salt: str | None = None

    def __post_init__(self) -> None:
        # Mirror ChatRequestOptionsMixin.check_generation_prompt.
        if self.continue_final_message and self.add_generation_prompt:
            raise VLLMValidationError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True.",
            )

        if self.cache_salt is not None:
            validate_cache_salt(self.cache_salt)

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        # Mirror ChatRequestOptionsMixin.build_chat_params.
        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                dict(
                    add_generation_prompt=self.add_generation_prompt,
                    continue_final_message=self.continue_final_message,
                ),
            ),
            media_io_kwargs=self.media_io_kwargs,
        )

    def build_prompt_extras(self) -> dict[str, Any] | None:
        """Pack the renderer-side `prompt_extras` payload."""
        extras: dict[str, Any] = {}
        if self.mm_processor_kwargs is not None:
            extras["mm_processor_kwargs"] = self.mm_processor_kwargs
        if self.cache_salt is not None:
            extras["cache_salt"] = self.cache_salt
        return extras or None

    def to_encode_context(
        self,
        *,
        inputs_seq: list[PoolingInput],
        pooling_params: PoolingParams | Sequence[PoolingParams] | None,
    ) -> "OfflineEncodeInputsContext":
        """Derive an `OfflineEncodeInputsContext` from this context."""
        # `late_chunking` is intentionally dropped: inputs are re-composed,
        #  so per-document source ranges no longer line up with the new inputs.
        return OfflineEncodeInputsContext(
            pooling_task=self.pooling_task,
            tokenization_kwargs=self.tokenization_kwargs,
            lora_request=self.lora_request,
            priorities=self.priorities,
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
            add_special_tokens=self.add_special_tokens,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs,
            chat_template_content_format=self.chat_template_content_format,
            media_io_kwargs=self.media_io_kwargs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            cache_salt=self.cache_salt,
            inputs_seq=inputs_seq,
            pooling_params=pooling_params,
        )


@dataclass
class OfflineEncodeInputsContext(OfflineInputsContext):
    inputs_seq: list[PoolingInput]
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


@dataclass
class OfflineScoringInputsContext(OfflineInputsContext):
    scoring_data: ScoringData
    pooling_params: PoolingParams

    def to_encode_context(
        self,
        *,
        inputs_seq: list[PoolingInput],
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    ) -> "OfflineEncodeInputsContext":
        # Default to the scoring ctx's own `pooling_params` so callers do not
        # have to pass it explicitly.
        return super().to_encode_context(
            inputs_seq=inputs_seq,
            pooling_params=(
                self.pooling_params if pooling_params is None else pooling_params
            ),
        )


@dataclass
class OfflinePluginInputsContext(OfflineInputsContext):
    prompts: DataPrompt
    pooling_params: PoolingParams | Sequence[PoolingParams] | None


AnyOfflineInputsContext: TypeAlias = (
    OfflineEncodeInputsContext
    | OfflineScoringInputsContext
    | OfflinePluginInputsContext
)


@dataclass
class OfflineOutputsContext:
    outputs: list[PoolingRequestOutput]

    ## for bi-encoder & late-interaction
    n_queries: int | None = None
    late_chunking: list[LateChunkingSlot | None] = field(default_factory=list)


class RenderParams(TypedDict):
    tok_params: TokenizeParams
    prompt_extras: dict[str, Any] | None
    skip_mm_cache: bool

    params: PoolingParams
    lora_requests: LoRARequest | None
    priorities: int
    late_chunking: NotRequired[LateChunkingSlot]


class EncodeCMPLRenderParams(RenderParams):
    prompts: DictPrompt


class EncodeChatRenderParams(RenderParams):
    conversations: list["ChatCompletionMessageParam"]
    chat_params: ChatParams


class ScoringRenderParams(RenderParams):
    data_1: ScoreData
    data_2: ScoreData
    chat_template: str | None
    chat_template_kwargs: NotRequired[dict[str, Any] | None]
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


def _is_chat_message(item: Any) -> bool:
    # harmony messages are pydantic models, not dicts
    if OpenAIHarmonyMessage is not None and isinstance(item, OpenAIHarmonyMessage):
        return True
    # OpenAI / Custom chat messages are TypedDicts (plain dicts at runtime)
    return isinstance(item, dict) and "role" in item


def is_conversations(prompt_or_conversations: PoolingInput) -> TypeGuard[ChatType]:
    """Return True if a single PoolingInput is a chat conversation."""
    if not isinstance(prompt_or_conversations, list):
        return False
    if len(prompt_or_conversations) == 0:
        return False
    return all(_is_chat_message(x) for x in prompt_or_conversations)


def input_to_seq(
    input_or_inputs: PoolingInput | list[PoolingInput],
) -> list[PoolingInput]:
    # Non-list input is a single prompt (str / prompt dict / embeds prompt).
    if not isinstance(input_or_inputs, list):
        return [input_or_inputs]

    # Empty list: nothing to normalize.
    if len(input_or_inputs) == 0:
        return input_or_inputs  # type: ignore[return-value]

    # list[int] is a single tokenized prompt, not a batch of inputs.
    if all(isinstance(x, int) for x in input_or_inputs):
        return [input_or_inputs]  # type: ignore[list-item]

    # list of messages is a single conversation, not a batch of inputs.
    if all(_is_chat_message(x) for x in input_or_inputs):
        return [input_or_inputs]  # type: ignore[list-item]

    # Otherwise it is already a batch of prompts / conversations.
    return input_or_inputs  # type: ignore[return-value]
