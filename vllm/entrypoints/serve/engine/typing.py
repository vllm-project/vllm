# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol, TypeAlias

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
    GenerateRequest,
    GenerateResponse,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.renderers import ChatParams, TokenizeParams


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | DerenderCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | BatchChatCompletionRequest
    | TokenizeChatRequest
    | DerenderChatRequest
)

SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest

AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse
    | ChatCompletionResponse
    | TranscriptionResponse
    | TokenizeResponse
    | GenerateResponse
)
