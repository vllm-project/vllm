# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeAlias

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
    DerenderChatStreamRequest,
    DerenderCompletionRequest,
    DerenderCompletionStreamRequest,
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

CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | DerenderCompletionRequest
    | DerenderCompletionStreamRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | BatchChatCompletionRequest
    | TokenizeChatRequest
    | DerenderChatRequest
    | DerenderChatStreamRequest
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
