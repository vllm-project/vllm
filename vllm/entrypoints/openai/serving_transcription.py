# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseStreamChoice,
    TranslationStreamResponse,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text import OpenAISpeechToText
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

logger = init_logger(__name__)


class OpenAIServingTranscription(OpenAISpeechToText):
    """Handles transcription requests."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            task_type="transcribe",
            log_error_stack=log_error_stack,
            enable_force_include_usage=enable_force_include_usage,
        )

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: Request
    ) -> TranscriptionResponse | AsyncGenerator[str, None] | ErrorResponse:
        """Transcription API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI transcription API.
        """
        return await self._create_speech_to_text(
            audio_data=audio_data,
            request=request,
            raw_request=raw_request,
            response_class=TranscriptionResponse,
            stream_generator_method=self.transcription_stream_generator,
        )

    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
    ) -> AsyncGenerator[str, None]:
        generator = self._speech_to_text_stream_generator(
            request=request,
            list_result_generator=result_generator,
            request_id=request_id,
            request_metadata=request_metadata,
            audio_duration_s=audio_duration_s,
            chunk_object_type="transcription.chunk",
            response_stream_choice_class=TranscriptionResponseStreamChoice,
            stream_response_class=TranscriptionStreamResponse,
        )
        async for chunk in generator:
            yield chunk


class OpenAIServingTranslation(OpenAISpeechToText):
    """Handles translation requests."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            task_type="translate",
            log_error_stack=log_error_stack,
            enable_force_include_usage=enable_force_include_usage,
        )

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: Request
    ) -> TranslationResponse | AsyncGenerator[str, None] | ErrorResponse:
        """Translation API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranslation
        for the API specification. This API mimics the OpenAI translation API.
        """
        return await self._create_speech_to_text(
            audio_data=audio_data,
            request=request,
            raw_request=raw_request,
            response_class=TranslationResponse,
            stream_generator_method=self.translation_stream_generator,
        )

    async def translation_stream_generator(
        self,
        request: TranslationRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
    ) -> AsyncGenerator[str, None]:
        generator = self._speech_to_text_stream_generator(
            request=request,
            list_result_generator=result_generator,
            request_id=request_id,
            request_metadata=request_metadata,
            audio_duration_s=audio_duration_s,
            chunk_object_type="translation.chunk",
            response_stream_choice_class=TranslationResponseStreamChoice,
            stream_response_class=TranslationStreamResponse,
        )
        async for chunk in generator:
            yield chunk
