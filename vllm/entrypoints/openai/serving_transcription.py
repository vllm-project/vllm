# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator
from typing import Optional, Union

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ErrorResponse, RequestResponseMetadata, TranscriptionRequest,
    TranscriptionResponse, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, TranslationRequest, TranslationResponse,
    TranslationResponseStreamChoice, TranslationStreamResponse)
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
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
<<<<<<< sangbumlikeagod/bugfix/fix-whisper-prompt-tag
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        processor = cached_get_processor(model_config.model)
        self.max_audio_clip_s = processor.feature_extractor.chunk_length
        self.model_sr = processor.feature_extractor.sampling_rate
        self.hop_length = processor.feature_extractor.hop_length

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)

    async def _preprocess_transcription(
        self,
        request: TranscriptionRequest,
        audio_data: bytes,
    ) -> tuple[list[PromptType], float]:
        # Validate request
        # TODO language should be optional and can be guessed.
        # For now we default to en. See
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
        lang_token = f"<|{request.language}|>" if request.language else "<|en|>"
        if request.language:
            if request.language in ISO639_1_SUPPORTED_LANGS:
                pass
            elif request.language in ISO639_1_OTHER_LANGS:
                logger.warning(
                    "The selected language %s has limited accuracy with"
                    " reported WER>=0.5. Results may be less accurate "
                    "for this choice.", request.language)
            else:
                raise ValueError(
                    f"Unsupported language: {request.language}."
                    "Language should be one of:" +
                    f" {list(ISO639_1_SUPPORTED_LANGS.values())}" +
                    f"or {list(ISO639_1_OTHER_LANGS.values())}")

        if len(audio_data) / 1024**2 > MAX_AUDIO_CLIP_FILESIZE_MB:
            raise ValueError("Maximum file size exceeded.")

        with io.BytesIO(audio_data) as bytes_:
            y, sr = librosa.load(bytes_)

        duration = librosa.get_duration(y=y, sr=sr)
        chunks = [y] if duration < 30 else self._split_audio(y, sr)
        prompts = []
        for i, chunk in enumerate(chunks):
            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (chunk, sr),
                    },
                },
                "decoder_prompt":
                f"<|prev|>{request.prompt}<|startoftranscript|>{lang_token}<|transcribe|><|notimestamps|>"
                if i == 0 else ""
            }
            prompts.append(cast(PromptType, prompt))
        return prompts, duration
=======
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         task_type="transcribe")
>>>>>>> main

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest,
        raw_request: Request
    ) -> Union[TranscriptionResponse, AsyncGenerator[str, None],
               ErrorResponse]:
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
            self, request: TranscriptionRequest,
            result_generator: list[AsyncGenerator[RequestOutput, None]],
            request_id: str, request_metadata: RequestResponseMetadata,
            audio_duration_s: float) -> AsyncGenerator[str, None]:
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
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         task_type="translate")

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest,
        raw_request: Request
    ) -> Union[TranslationResponse, AsyncGenerator[str, None], ErrorResponse]:
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
            self, request: TranslationRequest,
            result_generator: list[AsyncGenerator[RequestOutput, None]],
            request_id: str, request_metadata: RequestResponseMetadata,
            audio_duration_s: float) -> AsyncGenerator[str, None]:
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
