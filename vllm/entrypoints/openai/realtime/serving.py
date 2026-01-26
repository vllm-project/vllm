# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Literal, cast

import numpy as np

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.realtime.protocol import SessionUpdate
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsTranscription, supports_transcription
from vllm.v1.engine.async_llm import StreamingInput

logger = init_logger(__name__)


class OpenAIServingRealtime(OpenAIServing):
    """Realtime audio transcription service via WebSocket streaming.

    Provides streaming audio-to-text transcription by transforming audio chunks
    into StreamingInput objects that can be consumed by the engine.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        log_error_stack: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

        self.task_type: Literal["transcribe"] = "transcribe"
        self.enable_force_include_usage = enable_force_include_usage

        # Get speech-to-text configuration from the model
        # TODO(Patrick) - can we use the same config as before for
        # Speech-to-text
        self.asr_config = self.model_cls.get_speech_to_text_config(
            self.model_config, self.task_type
        )

        # TODO: Add warmup for audio preprocessing if needed
        # self._warmup_audio_preprocessing()

        logger.info("OpenAIServingRealtime initialized for task: %s", self.task_type)

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        """Get the model class that supports transcription."""
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        if not supports_transcription(model_cls):
            raise ValueError(
                f"Model {self.model_config.model} does not support transcription"
            )
        return cast(type[SupportsTranscription], model_cls)

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        config: SessionUpdate | None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Args:
            audio_stream: Async generator yielding float32 numpy audio arrays
            config: Session configuration with model and parameters

        Yields:
            StreamingInput objects containing audio prompts for the engine
        """
        # TODO: Validate model from config if provided
        # if config and config.model:
        #     error_check_ret = await self._check_model(request)
        #     if error_check_ret is not None:
        #         raise ValueError(f"Model validation failed: {error_check_ret}")

        # Get sampling params from config
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        sampling_params = SamplingParams.from_optional(
            temperature=config.temperature if config else 0.0,
            max_tokens=self.model_config.max_model_len,
            output_kind=RequestOutputKind.DELTA,
            skip_clone=True,
        )

        # Get and validate language from config
        language = config.language if config and config.language else None
        language = self.model_cls.validate_language(language)

        # Process each audio chunk from the stream
        async for audio_chunk in audio_stream:
            # TODO: Let models' adapt the audio_chunk
            # and yield adapted streaming input
            # TODO(Patrick) - add get_streaming_prompt
            # to voxtral
            prompt = self.model_cls.get_streaming_prompt(
                audio=audio_chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
            )

            # Yield as StreamingInput for the engine
            yield StreamingInput(
                prompt=prompt,
                sampling_params=sampling_params,
            )
