# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.exceptions import GenerationError
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

from ...serve.engine.protocol import ErrorResponse, UsageInfo
from ..base.serving import SpeechToTextBaseServing
from .protocol import (
    TranscriptionDiarizedSegment,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseDiarized,
    TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose,
    TranscriptionStreamResponse,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionUsageTokens,
)

logger = init_logger(__name__)


class OpenAIServingTranscription(SpeechToTextBaseServing):
    """Handles transcription requests."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_force_include_usage: bool = False,
        max_vad_workers: int = 0,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            task_type="transcribe",
            enable_force_include_usage=enable_force_include_usage,
            max_vad_workers=max_vad_workers,
        )

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> (
        TranscriptionResponse
        | TranscriptionResponseVerbose
        | TranscriptionResponseDiarized
        | AsyncGenerator[str, None]
        | ErrorResponse
    ):
        """Transcription API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI transcription API.
        """
        return await self._create_speech_to_text(
            audio_data=audio_data,
            request=request,
            raw_request=raw_request,
            response_class=(
                TranscriptionResponseVerbose
                if request.response_format == "verbose_json"
                else TranscriptionResponseDiarized
                if request.response_format == "diarized_json"
                else TranscriptionResponse
            ),
            stream_generator_method=(
                self.diarized_stream_generator
                if request.response_format == "diarized_json"
                else self.transcription_stream_generator
            ),
        )

    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        separator: str,
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
            separator=separator,
        )
        async for chunk in generator:
            yield chunk

    async def diarized_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        separator: str,
    ) -> AsyncGenerator[str, None]:
        """Stream each finalized segment as text delta and segment events,
        followed by a `transcript.text.done` event."""
        segment_texts: list[str] = []
        num_prompt_tokens = 0
        completion_tokens = 0

        def events_for_delta(text: str, *, finished: bool = False) -> Iterator[str]:
            for segment in parser.process_delta(text, finished=finished):
                event = TranscriptionDiarizedSegment(
                    id=f"seg_{len(segment_texts)}",
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=segment.speaker,
                )
                delta = TranscriptionTextDeltaEvent(
                    delta=(separator if segment_texts else "") + segment.text,
                    segment_id=event.id,
                )
                segment_texts.append(segment.text)
                yield f"data: {delta.model_dump_json()}\n\n"
                yield f"data: {event.model_dump_json()}\n\n"

        try:
            parser = self.model_cls.get_streaming_diarized_parser_cls()(
                self.model_cls, self.model_config
            )
            for index, chunk_generator in enumerate(result_generator):
                # Match the non-streaming path, which joins raw chunk texts.
                if index > 0:
                    for event in events_for_delta(separator):
                        yield event
                chunk_prompt_tokens = 0
                async with aclosing(chunk_generator):
                    async for res in chunk_generator:
                        if res.prompt_token_ids is not None:
                            # The decoder prompt already includes its audio positions.
                            chunk_prompt_tokens = len(res.prompt_token_ids)

                        assert len(res.outputs) == 1
                        output = res.outputs[0]
                        self._raise_if_error(output.finish_reason, request_id)
                        if output.finish_reason == "abort":
                            raise GenerationError("Transcription generation aborted")
                        completion_tokens += len(output.token_ids)
                        for event in events_for_delta(output.text):
                            yield event
                num_prompt_tokens += chunk_prompt_tokens

            for event in events_for_delta("", finished=True):
                yield event

            total_tokens = num_prompt_tokens + completion_tokens
            done = TranscriptionTextDoneEvent(
                text=separator.join(segment_texts),
                usage=TranscriptionUsageTokens(
                    input_tokens=num_prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            yield f"data: {done.model_dump_json()}\n\n"
        except Exception as e:
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
