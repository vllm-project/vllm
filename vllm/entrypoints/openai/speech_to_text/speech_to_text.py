# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io
import math
import time
import zlib
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Final, Literal, TypeAlias, TypeVar, cast

import numpy as np
from fastapi import Request
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing, SpeechToTextRequest
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponse,
    TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionStreamResponse,
    TranslationResponse,
    TranslationResponseStreamChoice,
    TranslationResponseVerbose,
    TranslationSegment,
    TranslationStreamResponse,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EncoderDecoderInput, EngineInput
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.model_executor.models import SupportsTranscription
from vllm.multimodal.audio import get_audio_duration, split_audio
from vllm.multimodal.media.audio import load_audio
from vllm.outputs import RequestOutput
from vllm.renderers.inputs import DictPrompt, EncoderDecoderDictPrompt
from vllm.renderers.inputs.preprocess import parse_enc_dec_prompt, parse_model_prompt
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import get_tokenizer

SpeechToTextResponse: TypeAlias = TranscriptionResponse | TranslationResponse
SpeechToTextResponseVerbose: TypeAlias = (
    TranscriptionResponseVerbose | TranslationResponseVerbose
)
SpeechToTextSegment: TypeAlias = TranscriptionSegment | TranslationSegment
T = TypeVar("T", bound=SpeechToTextResponse)
V = TypeVar("V", bound=SpeechToTextResponseVerbose)
S = TypeVar("S", bound=SpeechToTextSegment)

ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)

logger = init_logger(__name__)


class OpenAISpeechToText(OpenAIServing):
    """Base class for speech-to-text operations like transcription and
    translation."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.task_type: Final = task_type

        self.asr_config = self.model_cls.get_speech_to_text_config(
            self.model_config, task_type
        )

        self.enable_force_include_usage = enable_force_include_usage

        self.max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
        if self.model_cls.supports_segment_timestamp:
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                get_tokenizer(
                    tokenizer_name=self.model_config.tokenizer,
                    tokenizer_mode=self.model_config.tokenizer_mode,
                ),
            )

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params,
            )

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsTranscription], model_cls)

    async def _detect_language(
        self,
        audio_chunk: np.ndarray,
        request_id: str,
    ) -> str:
        """Auto-detect the spoken language from an audio chunk.

        Delegates prompt construction and output parsing to the model class
        via ``get_language_detection_prompt`` and
        ``parse_language_detection_output``.
        """
        prompt = self.model_cls.get_language_detection_prompt(
            audio_chunk,
            self.asr_config,
        )
        allowed_token_ids = self.model_cls.get_language_token_ids(
            self.tokenizer,
        )
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=allowed_token_ids,
        )

        result_generator = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
        )

        final_output: RequestOutput
        async for final_output in result_generator:
            if final_output.finished:
                break

        token_ids = list(final_output.outputs[0].token_ids)
        lang = self.model_cls.parse_language_detection_output(
            token_ids,
            self.tokenizer,
        )

        logger.info("Auto-detected language: '%s'", lang)
        return lang

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[EngineInput], float]:
        # Validate request
        language = self.model_cls.validate_language(request.language)
        # Skip to_language validation to avoid extra logging for Whisper.
        to_language = (
            self.model_cls.validate_language(request.to_language)
            if request.to_language
            else None
        )

        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=len(audio_data) / 1024**2,
            )

        # Decode audio bytes.  For container formats (MP4, M4A, WebM) that
        # soundfile cannot detect from a BytesIO stream, _load_audio_bytes
        # transparently falls back to ffmpeg via an in-memory fd.
        # NOTE resample to model SR here for efficiency. This is also a
        # pre-requisite for chunking, as it assumes Whisper SR.
        try:
            with io.BytesIO(audio_data) as buf:
                y, sr = load_audio(buf, sr=self.asr_config.sample_rate)
        except Exception as exc:
            raise ValueError("Invalid or unsupported audio file.") from exc

        duration = get_audio_duration(y=y, sr=sr)
        do_split_audio = self.asr_config.allow_audio_chunking and (
            self.asr_config.max_audio_clip_s is not None
            and duration > self.asr_config.max_audio_clip_s
        )

        if not do_split_audio:
            chunks = [y]
        else:
            assert self.asr_config.max_audio_clip_s is not None
            assert self.asr_config.min_energy_split_window_size is not None
            chunks = split_audio(
                audio_data=y,
                sample_rate=int(sr),
                max_clip_duration_s=self.asr_config.max_audio_clip_s,
                overlap_duration_s=self.asr_config.overlap_chunk_second,
                min_energy_window_size=self.asr_config.min_energy_split_window_size,
            )

        if language is None and getattr(
            self.model_cls, "supports_explicit_language_detection", False
        ):
            # Auto-detect language from the first chunk.
            language = await self._detect_language(
                chunks[0], f"{request_id}-lang_detect"
            )
            request.language = language

        parsed_prompts: list[DictPrompt] = []
        for chunk in chunks:
            # The model has control over the construction, as long as it
            # returns a valid PromptType.
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt,
                to_language=to_language,
            )

            parsed_prompt: DictPrompt
            if request.response_format == "verbose_json":
                parsed_prompt = parse_enc_dec_prompt(prompt)
                parsed_prompt = self._preprocess_verbose_prompt(parsed_prompt)
            else:
                parsed_prompt = parse_model_prompt(self.model_config, prompt)

            parsed_prompts.append(parsed_prompt)

        engine_inputs = await self.renderer.render_cmpl_async(parsed_prompts)

        return engine_inputs, duration

    def _preprocess_verbose_prompt(self, prompt: EncoderDecoderDictPrompt):
        dec_prompt = prompt["decoder_prompt"]

        if not (isinstance(dec_prompt, dict) and "prompt" in dec_prompt):
            raise VLLMValidationError(
                "Expected decoder_prompt to contain text",
                parameter="decoder_prompt",
                value=type(dec_prompt).__name__,
            )

        dec_prompt["prompt"] = dec_prompt["prompt"].replace(
            "<|notimestamps|>", "<|0.00|>"
        )

        return prompt

    @staticmethod
    def _get_decoder_prompt_len(engine_inputs: list[EngineInput]) -> int:
        """Get the length of the decoder prompt. Currently we need to offset
        by the decoder prompt length when running beam search because the mm
        encoder is not currently cached and runs on decode calls; because of
        this, we need to make sure the redundant encoder calls won't exceed
        the context :(

        FIXME (Alex) - this will be removed in the very near future once the
        encoder/decoder caching is implemented.
        """
        input_len = 0
        assert len(engine_inputs) > 0
        first_input = engine_inputs[0]

        if first_input.get("type") == "enc_dec":
            first_input = cast(EncoderDecoderInput, first_input)
            input_len = len(first_input["decoder_prompt"]["prompt_token_ids"])

        return input_len

    def _get_verbose_segments(
        self,
        tokens: tuple,
        log_probs: FlatLogprobs | list[dict[int, Logprob]],
        request: SpeechToTextRequest,
        segment_class: type[SpeechToTextSegment],
        start_time: float = 0,
    ) -> list[SpeechToTextSegment]:
        """
        Convert tokens to verbose segments.

        This method expects the model to produce
        timestamps as tokens (similar to Whisper).
        If the tokens do not include timestamp information,
        the segments may not be generated correctly.

        Note: No_speech_prob field is not supported
        in this implementation and will be None. See docs for details.
        """
        BASE_OFFSET = 0.02
        init_token = self.tokenizer.encode("<|0.00|>", add_special_tokens=False)[0]
        if tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]

        tokens_with_start = (init_token,) + tokens
        segments: list[SpeechToTextSegment] = []
        last_timestamp_start = 0

        if tokens_with_start[-2] < init_token and tokens_with_start[-1] >= init_token:
            tokens_with_start = tokens_with_start + (tokens_with_start[-1],)
        avg_logprob = 0.0
        for idx in range(1, len(tokens_with_start)):
            # Timestamp tokens (e.g., <|0.00|>) are assumed to be sorted.
            # If the ordering is violated, this slicing may produce incorrect results.
            token = tokens_with_start[idx]
            if token >= init_token and tokens_with_start[idx - 1] >= init_token:
                sliced_timestamp_tokens = tokens_with_start[last_timestamp_start:idx]
                start_timestamp = sliced_timestamp_tokens[0] - init_token
                end_timestamp = sliced_timestamp_tokens[-1] - init_token
                text = self.tokenizer.decode(sliced_timestamp_tokens[1:-1])
                text_bytes = text.encode("utf-8")

                casting_segment = cast(
                    SpeechToTextSegment,
                    segment_class(
                        id=len(segments),
                        seek=start_time,
                        start=start_time + BASE_OFFSET * start_timestamp,
                        end=start_time + BASE_OFFSET * end_timestamp,
                        temperature=request.temperature,
                        text=text,
                        # The compression ratio measures
                        # how compressible the generated text is.
                        # A higher ratio indicates more repetitive content,
                        # which is a strong sign of hallucination in outputs.
                        compression_ratio=len(text_bytes)
                        / len(zlib.compress(text_bytes)),
                        tokens=sliced_timestamp_tokens[1:-1],
                        avg_logprob=avg_logprob / (idx - last_timestamp_start),
                    ),
                )
                segments.append(casting_segment)
                last_timestamp_start = idx
                avg_logprob = 0
            else:
                avg_logprob += log_probs[idx - 1][token].logprob
        return segments

    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[ResponseType],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> T | V | AsyncGenerator[str, None] | ErrorResponse:
        """Base method for speech-to-text operations like transcription and
        translation."""
        if request.stream and request.use_beam_search:
            return self.create_error_response(
                "Streaming is not currently supported with beam search"
            )

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ["text", "json", "verbose_json"]:
            return self.create_error_response(
                "Currently only support response_format: "
                "`text`, `json` or `verbose_json`"
            )

        if (
            request.response_format == "verbose_json"
            and not self.model_cls.supports_segment_timestamp
        ):
            return self.create_error_response(
                f"Currently do not support verbose_json for {request.model}"
            )

        if request.response_format == "verbose_json" and request.stream:
            return self.create_error_response(
                "verbose_json format doesn't support streaming case"
            )
        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request)

        engine_inputs, duration_s = await self._preprocess_speech_to_text(
            request=request,
            audio_data=audio_data,
            request_id=request_id,
        )

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        list_result_generator: list[AsyncGenerator[RequestOutput, None]] | None = None

        input_len = (
            OpenAISpeechToText._get_decoder_prompt_len(engine_inputs)
            if request.use_beam_search
            else 0
        )

        # Unlike most decoder-only models, whisper generation length is not
        # constrained by the size of the input audio, which is mapped to a
        # fixed-size log-mel-spectogram. Still, allow for fewer tokens to be
        # generated by respecting the extra completion tokens arg.
        max_tokens = get_max_tokens(
            max_model_len,
            request.max_completion_tokens,
            input_len,
            self.default_sampling_params,
        )

        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(
                max_tokens, self.default_sampling_params
            )
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

        if request.response_format == "verbose_json":
            sampling_params.logprobs = 1

        list_result_generator = []
        for i, engine_input in enumerate(engine_inputs):
            request_id_item = f"{request_id}_{i}"

            self._log_inputs(
                request_id_item,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    params=sampling_params,
                    request_id=request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )

            list_result_generator.append(generator)

        if request.stream:
            return stream_generator_method(
                request, list_result_generator, request_id, request_metadata, duration_s
            )
        # Non-streaming response.
        total_segments = []
        text_parts = []
        try:
            assert list_result_generator is not None
            segments_types: dict[str, type[SpeechToTextSegment]] = {
                "transcribe": TranscriptionSegment,
                "translate": TranslationSegment,
            }
            segment_class: type[SpeechToTextSegment] = segments_types[self.task_type]
            text = ""
            chunk_size_in_s = self.asr_config.max_audio_clip_s
            if chunk_size_in_s is None:
                assert len(list_result_generator) == 1, (
                    "`max_audio_clip_s` is set to None, audio cannot be chunked"
                )
            for idx, result_generator in enumerate(list_result_generator):
                start_time = (
                    float(idx * chunk_size_in_s) if chunk_size_in_s is not None else 0.0
                )
                async for op in result_generator:
                    if request.response_format == "verbose_json":
                        assert op.outputs[0].logprobs
                        segments: list[SpeechToTextSegment] = (
                            self._get_verbose_segments(
                                tokens=tuple(op.outputs[0].token_ids),
                                segment_class=segment_class,
                                request=request,
                                start_time=start_time,
                                log_probs=op.outputs[0].logprobs,
                            )
                        )

                        total_segments.extend(segments)
                        text_parts.extend([seg.text for seg in segments])
                    else:
                        raw_text = op.outputs[0].text
                        text_parts.append(self.model_cls.post_process_output(raw_text))
            text = "".join(text_parts)
            if self.task_type == "transcribe":
                final_response: ResponseType
                # add usage in TranscriptionResponse.
                usage = {
                    "type": "duration",
                    # rounded up as per openAI specs
                    "seconds": int(math.ceil(duration_s)),
                }
                if request.response_format != "verbose_json":
                    final_response = cast(
                        T, TranscriptionResponse(text=text, usage=usage)
                    )
                else:
                    final_response = cast(
                        V,
                        TranscriptionResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            else:
                # no usage in response for translation task
                if request.response_format != "verbose_json":
                    final_response = cast(T, TranslationResponse(text=text))
                else:
                    final_response = cast(
                        V,
                        TranslationResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            return final_response
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        list_result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: type[TranscriptionResponseStreamChoice]
        | type[TranslationResponseStreamChoice],
        stream_response_class: type[TranscriptionStreamResponse]
        | type[TranslationStreamResponse],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = self.enable_force_include_usage or request.stream_include_usage
        include_continuous_usage = (
            request.stream_continuous_usage_stats
            if include_usage and request.stream_continuous_usage_stats
            else False
        )

        try:
            for result_generator in list_result_generator:
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if audio_tokens := self.model_cls.get_num_audio_tokens(
                            audio_duration_s, self.asr_config, self.model_config
                        ):
                            num_prompt_tokens += audio_tokens

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    # TODO: For models that output structured formats (e.g.,
                    # Qwen3-ASR with "language X<asr_text>" prefix), streaming
                    # would need buffering to strip the prefix properly since
                    # deltas may split the tag across chunks.
                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )

                    chunk = stream_response_class(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        except Exception as e:
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
