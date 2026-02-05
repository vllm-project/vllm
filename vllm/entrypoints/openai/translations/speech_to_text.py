# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io
import math
import time
import zlib
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Literal, TypeAlias, TypeVar, cast

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
from vllm.entrypoints.openai.translations.protocol import (
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
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import ExplicitEncoderDecoderPrompt, PromptType
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.model_executor.models import SupportsTranscription, supports_transcription
from vllm.outputs import RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

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
        log_error_stack: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.task_type = task_type

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

        # Warm up audio preprocessing to avoid first-request latency
        self._warmup_audio_preprocessing()
        # Warm up input processor with dummy audio
        self._warmup_input_processor()

    def _warmup_audio_preprocessing(self) -> None:
        """Warm up audio processing libraries to avoid first-request latency.

        The first call to librosa functions (load, get_duration, mel-spectrogram)
        triggers JIT compilation and library initialization which can take ~7s.
        This method warms up these operations during server initialization.
        """
        # Skip warmup if librosa is not installed (optional dependency)
        if isinstance(librosa, PlaceholderModule):
            return

        # Skip warmup if model doesn't support transcription
        if not supports_transcription(self.model_cls):
            return

        if getattr(self.model_cls, "skip_warmup_audio_preprocessing", False):
            return

        try:
            warmup_start = time.perf_counter()
            logger.info("Warming up audio preprocessing libraries...")

            # Create a minimal dummy audio (1 second of silence at target sample rate)
            dummy_audio = np.zeros(int(self.asr_config.sample_rate), dtype=np.float32)

            # Warm up librosa.load by using librosa functions on the dummy data
            # This initializes FFTW, numba JIT, and other audio processing libraries
            _ = librosa.get_duration(y=dummy_audio, sr=self.asr_config.sample_rate)

            # Warm up mel-spectrogram computation with model-specific parameters
            from vllm.transformers_utils.processor import cached_processor_from_config

            processor = cached_processor_from_config(self.model_config)
            feature_extractor = None
            if hasattr(processor, "feature_extractor"):
                feature_extractor = processor.feature_extractor
            elif hasattr(processor, "audio_processor"):
                # For models like GraniteSpeech that use audio_processor
                audio_proc = processor.audio_processor
                if hasattr(audio_proc, "feature_extractor"):
                    feature_extractor = audio_proc.feature_extractor
                # If audio_processor doesn't have feature_extractor,
                # skip mel-spectrogram warmup for these models

            if feature_extractor is not None:
                _ = librosa.feature.melspectrogram(
                    y=dummy_audio,
                    sr=self.asr_config.sample_rate,
                    n_mels=getattr(feature_extractor, "n_mels", 128),
                    n_fft=getattr(feature_extractor, "n_fft", 400),
                    hop_length=getattr(feature_extractor, "hop_length", 160),
                )

            warmup_elapsed = time.perf_counter() - warmup_start
            logger.info("Audio preprocessing warmup completed in %.2fs", warmup_elapsed)
        except Exception:
            # Don't fail initialization if warmup fails - log exception and continue
            logger.exception(
                "Audio preprocessing warmup failed (non-fatal): %s. "
                "First request may experience higher latency.",
            )

    def _warmup_input_processor(self) -> None:
        """Warm up input processor with dummy audio to avoid first-request latency.

        The first call to input_processor.process_inputs() with multimodal audio
        triggers multimodal processing initialization which can take ~2.5s.
        This method processes a dummy audio request to warm up the pipeline.
        """
        # Skip warmup if model doesn't support transcription
        if not supports_transcription(self.model_cls):
            return

        # Only warm up if model supports transcription methods
        if not hasattr(self.model_cls, "get_generation_prompt"):
            return

        try:
            from vllm.sampling_params import SamplingParams

            warmup_start = time.perf_counter()
            logger.info("Warming up multimodal input processor...")

            # Create minimal dummy audio (1 second of silence)
            dummy_audio = np.zeros(int(self.asr_config.sample_rate), dtype=np.float32)

            # Use the same method that _preprocess_speech_to_text uses
            # to create the prompt
            dummy_prompt = self.model_cls.get_generation_prompt(
                audio=dummy_audio,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language="en",
                task_type=self.task_type,
                request_prompt="",
                to_language=None,
            )

            # Create minimal sampling params
            dummy_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                skip_clone=True,  # Internal warmup, safe to skip clone
            )

            # Process the dummy input through the input processor
            # This will trigger all the multimodal processing initialization
            _ = self.input_processor.process_inputs(
                request_id="warmup",
                prompt=dummy_prompt,
                params=dummy_params,
            )

            warmup_elapsed = time.perf_counter() - warmup_start
            logger.info("Input processor warmup completed in %.2fs", warmup_elapsed)
        except Exception:
            # Don't fail initialization if warmup fails - log warning and continue
            logger.exception(
                "Input processor warmup failed (non-fatal): %s. "
                "First request may experience higher latency."
            )

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsTranscription], model_cls)

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
    ) -> tuple[list[PromptType], float]:
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

        with io.BytesIO(audio_data) as bytes_:
            # NOTE resample to model SR here for efficiency. This is also a
            # pre-requisite for chunking, as it assumes Whisper SR.
            y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)

        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (
            self.asr_config.allow_audio_chunking
            and duration > self.asr_config.max_audio_clip_s
        )
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
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
            if request.response_format == "verbose_json":
                if not (isinstance(prompt, dict) and "encoder_prompt" in prompt):
                    raise VLLMValidationError(
                        "Expected prompt to be an encoder-decoder prompt",
                        parameter="prompt",
                        value=type(prompt).__name__,
                    )

                prompt = self._preprocess_verbose_prompt(prompt)

            prompts.append(prompt)
        return prompts, duration

    def _repl_verbose_text(self, text: str):
        return text.replace("<|notimestamps|>", "<|0.00|>")

    def _preprocess_verbose_prompt(self, prompt: ExplicitEncoderDecoderPrompt):
        dec_prompt = prompt["decoder_prompt"]

        if isinstance(dec_prompt, str):
            prompt["decoder_prompt"] = self._repl_verbose_text(dec_prompt)
        elif isinstance(dec_prompt, dict) and "prompt" in dec_prompt:
            dec_prompt["prompt"] = self._repl_verbose_text(dec_prompt["prompt"])
        else:
            raise VLLMValidationError(
                "Expected decoder_prompt to contain text",
                parameter="decoder_prompt",
                value=type(dec_prompt).__name__,
            )

        return prompt

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
        response_class: type[T | V],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> T | V | AsyncGenerator[str, None] | ErrorResponse:
        """Base method for speech-to-text operations like transcription and
        translation."""
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

        try:
            lora_request = self._maybe_get_adapters(request)

            prompts, duration_s = await self._preprocess_speech_to_text(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)

        list_result_generator: list[AsyncGenerator[RequestOutput, None]] | None = None
        try:
            # Unlike most decoder-only models, whisper generation length is not
            # constrained by the size of the input audio, which is mapped to a
            # fixed-size log-mel-spectogram. Still, allow for fewer tokens to be
            # generated by respecting the extra completion tokens arg.
            if request.max_completion_tokens is None:
                default_max_tokens = self.model_config.max_model_len
            else:
                default_max_tokens = min(
                    self.model_config.max_model_len, request.max_completion_tokens
                )
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params
            )
            if request.response_format == "verbose_json":
                sampling_params.logprobs = 1

            self._log_inputs(
                request_id,
                # It will not display special tokens like <|startoftranscript|>
                request.prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            list_result_generator = [
                self.engine_client.generate(
                    prompt,
                    sampling_params,
                    f"{request_id}_{i}",
                    lora_request=lora_request,
                )
                for i, prompt in enumerate(prompts)
            ]
        except ValueError as e:
            return self.create_error_response(e)

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
        except ValueError as e:
            return self.create_error_response(e)

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

    def _split_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> list[np.ndarray]:
        assert self.asr_config.max_audio_clip_s is not None, (
            f"{self.asr_config.max_audio_clip_s=} cannot be None to"
            " split audio into chunks."
        )
        chunk_size = sample_rate * self.asr_config.max_audio_clip_s
        overlap_size = sample_rate * self.asr_config.overlap_chunk_second
        chunks = []
        i = 0
        while i < audio_data.shape[-1]:
            if i + chunk_size >= audio_data.shape[-1]:
                # handle last chunk
                chunks.append(audio_data[..., i:])
                break

            # Find the best split point in the overlap region
            search_start = i + chunk_size - overlap_size
            search_end = min(i + chunk_size, audio_data.shape[-1])
            split_point = self._find_split_point(audio_data, search_start, search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point
        return chunks

    def _find_split_point(self, wav: np.ndarray, start_idx: int, end_idx: int) -> int:
        """Find the best point to split audio by
        looking for silence or low amplitude.
        Args:
            wav: Audio tensor [1, T]
            start_idx: Start index of search region
            end_idx: End index of search region
        Returns:
            Index of best splitting point
        """
        segment = wav[start_idx:end_idx]

        # Calculate RMS energy in small windows
        min_energy = math.inf
        quietest_idx = 0
        min_energy_window = self.asr_config.min_energy_split_window_size
        assert min_energy_window is not None
        for i in range(0, len(segment) - min_energy_window, min_energy_window):
            window = segment[i : i + min_energy_window]
            energy = (window**2).mean() ** 0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx
