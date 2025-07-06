# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io
import math
import time
from collections.abc import AsyncGenerator
from functools import cached_property
from math import ceil
from typing import Callable, Literal, Optional, TypeVar, Union, cast

import numpy as np
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage, ErrorResponse, RequestResponseMetadata,
    TranscriptionResponse, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, TranslationResponse,
    TranslationResponseStreamChoice, TranslationStreamResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    SpeechToTextRequest)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_cls
from vllm.model_executor.models import SupportsTranscription
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

SpeechToTextResponse = Union[TranscriptionResponse, TranslationResponse]
T = TypeVar("T", bound=SpeechToTextResponse)

logger = init_logger(__name__)

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
MAX_AUDIO_CLIP_FILESIZE_MB = 25
MAX_AUDIO_CLIP_SECONDS = 30
OVERLAP_CHUNK_SECOND = 1
MIN_ENERGY_WINDOW_SIZE = 1600  # 1600 ~ 100ms for 16000 Hz audio


class OpenAISpeechToText(OpenAIServing):
    """Base class for speech-to-text operations like transcription and 
    translation."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        processor = cached_get_processor(model_config.model)
        self.max_audio_clip_s = processor.feature_extractor.chunk_length \
            if hasattr(processor.feature_extractor, 'chunk_length') \
            else MAX_AUDIO_CLIP_SECONDS
        self.model_sr = processor.feature_extractor.sampling_rate
        self.hop_length = processor.feature_extractor.hop_length
        self.task_type = task_type

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)

    @cached_property
    def model_cls(self):
        return get_model_cls(self.model_config)

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        previous_text :list[str],
    ) -> AsyncGenerator[tuple[PromptType, float]]:
        model_cls = cast(SupportsTranscription, self.model_cls)
        # Validate request
        # TODO language should be optional and can be guessed.
        # For now we default to en. See
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
        lang = request.language or "en"
        model_cls.validate_language(lang)

        if len(audio_data) / 1024**2 > MAX_AUDIO_CLIP_FILESIZE_MB:
            raise ValueError("Maximum file size exceeded.")

        with io.BytesIO(audio_data) as bytes_:
            # NOTE resample to model SR here for efficiency. This is also a
            # pre-requisite for chunking, as it assumes Whisper SR.
            y, sr = librosa.load(bytes_, sr=self.model_sr)

        duration = librosa.get_duration(y=y, sr=sr)
        chunks = [y
                  ] if duration < self.max_audio_clip_s else self._split_audio(
                      y, int(sr))
        for chunk in chunks:
            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (chunk, sr),
                    },
                },
                "decoder_prompt":
                self._create_prompt_with_previous_context(
                    system_prompt=model_cls.get_decoder_prompt
                        (lang, self.task_type,
                                request.prompt),
                    previous_text=previous_text,
                    lang_token=lang
                )
            }
            yield (cast(PromptType, prompt), duration)
            
    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[T],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> Union[T, AsyncGenerator[str, None], ErrorResponse]:
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

        if request.response_format not in ['text', 'json']:
            return self.create_error_response(
                "Currently only support response_format `text` or `json`")

        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            if lora_request:
                return self.create_error_response(
                    "Currently do not support LoRA for "
                    f"{self.task_type.title()}.")
            if prompt_adapter_request:
                return self.create_error_response(
                    f"Currently do not support PromptAdapter for "
                    f"{self.task_type.title()}.")
            previous_text = [""]
            asyncPromptGenerator = self._preprocess_speech_to_text(
                request=request,
                audio_data=audio_data,
                previous_text=previous_text
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        try:
            # Unlike most decoder-only models, whisper generation length is not
            # constrained by the size of the input audio, which is mapped to a
            # fixed-size log-mel-spectogram.
            default_max_tokens = self.model_config.max_model_len
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params)
            # streaming response.
            if request.stream:
                return stream_generator_method(request, asyncPromptGenerator,
                                            request_id, request_metadata,
                                            sampling_params, previous_text)
            
            # Non-streaming response.
            text= ""
            async for (prompt, _) in asyncPromptGenerator:
                partial_text = ""
                if text == "":
                    self._log_inputs(
                        request_id,
                        prompt['decoder_prompt'],  # type: ignore
                        params=sampling_params,
                        lora_request=None,
                        prompt_adapter_request=None
                    )
                    
                request_prompt = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                )
                async for op in request_prompt:
                    partial_text += op.outputs[0].text
                    
                previous_text[0] = partial_text.strip()
                text += partial_text

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return cast(T, response_class(text=text))

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        async_result_generator: AsyncGenerator[tuple[PromptType, float]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        sampling_params : SamplingParams,
        previous_context : list[str],
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: Union[
            type[TranscriptionResponseStreamChoice],
            type[TranslationResponseStreamChoice]],
        stream_response_class: Union[type[TranscriptionStreamResponse],
                                     type[TranslationStreamResponse]],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = request.stream_continuous_usage_stats\
            if include_usage and request.stream_continuous_usage_stats\
            else False

        try:
            async for (prompt, duration_s) in async_result_generator:
                partial_text = ""
                result_generator = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                )
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        # Do not account the 4-tokens 
                        # `<|startoftranscript|>..`
                        # Could be negative when language token
                        # is not specified.
                        num_prompt_tokens = max(
                            len(res.prompt_token_ids) - 4, 0)
                        # NOTE(NickLucche) user can't pass encoder
                        # prompts directly at least not to Whisper.
                        # One indicator of the encoder amount of processing
                        # is the log-mel spectogram length.
                        num_prompt_tokens += ceil(
                            duration_s * self.model_sr 
                            / self.hop_length)

                    # We need to do it here, 
                    # because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]
                    partial_text += res.outputs[0].text

                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(
                            delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)

                    chunk = stream_response_class(id=request_id,
                                                object=chunk_object_type,
                                                created=created_time,
                                                choices=[choice_data],
                                                model=model_name)

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens 
                            + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                    
                previous_context[0] = partial_text.strip()

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens)

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    def _split_audio(self, audio_data: np.ndarray,
                     sample_rate: int) -> list[np.ndarray]:
        chunk_size = sample_rate * self.max_audio_clip_s
        overlap_size = sample_rate * OVERLAP_CHUNK_SECOND
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
            split_point = self._find_split_point(audio_data, search_start,
                                                 search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point
        return chunks

    def _find_split_point(self, wav: np.ndarray, start_idx: int,
                          end_idx: int) -> int:
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
        for i in range(0,
                       len(segment) - MIN_ENERGY_WINDOW_SIZE,
                       MIN_ENERGY_WINDOW_SIZE):
            window = segment[i:i + MIN_ENERGY_WINDOW_SIZE]
            energy = (window**2).mean()**0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx
    
    def _create_prompt_with_previous_context(self, 
            system_prompt: str,
            previous_text :list[str],
            lang_token : str
        ) -> str:
        """
        According to the Whisper prompting guide:
        https://cookbook.openai.com/examples/whisper_prompting_guide

        The decoder prompt in Whisper is limited to 224 tokens.
        This means that both previous_text and the decoder prompt itself
        may need to be truncated to fit within this limit.

        Currently, the decoder prompt contains 4 special tokens,
        so the maximum length available
        for the rest of the prompt is 220 tokens.

        Token counting can vary by language:
        - In English, one token usually represents more than one letter.
        - In other languages, a single character may be 
        split into multiple tokens.

        To account for this, we set TOKEN_PER_CHAR = 1 for English,
        and TOKEN_PER_CHAR = 3 for other languages.

        MAX_PROMPT_LENGTH = 220 // TOKEN_PER_CHAR

        Additionally, to prevent hallucination,
        the prompt should be truncated at word boundaries.
        """

        TOKEN_PER_CHAR = 1 if lang_token == "en" else 3
        MAX_PROMPT_LENGTH = 220 // TOKEN_PER_CHAR
        
        ret_prompt = ""
        ret_prompt_len = 0
        
        request_prompt_list = system_prompt.split(' ')
        previous_text_list = previous_text[0].split(' ')
        for prompt in request_prompt_list:
            if len(prompt) + ret_prompt_len <= MAX_PROMPT_LENGTH:
                ret_prompt += prompt
                ret_prompt += " "
                ret_prompt_len += len(prompt)
            else:
                break
        
        previous_text_list_rev = []
        for previous_index in range(len(previous_text_list) - 1, 0, -1):
            prompt = previous_text_list[previous_index]
            if len(prompt) + ret_prompt_len <= MAX_PROMPT_LENGTH:
                previous_text_list_rev.append(prompt)
                ret_prompt_len += len(prompt)
            else: 
                break
            
        ret_prompt += ' '.join(reversed(previous_text_list_rev))
        return  (f"<|prev|>{ret_prompt}<|startoftranscript|>{lang_token}"
                f"<|{self.task_type}|><|notimestamps|>")
        
        