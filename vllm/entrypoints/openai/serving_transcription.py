# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import time
from collections.abc import AsyncGenerator
from math import ceil
from typing import Final, Optional, Union, cast

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage, ErrorResponse, RequestResponseMetadata, TranscriptionRequest,
    TranscriptionResponse, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

logger = init_logger(__name__)

# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages#supported-languages
# TODO these configs should live somewhere with the model so we can support
# additional ones

ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}
ISO639_1_OTHER_LANGS = {
    "lo": "Lao",
    "jw": "Javanese",
    "tk": "Turkmen",
    "yi": "Yiddish",
    "so": "Somali",
    "bn": "Bengali",
    "nn": "Norwegian Nynorsk",
    "si": "Sinhala",
    "yo": "Yoruba",
    "sa": "Sanskrit",
    "mi": "Māori",
    "fo": "Faroese",  # codespell:ignore
    "mt": "Maltese",
    "tg": "Tajik",
    "mg": "Malagasy",
    "haw": "Hawaiian",
    "km": "Khmer",
    "br": "Breton",
    "ps": "Pashto",
    "ln": "Lingala",
    "la": "Latin",
    "ml": "Malayalam",
    "sq": "Albanian",
    "su": "Sundanese",
    "eu": "Basque",
    "ka": "Georgian",
    "uz": "Uzbek",
    "sn": "Shona",
    "ht": "Haitian",
    "as": "Assamese",
    "mn": "Mongolian",
    "te": "Telugu",
    "pa": "Panjabi",
    "tt": "Tatar",
    "gu": "Gujarati",
    "oc": "Occitan",
    "ha": "Hausa",
    "ba": "Bashkir",
    "my": "Burmese",
    "sd": "Sindhi",
    "am": "Amharic",
    "lb": "Luxembourgish",
    "bo": "Tibetan"
}

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
MAX_AUDIO_CLIP_FILESIZE_MB = 25


class OpenAIServingTranscription(OpenAIServing):

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
    ) -> tuple[PromptType, float]:
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
        if duration > self.max_audio_clip_s:
            raise ValueError(
                f"Maximum clip duration ({self.max_audio_clip_s}s) "
                "exceeded.")

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            },
            "decoder_prompt":
            f"<|startoftranscript|>{lang_token}<|transcribe|><|notimestamps|>{request.prompt}"
        }
        return cast(PromptType, prompt), duration

    # TODO (varun) : Make verbose response work !
    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest,
        raw_request: Request
    ) -> Union[TranscriptionResponse, AsyncGenerator[str, None],
               ErrorResponse]:
        """Transcription API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI transcription API.
        """
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

        request_id = f"trsc-{self._base_request_id(raw_request)}"

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
                    "Currently do not support LoRA for Transcription.")
            if prompt_adapter_request:
                return self.create_error_response(
                    "Currently do not support PromptAdapter for Transcription."
                )

            prompt, duration_s = await self._preprocess_transcription(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        result_generator: Optional[AsyncGenerator[RequestOutput, None]] = None
        try:
            # Unlike most decoder-only models, whisper generation length is not
            # constrained by the size of the input audio, which is mapped to a
            # fixed-size log-mel-spectogram.
            default_max_tokens = self.model_config.max_model_len
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params)

            self._log_inputs(
                request_id,
                prompt['decoder_prompt'],  # type: ignore
                params=sampling_params,
                lora_request=None,
                prompt_adapter_request=None)

            result_generator = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if request.stream:
            return self.transcription_stream_generator(request,
                                                       result_generator,
                                                       request_id,
                                                       request_metadata,
                                                       duration_s)
        # Non-streaming response.
        try:
            assert result_generator is not None
            async for op in result_generator:
                result = op
            return TranscriptionResponse(text=result.outputs[0].text)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def transcription_stream_generator(
            self, request: TranscriptionRequest,
            result_generator: AsyncGenerator[RequestOutput, None],
            request_id: str, request_metadata: RequestResponseMetadata,
            audio_duration_s: float) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model
        chunk_object_type: Final = "transcription.chunk"

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = request.stream_continuous_usage_stats\
              if include_usage and request.stream_continuous_usage_stats\
                else False

        try:
            async for res in result_generator:
                # On first result.
                if res.prompt_token_ids is not None:
                    # Do not account the 4-tokens `<|startoftranscript|>..`
                    # Could be negative when language token is not specified.
                    num_prompt_tokens = max(len(res.prompt_token_ids) - 4, 0)
                    # NOTE(NickLucche) user can't pass encoder prompts directly
                    # at least not to Whisper. One indicator of the encoder
                    # amount of processing is the log-mel spectogram length.
                    num_prompt_tokens += ceil(audio_duration_s *
                                              self.model_sr / self.hop_length)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).

                # Just one output (n=1) supported.
                assert len(res.outputs) == 1
                output = res.outputs[0]

                delta_message = DeltaMessage(content=output.text)
                completion_tokens += len(output.token_ids)

                if output.finish_reason is None:
                    # Still generating, send delta update.
                    choice_data = TranscriptionResponseStreamChoice(
                        delta=delta_message)
                else:
                    # Model is finished generating.
                    choice_data = TranscriptionResponseStreamChoice(
                        delta=delta_message,
                        finish_reason=output.finish_reason,
                        stop_reason=output.stop_reason)

                chunk = TranscriptionStreamResponse(id=request_id,
                                                    object=chunk_object_type,
                                                    created=created_time,
                                                    choices=[choice_data],
                                                    model=model_name)

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
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)

                final_usage_chunk = TranscriptionStreamResponse(
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
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
