# SPDX-License-Identifier: Apache-2.0
import io
import asyncio
import numpy as np
from fastapi import Request
from typing import AsyncGenerator, Optional, Union, cast, Tuple


from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
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
    "cy": "Welsh",
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
    "bo": "Tibetan",
}

LANG_ID_TO_LANG_TOKEN = {
    "whisper-v3": {
        50259: "<|en|>",
        50260: "<|zh|>",
        50261: "<|de|>",
        50262: "<|es|>",
        50263: "<|ru|>",
        50264: "<|ko|>",
        50265: "<|fr|>",
        50266: "<|ja|>",
        50267: "<|pt|>",
        50268: "<|tr|>",
        50269: "<|pl|>",
        50270: "<|ca|>",
        50271: "<|nl|>",
        50272: "<|ar|>",
        50273: "<|sv|>",
        50274: "<|it|>",
        50275: "<|id|>",
        50276: "<|hi|>",
        50277: "<|fi|>",
        50278: "<|vi|>",
        50279: "<|he|>",
        50280: "<|uk|>",
        50281: "<|el|>",
        50282: "<|ms|>",
        50283: "<|cs|>",
        50284: "<|ro|>",
        50285: "<|da|>",
        50286: "<|hu|>",
        50287: "<|ta|>",
        50288: "<|no|>",
        50289: "<|th|>",
        50290: "<|ur|>",
        50291: "<|hr|>",
        50292: "<|bg|>",
        50293: "<|lt|>",
        50294: "<|la|>",
        50295: "<|mi|>",
        50296: "<|ml|>",
        50297: "<|cy|>",
        50298: "<|sk|>",
        50299: "<|te|>",
        50300: "<|fa|>",
        50301: "<|lv|>",
        50302: "<|bn|>",
        50303: "<|sr|>",
        50304: "<|az|>",
        50305: "<|sl|>",
        50306: "<|kn|>",
        50307: "<|et|>",
        50308: "<|mk|>",
        50309: "<|br|>",
        50310: "<|eu|>",
        50311: "<|is|>",
        50312: "<|hy|>",
        50313: "<|ne|>",
        50314: "<|mn|>",
        50315: "<|bs|>",
        50316: "<|kk|>",
        50317: "<|sq|>",
        50318: "<|sw|>",
        50319: "<|gl|>",
        50320: "<|mr|>",
        50321: "<|pa|>",
        50322: "<|si|>",
        50323: "<|km|>",
        50324: "<|sn|>",
        50325: "<|yo|>",
        50326: "<|so|>",
        50327: "<|af|>",
        50328: "<|oc|>",
        50329: "<|ka|>",
        50330: "<|be|>",
        50331: "<|tg|>",
        50332: "<|sd|>",
        50333: "<|gu|>",
        50334: "<|am|>",
        50335: "<|yi|>",
        50336: "<|lo|>",
        50337: "<|uz|>",
        50338: "<|fo|>",
        50339: "<|ht|>",
        50340: "<|ps|>",
        50341: "<|tk|>",
        50342: "<|nn|>",
        50343: "<|mt|>",
        50344: "<|sa|>",
        50345: "<|lb|>",
        50346: "<|my|>",
        50347: "<|bo|>",
        50348: "<|tl|>",
        50349: "<|mg|>",
        50350: "<|as|>",
        50351: "<|tt|>",
        50352: "<|haw|>",
        50353: "<|ln|>",
        50354: "<|ha|>",
        50355: "<|ba|>",
        50356: "<|jw|>",
        50357: "<|su|>",
        50358: "<|yue|>",
    },
}

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
MAX_AUDIO_CLIP_FILESIZE_MB = 25
# TODO get from processor.feature_extractor.chunk_length
MAX_AUDIO_CLIP_DURATION_S = 30


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
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        diff_sampling_param = self.model_config.get_diff_sampling_param()
        if diff_sampling_param:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                diff_sampling_param,
            )

    async def _detect_language(
        self,
        audio_data: Tuple[np.ndarray, int],
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> str:
        # TODO language should be optional and can be guessed.
        # For now we default to en. See
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520

        if request.language:
            if request.language in ISO639_1_SUPPORTED_LANGS:
                pass
            elif request.language in ISO639_1_OTHER_LANGS:
                logger.warning(
                    "The selected language %s has limited accuracy with"
                    " reported WER>=0.5. Results may be less accurate "
                    "for this choice.",
                    request.language,
                )
            else:
                raise ValueError(
                    f"Unsupported language: {request.language}."
                    "Language should be one of:"
                    + f" {list(ISO639_1_SUPPORTED_LANGS.values())}"
                    + f"or {list(ISO639_1_OTHER_LANGS.values())}"
                )
            return f"<|{request.language}|>"  # Corrected: Use f-string

        default_lang_token = "<|en|>"

        if (
            "v3" in self.model_config.model.lower()
            and self.model_config.hf_config.model_type.lower() == "whisper"
        ):
            id2token = LANG_ID_TO_LANG_TOKEN["whisper-v3"]
        else:
            return default_lang_token

        request_id = f"trsc-dl-{self._base_request_id(raw_request)}"
        prompt = cast(
            PromptType,
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": audio_data,
                    },
                },
                "decoder_prompt": "<|startoftranscript|>",
            },
        )
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=1,
            allowed_token_ids=list(id2token.keys()),
        )
        result_generator = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
        )

        try:
            async for result in result_generator:
                lang_id = result.outputs[0].token_ids[0]
                lang_token = id2token[lang_id]
                break
            return lang_token
        except Exception as e:  # catch other exception during generation
            logger.error(f"An error occurred during language detection: {e}")

        return default_lang_token  # This line should not be reachable.

    async def _preprocess_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: Request
    ) -> PromptType:
        # Validate request
        if len(audio_data) / 1024**2 > MAX_AUDIO_CLIP_FILESIZE_MB:
            raise ValueError("Maximum file size exceeded.")

        with io.BytesIO(audio_data) as bytes_:
            y, sr = librosa.load(bytes_)

        if librosa.get_duration(y=y, sr=sr) > MAX_AUDIO_CLIP_DURATION_S:
            raise ValueError(
                f"Maximum clip duration ({MAX_AUDIO_CLIP_DURATION_S}s) " "exceeded."
            )

        lang_token = await self._detect_language((y, sr), request, raw_request)

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            },
            "decoder_prompt": f"<|startoftranscript|>{lang_token}<|transcribe|><|notimestamps|>{request.prompt}",
        }
        return cast(PromptType, prompt)

    # TODO (varun) : Make verbose response work !
    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: Request
    ) -> Union[TranscriptionResponse, TranscriptionResponseVerbose, ErrorResponse]:
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

        if request.response_format not in ["text", "json"]:
            return self.create_error_response(
                "Currently only support response_format `text` or `json`"
            )

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            if lora_request:
                return self.create_error_response(
                    "Currently do not support LoRA for Transcription."
                )
            if prompt_adapter_request:
                return self.create_error_response(
                    "Currently do not support PromptAdapter for Transcription."
                )

            prompt = await self._preprocess_transcription(
                request=request, audio_data=audio_data, raw_request=raw_request
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # TODO cmpl->transcription?
        request_id = f"trsc-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        result_generator: Optional[AsyncGenerator[RequestOutput, None]] = None
        try:
            # TODO(rob): subtract len of tokenized prompt.
            default_max_tokens = self.model_config.max_model_len
            default_params = self.model_config.get_diff_sampling_param()
            sampling_params = request.to_sampling_params(
                default_max_tokens, default_params
            )

            self._log_inputs(
                request_id,
                prompt["decoder_prompt"],  # type: ignore
                params=sampling_params,
                lora_request=None,
                prompt_adapter_request=None,
            )

            result_generator = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # TODO(rob): figure out a way to pipe streaming in.
        # Non-streaming response.
        try:
            async for op in result_generator:
                result = op
            return TranscriptionResponse(text=result.outputs[0].text)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
