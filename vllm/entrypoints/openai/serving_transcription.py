import asyncio
import io
from typing import Any, AsyncGenerator, Dict, Optional, Union

import librosa
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (RequestResponseMetadata,
                                              TranscriptionRequest,
                                              TranscriptionResponse,
                                              TranscriptionResponseVerbose)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

logger = init_logger(__name__)


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

        diff_sampling_param = self.model_config.get_diff_sampling_param()
        if diff_sampling_param:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                diff_sampling_param)

    async def _preprocess_transcription(
        self,
        request: TranscriptionRequest,
        audio_data: bytes,
    ) -> Dict[Any, Any]:
        return {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": librosa.load(io.BytesIO(audio_data)),
                },
            },
            # TODO(rob): tokenize here.
            "decoder_prompt":
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            # "decoder_prompt": f"{request.prompt}",
        }

    # TODO (varun) : Make verbose response work !
    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest,
        raw_request: Request
    ) -> Union[TranscriptionResponse, TranscriptionResponseVerbose]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI completion API.
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

        request_id = f"cmpl-{self._base_request_id(raw_request)}"

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

            prompt = await self._preprocess_transcription(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        result_generator: AsyncGenerator[RequestOutput, None] = None
        try:
            # TODO(rob): subtract len of tokenized prompt.
            default_max_tokens = self.model_config.max_model_len
            default_params = self.model_config.get_diff_sampling_param()
            sampling_params = request.to_sampling_params(
                default_max_tokens, default_params)

            self._log_inputs(request_id,
                             prompt['decoder_prompt'],
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

        # TODO(rob): figure out a way to pipe streaming in.
        stream = False
        if stream:
            return None

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
