# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import Request

from vllm import PromptType, SamplingParams, envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.pooling.typing import AnyPoolingRequest
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.exceptions import VLLMNotFoundError
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.sampling_params import BeamSearchParams
from vllm.utils import random_uuid


class BaseServing:
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        model_config: ModelConfig,
        request_logger: RequestLogger | None = None,
    ):
        self.models = models
        self.model_config = model_config
        self.request_logger = request_logger

    async def _check_model(
        self,
        request: AnyRequest | AnyPoolingRequest,
    ) -> ErrorResponse | None:
        error_response = None

        if self._is_model_supported(request.model):
            return None
        if request.model in self.models.lora_requests:
            return None
        if (
            envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING
            and request.model
            and (load_result := await self.models.resolve_lora(request.model))
        ):
            if isinstance(load_result, LoRARequest):
                return None
            if (
                isinstance(load_result, ErrorResponse)
                and load_result.error.code == HTTPStatus.BAD_REQUEST.value
            ):
                error_response = load_result

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        if envs.VLLM_SKIP_MODEL_NAME_VALIDATION:
            return True
        return self.models.is_base_model(model_name)

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def _extract_prompt_components(self, prompt: PromptType | EngineInput):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: PromptType | EngineInput):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: EngineInput):
        return extract_prompt_len(self.model_config, prompt)

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | EngineInput,
        params: SamplingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
            lora_request=lora_request,
        )

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    def _get_message_types(self, request: AnyRequest | AnyPoolingRequest) -> set[str]:
        """Retrieve the set of types from message content dicts up
        until `_`; we use this to match potential multimodal data
        with default per modality loras.
        """
        message_types: set[str] = set()

        if not hasattr(request, "messages"):
            return message_types

        messages = request.messages
        if messages is None or isinstance(messages, (str, bytes)):
            return message_types

        for message in messages:
            if (
                isinstance(message, dict)
                and "content" in message
                and isinstance(message["content"], list)
            ):
                for content_dict in message["content"]:
                    if "type" in content_dict:
                        message_types.add(content_dict["type"].split("_")[0])
        return message_types

    def _get_active_default_mm_loras(
        self, request: AnyRequest | AnyPoolingRequest
    ) -> LoRARequest | None:
        """Determine if there are any active default multimodal loras."""
        # TODO: Currently this is only enabled for chat completions
        # to be better aligned with only being enabled for .generate
        # when run offline. It would be nice to support additional
        # tasks types in the future.
        message_types = self._get_message_types(request)
        default_mm_loras = set()

        for lora in self.models.lora_requests.values():
            # Best effort match for default multimodal lora adapters;
            # There is probably a better way to do this, but currently
            # this matches against the set of 'types' in any content lists
            # up until '_', e.g., to match audio_url -> audio
            if lora.lora_name in message_types:
                default_mm_loras.add(lora)

        # Currently only support default modality specific loras if
        # we have exactly one lora matched on the request.
        if len(default_mm_loras) == 1:
            return default_mm_loras.pop()
        return None

    def _maybe_get_adapters(
        self,
        request: AnyRequest | AnyPoolingRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        # Currently only support default modality specific loras
        # if we have exactly one lora matched on the request.
        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                return default_mm_lora

        if self._is_model_supported(request.model):
            return None

        # if _check_model has been called earlier, this will be unreachable
        raise VLLMNotFoundError(f"The model `{request.model}` does not exist.")
