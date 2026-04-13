# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from http import HTTPStatus
from typing import ClassVar

from fastapi import Request
from fastapi.responses import Response
from starlette.datastructures import Headers

from vllm import PoolingParams, PoolingRequestOutput, envs
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateConfig,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.typing import AnyPoolingRequest, PoolingServeContext
from vllm.exceptions import VLLMNotFoundError
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import merge_async_iterators

from .io_processor import PoolingIOProcessor


class PoolingServingBase(ABC):
    request_id_prefix: ClassVar[str]

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ):
        self.engine_client = engine_client
        self.models = models
        self.model_config = models.model_config
        self.renderer = models.renderer
        self.vllm_config = engine_client.vllm_config
        self.max_model_len = self.model_config.max_model_len
        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.log_error_stack = log_error_stack
        self.chat_template_config = ChatTemplateConfig(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            trust_request_chat_template=trust_request_chat_template,
        )

    @abstractmethod
    async def __call__(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ) -> Response:
        raise NotImplementedError

    async def _init_ctx(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ):
        model_name = self.models.model_name()
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"
        await self._check_model(request)

        ctx = PoolingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
        )

        self._validate_request(ctx)
        self._maybe_get_adapters(ctx)
        return ctx

    async def _prepare_generators(
        self,
        ctx: PoolingServeContext,
    ):
        if ctx.engine_inputs is None:
            raise ValueError("Engine prompts not available")

        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        trace_headers = (
            None
            if ctx.raw_request is None
            else await self._get_trace_headers(ctx.raw_request.headers)
        )

        assert ctx.pooling_params is not None
        pooling_params = ctx.pooling_params

        if isinstance(pooling_params, list):
            for params in pooling_params:
                params.verify(self.model_config)
        else:
            pooling_params.verify(self.model_config)

        for i, engine_input in enumerate(ctx.engine_inputs):
            prompt_request_id = (
                f"{ctx.request_id}-{i}"
                if ctx.prompt_request_ids is None
                else ctx.prompt_request_ids[i]
            )

            params = (
                pooling_params[i]
                if isinstance(pooling_params, list)
                else pooling_params
            )

            self._log_inputs(
                prompt_request_id,
                engine_input,
                params=params,
                lora_request=ctx.lora_request,
            )

            generator = self.engine_client.encode(
                engine_input,
                params,
                prompt_request_id,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
            )

            generators.append(generator)

        ctx.result_generator = merge_async_iterators(*generators)

    async def _collect_batch(
        self,
        ctx: PoolingServeContext,
    ):
        if ctx.engine_inputs is None:
            raise ValueError("Engine prompts not available")

        if ctx.result_generator is None:
            raise ValueError("Result generator not available")

        num_inputs = len(ctx.engine_inputs)
        final_res_batch: list[PoolingRequestOutput | None]
        final_res_batch = [None] * num_inputs

        async for i, res in ctx.result_generator:
            final_res_batch[i] = res

        if None in final_res_batch:
            raise ValueError("Failed to generate results for all prompts")

        ctx.final_res_batch = [res for res in final_res_batch if res is not None]

    @abstractmethod
    async def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        raise NotImplementedError

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

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)

    async def _check_model(
        self,
        request: AnyPoolingRequest,
    ) -> ErrorResponse | None:
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
                raise ValueError(load_result.error.message)
        return None

    def _validate_request(self, ctx: PoolingServeContext) -> None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.max_model_len
        ):
            raise ValueError(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please request a smaller truncation size."
            )

        return None

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    def _maybe_get_adapters(
        self,
        ctx: PoolingServeContext,
        supports_default_mm_loras: bool = False,
    ):
        request = ctx.request
        if request.model in self.models.lora_requests:
            ctx.lora_request = self.models.lora_requests[request.model]

        # Currently only support default modality specific loras
        # if we have exactly one lora matched on the request.
        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                ctx.lora_request = default_mm_lora

        if self._is_model_supported(request.model):
            return None

        # if _check_model has been called earlier, this will be unreachable
        raise VLLMNotFoundError(f"The model `{request.model}` does not exist.")

    def _get_active_default_mm_loras(
        self, request: AnyPoolingRequest
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

    def _get_message_types(self, request: AnyPoolingRequest) -> set[str]:
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

    def _log_inputs(
        self,
        request_id: str,
        inputs: EngineInput,
        params: PoolingParams,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = extract_prompt_components(self.model_config, inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
            lora_request=lora_request,
        )


class PoolingServing(PoolingServingBase, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.io_processor = self.init_io_processor(
            vllm_config=self.vllm_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )

    @abstractmethod
    def init_io_processor(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> PoolingIOProcessor:
        raise NotImplementedError

    async def __call__(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ) -> Response:
        ctx = await self._init_ctx(request, raw_request)
        await self.io_processor.pre_process_online_async(ctx)

        if ctx.pooling_params is None:
            ctx.pooling_params = self.io_processor.create_pooling_params(request)

        await self._prepare_generators(ctx)
        await self._collect_batch(ctx)
        await self.io_processor.post_process_online_async(ctx)
        return await self._build_response(ctx)
