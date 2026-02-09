# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator, Mapping
from http import HTTPStatus
from typing import (
    ClassVar,
    TypeAlias,
    assert_never,
)
from wsgiref.headers import Headers

from fastapi import Request
from starlette.responses import JSONResponse

from vllm import (
    PoolingParams,
    PoolingRequestOutput,
    PromptType,
    SamplingParams,
    envs,
)
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.entrypoints.openai.engine.serving import (
    AnyRequest,
    AnyResponse,
    ServeContext,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.utils import create_error_response
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.lora.request import LoRARequest
from vllm.sampling_params import BeamSearchParams
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import merge_async_iterators

ClassificationServeContext: TypeAlias = ServeContext[ClassificationRequest]


class PoolingServing:
    request_id_prefix: ClassVar[str]

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client

        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self.log_error_stack = True

        self.input_processor = self.models.input_processor
        self.io_processor = self.models.io_processor
        self.renderer = self.models.renderer
        self.model_config = self.models.model_config
        self.max_model_len = self.model_config.max_model_len

    async def __call__(
        self,
        request: OpenAIBaseModel,
        raw_request: Request,
    ):
        try:
            model_name = self.models.model_name()
            request_id = (
                f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"
            )

            await self._check_model(request, raw_request)

            ctx = ClassificationServeContext(
                request=request,
                raw_request=raw_request,
                model_name=model_name,
                request_id=request_id,
            )

            self._validate_request(ctx)
            ctx.lora_request = self._maybe_get_adapters(ctx.request)

            await self._preprocess(ctx)
            await self._prepare_generators(ctx)
            await self._collect_batch(ctx)
            generator = await self._build_response(ctx)
        except Exception as e:
            generator = self.create_error_response(e)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.error.code
            )
        elif isinstance(generator, OpenAIBaseModel):
            return JSONResponse(content=generator.model_dump())

        assert_never(generator)

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

    def _create_pooling_params(
        self,
        ctx: ServeContext,
    ) -> PoolingParams:
        if not hasattr(ctx.request, "to_pooling_params"):
            raise ValueError("Request type does not support pooling parameters")

        return ctx.request.to_pooling_params()

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Schedule the request and get the result generator."""
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        trace_headers = (
            None
            if ctx.raw_request is None
            else await self._get_trace_headers(ctx.raw_request.headers)
        )

        pooling_params = self._create_pooling_params(ctx)

        for i, engine_prompt in enumerate(ctx.engine_prompts):
            request_id_item = f"{ctx.request_id}-{i}"

            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=pooling_params,
                lora_request=ctx.lora_request,
            )

            generator = self.engine_client.encode(
                engine_prompt,
                pooling_params,
                request_id_item,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
            )

            generators.append(generator)

        ctx.result_generator = merge_async_iterators(*generators)

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Collect batch results from the result generator."""

        num_prompts = len(ctx.engine_prompts)
        final_res_batch: list[PoolingRequestOutput | None]
        final_res_batch = [None] * num_prompts

        async for i, res in ctx.result_generator:
            final_res_batch[i] = res

        if None in final_res_batch:
            raise ValueError("Failed to generate results for all prompts")

        ctx.final_res_batch = [res for res in final_res_batch if res is not None]

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> None:
        raise NotImplementedError

    async def _build_response(
        self,
        ctx: ServeContext,
    ) -> AnyResponse:
        raise NotImplementedError

    #########################################################
    #########################################################

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
        request: AnyRequest,
        raw_request: Request,
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
                raise ValueError(load_result.error.message)
        return None

    def _validate_request(self, ctx: ServeContext) -> None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.max_model_len
        ):
            raise ValueError(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size."
            )
        return None

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str | None = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(
            message=message,
            err_type=err_type,
            status_code=status_code,
            param=param,
            log_error_stack=self.log_error_stack,
        )

    def _maybe_get_adapters(
        self,
        request: AnyRequest,
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
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _get_active_default_mm_loras(self, request: AnyRequest) -> LoRARequest | None:
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

    def _get_message_types(self, request: AnyRequest) -> set[str]:
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
        inputs: PromptType,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        prompt, prompt_token_ids, prompt_embeds = get_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            prompt,
            prompt_token_ids,
            prompt_embeds,
            params=params,
            lora_request=lora_request,
        )
