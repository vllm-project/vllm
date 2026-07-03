# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import Executor
from http import HTTPStatus
from typing import ClassVar

import torch
from fastapi import Request
from fastapi.responses import Response
from starlette.datastructures import Headers

from vllm import PoolingRequestOutput, envs
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.lora.request import LoRARequest
from vllm.renderers.base import BaseRenderer
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils.async_utils import make_async, merge_async_iterators

from ..typing import AnyPoolingRequest, PoolingServeContext
from .io_processor import PoolingIOProcessor


class PoolingBaseServing(ABC, BaseServing):
    request_id_prefix: ClassVar[str]

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template_config: ChatTemplateConfig,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__(
            models=models,
            model_config=models.model_config,
            request_logger=request_logger,
        )

        self.engine_client = engine_client
        self.renderer = engine_client.renderer
        self.vllm_config = engine_client.vllm_config
        self.max_model_len = self.model_config.max_model_len
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.log_error_stack = log_error_stack
        self.chat_template_config = chat_template_config

        # Shared thread pool executor for preprocessing and postprocessing.
        self._executor: Executor = self.renderer._executor
        self._preprocessing_async = make_async(
            self._preprocessing, executor=self._executor
        )
        self._postprocessing_async = make_async(
            self._postprocessing, executor=self._executor
        )

    async def __call__(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ) -> Response:
        io_processor = self.get_io_processor(request)
        ctx = await self._init_ctx(io_processor, request, raw_request)
        await self._preprocessing_async(io_processor, ctx)
        await self._prepare_generators(ctx)
        await self._collect_batch(ctx)
        return await self._postprocessing_async(io_processor, ctx)

    @abstractmethod
    def get_io_processor(self, request: AnyPoolingRequest) -> PoolingIOProcessor:
        raise NotImplementedError

    @torch.inference_mode()
    def _preprocessing(
        self, io_processor: PoolingIOProcessor, ctx: PoolingServeContext
    ):
        return io_processor.pre_process_online(ctx)

    @torch.inference_mode()
    def _postprocessing(
        self, io_processor: PoolingIOProcessor, ctx: PoolingServeContext
    ):
        io_processor.post_process_online(ctx)
        return self._build_response(ctx)

    async def _init_ctx(
        self,
        io_processor: PoolingIOProcessor,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ):
        model_name = self.models.model_name()
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"
        await self._check_model(request)

        pooling_params = io_processor.create_pooling_params(request)
        ctx = PoolingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            pooling_params=pooling_params,
            request_id=request_id,
        )

        self._validate_request(ctx)
        ctx.lora_request = self._maybe_get_adapters(ctx.request)
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
    def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        raise NotImplementedError

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


class PoolingServing(PoolingBaseServing, ABC):
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

    def get_io_processor(self, request: AnyPoolingRequest) -> PoolingIOProcessor:
        return self.io_processor
