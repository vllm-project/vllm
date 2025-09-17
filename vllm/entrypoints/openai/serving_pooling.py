# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import time
from collections.abc import AsyncGenerator
from typing import Final, Literal, Optional, Union, cast

import jinja2
import numpy as np
import torch
from fastapi import Request
from typing_extensions import assert_never

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
# yapf: disable
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              IOProcessorRequest,
                                              IOProcessorResponse,
                                              PoolingChatRequest,
                                              PoolingCompletionRequest,
                                              PoolingRequest, PoolingResponse,
                                              PoolingResponseData, UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.renderer import RenderConfig
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.utils import merge_async_iterators

logger = init_logger(__name__)


def _get_data(
    output: PoolingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return output.data.tolist()
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        pt_float32 = output.data.to(dtype=torch.float32)
        pooling_bytes = np.array(pt_float32, dtype="float32").tobytes()
        return base64.b64encode(pooling_bytes).decode("utf-8")

    assert_never(encoding_format)


class OpenAIServingPooling(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        vllm_config: VllmConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=vllm_config.model_config,
                         models=models,
                         request_logger=request_logger,
                         log_error_stack=log_error_stack)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        io_processor_plugin = self.model_config.io_processor_plugin
        self.io_processor = get_io_processor(vllm_config, io_processor_plugin)

    async def create_pooling(
        self,
        request: PoolingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[PoolingResponse, IOProcessorResponse, ErrorResponse]:
        """
        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        model_name = self.models.model_name()

        request_id = f"pool-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        is_io_processor_request = isinstance(request, IOProcessorRequest)
        try:
            lora_request = self._maybe_get_adapters(request)

            if self.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = await self.engine_client.get_tokenizer()
            renderer = self._get_renderer(tokenizer)

            if getattr(request, "dimensions", None) is not None:
                return self.create_error_response(
                    "dimensions is currently not supported")

            truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens",
                                             None)
            truncate_prompt_tokens = _validate_truncation_size(
                self.max_model_len, truncate_prompt_tokens)

            if is_io_processor_request:
                if self.io_processor is None:
                    raise ValueError(
                        "No IOProcessor plugin installed. Please refer "
                        "to the documentation and to the "
                        "'prithvi_geospatial_mae_io_processor' "
                        "offline inference example for more details.")

                validated_prompt = self.io_processor.parse_request(request)

                engine_prompts = await self.io_processor.pre_process_async(
                    prompt=validated_prompt, request_id=request_id)

            elif isinstance(request, PoolingChatRequest):
                (
                    _,
                    _,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    # In pooling requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    add_special_tokens=request.add_special_tokens,
                )
            elif isinstance(request, PoolingCompletionRequest):
                engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=request.input,
                    config=self._build_render_config(request),
                )
            else:
                raise ValueError(
                    f"Unsupported request of type {type(request)}")
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()

            try:
                pooling_params.verify("encode", self.model_config)
            except ValueError as e:
                return self.create_error_response(str(e))

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 engine_prompt,
                                 params=pooling_params,
                                 lora_request=lora_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        if is_io_processor_request:
            assert self.io_processor is not None
            output = await self.io_processor.post_process_async(
                model_output=result_generator,
                request_id=request_id,
            )
            return self.io_processor.output_to_response(output)

        assert isinstance(request,
                          (PoolingCompletionRequest, PoolingChatRequest))
        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: list[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(list[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_pooling_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
                request.encoding_format,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def request_output_to_pooling_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
    ) -> PoolingResponse:
        items: list[PoolingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = PoolingResponseData(
                index=idx,
                data=_get_data(final_res.outputs, encoding_format),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return PoolingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )

    def _build_render_config(
            self, request: PoolingCompletionRequest) -> RenderConfig:
        return RenderConfig(
            max_length=self.max_model_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens)
