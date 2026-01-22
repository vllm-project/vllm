# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Final, cast

import jinja2
from fastapi import Request
from typing_extensions import assert_never

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    IOProcessorResponse,
    PoolingBytesResponse,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
)
from vllm.entrypoints.renderer import RenderConfig
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.tasks import PoolingTask, SupportedTask
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.serial_utils import (
    EmbedDType,
    EncodingFormat,
    Endianness,
    encode_pooling_bytes,
    encode_pooling_output,
)

logger = init_logger(__name__)


class OpenAIServingPooling(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        supported_tasks: tuple[SupportedTask, ...],
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

        self.supported_tasks = supported_tasks
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

    async def create_pooling(
        self,
        request: PoolingRequest,
        raw_request: Request | None = None,
    ) -> PoolingResponse | IOProcessorResponse | PoolingBytesResponse | ErrorResponse:
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
                    "dimensions is currently not supported"
                )

            truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens", None)
            truncate_prompt_tokens = _validate_truncation_size(
                self.max_model_len, truncate_prompt_tokens
            )

            if is_io_processor_request:
                if self.io_processor is None:
                    raise ValueError(
                        "No IOProcessor plugin installed. Please refer "
                        "to the documentation and to the "
                        "'prithvi_geospatial_mae_io_processor' "
                        "offline inference example for more details."
                    )

                validated_prompt = self.io_processor.parse_request(request)

                engine_prompts = await self.io_processor.pre_process_async(
                    prompt=validated_prompt, request_id=request_id
                )
                if not isinstance(engine_prompts, Sequence) or isinstance(
                    engine_prompts, (str, bytes, bytearray)
                ):
                    engine_prompts = [engine_prompts]

            elif isinstance(request, PoolingChatRequest):
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                _, engine_prompts = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    add_special_tokens=request.add_special_tokens,
                )
            elif isinstance(request, PoolingCompletionRequest):
                engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=request.input,
                    config=self._build_render_config(request),
                )
            else:
                raise ValueError(f"Unsupported request of type {type(request)}")
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            if is_io_processor_request:
                assert self.io_processor is not None and isinstance(
                    request, IOProcessorRequest
                )
                pooling_params = self.io_processor.validate_or_generate_params()
            else:
                pooling_params = request.to_pooling_params()

            pooling_task: PoolingTask
            if request.task is None:
                if "token_embed" in self.supported_tasks:
                    pooling_task = "token_embed"
                elif "token_classify" in self.supported_tasks:
                    pooling_task = "token_classify"
                elif "plugin" in self.supported_tasks:
                    pooling_task = "plugin"
                else:
                    return self.create_error_response(
                        f"pooling_task must be one of {self.supported_tasks}."
                    )
            else:
                pooling_task = request.task

            if pooling_task not in self.supported_tasks:
                return self.create_error_response(
                    f"Task {pooling_task} is not supported, it"
                    f" must be one of {self.supported_tasks}."
                )

            try:
                pooling_params.verify(pooling_task, self.model_config)
            except ValueError as e:
                return self.create_error_response(str(e))

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=pooling_params,
                    lora_request=lora_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

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

        assert isinstance(request, (PoolingCompletionRequest, PoolingChatRequest))
        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: list[PoolingRequestOutput | None]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(list[PoolingRequestOutput], final_res_batch)

            response = self.request_output_to_pooling_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
                request.encoding_format,
                request.embed_dtype,
                request.endianness,
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
        encoding_format: EncodingFormat,
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingResponse | PoolingBytesResponse:
        def encode_float_base64():
            items: list[PoolingResponseData] = []
            num_prompt_tokens = 0

            for idx, final_res in enumerate(final_res_batch):
                item = PoolingResponseData(
                    index=idx,
                    data=encode_pooling_output(
                        final_res,
                        encoding_format=encoding_format,
                        embed_dtype=embed_dtype,
                        endianness=endianness,
                    ),
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

        def encode_bytes(bytes_only: bool) -> PoolingBytesResponse:
            content, items, usage = encode_pooling_bytes(
                pooling_outputs=final_res_batch,
                embed_dtype=embed_dtype,
                endianness=endianness,
            )

            headers = (
                None
                if bytes_only
                else {
                    "metadata": json.dumps(
                        {
                            "id": request_id,
                            "created": created_time,
                            "model": model_name,
                            "data": items,
                            "usage": usage,
                        }
                    )
                }
            )

            return PoolingBytesResponse(
                content=content,
                headers=headers,
            )

        if encoding_format == "float" or encoding_format == "base64":
            return encode_float_base64()
        elif encoding_format == "bytes" or encoding_format == "bytes_only":
            return encode_bytes(bytes_only=encoding_format == "bytes_only")
        else:
            assert_never(encoding_format)

    def _build_render_config(self, request: PoolingCompletionRequest) -> RenderConfig:
        return RenderConfig(
            max_length=self.max_model_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )
