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

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              PoolingChatRequest,
                                              PoolingRequest, PoolingResponse,
                                              PoolingResponseData, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
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
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_pooling(
        self,
        request: PoolingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[PoolingResponse, ErrorResponse]:
        """
        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = self._get_model_name(request.model)
        request_id = f"pool-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        truncate_prompt_tokens = request.truncate_prompt_tokens

        try:
            truncate_prompt_tokens = _validate_truncation_size(
                self.max_model_len, truncate_prompt_tokens)
            lora_request = self._maybe_get_adapters(request)

            if self.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = await self.engine_client.get_tokenizer(lora_request
                                                                   )

            if isinstance(request, PoolingChatRequest):
                (
                    _,
                    request_prompts,
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
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.input,
                     truncate_prompt_tokens=truncate_prompt_tokens,
                     add_special_tokens=request.add_special_tokens,
                 )
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
                                 request_prompts[i],
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
                encoding_format,
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
