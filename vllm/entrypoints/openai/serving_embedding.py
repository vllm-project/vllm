import asyncio
import base64
import time
from typing import AsyncGenerator, List, Literal, Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import BaseModelPath, OpenAIServing
from vllm.logger import init_logger
from vllm.outputs import EmbeddingOutput, EmbeddingRequestOutput
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[List[float], str]:
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


def request_output_to_embedding_response(
        final_res_batch: List[EmbeddingRequestOutput], request_id: str,
        created_time: int, model_name: str,
        encoding_format: Literal["float", "base64"]) -> EmbeddingResponse:
    data: List[EmbeddingResponseData] = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        prompt_token_ids = final_res.prompt_token_ids
        embedding = _get_embedding(final_res.outputs, encoding_format)
        embedding_data = EmbeddingResponseData(index=idx, embedding=embedding)
        data.append(embedding_data)

        num_prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return EmbeddingResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
        usage=usage,
    )


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)

        self.chat_template = load_chat_template(chat_template)

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Embedding API similar to OpenAI's API.

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

        model_name = request.model
        request_id = f"embd-{random_uuid()}"
        created_time = int(time.monotonic())

        truncate_prompt_tokens = None

        if request.truncate_prompt_tokens is not None:
            if request.truncate_prompt_tokens <= self.max_model_len:
                truncate_prompt_tokens = request.truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

            if isinstance(request, EmbeddingChatRequest):
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                request_prompts, engine_prompts = self._preprocess_completion(
                    request,
                    tokenizer,
                    request.input,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[EmbeddingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

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

        result_generator = merge_async_iterators(
            *generators,
            is_cancelled=raw_request.is_disconnected if raw_request else None,
        )

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[EmbeddingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        try:
            for final_res in final_res_batch:
                assert final_res is not None

            final_res_batch_checked = cast(List[EmbeddingRequestOutput],
                                           final_res_batch)

            response = request_output_to_embedding_response(
                final_res_batch_checked, request_id, created_time, model_name,
                encoding_format)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response
