import asyncio
import base64
import time
from typing import (AsyncGenerator, AsyncIterator, List, Optional, Tuple,
                    Union, cast)

import numpy as np
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)

TypeTokenIDs = List[int]


def request_output_to_embedding_response(
        final_res_batch: List[EmbeddingRequestOutput], request_id: str,
        created_time: int, model_name: str,
        encoding_format: str) -> EmbeddingResponse:
    data: List[EmbeddingResponseData] = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        prompt_token_ids = final_res.prompt_token_ids
        embedding = final_res.outputs.embedding
        if encoding_format == "base64":
            embedding_bytes = np.array(embedding).tobytes()
            embedding = base64.b64encode(embedding_bytes).decode("utf-8")
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
        async_engine_client: AsyncEngineClient,
        model_config: ModelConfig,
        served_model_names: List[str],
        *,
        request_logger: Optional[RequestLogger],
    ):
        super().__init__(async_engine_client=async_engine_client,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)
        self._check_embedding_mode(model_config.embedding_mode)

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, EmbeddingResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = (request.encoding_format
                           if request.encoding_format else "float")
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"embd-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[EmbeddingRequestOutput, None]] = []
        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.async_engine_client.get_tokenizer(
                lora_request)

            pooling_params = request.to_pooling_params()

            prompts = list(
                self._tokenize_prompt_input_or_inputs(
                    request,
                    tokenizer,
                    request.input,
                ))

            for i, prompt_inputs in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 prompt_inputs,
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                if prompt_adapter_request is not None:
                    raise NotImplementedError(
                        "Prompt adapter is not supported "
                        "for embedding models")

                generator = self.async_engine_client.encode(
                    {"prompt_token_ids": prompt_inputs["prompt_token_ids"]},
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, EmbeddingRequestOutput]] = merge_async_iterators(
                *generators,
                is_cancelled=raw_request.is_disconnected
                if raw_request else None)

        # Non-streaming response
        final_res_batch: List[Optional[EmbeddingRequestOutput]]
        final_res_batch = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for final_res in final_res_batch:
                assert final_res is not None

            final_res_batch_checked = cast(List[EmbeddingRequestOutput],
                                           final_res_batch)

            response = request_output_to_embedding_response(
                final_res_batch_checked, request_id, created_time, model_name,
                encoding_format)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def _check_embedding_mode(self, embedding_mode: bool):
        if not embedding_mode:
            logger.warning(
                "embedding_mode is False. Embedding API will not work.")
        else:
            logger.info("Activating the server engine with embedding enabled.")
