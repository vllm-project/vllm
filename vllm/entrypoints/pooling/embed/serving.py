# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, cast

from fastapi.responses import JSONResponse, StreamingResponse
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.entrypoints.pooling.utils import (
    encode_pooling_bytes,
    encode_pooling_output_base64,
    encode_pooling_output_float,
    get_json_response_cls,
)
from vllm.outputs import PoolingRequestOutput
from vllm.renderers import BaseRenderer
from vllm.utils.serial_utils import EmbedDType, Endianness

JSONResponseCLS = get_json_response_cls()

EmbeddingServeContext: TypeAlias = PoolingServeContext[EmbeddingRequest]


class ServingEmbedding(PoolingServing):
    """
    Embedding API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/embeddings/create
    for the API specification. This API mimics the OpenAI Embedding API.
    """

    request_id_prefix = "embd"

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> EmbedIOProcessor:
        return EmbedIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    async def _build_response(
        self,
        ctx: EmbeddingServeContext,
    ) -> JSONResponse | StreamingResponse:
        encoding_format = ctx.request.encoding_format
        embed_dtype = ctx.request.embed_dtype
        endianness = ctx.request.endianness

        if encoding_format == "float" or encoding_format == "base64":
            return self._request_output_to_embed_json_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        if encoding_format == "bytes" or encoding_format == "bytes_only":
            return self._request_output_to_to_embed_bytes_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        assert_never(encoding_format)

    def _request_output_to_embed_json_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> JSONResponse:
        encode_fn = cast(
            Callable[[PoolingRequestOutput], list[float] | str],
            (
                encode_pooling_output_float
                if encoding_format == "float"
                else partial(
                    encode_pooling_output_base64,
                    embed_dtype=embed_dtype,
                    endianness=endianness,
                )
            ),
        )

        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = EmbeddingResponseData(
                index=idx,
                embedding=encode_fn(final_res),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        response = EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
        return JSONResponseCLS(content=response.model_dump())

    def _request_output_to_to_embed_bytes_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["bytes", "bytes_only"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> StreamingResponse:
        content, items, usage = encode_pooling_bytes(
            pooling_outputs=final_res_batch,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )

        headers = (
            None
            if encoding_format == "bytes_only"
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

        response = EmbeddingBytesResponse(content=content, headers=headers)
        return StreamingResponse(
            content=response.content,
            headers=response.headers,
            media_type=response.media_type,
        )
