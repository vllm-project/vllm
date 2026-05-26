# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, cast

from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput, PoolingRequestOutput
from vllm.utils.serial_utils import EmbedDType, Endianness

from ..base.serving import PoolingServing
from ..typing import PoolingServeContext
from ..utils import (
    encode_pooling_bytes,
    encode_pooling_output_base64,
    get_json_response_cls,
)
from .io_processor import EmbedIOProcessor
from .protocol import (
    CohereBilledUnits,
    CohereEmbedRequest,
    CohereEmbedResponse,
    CohereMeta,
    EmbeddingBytesResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    build_typed_embeddings,
)

logger = init_logger(__name__)


EmbeddingServeContext: TypeAlias = PoolingServeContext[EmbeddingRequest]


class ServingEmbedding(PoolingServing):
    """Embedding API supporting both OpenAI and Cohere formats."""

    request_id_prefix = "embd"
    io_processor: EmbedIOProcessor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.json_response_cls = get_json_response_cls()

    def init_io_processor(self, *args, **kwargs) -> EmbedIOProcessor:
        return EmbedIOProcessor(*args, **kwargs)

    def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        if isinstance(ctx.request, CohereEmbedRequest):
            return self._build_cohere_response_from_ctx(ctx)
        return self._build_openai_response(ctx)

    def _build_openai_response(
        self,
        ctx: EmbeddingServeContext,
    ) -> JSONResponse | StreamingResponse:
        encoding_format = ctx.request.encoding_format
        embed_dtype = ctx.request.embed_dtype
        endianness = ctx.request.endianness

        if encoding_format == "float" or encoding_format == "base64":
            return self._openai_json_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        if encoding_format == "bytes" or encoding_format == "bytes_only":
            return self._openai_bytes_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        assert_never(encoding_format)

    def _openai_json_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> JSONResponse:
        encoded_outputs: list[list[float] | str]
        if encoding_format == "float":
            embedding_batch = EmbeddingRequestOutput.from_base_batch(final_res_batch)
            encoded_outputs = [item.outputs.embedding for item in embedding_batch]
        else:
            encode_fn = cast(
                Callable[[PoolingRequestOutput], str],
                partial(
                    encode_pooling_output_base64,
                    embed_dtype=embed_dtype,
                    endianness=endianness,
                ),
            )
            encoded_outputs = [encode_fn(final_res) for final_res in final_res_batch]

        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        for idx, (final_res, encoded_output) in enumerate(
            zip(final_res_batch, encoded_outputs)
        ):
            item = EmbeddingResponseData(
                index=idx,
                embedding=encoded_output,
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
        return self.json_response_cls(content=response.model_dump())

    def _openai_bytes_response(
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

    def _build_cohere_response_from_ctx(
        self,
        ctx: PoolingServeContext,
    ) -> JSONResponse:
        request = ctx.request
        assert isinstance(request, CohereEmbedRequest)

        embedding_batch = EmbeddingRequestOutput.from_base_batch(ctx.final_res_batch)
        all_floats = [item.outputs.embedding for item in embedding_batch]
        total_tokens = sum(len(out.prompt_token_ids) for out in ctx.final_res_batch)

        image_tokens = total_tokens if request.images is not None else 0
        texts_echo = request.texts

        embedding_types = request.embedding_types or ["float"]
        embeddings_obj = build_typed_embeddings(all_floats, embedding_types)

        input_tokens = total_tokens - image_tokens
        response = CohereEmbedResponse(
            id=ctx.request_id,
            embeddings=embeddings_obj,
            texts=texts_echo,
            meta=CohereMeta(
                billed_units=CohereBilledUnits(
                    input_tokens=input_tokens,
                    image_tokens=image_tokens,
                ),
            ),
        )
        return self.json_response_cls(content=response.model_dump(exclude_none=True))
