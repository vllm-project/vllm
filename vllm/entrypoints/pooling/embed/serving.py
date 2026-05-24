# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, cast

from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing_extensions import assert_never

from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.utils.serial_utils import EmbedDType, Endianness

from ..base.serving import PoolingServing
from ..typing import PoolingServeContext
from ..utils import (
    BytesEncodingFormat,
    JsonEncodingFormat,
    build_pooling_bytes_streaming_response,
    encode_pooling_output_float,
    encode_pooling_output_float_or_ndarray,
    get_json_response_cls,
    get_pooling_output_encoder,
    get_pooling_usage,
)
from .io_processor import EmbedIOProcessor
from .protocol import (
    CohereBilledUnits,
    CohereEmbedRequest,
    CohereEmbedResponse,
    CohereMeta,
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
        encoding_format: JsonEncodingFormat,
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> JSONResponse:
        use_ndarray_response = (
            encoding_format == "float"
            and self.json_response_cls.__name__ == "ORJSONResponse"
        )
        if use_ndarray_response:
            ndarray_items: list[dict[str, object]] = []

            for idx, final_res in enumerate(final_res_batch):
                item_dict = EmbeddingResponseData(
                    index=idx,
                    embedding=[],
                ).model_dump()
                item_dict["embedding"] = encode_pooling_output_float_or_ndarray(
                    final_res
                )
                ndarray_items.append(item_dict)
            ndarray_response = EmbeddingResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                data=[],  # type: ignore[arg-type]
                usage=get_pooling_usage(final_res_batch),
            ).model_dump()
            ndarray_response["data"] = ndarray_items

            return self.json_response_cls(content=ndarray_response)

        encode_fn = get_pooling_output_encoder(
            encoding_format=encoding_format,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )

        items: list[EmbeddingResponseData] = []

        for idx, final_res in enumerate(final_res_batch):
            item = EmbeddingResponseData(
                index=idx,
                embedding=cast(list[float] | str, encode_fn(final_res)),
            )

            items.append(item)

        response = EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=get_pooling_usage(final_res_batch),
        )
        return self.json_response_cls(content=response.model_dump())

    def _openai_bytes_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: BytesEncodingFormat,
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> StreamingResponse:
        return build_pooling_bytes_streaming_response(
            pooling_outputs=final_res_batch,
            request_id=request_id,
            created_time=created_time,
            model_name=model_name,
            encoding_format=encoding_format,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )

    def _build_cohere_response_from_ctx(
        self,
        ctx: PoolingServeContext,
    ) -> JSONResponse:
        request = ctx.request
        assert isinstance(request, CohereEmbedRequest)

        all_floats = [
            cast(list[float], encode_pooling_output_float(out))
            for out in ctx.final_res_batch
        ]
        total_tokens = get_pooling_usage(ctx.final_res_batch).prompt_tokens

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
