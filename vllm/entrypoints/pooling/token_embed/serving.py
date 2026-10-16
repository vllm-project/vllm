# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi.responses import JSONResponse

from ..base.serving import PoolingServing
from ..factories import init_pooling_io_processors
from ..typing import PoolingServeContext
from ..utils import (
    encode_pooling_output_float,
    encode_pooling_output_float_or_ndarray,
    get_json_response_cls,
    get_pooling_usage,
)
from .protocol import TokenEmbeddingResponse, TokenEmbeddingResponseData


class ServingTokenEmbedding(PoolingServing):
    """Serve token embeddings and late chunking through /v1/embeddings."""

    request_id_prefix = "embd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_response_cls = get_json_response_cls()

    def init_io_processor(self, *args, **kwargs):
        return init_pooling_io_processors(("token_embed",), *args, **kwargs)[
            "token_embed"
        ]

    def _build_response(self, ctx: PoolingServeContext) -> JSONResponse:
        encode = (
            encode_pooling_output_float_or_ndarray
            if self.json_response_cls.__name__ == "ORJSONResponse"
            else encode_pooling_output_float
        )
        items = []
        for index, output in enumerate(ctx.final_res_batch):
            item = TokenEmbeddingResponseData(
                index=index,
                embedding=[],
                late_chunking=output.late_chunking,
            ).model_dump(
                exclude={"late_chunking"} if output.late_chunking is None else None
            )
            # Keep the vector matrix out of Pydantic's list conversion.
            item["embedding"] = encode(output)
            items.append(item)
        response = TokenEmbeddingResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=[],
            usage=get_pooling_usage(ctx.final_res_batch),
        ).model_dump()
        response["data"] = items
        return self.json_response_cls(content=response)
