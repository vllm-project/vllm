# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing_extensions import assert_never

from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.tasks import SupportedTask
from vllm.utils.serial_utils import EmbedDType, Endianness

from ..base.io_processor import PoolingIOProcessor
from ..base.serving import PoolingServingBase
from ..factories import init_pooling_io_processors
from ..typing import AnyPoolingRequest, PoolingServeContext
from ..utils import (
    BytesEncodingFormat,
    JsonEncodingFormat,
    build_pooling_bytes_streaming_response,
    get_json_response_cls,
    get_pooling_output_encoder,
    get_pooling_usage,
)
from .protocol import (
    IOProcessorRequest,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
)

logger = init_logger(__name__)


class ServingPooling(PoolingServingBase):
    request_id_prefix = "pooling"

    def __init__(
        self,
        *args,
        supported_tasks: tuple[SupportedTask, ...],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.supported_tasks = supported_tasks
        self.pooling_task = self.model_config.get_pooling_task(supported_tasks)
        self.io_processors = init_pooling_io_processors(
            supported_tasks=supported_tasks,
            vllm_config=self.vllm_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )
        self.json_response_cls = get_json_response_cls()

    def get_io_processor(self, request: AnyPoolingRequest) -> PoolingIOProcessor:
        assert isinstance(request, PoolingRequest)
        pooling_task = self._verify_pooling_task(request)
        return self.io_processors[pooling_task]

    def _verify_pooling_task(self, request: PoolingRequest) -> str:
        if getattr(request, "dimensions", None) is not None:
            raise ValueError("dimensions is currently not supported")

        if request.task is None:
            request.task = self.pooling_task

        if isinstance(request, IOProcessorRequest):
            request.task = "plugin"

        assert request.task is not None
        pooling_task = request.task

        # plugin task uses io_processor.parse_request to verify inputs
        if pooling_task != "plugin" and pooling_task != self.pooling_task:
            if pooling_task not in self.supported_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_task!r} "
                    f"Supported tasks: {self.supported_tasks}"
                )
            else:
                raise ValueError(
                    "Try switching the model's pooling_task "
                    f"via --pooler-config.task {request.task}."
                )

        if pooling_task == "plugin" and "plugin" not in self.io_processors:
            raise ValueError(
                "No IOProcessor plugin installed. Please refer "
                "to the documentation and to the "
                "'prithvi_geospatial_mae_io_processor' "
                "offline inference example for more details."
            )

        return pooling_task

    def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        if ctx.response is not None:
            # for IOProcessorResponse
            return self.json_response_cls(content=ctx.response.model_dump())

        encoding_format = ctx.request.encoding_format
        embed_dtype = ctx.request.embed_dtype
        endianness = ctx.request.endianness

        if encoding_format == "float" or encoding_format == "base64":
            return self.request_output_to_pooling_json_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        if encoding_format == "bytes" or encoding_format == "bytes_only":
            return self.request_output_to_pooling_bytes_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        assert_never(encoding_format)

    def request_output_to_pooling_json_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: JsonEncodingFormat,
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> JSONResponse:
        encode_fn = get_pooling_output_encoder(
            encoding_format=encoding_format,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )

        items: list[PoolingResponseData] = []

        for idx, final_res in enumerate(final_res_batch):
            item = PoolingResponseData(
                index=idx,
                data=encode_fn(final_res),
            )

            items.append(item)

        response = PoolingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=get_pooling_usage(final_res_batch),
        )
        return self.json_response_cls(content=response.model_dump())

    def request_output_to_pooling_bytes_response(
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
