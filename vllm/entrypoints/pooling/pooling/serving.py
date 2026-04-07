# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

from fastapi import Request
from fastapi.responses import Response
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServingBase
from vllm.entrypoints.pooling.io_processor_factories import init_pooling_io_processors
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingBytesResponse,
    PoolingResponse,
    PoolingResponseData,
)
from vllm.entrypoints.pooling.typing import AnyPoolingRequest, PoolingServeContext
from vllm.entrypoints.pooling.utils import (
    encode_pooling_bytes,
    encode_pooling_output_base64,
    encode_pooling_output_float,
)
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.tasks import SupportedTask
from vllm.utils.serial_utils import EmbedDType, Endianness

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

    async def __call__(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ) -> Response:
        ctx = await self._init_ctx(request, raw_request)

        if request.task is None:
            request.task = self.pooling_task

        if getattr(request, "dimensions", None) is not None:
            raise ValueError("dimensions is currently not supported")

        # plugin task uses io_processor.parse_request to verify inputs
        if request.task != "plugin" and request.task != self.pooling_task:
            if request.task not in self.supported_tasks:
                raise ValueError(
                    f"Unsupported task: {request.task!r} "
                    f"Supported tasks: {self.supported_tasks}"
                )
            else:
                logger.warning_once(
                    "Pooling multitask support is deprecated and will be removed "
                    "in v0.20. When the default pooling task is not what you want, you "
                    "need to manually specify it via --pooler-config.task %s. ",
                    request.task,
                )

        if isinstance(request, IOProcessorRequest):
            if "plugin" not in self.io_processors:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details."
                )

            io_processor = self.io_processors["plugin"]
        else:
            io_processor = self.io_processors[request.task]
            await io_processor.pre_process_online_async(ctx)

            if ctx.pooling_params is None:
                ctx.pooling_params = io_processor.create_pooling_params(request)

        await self._prepare_generators(ctx)
        await self._collect_batch(ctx)

        await io_processor.post_process_online_async(ctx)
        return await self._build_response(ctx)

    async def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
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
        encoding_format: Literal["float", "base64"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingResponse:
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

        items: list[PoolingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = PoolingResponseData(
                index=idx,
                data=encode_fn(final_res),
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

    def request_output_to_pooling_bytes_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["bytes", "bytes_only"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> PoolingBytesResponse:
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

        return PoolingBytesResponse(content=content, headers=headers)
