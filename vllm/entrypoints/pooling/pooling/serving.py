# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import assert_never, cast

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from vllm.distributed.io_transfer import (
    ensure_io_transfer_initialized,
    get_io_transfer,
    has_io_transfer,
)
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.pooling.pooling.tensor_serde import encode_tensor
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import decode_mm_kwargs_item
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import create_json_response
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import mm_input
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.tasks import SupportedTask
from vllm.utils.serial_utils import EmbedDType, Endianness

from ..base.io_processor import PoolingIOProcessor
from ..base.serving import PoolingBaseServing
from ..factories import init_pooling_io_processors
from ..typing import AnyPoolingRequest, PoolingServeContext
from ..utils import (
    BytesEncodingFormat,
    JsonEncodingFormat,
    build_pooling_bytes_streaming_response,
    get_pooling_output_encoder,
    get_pooling_usage,
)
from .protocol import (
    DataLocation,
    DataObject,
    DataRef,
    GeneratePoolingRequest,
    GeneratePoolingResponse,
    IOProcessorRequest,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
)

logger = init_logger(__name__)


class ServingPooling(PoolingBaseServing):
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

        # Inference instances reuse the renderer executor for connector calls
        # rather than creating a dedicated IO thread pool.
        ensure_io_transfer_initialized(self.vllm_config)
        self.io_connector = get_io_transfer() if has_io_transfer() else None

    def get_io_processor(self, request: AnyPoolingRequest) -> PoolingIOProcessor:
        request = cast(PoolingRequest, request)
        pooling_task = self._verify_pooling_task(request)
        return self.io_processors[pooling_task]

    def _verify_pooling_task(self, request: PoolingRequest) -> str:
        if getattr(request, "dimensions", None) is not None:
            raise VLLMValidationError(
                "dimensions is currently not supported", parameter="dimensions"
            )

        if request.task is None:
            request.task = self.pooling_task

        if isinstance(request, IOProcessorRequest):
            request.task = "plugin"

        assert request.task is not None
        pooling_task = request.task

        # plugin task uses io_processor.parse_data to verify inputs
        if pooling_task != "plugin" and pooling_task != self.pooling_task:
            if pooling_task not in self.supported_tasks:
                raise VLLMValidationError(
                    f"Unsupported task: {pooling_task!r} "
                    f"Supported tasks: {self.supported_tasks}",
                    parameter="task",
                )
            else:
                raise VLLMValidationError(
                    "Try switching the model's pooling_task "
                    f"via --pooler-config.task {request.task}.",
                    parameter="task",
                )

        if pooling_task == "plugin" and "plugin" not in self.io_processors:
            raise VLLMValidationError(
                "No IOProcessor plugin installed. Please refer "
                "to the documentation and to the "
                "'prithvi_geospatial_mae_io_processor' "
                "offline inference example for more details.",
                parameter="task",
            )

        return pooling_task

    def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        if ctx.response is not None:
            # for IOProcessorResponse
            return create_json_response(ctx.response)

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
    ) -> Response:
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
        return create_json_response(response)

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

    async def _generate_pooling_engine_call(
        self,
        ctx: PoolingServeContext,
        request: GeneratePoolingRequest,
    ) -> GeneratePoolingResponse:
        """Execute the engine call for a GeneratePoolingRequest.
        :meth:`PoolingServingBase._engine_call`.
        """
        # Derive output object_id from the inbound data_ref so the
        # derender side can locate it without recomputing any hash.
        # Replace '/' in model name so it is safe to embed in a
        # filename on both sender and receiver
        assert request.data_ref is not None
        assert isinstance(self.vllm_config.model_config.served_model_name, str)
        safe_model_name = self.vllm_config.model_config.served_model_name.replace(
            "/", "_"
        )
        output_object_id = f"{request.data_ref.object_id}:{safe_model_name}:output"

        mm_placeholders = {"image": [PlaceholderRange(offset=0, length=0)]}
        mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
        if request.data is not None and request.data_ref is not None:
            mm_kwargs = {"image": [decode_mm_kwargs_item(request.data.data)]}
        elif request.data_ref is not None and self.io_connector:
            data_obj = await self.io_connector.get_data_async(
                request.data_ref, executor=self._executor
            )
            # The connector owns serialization: get_data() returns the
            # raw MultiModalKwargsItem — no base64 decode needed.
            mm_kwargs = {"image": [cast(MultiModalKwargsItem, data_obj.data)]}
        else:
            raise KeyError("Couldn't find data.")

        mm_hashes = {"image": [request.data_ref.object_id]}

        engine_input = mm_input(
            prompt_token_ids=request.token_ids,
            mm_kwargs=MultiModalKwargsItems(mm_kwargs),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            cache_salt=request.cache_salt,
        )

        assert self.engine_client is not None

        generator = self.engine_client.encode(
            engine_input,
            request.to_pooling_params(),
            request.request_id,
            priority=request.priority,
        )

        assert generator is not None

        pooling_request_output: PoolingRequestOutput[PoolingOutput] = await anext(
            generator
        )

        tensor = pooling_request_output.outputs.data

        output_data_obj: DataObject | None
        if self.io_connector:
            # Pass the raw tensor — the connector owns serialization
            # and will use MsgpackEncoder for a zero-copy ZMQ transfer.
            data_ref = await self.io_connector.put_data_async(
                output_object_id,
                tensor,
                executor=self._executor,
            )
            output_data_obj = None
        else:
            # No connector: encode to base64 for JSON transport.
            data_ref = DataRef(
                object_id=output_object_id, location=DataLocation(locator="")
            )
            output_data_obj = DataObject(data=encode_tensor(tensor))

        return GeneratePoolingResponse(
            request_id=request.request_id,
            output_data_ref=data_ref,
            output_metadata_ref=request.metadata_ref,  # Forwarding it to the next stage
            output_data=output_data_obj,
            output_metadata=request.metadata,  # Forwarding it to the next stage
        )

    async def serve_pooling_embeds(
        self,
        request: GeneratePoolingRequest,
        raw_request: Request | None = None,
    ) -> GeneratePoolingResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        if self.engine_client and self.engine_client.errored:
            raise self.engine_client.dead_error

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        io_processor = self.io_processors["plugin"]
        model_name = self.models.model_name()

        pooling_params = io_processor.create_pooling_params(request)
        ctx = PoolingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            pooling_params=pooling_params,
            request_id=request.request_id,
            lora_request=self._maybe_get_adapters(request),
            priorities=request.priority,
            prompt_extras=None,
        )

        self._validate_request(ctx)

        return await self._generate_pooling_engine_call(ctx, request)
