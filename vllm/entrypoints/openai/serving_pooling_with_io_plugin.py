# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Optional, Union

from fastapi import Request

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              IOProcessorRequest,
                                              IOProcessorResponse)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)


class ServingPoolingWithIOPlugin(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        vllm_config: VllmConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=vllm_config.model_config,
            models=models,
            request_logger=request_logger,
        )

    async def create_pooling_with_io_plugin(
        self,
        request: IOProcessorRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[IOProcessorResponse, ErrorResponse]:

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"io-processor-{self._base_request_id(raw_request)}"

        try:
            pooling_params = request.to_pooling_params()
            trace_headers = (None if raw_request is None else await
                             self._get_trace_headers(raw_request.headers))

            output = (await self.engine_client.encode_with_io_processor(
                request,
                pooling_params,
                request_id,
                trace_headers=trace_headers,
                priority=request.priority,
            ))
        except ValueError as e:
            return self.create_error_response(str(e))
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        io_processor = await self.engine_client.get_io_processor()
        response = io_processor.output_to_response(output)

        return response
