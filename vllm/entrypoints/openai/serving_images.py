# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from typing import Optional, Union

from fastapi import Request

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse, ImageData,
                                              ImagesGenerationResponse,
                                              ImagesPredictionRequest)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.plugins.multimodal_data_processors.types import ImagePrompt

logger = init_logger(__name__)


class ServingImagesPrediction(OpenAIServing):

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

    def _request_to_model_prompt(
            self, request: ImagesPredictionRequest) -> PromptType:

        img_prompt = ImagePrompt(
            data=request.image.data,
            data_format=request.image.data_format,
            image_format=request.image_format,
            out_format=request.response_format,
        )
        return {
            "prompt_token_ids": [1],
            "multi_modal_data": {
                "image": dict(img_prompt)
            },
        }

    async def create_images_prediction(
        self,
        request: ImagesPredictionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ImagesGenerationResponse, ErrorResponse]:

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"images-prediction-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        # Schedule the request and get the result generator.
        # Note that at the moment, models capable of generating images
        # are piggybacking on the pooling models support.
        # See the PrithviMAEGeospatial model
        try:
            pooling_params = request.to_pooling_params()
            model_prompt = self._request_to_model_prompt(request)
            trace_headers = (None if raw_request is None else await
                             self._get_trace_headers(raw_request.headers))

            output = await self.engine_client.encode_with_mm_data_plugin(
                model_prompt,
                pooling_params,
                request_id,
                trace_headers=trace_headers,
                priority=request.priority,
            )
        except ValueError as e:
            return self.create_error_response(str(e))
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if output.task_output is None:
            return self.create_error_response(
                "Error post-processing poolin data,"
                " task_output should not be none")

        output_image = ImageData(data=output.task_output.data,
                                 data_format=output.task_output.type)

        return ImagesGenerationResponse(
            created=created_time,
            image=output_image,
            image_format=request.image_format,
        )
