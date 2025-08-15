# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, Sequence
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
from vllm.inputs.data import ImagePrompt, PromptType
from vllm.logger import init_logger
from vllm.outputs import ImageRequestOutput, PoolingRequestOutput
from vllm.plugins.multimodal_data_processors import (
    get_multimodal_data_processor)
from vllm.utils import merge_async_iterators

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

        # Load the multimodal data processing plugin
        self.multimodal_data_processor = get_multimodal_data_processor(
            vllm_config)

    async def _preprocess_request(
        self,
        request: ImagesPredictionRequest,
        request_id: str,
    ) -> Sequence[PromptType]:

        prompt = ImagePrompt(
            data=request.image.data,
            data_format=request.image.data_format,
            image_format=request.image_format,
        )

        model_prompts = (await
                         self.multimodal_data_processor.pre_process_async(
                             prompt=prompt,
                             request_id=request_id,
                         ))
        return model_prompts

    async def _postprocess_request(
        self,
        request: ImagesPredictionRequest,
        request_id: str,
        model_out: Sequence[Optional[PoolingRequestOutput]],
    ) -> ImageData:

        processor_output = (await
                            self.multimodal_data_processor.post_process_async(
                                model_out=model_out,
                                request_id=request_id,
                                out_format=request.response_format))

        assert isinstance(processor_output, ImageRequestOutput)

        return ImageData(data=processor_output.data,
                         data_format=request.response_format)

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

        try:
            # Here I am assuming that the image prediction request might
            # be split in multiple prompts because of tiling
            prompts = await self._preprocess_request(request=request,
                                                     request_id=request_id)

        except Exception as e:
            logger.exception("Error in preprocessing image")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        # Note that at the moment, models capable of generating images
        # are piggybacking on the pooling models support.
        # See the PrithviMAEGeospatial model
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()
            for i, prompt in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    prompt,
                    pooling_params,
                    request_id_item,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)
        num_prompts = len(prompts)

        final_res_batch: Sequence[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        try:
            response_image_data = await self._postprocess_request(
                request=request,
                request_id=request_id,
                model_out=final_res_batch,
            )
        except Exception as e:
            logger.exception("Error in post-processing model output")
            return self.create_error_response(str(e))

        return ImagesGenerationResponse(
            created=created_time,
            image=response_image_data,
            image_format=request.image_format,
        )
