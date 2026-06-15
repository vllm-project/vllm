# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.api_router import models
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@sagemaker_standards.register_load_adapter_handler(
    request_shape={
        "lora_name": "body.name",
        "lora_path": "body.src",
        "load_inplace": "body.load_inplace || `false`",
        "is_3d_lora_weight": "body.is_3d_lora_weight || `false`",
    },
)
async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
    handler: OpenAIServingModels = models(raw_request)
    response = await handler.load_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )

    return Response(status_code=200, content=response)


@sagemaker_standards.register_unload_adapter_handler(
    request_shape={
        "lora_name": "path_params.adapter_name",
    }
)
async def unload_lora_adapter(request: UnloadLoRAAdapterRequest, raw_request: Request):
    handler: OpenAIServingModels = models(raw_request)
    response = await handler.unload_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )

    return Response(status_code=200, content=response)
