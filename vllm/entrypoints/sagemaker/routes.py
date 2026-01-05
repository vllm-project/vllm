# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

import model_hosting_container_standards.sagemaker as sagemaker_standards
import pydantic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.api_server import (
    base,
    chat,
    completion,
    create_chat_completion,
    create_completion,
    validate_json_request,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.pooling.classify.api_router import classify, create_classify
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.entrypoints.pooling.embed.api_router import create_embedding, embedding
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.pooling.pooling.api_router import create_pooling, pooling
from vllm.entrypoints.pooling.pooling.protocol import PoolingRequest
from vllm.entrypoints.pooling.score.api_router import (
    create_score,
    do_rerank,
    rerank,
    score,
)
from vllm.entrypoints.pooling.score.protocol import RerankRequest, ScoreRequest
from vllm.entrypoints.serve.instrumentator.health import health

# TODO: RequestType = TypeForm[BaseModel] when recognized by type checkers
# (requires typing_extensions >= 4.13)
RequestType = Any
GetHandlerFn = Callable[[Request], OpenAIServing | None]
EndpointFn = Callable[[RequestType, Request], Awaitable[Any]]

# NOTE: Items defined earlier take higher priority
INVOCATION_TYPES: list[tuple[RequestType, tuple[GetHandlerFn, EndpointFn]]] = [
    (ChatCompletionRequest, (chat, create_chat_completion)),
    (CompletionRequest, (completion, create_completion)),
    (EmbeddingRequest, (embedding, create_embedding)),
    (ClassificationRequest, (classify, create_classify)),
    (ScoreRequest, (score, create_score)),
    (RerankRequest, (rerank, do_rerank)),
    (PoolingRequest, (pooling, create_pooling)),
]

# NOTE: Construct the TypeAdapters only once
INVOCATION_VALIDATORS = [
    (pydantic.TypeAdapter(request_type), (get_handler, endpoint))
    for request_type, (get_handler, endpoint) in INVOCATION_TYPES
]


def register_sagemaker_routes(router: APIRouter):
    @router.post("/ping", response_class=Response)
    @router.get("/ping", response_class=Response)
    @sagemaker_standards.register_ping_handler
    async def ping(raw_request: Request) -> Response:
        """Ping check. Endpoint required for SageMaker"""
        return await health(raw_request)

    @router.post(
        "/invocations",
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        },
    )
    @sagemaker_standards.register_invocation_handler
    @sagemaker_standards.stateful_session_manager()
    @sagemaker_standards.inject_adapter_id(adapter_path="model")
    async def invocations(raw_request: Request):
        """For SageMaker, routes requests based on the request type."""
        try:
            body = await raw_request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            ) from e

        valid_endpoints = [
            (validator, endpoint)
            for validator, (get_handler, endpoint) in INVOCATION_VALIDATORS
            if get_handler(raw_request) is not None
        ]

        for request_validator, endpoint in valid_endpoints:
            try:
                request = request_validator.validate_python(body)
            except pydantic.ValidationError:
                continue

            return await endpoint(request, raw_request)

        type_names = [
            t.__name__ if isinstance(t := validator._type, type) else str(t)
            for validator, _ in valid_endpoints
        ]
        msg = f"Cannot find suitable handler for request. Expected one of: {type_names}"
        res = base(raw_request).create_error_response(message=msg)
        return JSONResponse(content=res.model_dump(), status_code=res.error.code)

    return router
