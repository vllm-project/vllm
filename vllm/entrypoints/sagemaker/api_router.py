# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

import model_hosting_container_standards.sagemaker as sagemaker_standards
import pydantic
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.api_server import base
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.instrumentator.health import health
from vllm.tasks import POOLING_TASKS, SupportedTask

# TODO: RequestType = TypeForm[BaseModel] when recognized by type checkers
# (requires typing_extensions >= 4.13)
RequestType = Any
GetHandlerFn = Callable[[Request], OpenAIServing | None]
EndpointFn = Callable[[RequestType, Request], Awaitable[Any]]


def get_invocation_types(supported_tasks: tuple["SupportedTask", ...]):
    # NOTE: Items defined earlier take higher priority
    INVOCATION_TYPES: list[tuple[RequestType, tuple[GetHandlerFn, EndpointFn]]] = []

    if "generate" in supported_tasks:
        from vllm.entrypoints.openai.chat_completion.api_router import (
            chat,
            create_chat_completion,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.completion.api_router import (
            completion,
            create_completion,
        )
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest

        INVOCATION_TYPES += [
            (ChatCompletionRequest, (chat, create_chat_completion)),
            (CompletionRequest, (completion, create_completion)),
        ]

    if "embed" in supported_tasks:
        from vllm.entrypoints.pooling.embed.api_router import (
            create_embedding,
            embedding,
        )
        from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest

        INVOCATION_TYPES += [
            (EmbeddingRequest, (embedding, create_embedding)),
        ]

    if "classify" in supported_tasks:
        from vllm.entrypoints.pooling.classify.api_router import (
            classify,
            create_classify,
        )
        from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest

        INVOCATION_TYPES += [
            (ClassificationRequest, (classify, create_classify)),
        ]

    if "score" in supported_tasks:
        from vllm.entrypoints.pooling.score.api_router import do_rerank, rerank
        from vllm.entrypoints.pooling.score.protocol import RerankRequest

        INVOCATION_TYPES += [
            (RerankRequest, (rerank, do_rerank)),
        ]

    if "score" in supported_tasks or "embed" in supported_tasks:
        from vllm.entrypoints.pooling.score.api_router import create_score, score
        from vllm.entrypoints.pooling.score.protocol import ScoreRequest

        INVOCATION_TYPES += [
            (ScoreRequest, (score, create_score)),
        ]

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling.pooling.api_router import create_pooling, pooling
        from vllm.entrypoints.pooling.pooling.protocol import PoolingRequest

        INVOCATION_TYPES += [
            (PoolingRequest, (pooling, create_pooling)),
        ]

    return INVOCATION_TYPES


def attach_router(app: FastAPI, supported_tasks: tuple["SupportedTask", ...]):
    router = APIRouter()

    # NOTE: Construct the TypeAdapters only once
    INVOCATION_TYPES = get_invocation_types(supported_tasks)
    INVOCATION_VALIDATORS = [
        (pydantic.TypeAdapter(request_type), (get_handler, endpoint))
        for request_type, (get_handler, endpoint) in INVOCATION_TYPES
    ]

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

    app.include_router(router)


def sagemaker_standards_bootstrap(app: FastAPI) -> FastAPI:
    return sagemaker_standards.bootstrap(app)
