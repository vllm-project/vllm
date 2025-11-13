# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from http import HTTPStatus

import model_hosting_container_standards.sagemaker as sagemaker_standards
import pydantic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.api_server import (
    INVOCATION_VALIDATORS,
    base,
    health,
    validate_json_request,
)
from vllm.entrypoints.openai.protocol import ErrorResponse


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
