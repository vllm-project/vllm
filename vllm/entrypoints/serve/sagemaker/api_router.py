# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
import types
import typing
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

import pydantic
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from vllm.config import ModelConfig
from vllm.entrypoints.generate.factories import get_generate_invocation_types
from vllm.entrypoints.pooling.factories import get_pooling_invocation_types
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.instrumentator.basic import base
from vllm.entrypoints.serve.instrumentator.health import health
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.exceptions import VLLMValidationError
from vllm.tasks import SupportedTask

# TODO: RequestType = TypeForm[BaseModel] when recognized by type checkers
# (requires typing_extensions >= 4.13)
RequestType = Any
GetHandlerFn = Callable[[Request], BaseServing | None]
EndpointFn = Callable[[RequestType, Request], Awaitable[Any]]


def _required_fields(request_type: RequestType) -> list[frozenset[str]]:
    """The required top-level fields of each model that `request_type`
    accepts. A union contributes one set per member."""
    origin = typing.get_origin(request_type)
    if origin is typing.Annotated:
        return _required_fields(typing.get_args(request_type)[0])
    if origin in (typing.Union, types.UnionType):
        return [
            fields
            for member in typing.get_args(request_type)
            for fields in _required_fields(member)
        ]
    model_fields = getattr(request_type, "model_fields", {})
    return [
        frozenset(name for name, field in model_fields.items() if field.is_required())
    ]


def _snapshot_handler_levels() -> list[tuple[logging.Handler, int]]:
    """Snapshot handler levels for loggers that may be affected by third-party
    logging configuration side effects (e.g. model_hosting_container_standards
    calling configure_root_logger() at import time)."""
    loggers = [logging.getLogger(), logging.getLogger("vllm")]
    seen: set[int] = set()
    snapshot: list[tuple[logging.Handler, int]] = []
    for logger in loggers:
        for handler in logger.handlers:
            if id(handler) not in seen:
                seen.add(id(handler))
                snapshot.append((handler, handler.level))
    return snapshot


def _restore_handler_levels(
    snapshot: list[tuple[logging.Handler, int]],
) -> None:
    """Restore handler levels from a snapshot."""
    for handler, level in snapshot:
        handler.setLevel(level)


def attach_router(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    """Attach the SageMaker hosting endpoints to the API server.

    Handler levels are snapshotted and restored because importing
    model_hosting_container_standards may reconfigure root logging.
    """
    snapshot = _snapshot_handler_levels()
    try:
        _attach_router(app, supported_tasks, model_config)
    finally:
        _restore_handler_levels(snapshot)


def _attach_router(
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    """Register the SageMaker hosting routes (/ping, /invocations) on the
    app."""
    import model_hosting_container_standards.sagemaker as sagemaker_standards

    router = APIRouter()

    # NOTE: Construct the TypeAdapters only once
    INVOCATION_TYPES = get_generate_invocation_types(
        supported_tasks, model_config
    ) + get_pooling_invocation_types(supported_tasks, model_config)

    INVOCATION_VALIDATORS = [
        (
            pydantic.TypeAdapter(request_type),
            _required_fields(request_type),
            (get_handler, endpoint),
        )
        for request_type, (get_handler, endpoint) in INVOCATION_TYPES
    ]

    @router.post("/ping", response_class=Response)
    @router.get("/ping", response_class=Response)
    @sagemaker_standards.register_ping_handler
    async def ping(raw_request: Request) -> Response:
        """Ping check. Endpoint required for SageMaker."""
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
            (validator, required_fields, endpoint)
            for validator, required_fields, (get_handler, endpoint) in (
                INVOCATION_VALIDATORS
            )
            if get_handler(raw_request) is not None
        ]

        # The error of the first request type that the body has all the
        # required fields of, i.e. the type the client most likely meant.
        error: Exception | None = None
        for request_validator, required_fields, endpoint in valid_endpoints:
            try:
                request = request_validator.validate_python(body)
            except (pydantic.ValidationError, VLLMValidationError) as e:
                # Pydantic does not wrap a `VLLMValidationError` raised by a
                # request type's validators, so it must be caught as well.
                if (
                    error is None
                    and isinstance(body, dict)
                    and any(fields.issubset(body) for fields in required_fields)
                ):
                    error = e
                continue

            return await endpoint(request, raw_request)

        # Report the error in the same form as that type's own endpoint.
        if isinstance(error, pydantic.ValidationError):
            raise RequestValidationError(
                [
                    {**err, "loc": ("body", *err["loc"])}
                    for err in error.errors(include_url=False)
                ]
            )
        if error is not None:
            raise error

        type_names = [
            t.__name__ if isinstance(t := validator._type, type) else str(t)
            for validator, _, _ in valid_endpoints
        ]
        msg = f"Cannot find suitable handler for request. Expected one of: {type_names}"
        res = base(raw_request).create_error_response(message=msg)
        return JSONResponse(content=res.model_dump(), status_code=res.error.code)

    app.include_router(router)


def sagemaker_standards_bootstrap(app: FastAPI) -> FastAPI:
    """Bootstrap the app with the SageMaker hosting standards.

    Handler levels are restored right after the import because importing
    model_hosting_container_standards may reconfigure root logging, and
    bootstrap must run with the original levels.
    """
    snapshot = _snapshot_handler_levels()
    try:
        import model_hosting_container_standards.sagemaker as sagemaker_standards

        app = sagemaker_standards.bootstrap(app)
    finally:
        _restore_handler_levels(snapshot)
    return app
