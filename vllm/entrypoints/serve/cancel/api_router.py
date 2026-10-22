# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.cancel.protocol import (
    CancelRequest,
    CancelResponse,
    SingleCancelResponse,
)
from vllm.entrypoints.serve.cancel.registry import CancellationRegistry
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> "EngineClient":
    return request.app.state.engine_client


def cancellation_registry(request: Request) -> CancellationRegistry:
    return request.app.state.cancellation_registry


async def _abort(raw_request: Request, request_id: str) -> bool:
    """Abort every engine request created for a client-visible request id."""
    engine_req_ids = cancellation_registry(raw_request).resolve(request_id)
    return bool(await engine_client(raw_request).abort(engine_req_ids))


@router.post(
    "/v1/requests/{request_id}/cancel",
    responses={
        HTTPStatus.OK.value: {"model": SingleCancelResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    },
)
@router.delete(
    "/v1/requests/{request_id}",
    responses={
        HTTPStatus.OK.value: {"model": SingleCancelResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    },
)
async def cancel_request(request_id: str, raw_request: Request):
    """Cancel an in-flight request by its request ID."""
    if not await _abort(raw_request, request_id):
        error = create_error_response(
            f"Request {request_id} not found or already completed.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return JSONResponse(
            content=error.model_dump(),
            status_code=HTTPStatus.NOT_FOUND.value,
        )

    return JSONResponse(
        content=SingleCancelResponse(request_id=request_id).model_dump(),
        status_code=HTTPStatus.OK.value,
    )


@router.post(
    "/v1/requests/cancel",
    responses={
        HTTPStatus.OK.value: {"model": CancelResponse},
    },
)
async def cancel_requests_batch(
    request: CancelRequest,
    raw_request: Request,
):
    """Cancel a list of in-flight requests."""
    if not request.request_ids:
        return JSONResponse(
            content=CancelResponse(cancelled_request_ids=[]).model_dump(),
            status_code=HTTPStatus.OK.value,
        )

    cancelled = [
        req_id for req_id in request.request_ids if await _abort(raw_request, req_id)
    ]

    return JSONResponse(
        content=CancelResponse(cancelled_request_ids=cancelled).model_dump(),
        status_code=HTTPStatus.OK.value,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
