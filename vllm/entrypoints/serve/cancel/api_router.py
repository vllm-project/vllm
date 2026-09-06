# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
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
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> "EngineClient":
    return request.app.state.engine_client


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
    engine = engine_client(raw_request)

    # If engine provides has_request, check before aborting
    has_req_fn = getattr(engine, "has_request", None)
    if callable(has_req_fn) and not has_req_fn(request_id):
        error = create_error_response(
            f"Request {request_id} not found or already completed.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return JSONResponse(
            content=error.model_dump(),
            status_code=HTTPStatus.NOT_FOUND.value,
        )

    aborted = await engine.abort(request_id)
    if isinstance(aborted, list) and len(aborted) == 0:
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
    engine = engine_client(raw_request)
    if not request.request_ids:
        return JSONResponse(
            content=CancelResponse(cancelled_request_ids=[]).model_dump(),
            status_code=HTTPStatus.OK.value,
        )

    cancelled: list[str] = []
    for req_id in request.request_ids:
        res = await engine.abort(req_id)
        if not isinstance(res, list) or len(res) > 0:
            cancelled.append(req_id)

    return JSONResponse(
        content=CancelResponse(cancelled_request_ids=cancelled).model_dump(),
        status_code=HTTPStatus.OK.value,
    )


@router.post(
    "/abort_requests",
    responses={
        HTTPStatus.OK.value: {"model": CancelResponse},
    },
)
async def abort_requests(raw_request: Request):
    """Abort in-flight requests without pausing the scheduler.

    Accepts an optional JSON body with ``request_ids``. If empty or missing,
    aborts all in-flight requests.
    """
    engine = engine_client(raw_request)

    body = {}
    with contextlib.suppress(Exception):
        body = await raw_request.json()

    request_ids = body.get("request_ids") if isinstance(body, dict) else None

    if request_ids:
        cancelled: list[str] = []
        for req_id in request_ids:
            res = await engine.abort(req_id)
            if not isinstance(res, list) or len(res) > 0:
                cancelled.append(req_id)
        return JSONResponse(
            content=CancelResponse(cancelled_request_ids=cancelled).model_dump(),
            status_code=HTTPStatus.OK.value,
        )

    # Empty or missing request_ids: abort all in-flight requests
    op = getattr(engine, "output_processor", None)
    if op is not None:
        all_ids = [
            *getattr(op, "request_states", {}).keys(),
            *getattr(op, "parent_requests", {}).keys(),
        ]
        aborted = await engine.abort(all_ids, internal=True)
        return JSONResponse(
            content=CancelResponse(
                cancelled_request_ids=aborted if isinstance(aborted, list) else all_ids
            ).model_dump(),
            status_code=HTTPStatus.OK.value,
        )

    return JSONResponse(
        content=CancelResponse(cancelled_request_ids=[]).model_dump(),
        status_code=HTTPStatus.OK.value,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
