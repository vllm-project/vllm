# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.logger import init_logger

from ..utils import sanitize_message

logger = init_logger(__name__)


async def http_exception_handler(req: Request, exc: HTTPException):
    if req.app.state.args.log_error_stack:
        logger.exception(
            "HTTPException caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )
    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(exc.detail),
            type=HTTPStatus(exc.status_code).phrase,
            code=exc.status_code,
        )
    )
    return JSONResponse(err.model_dump(), status_code=exc.status_code)
