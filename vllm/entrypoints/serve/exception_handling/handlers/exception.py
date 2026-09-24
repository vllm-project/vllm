# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from starlette.responses import JSONResponse

from vllm.logger import init_logger

from ..error_response import create_error_response

logger = init_logger(__name__)


async def exception_handler(req: Request, exc: Exception):
    if req.app.state.args.log_error_stack:
        logger.error(
            "Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)
