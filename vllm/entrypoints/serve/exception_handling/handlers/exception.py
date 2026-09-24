# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from starlette.responses import JSONResponse

from vllm.entrypoints.serve.utils.request_id import get_external_request_id
from vllm.logger import bind_external_request_id, init_logger

from ..error_response import create_error_response

logger = init_logger(__name__)


async def exception_handler(req: Request, exc: Exception):
    if req.app.state.args.log_error_stack:
        request_id = get_external_request_id(req)
        bind_external_request_id(logger, request_id).error(
            "Exception caught. Request id: %s",
            request_id,
        )

    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)
