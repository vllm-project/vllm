# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from starlette.responses import JSONResponse

from vllm.entrypoints.launchers.launcher import terminate_if_errored
from vllm.exceptions import GenerationError, VLLMError
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from ..error_response import create_error_response
from .exception import exception_handler

logger = init_logger(__name__)


async def vllm_error_handler(req: Request, exc: VLLMError):
    """Dispatch a vLLM-specific error to the appropriate handler."""
    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        return await engine_error_handler(req, exc)
    elif isinstance(exc, GenerationError):
        return await generation_error_handler(req, exc)
    else:
        return await exception_handler(req, exc)


async def engine_error_handler(
    req: Request, exc: EngineDeadError | EngineGenerateError
):
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.

    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).

    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.

    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    If an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down. Instead, we use the watchdog
    background task for check for errored state.
    """

    if req.app.state.args.log_error_stack:
        logger.exception(
            "Engine Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=req.app.state.engine_client,
    )
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


async def generation_error_handler(req: Request, exc: GenerationError):
    """Handle GenerationError without logging stack traces.

    GenerationError is a known, expected error (e.g. KV cache load failure)
    that should be returned to the client as a 500 response without polluting
    server logs with stack traces.
    """
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)
