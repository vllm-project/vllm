# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.logger import init_logger

logger = init_logger(__name__)


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    exc: Exception | None = None

    if isinstance(message, Exception):
        exc = message
        logger.debug(
            "create_error_response called with %s: %s", type(exc).__name__, exc
        )

        from vllm.exceptions import (
            VLLMNotFoundError,
            VLLMUnprocessableEntityError,
            VLLMValidationError,
        )

        if isinstance(exc, VLLMValidationError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = exc.parameter
        elif isinstance(exc, VLLMUnprocessableEntityError):
            err_type = "UnprocessableEntityError"
            status_code = HTTPStatus.UNPROCESSABLE_ENTITY
            param = exc.parameter
        elif isinstance(exc, VLLMNotFoundError):
            err_type = "NotFoundError"
            status_code = HTTPStatus.NOT_FOUND
            param = None
        elif isinstance(exc, (ValueError, TypeError, OverflowError)):
            # Common validation errors from user input
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "NotImplementedError"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        elif isinstance(exc, GenerationError):
            err_type = "InternalServerError"
            status_code = exc.status_code
            param = None
        elif any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__):
            # jinja2.TemplateError and its subclasses (avoid importing jinja2)
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        else:
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None

        message = str(exc)

    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=err_type,
            code=status_code.value,
            param=param,
        )
    )
