# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for vLLM."""
import sys
import traceback
from http import HTTPStatus
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
else:
    ErrorResponse = Any


class VLLMValidationError(ValueError):
    """vLLM-specific validation error for request validation failures.

    Args:
        message: The error message describing the validation failure.
        parameter: Optional parameter name that failed validation.
        value: Optional value that was rejected during validation.
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    def __str__(self):
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base



def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
    log_error_stack: bool = False,
) -> ErrorResponse:
    exc: Exception | None = None

    from vllm.entrypoints.utils import sanitize_message
    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

    if isinstance(message, Exception):
        exc = message

        if isinstance(exc, VLLMValidationError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = exc.parameter
        elif isinstance(exc, (ValueError, TypeError, RuntimeError, OverflowError)):
            # Common validation errors from user input
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "NotImplementedError"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        elif exc.__class__.__name__ == "TemplateError":
            # jinja2.TemplateError (avoid importing jinja2)
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        else:
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None

        message = str(exc)

    if log_error_stack:
        exc_type, _, _ = sys.exc_info()
        if exc_type is not None:
            traceback.print_exc()
        else:
            traceback.print_stack()

    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=err_type,
            code=status_code.value,
            param=param,
        )
    )
