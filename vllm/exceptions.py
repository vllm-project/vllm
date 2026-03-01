# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for vLLM."""

from http import HTTPStatus
from typing import Any


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


class GenerationError(Exception):
    """Raised when finish_reason indicates internal server error (500)"""

    def __init__(
        self,
        message: str = "Internal server error",
        err_type: str = "InternalServerError",
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message)
        self.err_type = err_type
        self.status_code = status_code


class RequestRejectedError(Exception):
    """Raised when finish_reason indicates the request was rejected
    (e.g., queue full, rate-limited, etc)."""

    def __init__(
        self,
        message: str = "Request was rejected",
        err_type: str = "ServiceUnavailableError",
        status_code: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE,
    ):
        super().__init__(message)
        self.err_type = err_type
        self.status_code = status_code
