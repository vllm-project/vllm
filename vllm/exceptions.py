# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for vLLM."""

from http import HTTPStatus
from typing import Any


class VLLMError(Exception):
    """Base class for all vLLM-specific errors.

    Subclasses are split into `VLLMClientError` (caused by the request, mapped
    to 4xx) and `VLLMServerError` (caused by the server, mapped to 5xx).
    Dispatching on this hierarchy lets the entrypoints decide the HTTP status
    without relying on raw Python exception types such as `ValueError`.
    """


class VLLMClientError(VLLMError):
    """Base class for errors caused by the client request (4xx)."""


class VLLMServerError(VLLMError):
    """Base class for errors caused by the server (5xx)."""


class VLLMValidationError(VLLMClientError):
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


class VLLMNotFoundError(VLLMClientError):
    """vLLM-specific NotFoundError"""

    pass


class LoRAAdapterNotFoundError(VLLMNotFoundError):
    """Exception raised when a LoRA adapter is not found.

    This exception is thrown when a requested LoRA adapter does not exist
    in the system.

    Attributes:
        message: The error message string describing the exception
    """

    message: str

    def __init__(
        self,
        lora_name: str,
        lora_path: str,
    ) -> None:
        message = f"Loading lora {lora_name} failed: No adapter found for {lora_path}"
        self.message = message

    def __str__(self):
        return self.message


class VLLMUnprocessableEntityError(VLLMClientError):
    """vLLM-specific error for unprocessable entity requests.

    This exception is raised when the request content is invalid or cannot be
    processed, such as when an image URL points to a non-existent or inaccessible
    resource (404, 403, DNS failure, etc.).

    Args:
        message: The error message describing the unprocessable entity.
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


class GracefulHTTPError(VLLMError):
    """Exception that should be translated into an HTTP error response.

    These are expected to occur during normal operation (e.g. admission
    control rejections) and should be surfaced to the client with the
    explicit HTTP status code they carry, rather than being mapped to a
    generic 4xx/5xx by the client/server split.
    """

    def __init__(self, message: str, http_status: HTTPStatus):
        super().__init__(message)
        self.message = message
        self.http_status = http_status


class QueueOverflowError(GracefulHTTPError):
    """Raised when admitting a request would exceed the request queue limit.

    Returns HTTP 503 (Service Unavailable) so that load balancers and
    client SDKs retry the request on a different instance.
    """

    def __init__(self):
        super().__init__(
            "The engine is currently busy and cannot accept new requests. "
            "Please try again later or on a different instance.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class MaxQueuedTokensError(GracefulHTTPError):
    """Raised when the pending prefill tokens exceed the configured limit.

    Returns HTTP 503 (Service Unavailable) so that load balancers and
    client SDKs retry the request on a different instance.
    """

    def __init__(self):
        super().__init__(
            "The engine has reached its prefill token backlog limit. "
            "Please try again later or on a different instance.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )
