# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom exceptions for vLLM."""

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
