# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus


class GracefulHTTPError(ValueError):
    """Represent exceptions that should be propagated gracefully to the user.

    These are exceptions that are expected to occur during normal operation.
    They will be caught and translated into a proper HTTP response.

    vLLM by default only catches ValueError in most places.
    To avoid breaking too many assumptions, we subclass ValueError.
    """

    def __init__(self, message: str, http_status: HTTPStatus):
        super().__init__(message)
        self.message = message
        self.http_status = http_status


class QueueOverflowError(GracefulHTTPError):
    def __init__(self):
        super().__init__("Request queue full.", HTTPStatus.SERVICE_UNAVAILABLE)


class MaxPendingTokensError(GracefulHTTPError):
    def __init__(self):
        super().__init__(
            "Request queue reached max pending tokens limit.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )


class MaxTierPendingTokensError(GracefulHTTPError):
    def __init__(self):
        super().__init__(
            "Request queue reached max pending tokens for tier.",
            HTTPStatus.TOO_MANY_REQUESTS,
        )


class TooManyRequestsError(GracefulHTTPError):
    def __init__(self):
        super().__init__(
            "Rate limit reached for given tier.", HTTPStatus.TOO_MANY_REQUESTS
        )


class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
