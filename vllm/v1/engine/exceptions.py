# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.exceptions import VLLMServerError


class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable.

    `str(exc)` is the log-oriented message for server logs and offline
    tracebacks; `CLIENT_MESSAGE` is what API clients receive.
    """

    ENGINE_DEAD_MESSAGE = (
        "EngineCore encountered an issue. See stack trace (above) for the root cause."
    )
    CLIENT_MESSAGE = "The server had an error while processing your request."

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(self.ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


class EngineUnhealthyError(VLLMServerError):
    """Raised when the engine is alive but not ready to serve traffic.

    Args:
        message: Human-readable description.
        reason: Machine-readable not-ready reason reported by ``/health_generate``.
        details: Extra JSON-serializable fields reported by ``/health_generate``.

    """

    def __init__(self, message: str = "", reason: str = "unhealthy", **details: Any):
        super().__init__(message)
        self.reason = reason
        self.details = details


class EngineSleepingError(EngineUnhealthyError):
    """Raised when the engine is intentionally sleeping or paused."""

    def __init__(self, message: str = "", **details: Any):
        super().__init__(message, reason="sleeping", **details)
