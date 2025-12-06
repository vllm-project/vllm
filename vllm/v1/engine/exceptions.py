# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineSleepingError(Exception):
    """Raised when the engine is sleeping and cannot process requests.
    
    This is a recoverable error - the user can call /wake_up to resume
    the engine and retry their request. Returns HTTP 503 to indicate
    the service is temporarily unavailable.
    """

    def __init__(self, message: str = "Engine is sleeping. Call /wake_up to resume."):
        super().__init__(message)


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
