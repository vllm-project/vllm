# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineRequestTimeoutError(EngineGenerateError):
    """Raised when a request exceeds the ``request_timeout_s`` or
    ``request_stall_timeout_s`` limits set in ``SchedulerConfig``.
    The request has been aborted.

    Subclasses EngineGenerateError so existing handlers for recoverable
    generate failures still match, while callers that need to distinguish
    a timeout can catch this more-specific type."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
