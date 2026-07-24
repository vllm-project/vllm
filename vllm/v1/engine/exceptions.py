# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        # Use the custom message as the sole message, ignoring any user-supplied message in args
        # to avoid duplication and confusion.
        super().__init__(ENGINE_DEAD_MESSAGE, **kwargs)
        # Properly suppress the context for cleaner tracebacks.
        self.__suppress_context__ = suppress_context
