# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        import traceback
        
        # Capture stack trace at creation point for debugging
        creation_stack = ''.join(traceback.format_stack())
        
        ENGINE_DEAD_MESSAGE = (
            "EngineCore encountered an issue. See stack trace (above) for the root cause.\n"
            f"EngineDeadError was created at:\n{creation_stack}"
        )

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
        
        # Store creation stack for logging
        self.creation_stack = creation_stack
