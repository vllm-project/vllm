class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Maybe recoverable."""
    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(args, kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
