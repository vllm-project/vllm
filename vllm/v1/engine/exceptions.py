class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Maybe recoverable."""
    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(args, kwargs)

        # If we get an EngineDead signal when using LLMEngine,
        # we often shutdown the EngineCore while the main
        # process is still using ZMQ. This makes the root
        # cause clear in the stack trace.
        if suppress_context:
            self.__suppress_context__ = True
