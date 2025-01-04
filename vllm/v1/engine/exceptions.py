# There exceptions are raised by the LLMEngine and AsyncLLM
# when errors occur. See vllm/entrypoints/launcher.py for the 
# handlers of these exceptions in the API Server.

# Raised when a AsyncLLM.generate() fails. Possibly recoverable.
class EngineGenerateError(Exception):
    pass

# Raised when the EngineCore dies. Unrecoverable.
class EngineDeadError(Exception):
    pass

def engine_dead_error_guard(func):
    """
    Decorator to be used by functions that call engine_core.
    engine_core runs in a background process and sends a fatal
    signal to the LLMEngine if it encounters an error. The
    LLMEngine handles this signal, sets self._errored, and then
    calls self.shutdown(), which kills engine_core.

    After the signal is handled, we will get an exception if
    we try to interact with the engine_core. This decorator
    catches the exception and raises an a more accurate 
    EngineDeadError exception to make the fundamental issue
    clearer to the end user.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # NOTE: args[0] is self (EngineCoreMPClient)
            if not args[0]._errored:
                raise e
            else:
                new_e = EngineDeadError(
                    "Engine got error in background worker process. "
                    "See stack trace for root cause issue.")
                # Convert the exception to EngineDeadError to give the
                # user a clear failure reason, suppressing.
                # https://docs.python.org/3/library/exceptions.html#exception-context # noqa: E501
                new_e.__suppress_context__ = True
                raise new_e from None

    return wrapper