'''Example code utils'''

import os
from contextlib import contextmanager
from typing import Generator

from vllm.utils import STR_BACKEND_ENV_VAR


@contextmanager
def override_backend_env_var_context_manager(
    backend_name: str, ) -> Generator[None, None, None]:
    '''
    Override the environment variable indicating the vLLM backend temporarily,
    in a context where pytest monkeypatch is not available (i.e. *outside*
    the context of a unit test, such as in an example code file.)

    Accomplish this using a custom context manager.

    Arguments:

    * backend_name: attention backend name to force

    Returns:

    * Generator
    '''

    key = STR_BACKEND_ENV_VAR

    # Save the current state of the environment variable (if it exists)
    original_value = os.environ.get(key, None)

    # Set the new value of the environment variable
    os.environ[key] = backend_name

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the environment variable to its original state
        if original_value is None:
            os.environ.pop(
                key, None)  # Remove the variable if it wasn't originally set
        else:
            os.environ[
                key] = original_value  # Revert back to the original value
