"""Kernel test utils"""

from tests.utils import env_var_fixture
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def backend_override_fixture(backend_name: str) -> Iterator[None]:
    '''
    Text fixture, temporarily configures the vLLM backend by setting
    VLLM_ATTENTION_BACKEND, then resets the environment outside of
    the fixture.

    Usage:

        with backend_override_fixture("backend_name"):
            # code that depends on vLLM backend

        # VLLM_ATTENTION_BACKEND is returned to original value
        # or unset
    '''
    with env_var_fixture('VLLM_ATTENTION_BACKEND', backend_name):
        yield
