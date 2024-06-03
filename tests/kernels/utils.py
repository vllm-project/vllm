"""Kernel test utils"""

import pytest

STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"


def override_backend(mpatch: pytest.MonkeyPatch, backend_name: str) -> None:
    '''
    Override vLLM attention backend temporarily,
    using pytest monkeypatch to ensure that the env vars get
    reset once the test context exits.

    Arguments:

    * mpatch: pytest monkeypatch instance
    * backend_name: attention backend name to force
    '''
    mpatch.setenv(STR_BACKEND_ENV_VAR, backend_name)
