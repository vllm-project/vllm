"""Kernel test utils"""

import pytest

STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"


def override_backend_env_variable(mpatch: pytest.MonkeyPatch,
                                  backend_name: str) -> None:
    '''
    Override the environment variable indicating the vLLM backend temporarily,
    using pytest monkeypatch to ensure that the env vars get
    reset once the test context exits.

    Arguments:

    * mpatch: pytest monkeypatch instance
    * backend_name: attention backend name to force
    '''
    mpatch.setenv(STR_BACKEND_ENV_VAR, backend_name)
