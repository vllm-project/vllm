"""Kernel test utils"""

STR_BACKEND_ENV_VAR = "VLLM_ATTENTION_BACKEND"
STR_FLASH_ATTN_VAL = "FLASH_ATTN"
STR_INVALID_VAL = "INVALID"


def override_backend(mpatch, backend_name):
    mpatch.setenv(STR_BACKEND_ENV_VAR, backend_name)
