# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_BACKEND_ENV_VAR


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


def test_selector(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, "ROCM_FLASH")

        # Set the current platform to ROCm using monkeypatch
        monkeypatch.setattr("vllm.attention.selector.current_platform",
                            RocmPlatform())

        # Test standard ROCm attention
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
        assert (backend.get_name() == "ROCM_FLASH"
                or backend.get_name() == "TRITON_ATTN_VLLM_V1")

        # mla test for deepseek related
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False,
                                   False, True)
        assert backend.get_name() == "TRITON_MLA"
