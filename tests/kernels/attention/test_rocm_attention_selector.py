# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

        # MLA test for deepseek related

        # change the attention backend to triton MLA
        m.setenv(STR_BACKEND_ENV_VAR, "TRITON_MLA")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False,
                                   False, True)
        assert backend.get_name() == "TRITON_MLA"

        # If attention backend is None
        # If use_mla is true
        # The selected backend is triton MLA
        m.setenv(STR_BACKEND_ENV_VAR, None)
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False,
                                   False, True)
        assert backend.get_name() == "TRITON_MLA"

        # change the attention backend to AITER MLA
        m.setenv(STR_BACKEND_ENV_VAR, "ROCM_AITER_MLA")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 1, False,
                                   False, True)
        assert (backend.get_name() == "ROCM_AITER_MLA"
                or backend.get_name() == "ROCM_AITER_MLA_VLLM_V1")

        # If attention backend is None
        # If use_mla is true
        # If VLLM_ROCM_USE_AITER is enabled
        # The selected backend is ROCM_AITER_MLA
        m.setenv(STR_BACKEND_ENV_VAR, None)
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 1, False,
                                   False, True)
        assert (backend.get_name() == "ROCM_AITER_MLA"
                or backend.get_name() == "ROCM_AITER_MLA_VLLM_V1")
