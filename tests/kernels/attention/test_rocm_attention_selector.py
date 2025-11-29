# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.rocm import RocmPlatform


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


@pytest.mark.skip(reason="Skipped for now. Should be revisited.")
def test_selector(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", "ROCM_ATTN")

        # Set the current platform to ROCm using monkeypatch
        monkeypatch.setattr("vllm.attention.selector.current_platform", RocmPlatform())

        # Test standard ROCm attention
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
        assert backend.get_name() == "ROCM_FLASH" or backend.get_name() == "TRITON_ATTN"

        # MLA test for deepseek related

        # change the attention backend to triton MLA
        m.setenv("VLLM_ATTENTION_BACKEND", "TRITON_MLA")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False, use_mla=True)
        assert backend.get_name() == "TRITON_MLA"

        # If attention backend is None
        # If use_mla is true
        # The selected backend is triton MLA
        m.setenv("VLLM_ATTENTION_BACKEND", "")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False, use_mla=True)
        assert backend.get_name() == "TRITON_MLA"

        # change the attention backend to AITER MLA
        m.setenv("VLLM_ATTENTION_BACKEND", "ROCM_AITER_MLA")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 1, False, use_mla=True)
        assert backend.get_name() == "ROCM_AITER_MLA"

        # If attention backend is None
        # If use_mla is true
        # If VLLM_ROCM_USE_AITER is enabled
        # The selected backend is ROCM_AITER_MLA
        m.setenv("VLLM_ATTENTION_BACKEND", "")
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        backend = get_attn_backend(576, torch.bfloat16, "auto", 1, False, use_mla=True)
        assert backend.get_name() == "ROCM_AITER_MLA"
