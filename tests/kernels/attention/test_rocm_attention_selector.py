# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend
from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only test")


def test_rocm_backends_do_not_support_mm_prefix():
    """Regression test: ROCm backends must not claim mm_prefix support.

    If they do, prefix-LM multimodal models that attend vision tokens
    bidirectionally (e.g. Gemma3 4B) will use bidirectional attention
    via the ROCm backend instead of falling back to Triton, producing
    incorrect results. Remove this test only if support is properly added.
    """
    assert not RocmAttentionBackend.supports_mm_prefix(), (
        "RocmAttentionBackend does not support mm_prefix — "
        "defer to Triton unified attention instead for now"
    )
    assert not RocmAiterUnifiedAttentionBackend.supports_mm_prefix(), (
        "RocmAiterUnifiedAttentionBackend does not support mm_prefix — "
        "defer to Triton unified attention instead for now"
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


@pytest.mark.skip(reason="Skipped for now. Should be revisited.")
def test_selector(monkeypatch: pytest.MonkeyPatch):
    # Set the current platform to ROCm using monkeypatch
    monkeypatch.setattr("vllm.v1.attention.selector.current_platform", RocmPlatform())

    # Test standard ROCm attention
    attention_config = AttentionConfig(backend=AttentionBackendEnum.ROCM_ATTN)
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
        assert backend.get_name() == "ROCM_FLASH" or backend.get_name() == "TRITON_ATTN"

    # MLA test for deepseek related
    # Change the attention backend to triton MLA
    attention_config = AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA)
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False, use_mla=True)
        assert backend.get_name() == "TRITON_MLA"

    # If attention backend is None
    # If use_mla is true
    # The selected backend is triton MLA
    attention_config = AttentionConfig(backend=None)
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        backend = get_attn_backend(576, torch.bfloat16, "auto", 16, False, use_mla=True)
        assert backend.get_name() == "TRITON_MLA"

    # Change the attention backend to AITER MLA
    attention_config = AttentionConfig(backend=AttentionBackendEnum.ROCM_AITER_MLA)
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        backend = get_attn_backend(576, torch.bfloat16, "auto", 1, False, use_mla=True)
        assert backend.get_name() == "ROCM_AITER_MLA"

    # If attention backend is None
    # If use_mla is true
    # If VLLM_ROCM_USE_AITER is enabled
    # The selected backend is ROCM_AITER_MLA
    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")

        attention_config = AttentionConfig(backend=None)
        vllm_config = VllmConfig(attention_config=attention_config)

        with set_current_vllm_config(vllm_config):
            backend = get_attn_backend(
                576, torch.bfloat16, "auto", 1, False, use_mla=True
            )
            assert backend.get_name() == "ROCM_AITER_MLA"
