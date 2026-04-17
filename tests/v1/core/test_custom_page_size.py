# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `custom_page_size` seam on AttentionSpec.

These tests verify that:
1. AttentionBackend.get_kv_cache_page_size() defaults to None.
2. An AttentionSpec with custom_page_size returns that value from
   real_page_size_bytes instead of the head_size-based formula.
3. FullAttentionSpec.merge and MLAAttentionSpec.merge preserve
   custom_page_size.
"""
import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec


class _QuantizedStubBackend(AttentionBackend):
    """Minimal backend that reports a smaller page size than head_size*dtype."""

    @classmethod
    def get_kv_cache_page_size(
        cls, block_size, num_kv_heads, head_size, dtype, cache_dtype_str="auto"
    ):
        # Pretend 4-bit quantization: 4 bits per element → head_size / 2 bytes.
        return 2 * block_size * num_kv_heads * (head_size // 2)


def test_default_backend_returns_none():
    # The base class returns None; overriding is opt-in.
    assert AttentionBackend.get_kv_cache_page_size(
        block_size=16, num_kv_heads=8, head_size=128,
        dtype=torch.float16, cache_dtype_str="auto",
    ) is None


def test_stub_backend_returns_custom_size():
    got = _QuantizedStubBackend.get_kv_cache_page_size(
        block_size=16, num_kv_heads=8, head_size=128,
        dtype=torch.float16, cache_dtype_str="fp8",
    )
    # 2 * 16 * 8 * 64 = 16384 (half of the fp16 default 32768).
    assert got == 16384


def test_spec_custom_page_size_overrides_formula():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128,
        dtype=torch.float16,
        custom_page_size=12345,
    )
    assert spec.real_page_size_bytes == 12345


def test_spec_default_still_uses_formula():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128,
        dtype=torch.float16,
    )
    # block_size * num_kv_heads * head_size * dtype_size
    # 16 * 8 * 128 * 2 = 32768 (FullAttentionSpec formula)
    assert spec.real_page_size_bytes == 32768


def test_full_attention_spec_merge_preserves_custom_page_size():
    a = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128,
        dtype=torch.float16, custom_page_size=9999,
    )
    merged = FullAttentionSpec.merge([a, a])
    assert merged.custom_page_size == 9999


def test_mla_attention_spec_merge_preserves_custom_page_size():
    a = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576,
        dtype=torch.bfloat16, cache_dtype_str="auto",
        custom_page_size=7777,
    )
    merged = MLAAttentionSpec.merge([a])
    assert merged.custom_page_size == 7777


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
