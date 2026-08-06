# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


def test_sm90_fa4_fp8_kv_uses_exact_kernel_block_size(monkeypatch):
    monkeypatch.setattr(
        FlashAttentionBackend,
        "_get_sm90_fa4_fp8_kv_block_size",
        staticmethod(lambda: 64),
    )

    assert FlashAttentionBackend.get_supported_kernel_block_sizes() == [64]
    assert FlashAttentionBackend.get_preferred_block_size(16) == 64
    assert FlashAttentionBackend.get_preferred_block_size(128) == 128


def test_generic_flash_attn_accepts_multiples_of_16(monkeypatch):
    monkeypatch.setattr(
        FlashAttentionBackend,
        "_get_sm90_fa4_fp8_kv_block_size",
        staticmethod(lambda: None),
    )

    (supported_size,) = FlashAttentionBackend.get_supported_kernel_block_sizes()
    assert isinstance(supported_size, MultipleOf)
    assert supported_size.base == 16
