# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test():
    return False


def test_triton_attn_backend_supports_inline_scales_flag():
    """Verify default class attribute on backend and customization behavior."""
    assert AttentionBackend.supports_inline_scales is True
    assert TritonAttentionBackend.supports_inline_scales is True

    # Per-token-head INT8 spec with head_size=256
    spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )

    # 1. Default: inline scales pack 4-byte fp32 scale per head
    custom_spec = TritonAttentionBackend.customize_spec(spec)
    # hs_k=256, hs_v=256, scale_bytes=4 -> (256 + 256) * 1 + 2 * 4 = 520 bytes
    assert custom_spec.state_content_bytes == 520
    assert custom_spec.state_content_bytes % 128 != 0

    # 2. Gather / sparse backend opting out of inline scales
    class MockGatherAttentionBackend(TritonAttentionBackend):
        supports_inline_scales: ClassVar[bool] = False

    decoupled_spec = MockGatherAttentionBackend.customize_spec(spec)
    # Preserves natural unpadded spec (512 bytes per slot, 128B aligned)
    assert decoupled_spec.state_content_bytes is None


def test_triton_attn_impl_inline_scale_caches():
    """Verify inline scale cache views when supports_inline_scales=True."""
    num_blocks = 2
    nkv = 2
    block_size = 64
    head_size = 256
    scale_pad = 4  # sizeof(float32) / sizeof(int8)
    content = 2 * (head_size + scale_pad)  # 520 bytes

    # Inline packed KV cache: (num_blocks, nkv, block_size, 520)
    kv_cache = torch.zeros(
        (num_blocks, nkv, block_size, content),
        dtype=torch.int8,
    )

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=nkv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int8_per_token_head",
        supports_inline_scales=True,
    )
    assert impl.supports_inline_scales is True

    key_cache, value_cache = impl._pth_key_value_caches(kv_cache)
    # Padded head size: 260
    assert key_cache.shape == (num_blocks, block_size, nkv, 260)
    assert value_cache.shape == (num_blocks, block_size, nkv, 260)
    assert impl._k_scale_cache is not None
    assert impl._v_scale_cache is not None
    assert impl._k_scale_cache.shape == (num_blocks, block_size, nkv)
    assert impl._v_scale_cache.shape == (num_blocks, block_size, nkv)


def test_triton_attn_impl_decoupled_side_tensor_scale_caches():
    """Verify decoupled side-tensor scale caches when supports_inline_scales=False."""
    num_blocks = 2
    nkv = 2
    block_size = 64
    head_size = 256
    # Natural unpadded KV cache: (num_blocks, nkv, block_size, 512)
    content = 2 * head_size  # 512 bytes (256B K + 256B V)

    kv_cache = torch.zeros(
        (num_blocks, nkv, block_size, content),
        dtype=torch.int8,
    )

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=nkv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int8_per_token_head",
        supports_inline_scales=False,
    )
    assert impl.supports_inline_scales is False

    key_cache, value_cache = impl._pth_key_value_caches(kv_cache)

    # 1. KV cache split preserves natural 128B alignment
    assert key_cache.shape == (num_blocks, block_size, nkv, head_size)
    assert value_cache.shape == (num_blocks, block_size, nkv, head_size)
    assert key_cache.shape[-1] % 128 == 0
    assert value_cache.shape[-1] % 128 == 0

    # 2. Scale caches allocated as decoupled side tensors with float32
    assert impl._k_scale_cache is not None
    assert impl._v_scale_cache is not None
    assert impl._k_scale_cache.shape == (num_blocks, block_size, nkv)
    assert impl._v_scale_cache.shape == (num_blocks, block_size, nkv)
    assert impl._k_scale_cache.dtype == torch.float32
    assert impl._v_scale_cache.dtype == torch.float32
    assert torch.all(impl._k_scale_cache == 1.0)
    assert torch.all(impl._v_scale_cache == 1.0)

    # 3. Decoupled tensors have distinct storage from kv_cache
    assert impl._k_scale_cache.data_ptr() != kv_cache.data_ptr()
    assert impl._v_scale_cache.data_ptr() != kv_cache.data_ptr()


def test_attention_layer_forwarding_supports_inline_scales():
    """Verify backend supports_inline_scales forwards to AttentionImpl."""
    import inspect

    class MockGatherAttentionBackend(TritonAttentionBackend):
        supports_inline_scales: ClassVar[bool] = False

    backend = MockGatherAttentionBackend()
    impl_cls = backend.get_impl_cls()
    sig = inspect.signature(impl_cls.__init__)
    extra_impl_args = {}
    if (
        hasattr(backend, "supports_inline_scales")
        and "supports_inline_scales" in sig.parameters
    ):
        extra_impl_args.setdefault(
            "supports_inline_scales", backend.supports_inline_scales
        )

    impl = impl_cls(
        num_heads=4,
        head_size=256,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int8_per_token_head",
        **extra_impl_args,
    )
    assert impl.supports_inline_scales is False
