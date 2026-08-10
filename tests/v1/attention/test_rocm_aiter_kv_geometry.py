# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache geometry contract for the AITER attention backends.

The AITER fused QK-norm+RoPE+cache kernel addresses each side as
``block_id * stride(0) + token * (H*hs) + head * hs`` and hard-asserts that
each side is contiguous from dim 1 (``k_cache/v_cache must be contiguous
within a block``). Violating it aborts the process rather than raising, so
this pins the allocated shape and the strides the split views hand to the
kernel.
"""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
)
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
    RocmAiterUnifiedAttentionImpl,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 128

pytestmark = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)


def is_contiguous_from_dim1(t: torch.Tensor) -> bool:
    """Mirror of the AITER-side predicate guarding the fused kernel.

    The block stride (dim 0) is free -- the kernel reads ``stride(0)`` per
    side -- but everything inside a block must be densely packed.
    """
    expected = 1
    for dim in range(t.dim() - 1, 0, -1):
        if t.stride(dim) != expected:
            return False
        expected *= t.size(dim)
    return True


def _make_impl(impl_cls):
    return impl_cls(
        num_heads=NUM_KV_HEADS * 2,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


@pytest.fixture
def aiter_env(monkeypatch: pytest.MonkeyPatch):
    """Enable AITER with the shuffle layout off (separate K/V head groups)."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "0")
        rocm_aiter_ops.refresh_env_variables()
        yield
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    "backend_cls, impl_cls",
    [
        (AiterFlashAttentionBackend, AiterFlashAttentionImpl),
        (RocmAiterUnifiedAttentionBackend, RocmAiterUnifiedAttentionImpl),
    ],
)
def test_split_sides_satisfy_fused_kernel_contract(aiter_env, backend_cls, impl_cls):
    """Each split side must be block-contiguous with token/head strides."""
    shape = backend_cls.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    )
    assert shape == (NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS * HEAD_SIZE)

    kv_cache = torch.zeros(shape, dtype=torch.bfloat16, device="cpu")
    key_cache, value_cache = _make_impl(impl_cls)._split_kv_cache(kv_cache)

    for side in (key_cache, value_cache):
        assert side.shape == (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        assert is_contiguous_from_dim1(side)
        # slot stride spans one token's head group, head stride one head.
        assert side.stride(1) == NUM_KV_HEADS * HEAD_SIZE
        assert side.stride(2) == HEAD_SIZE
        # Each side's block stride spans both halves.
        assert side.stride(0) == 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE

    # The two halves must not alias.
    key_cache[0, 0, 0, 0] = 1.0
    assert value_cache[0, 0, 0, 0] == 0.0


def test_packed_layout_retained_under_shuffle(monkeypatch: pytest.MonkeyPatch):
    """The shuffle path keeps its own x-packed interior, so it stays packed."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1")
        rocm_aiter_ops.refresh_env_variables()
        shape = AiterFlashAttentionBackend.get_kv_cache_shape(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
        )
        assert shape == (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
        impl = _make_impl(AiterFlashAttentionImpl)
        assert not impl.fused_qk_norm_rope_kvcache_supported()
    rocm_aiter_ops.refresh_env_variables()


def test_packed_split_would_violate_the_contract():
    """Guards the premise: the packed layout cannot satisfy the kernel.

    Splitting the packed content dim leaves the head dim strided by the whole
    block, which is what made the fused path abort.
    """
    packed = torch.zeros(
        NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=torch.bfloat16
    )
    key_cache, value_cache = packed.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    assert not is_contiguous_from_dim1(key_cache)
    assert not is_contiguous_from_dim1(value_cache)
