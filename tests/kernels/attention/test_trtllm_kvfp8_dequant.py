# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone unit tests for trtllm_prefill_attn_kvfp8_dequant.

Tests KV cache layouts against a pure-PyTorch reference implementation.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheLayout

if current_platform.is_rocm():
    pytest.skip(
        "trtllm kvfp8 dequant is not supported on ROCm.",
        allow_module_level=True,
    )

FP8_DTYPE = current_platform.fp8_dtype()

NUM_BLOCKS = 128


def to_float8(x, dtype=None):
    if dtype is None:
        dtype = FP8_DTYPE
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def make_random_kv_cache(
    num_blocks, num_kv_heads, block_size, head_size, layout=KVCacheLayout.LBHNC
):
    """Create a random fp8 KV cache in 5D ``(B, 2, H, N, hs)`` format.

    The 4D cache ``(B, H, N, 2*hs)`` is allocated with the physical stride
    order from *layout*, then reshaped to 5D. Non-identity layouts produce
    non-contiguous strides on inner dims, matching the actual forward path.
    """
    logical_4d = (num_blocks, num_kv_heads, block_size, 2 * head_size)
    stride_order = layout.layer_stride_order
    physical_4d = tuple(logical_4d[i] for i in stride_order)
    inv_order = [stride_order.index(i) for i in range(4)]

    raw_phys = torch.randn(*physical_4d, dtype=torch.bfloat16, device="cuda")
    fp8_phys, scale = to_float8(raw_phys)
    fp8_4d = fp8_phys.permute(*inv_order)
    kv_5d = fp8_4d.view(
        num_blocks,
        num_kv_heads,
        block_size,
        2,
        head_size,
    ).permute(0, 3, 1, 2, 4)
    return kv_5d, scale


def ref_dequant(kv_cache, block_tables, k_scale, v_scale, dequant_dtype):
    """Pure PyTorch reference: gather pages and dequantize fp8 -> dequant_dtype."""
    batch_size, num_pages_per_seq = block_tables.shape
    s = kv_cache.shape
    out = torch.zeros(
        batch_size * num_pages_per_seq + 1,
        s[1],
        s[2],
        s[3],
        s[4],
        dtype=dequant_dtype,
        device=kv_cache.device,
    )
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            page_idx = block_tables[b, p].item()
            if page_idx <= 0:
                continue
            mock_idx = b * num_pages_per_seq + p + 1
            out[mock_idx, 0] = (kv_cache[page_idx, 0].float() * k_scale.item()).to(
                dequant_dtype
            )
            out[mock_idx, 1] = (kv_cache[page_idx, 1].float() * v_scale.item()).to(
                dequant_dtype
            )
    return out


@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_pages_per_seq", [3, 8])
@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
@torch.inference_mode()
def test_trtllm_kvfp8_dequant(
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    batch_size: int,
    num_pages_per_seq: int,
    layout: KVCacheLayout,
):
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")

    kv_cache, scale = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
        layout=layout,
    )

    k_scale = scale.clone()
    v_scale = scale.clone()

    block_tables = torch.randint(
        1,
        NUM_BLOCKS,
        (batch_size, num_pages_per_seq),
        dtype=torch.int32,
    )

    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )

    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    expected_bt = torch.arange(
        1,
        batch_size * num_pages_per_seq + 1,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size, num_pages_per_seq)
    torch.testing.assert_close(mock_block_table, expected_bt)

    # Page 0 is padding (never written), compare only pages 1+
    torch.testing.assert_close(mock_kv_cache[1:], ref[1:], atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_block_tables_with_zero_pages():
    """Pages with index <= 0 must be skipped (early return in kernel)."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 8, 16, 64

    kv_cache, scale = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = v_scale = scale.clone()

    # Mix of valid pages and zeros (padding)
    block_tables = torch.tensor(
        [[5, 0, 10], [0, 0, 0], [3, 7, 0]],
        dtype=torch.int32,
        device="cuda",
    )

    mock_kv_cache, _ = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    # Only compare pages that were actually written (non-zero page indices)
    for b in range(block_tables.shape[0]):
        for p in range(block_tables.shape[1]):
            if block_tables[b, p].item() > 0:
                idx = b * block_tables.shape[1] + p + 1
                torch.testing.assert_close(
                    mock_kv_cache[idx],
                    ref[idx],
                    atol=1e-3,
                    rtol=1e-3,
                )


@torch.inference_mode()
def test_all_zero_block_tables():
    """All-zero block_tables: kernel should write nothing."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 4, 16, 64

    kv_cache, scale = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = v_scale = scale.clone()

    block_tables = torch.zeros(2, 4, dtype=torch.int32, device="cuda")

    # Should not crash even though no pages are valid
    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    assert mock_kv_cache.shape[0] == 2 * 4 + 1
    assert mock_block_table.shape == (2, 4)


@torch.inference_mode()
def test_different_k_v_scales():
    """Verify K and V are dequantized with independent scales."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 8, 16, 64

    kv_cache, _ = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([2.0], dtype=torch.float32, device="cuda")

    block_tables = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    mock_kv_cache, _ = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    torch.testing.assert_close(mock_kv_cache[1:], ref[1:], atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_single_page_per_seq():
    """Minimum grid dim 1 = 1 page per sequence."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 8, 16, 128

    kv_cache, scale = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = v_scale = scale.clone()

    block_tables = torch.tensor([[5], [10], [20]], dtype=torch.int32, device="cuda")

    mock_kv_cache, _ = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    torch.testing.assert_close(mock_kv_cache[1:], ref[1:], atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_large_page_indices():
    """Page indices near the top of the buffer stress offset arithmetic."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 8, 16, 128
    large_num_blocks = 32768

    kv_cache, scale = make_random_kv_cache(
        large_num_blocks,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = v_scale = scale.clone()

    # Use page indices near the top of the buffer
    block_tables = torch.tensor(
        [[large_num_blocks - 1, large_num_blocks - 2, 1]],
        dtype=torch.int32,
        device="cuda",
    )

    mock_kv_cache, _ = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    torch.testing.assert_close(mock_kv_cache[1:], ref[1:], atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_large_block_size():
    """block_size=64 -> HEAD_STRIDE=8192, large tl.arange per thread block."""
    from vllm.v1.attention.backends.flashinfer import (
        trtllm_prefill_attn_kvfp8_dequant,
    )

    torch.set_default_device("cuda")
    num_kv_heads, block_size, head_size = 4, 64, 128

    kv_cache, scale = make_random_kv_cache(
        NUM_BLOCKS,
        num_kv_heads,
        block_size,
        head_size,
    )
    k_scale = v_scale = scale.clone()

    block_tables = torch.randint(
        1,
        NUM_BLOCKS,
        (2, 4),
        dtype=torch.int32,
        device="cuda",
    )

    mock_kv_cache, _ = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables,
        k_scale,
        v_scale,
        torch.bfloat16,
    )
    ref = ref_dequant(kv_cache, block_tables, k_scale, v_scale, torch.bfloat16)

    torch.testing.assert_close(mock_kv_cache[1:], ref[1:], atol=1e-3, rtol=1e-3)
