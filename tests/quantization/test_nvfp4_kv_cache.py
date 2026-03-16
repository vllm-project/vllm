# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 KV cache quantization.

Covers:
- is_quantized_kv_cache for "nvfp4"
- AttentionSpec / FullAttentionSpec NVFP4 page size and shape
- Triton reshape_and_cache_nvfp4 kernel correctness
- NVFP4 round-trip quantize/dequantize accuracy
- Negative slot mapping (padding tokens skipped)
- process_weights_after_loading NVFP4 paths
- calc_kv_scales NVFP4 path
- Triton unified_attention with fused NVFP4 dequant

Run: pytest tests/quantization/test_nvfp4_kv_cache.py -v -s
"""

import random
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import is_quantized_kv_cache
from vllm.v1.kv_cache_interface import NVFP4_QUANT_BLOCK_SIZE

# Skip entire module if no CUDA GPU available (NVFP4 uses tl.float8e4nv)
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="NVFP4 KV cache tests require CUDA GPU.",
    ),
]

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
NUM_TOKENS = [1, 7, 32]
NUM_KV_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16]
SEEDS = [0]

# FP4 E2M1 representable magnitudes
FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E2M1_MAX = 6.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nvfp4_effective_head_size(head_size: int) -> int:
    """Compute the effective head size in bytes for NVFP4 cache."""
    return head_size // 2 + head_size // NVFP4_QUANT_BLOCK_SIZE


def _nvfp4_dequant_ref(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    head_size: int,
) -> torch.Tensor:
    """Reference NVFP4 dequantization.

    Args:
        packed: uint8 tensor of shape [..., head_size // 2]
        scales: uint8 tensor of shape [..., head_size // NVFP4_QUANT_BLOCK_SIZE]
            (FP8 E4M3 bitcast)
        global_scale: second-level scale factor
        head_size: original head dimension

    Returns:
        float32 tensor of shape [..., head_size]
    """
    # FP4 E2M1 LUT
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )

    # Unpack nibbles
    lo = packed.to(torch.int32) & 0xF
    hi = (packed.to(torch.int32) >> 4) & 0xF

    # Decode via LUT
    lo_val = lut[lo]
    hi_val = lut[hi]

    # Decode FP8 E4M3 scales
    scales_f32 = scales.view(torch.float8_e4m3fn).to(torch.float32)

    # Apply per-group scales
    group_size = NVFP4_QUANT_BLOCK_SIZE
    packed_per_group = group_size // 2
    num_groups = head_size // group_size

    # Expand scales to match packed data
    batch_shape = packed.shape[:-1]
    result_even = torch.zeros(
        *batch_shape, head_size // 2, dtype=torch.float32, device=packed.device
    )
    result_odd = torch.zeros_like(result_even)

    for g in range(num_groups):
        start = g * packed_per_group
        end = start + packed_per_group
        s = scales_f32[..., g : g + 1] * global_scale
        result_even[..., start:end] = lo_val[..., start:end] * s
        result_odd[..., start:end] = hi_val[..., start:end] * s

    # Interleave even/odd
    result = torch.zeros(
        *batch_shape, head_size, dtype=torch.float32, device=packed.device
    )
    result[..., 0::2] = result_even
    result[..., 1::2] = result_odd
    return result


# ===========================================================================
# 1. is_quantized_kv_cache
# ===========================================================================
class TestIsQuantizedKvCache:
    def test_nvfp4(self):
        assert is_quantized_kv_cache("nvfp4")

    def test_fp8(self):
        assert is_quantized_kv_cache("fp8")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")


# ===========================================================================
# 2. AttentionSpec / FullAttentionSpec NVFP4 page size
# ===========================================================================
class TestNvfp4CacheSpec:
    def test_attention_spec_is_nvfp4(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4",
        )
        assert spec.is_nvfp4
        assert not AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        ).is_nvfp4

    def test_nvfp4_head_size_bytes(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4",
        )
        # 128 // 2 + 128 // 16 = 64 + 8 = 72
        assert spec.nvfp4_head_size_bytes == 72

    def test_nvfp4_page_size_bytes(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4",
        )
        # 2 * 16 * 8 * 72 = 18432
        assert spec.page_size_bytes == 2 * 16 * 8 * 72

    def test_full_attention_spec_nvfp4(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4",
        )
        assert spec.is_nvfp4
        # FullAttentionSpec should also compute NVFP4 page size correctly
        assert spec.page_size_bytes == 2 * 16 * 8 * 72

    def test_compression_ratio(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        nvfp4 = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4",
        )
        bf16 = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
        ratio = bf16.page_size_bytes / nvfp4.page_size_bytes
        # BF16: 2 * 16 * 8 * 128 * 2 = 65536
        # NVFP4: 2 * 16 * 8 * 72 = 18432
        assert ratio == pytest.approx(65536 / 18432, rel=1e-3)

    def test_get_kv_cache_shape_nvfp4(self):
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

        shape = TritonAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="nvfp4",
        )
        assert shape == (100, 2, 16, 8, 72)


# ===========================================================================
# 3. Triton reshape_and_cache_nvfp4 kernel
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_nvfp4(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_nvfp4 stores packed FP4 data + scales."""
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_nvfp4,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    eff_head_size = _nvfp4_effective_head_size(head_size)
    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, eff_head_size, dtype=torch.uint8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, eff_head_size, dtype=torch.uint8
    )

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    triton_reshape_and_cache_nvfp4(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    # Verify non-zero data was written for valid slots
    packed_head = head_size // 2
    scale_head = head_size // NVFP4_QUANT_BLOCK_SIZE
    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size
        k_data = key_cache[blk, off]
        v_data = value_cache[blk, off]

        # At least some packed bytes should be non-zero (unless input was zero)
        if key[i].abs().max() > 0:
            assert k_data[:packed_head].any(), f"Key packed data all zeros at token {i}"
        if value[i].abs().max() > 0:
            assert v_data[:packed_head].any(), (
                f"Value packed data all zeros at token {i}"
            )

        # Block scales should be non-zero for non-zero inputs
        k_scales = k_data[packed_head : packed_head + scale_head]
        v_scales = v_data[packed_head : packed_head + scale_head]
        if key[i].abs().max() > 0:
            assert k_scales.any(), f"Key scales all zeros at token {i}"
        if value[i].abs().max() > 0:
            assert v_scales.any(), f"Value scales all zeros at token {i}"


# ===========================================================================
# 4. NVFP4 round-trip accuracy
# ===========================================================================
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_heads", [4])
@torch.inference_mode()
def test_nvfp4_round_trip_accuracy(head_size: int, num_heads: int):
    """Verify NVFP4 round-trip: quantize via kernel → dequant via kernel,
    compare to original with expected FP4 tolerance."""
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_nvfp4,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        nvfp4_dequant_kv_cache,
    )

    set_random_seed(42)
    torch.set_default_device("cuda")

    num_tokens = 16
    block_size = 16
    eff_head_size = _nvfp4_effective_head_size(head_size)
    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    # Use moderate values so quantization isn't too lossy
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, eff_head_size, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)

    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")
    k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    # Quantize
    triton_reshape_and_cache_nvfp4(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    # Dequantize
    key_deq = nvfp4_dequant_kv_cache(key_cache, k_scale, head_size)
    value_deq = nvfp4_dequant_kv_cache(value_cache, v_scale, head_size)

    # Compare: FP4 E2M1 has only 8 representable magnitudes, so we expect
    # relatively coarse quantization. Relative error can be 20-30%.
    for i in range(num_tokens):
        blk = i // block_size
        off = i % block_size

        k_orig = key[i].float()
        k_rt = key_deq[blk, off].float()
        v_orig = value[i].float()
        v_rt = value_deq[blk, off].float()

        # Absolute error should be bounded by the quantization step size
        # For global_scale=1.0, the max value is 6.0 * block_scale
        # Allow generous tolerance for FP4
        k_err = (k_orig - k_rt).abs()
        v_err = (v_orig - v_rt).abs()
        k_range = k_orig.abs().max()
        v_range = v_orig.abs().max()

        # Error should be within ~50% of the range (FP4 is very coarse)
        if k_range > 0.01:
            assert k_err.max() < k_range * 0.6, (
                f"Key round-trip error too large at token {i}: "
                f"max_err={k_err.max():.4f}, range={k_range:.4f}"
            )
        if v_range > 0.01:
            assert v_err.max() < v_range * 0.6, (
                f"Value round-trip error too large at token {i}: "
                f"max_err={v_err.max():.4f}, range={v_range:.4f}"
            )


# ===========================================================================
# 5. Negative slot mapping
# ===========================================================================
@torch.inference_mode()
def test_nvfp4_negative_slot_skipped():
    """Tokens with slot_mapping=-1 should leave the cache unchanged."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_nvfp4,
    )

    torch.set_default_device("cuda")
    num_tokens = 4
    num_heads = 2
    head_size = 64
    block_size = 16
    num_blocks = 2
    eff_head_size = _nvfp4_effective_head_size(head_size)

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, eff_head_size, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache_before = key_cache.clone()
    val_cache_before = value_cache.clone()

    slot_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.long, device="cuda")
    k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    triton_reshape_and_cache_nvfp4(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    # Slots 0 and 1 should have been written
    assert not torch.equal(key_cache[0, 0], key_cache_before[0, 0])
    assert not torch.equal(key_cache[0, 1], key_cache_before[0, 1])

    # All other slots should be unchanged
    assert torch.equal(key_cache[0, 2:], key_cache_before[0, 2:])
    assert torch.equal(key_cache[1], key_cache_before[1])
    assert torch.equal(value_cache[0, 2:], val_cache_before[0, 2:])


# ===========================================================================
# 6. process_weights_after_loading — NVFP4 paths
# ===========================================================================
class TestProcessWeightsAfterLoadingNvfp4:
    def _make_layer(self, k_scale_val, v_scale_val):
        layer = MagicMock()
        layer.kv_cache_dtype = "nvfp4"
        layer.calculate_kv_scales = False
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(k_scale_val, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(v_scale_val, dtype=torch.float32), requires_grad=False
        )
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer._k_scale = torch.tensor(0.0)
        layer._v_scale = torch.tensor(0.0)
        layer._q_scale = torch.tensor(0.0)
        layer._prob_scale = torch.tensor(0.0)
        layer._k_scale_float = 0.0
        layer._v_scale_float = 0.0
        layer._q_scale_float = 0.0
        return layer

    def test_nvfp4_positive_scales(self):
        """NVFP4 with positive scales should store them."""
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod,
        )

        layer = self._make_layer(2.5, 1.5)
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()
        method.process_weights_after_loading(layer)

        assert abs(layer._k_scale_float - 2.5) < 1e-6
        assert abs(layer._v_scale_float - 1.5) < 1e-6

    def test_nvfp4_no_scales_defaults_to_1(self):
        """NVFP4 with no checkpoint scales should default to 1.0."""
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod,
        )

        layer = self._make_layer(-1.0, -1.0)
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()
        method.process_weights_after_loading(layer)

        assert abs(layer._k_scale_float - 1.0) < 1e-6
        assert abs(layer._v_scale_float - 1.0) < 1e-6


# ===========================================================================
# 7. calc_kv_scales — NVFP4 path
# ===========================================================================
@torch.inference_mode()
def test_calc_kv_scales_nvfp4():
    """Test that calc_kv_scales produces correct global scales for NVFP4."""
    num_tokens = 8
    num_kv_heads = 4
    head_size = 64

    layer = MagicMock()
    layer.kv_cache_dtype = "nvfp4"
    layer.num_kv_heads = num_kv_heads
    layer._k_scale = torch.tensor(0.0, device="cuda")
    layer._v_scale = torch.tensor(0.0, device="cuda")
    layer._q_scale = torch.tensor(0.0, device="cuda")
    layer._k_scale_float = 0.0
    layer._v_scale_float = 0.0
    layer._q_scale_float = 0.0
    layer.q_range = torch.tensor(127.0, device="cuda")
    layer.k_range = torch.tensor(127.0, device="cuda")
    layer.v_range = torch.tensor(127.0, device="cuda")
    layer.calculate_kv_scales = True

    query = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")
    key = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")
    value = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")

    from vllm.model_executor.layers.attention.attention import Attention

    Attention.calc_kv_scales(layer, query, key, value)

    # Verify scale = absmax / (FP4_MAX * FP8_E4M3_MAX) = absmax / (6 * 448)
    k_absmax = torch.abs(key).max()
    v_absmax = torch.abs(value).max()
    expected_k = k_absmax / (6.0 * 448.0)
    expected_v = v_absmax / (6.0 * 448.0)

    torch.testing.assert_close(layer._k_scale, expected_k, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(layer._v_scale, expected_v, atol=1e-6, rtol=1e-5)
    assert layer.calculate_kv_scales is False


# ===========================================================================
# 8. Triton unified_attention with NVFP4 fused dequant
# ===========================================================================
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_nvfp4_unified_attention_integration(
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: quantize KV to NVFP4, run attention with fused dequant,
    compare to BF16 reference."""
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_nvfp4,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )

    set_random_seed(0)
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    eff_head_size = _nvfp4_effective_head_size(head_size)
    num_blocks = 64
    seq_len = 64
    num_seqs = 2
    query_len = 1

    # Create query
    total_q = num_seqs * query_len
    query = torch.randn(total_q, num_query_heads, head_size, dtype=torch.bfloat16)

    # Create BF16 KV data and quantize to NVFP4
    key_bf16 = (
        torch.randn(
            num_blocks * block_size, num_kv_heads, head_size, dtype=torch.bfloat16
        )
        * 0.3
    )
    value_bf16 = (
        torch.randn(
            num_blocks * block_size, num_kv_heads, head_size, dtype=torch.bfloat16
        )
        * 0.3
    )

    # NVFP4 cache
    key_cache_nvfp4 = torch.zeros(
        num_blocks, block_size, num_kv_heads, eff_head_size, dtype=torch.uint8
    )
    value_cache_nvfp4 = torch.zeros_like(key_cache_nvfp4)

    # BF16 cache (for reference)
    key_cache_bf16 = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache_bf16 = torch.zeros_like(key_cache_bf16)

    # Fill both caches with the same data
    all_slots = torch.arange(num_blocks * block_size, dtype=torch.long, device="cuda")
    k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    triton_reshape_and_cache_nvfp4(
        key_bf16,
        value_bf16,
        key_cache_nvfp4,
        value_cache_nvfp4,
        all_slots,
        k_scale,
        v_scale,
    )

    # Fill BF16 cache directly
    for i in range(num_blocks * block_size):
        blk = i // block_size
        off = i % block_size
        key_cache_bf16[blk, off] = key_bf16[i]
        value_cache_bf16[blk, off] = value_bf16[i]

    # Attention metadata
    cu_query_lens = torch.tensor(
        [0] + [query_len] * num_seqs, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor([seq_len] * num_seqs, dtype=torch.int32)

    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_tables = torch.zeros(
        num_seqs, max_blocks_per_seq, dtype=torch.int32, device="cuda"
    )
    for s in range(num_seqs):
        for b in range(max_blocks_per_seq):
            block_tables[s, b] = s * max_blocks_per_seq + b

    descale_shape = (num_seqs, num_kv_heads)

    # Run NVFP4 attention (fused dequant)
    output_nvfp4 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_nvfp4,
        v=value_cache_nvfp4,
        out=output_nvfp4,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=seq_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_scale.expand(descale_shape),
        v_descale=v_scale.expand(descale_shape),
        nvfp4_head_size=head_size,
    )

    # Run BF16 reference attention
    output_bf16 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_bf16,
        v=value_cache_bf16,
        out=output_bf16,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=seq_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    # FP4 quantization is very coarse — generous tolerance
    torch.testing.assert_close(output_nvfp4, output_bf16, atol=0.15, rtol=0.15)
