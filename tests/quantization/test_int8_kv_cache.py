# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for INT8 KV cache quantization.

Covers:
- Per-token per-head Triton reshape-and-cache kernel
- Round-trip quantize/dequantize accuracy
- process_weights_after_loading INT8 early-return path in kv_cache.py
- End-to-end integration with Triton unified attention kernel

Run: pytest tests/quantization/test_int8_kv_cache.py -v -s
"""

import random
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import KVQuantMode, is_quantized_kv_cache

# Skip entire module if no CUDA/ROCm GPU available
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="INT8 KV cache tests require CUDA or ROCm GPU.",
    ),
]

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
NUM_TOKENS = [1, 7, 42]
NUM_KV_HEADS = [1, 4, 8]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16]
SEEDS = [0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quantize_per_token_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token INT8 quantization (one scale per token).

    Returns (quantized_int8, scales) where scales is [num_tokens].
    """
    # absmax across all heads and head dims → one scalar per token
    absmax = data.float().abs().amax(dim=(1, 2))  # [num_tokens]
    scales = (absmax / 127.0).clamp(min=1e-6)
    # scales[:, None, None] broadcasts to [num_tokens, num_heads, head_size]
    q = (data.float() / scales[:, None, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scales


# ===========================================================================
# 1. is_quantized_kv_cache
# ===========================================================================
class TestIsQuantizedKvCache:
    def test_fp8_variants(self):
        assert is_quantized_kv_cache("fp8")
        assert is_quantized_kv_cache("fp8_e4m3")
        assert is_quantized_kv_cache("fp8_e5m2")

    def test_int8_per_token(self):
        assert is_quantized_kv_cache("int8_per_token")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")


# ===========================================================================
# 2. Triton per-token per-head INT8 kernel
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_int8_per_token(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_flash_per_token_quant kernel."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    k_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    triton_reshape_and_cache_flash_per_token_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    # Reference
    ref_k_quant, ref_k_scales = _quantize_per_token_ref(key)
    ref_v_quant, ref_v_scales = _quantize_per_token_ref(value)

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size

        # Check quantized values
        torch.testing.assert_close(
            key_cache[blk, off].float(), ref_k_quant[i].float(), atol=1.0, rtol=0.0
        )
        torch.testing.assert_close(
            value_cache[blk, off].float(), ref_v_quant[i].float(), atol=1.0, rtol=0.0
        )

        # Check scales
        torch.testing.assert_close(
            k_scale_cache[blk, off], ref_k_scales[i], atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            v_scale_cache[blk, off], ref_v_scales[i], atol=1e-4, rtol=1e-3
        )


# ===========================================================================
# 3. Per-token INT8 round-trip accuracy (quantize → dequantize)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_int8_per_token_round_trip_accuracy(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
):
    """Verify per-token INT8 round-trip: kernel dequant matches
    a reference that replicates the kernel's truncation semantics.

    The Triton kernel quantises via tl.clamp(x / scale, -128, 127)
    then tl.store to int8, which **truncates** toward zero (not
    rounds).  We replicate that here with torch.trunc() so the
    reference is exact, then compare dequantised outputs.
    """
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    # Realistic attention-scale values (small magnitudes)
    key = (
        torch.randn(
            num_tokens,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
        * 0.5
    )
    value = (
        torch.randn(
            num_tokens,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
        * 0.5
    )

    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.int8,
    )
    value_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.int8,
    )
    k_scale_cache = torch.ones(
        num_blocks,
        block_size,
        dtype=torch.float32,
    )
    v_scale_cache = torch.ones(
        num_blocks,
        block_size,
        dtype=torch.float32,
    )

    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    triton_reshape_and_cache_flash_per_token_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    # Build a reference that exactly matches the kernel:
    #   scale = max(absmax / 127, 1e-6)
    #   q_int8 = trunc(clamp(x / scale, -128, 127))
    #   dequant = q_int8 * scale
    for i in range(num_tokens):
        blk = i // block_size
        off = i % block_size

        for label, data, cache, sc in [
            ("key", key, key_cache, k_scale_cache),
            ("val", value, value_cache, v_scale_cache),
        ]:
            orig = data[i].float()  # [num_heads, head_size]
            # Per-token: absmax across all heads and dims → scalar
            absmax = orig.abs().amax()
            ref_scale = (absmax / 127.0).clamp(min=1e-6)

            # Triton truncates on float→int8 store
            ref_q = (orig / ref_scale).clamp(-128.0, 127.0).trunc().to(torch.int8)
            ref_deq = ref_q.float() * ref_scale

            actual_q = cache[blk, off]
            actual_sc = sc[blk, off]  # scalar
            actual_deq = actual_q.float() * actual_sc

            # Scales must match exactly (both float32)
            torch.testing.assert_close(
                actual_sc,
                ref_scale,
                atol=1e-5,
                rtol=1e-5,
            )
            # Quantised int8 values: allow +-1 for
            # bf16→f32 representation differences
            torch.testing.assert_close(
                actual_q.float(),
                ref_q.float(),
                atol=1.0,
                rtol=0.0,
            )
            # Dequantised values: error <= 1 * scale
            # (from the +-1 int8 tolerance above)
            err = (actual_deq - ref_deq).abs()
            bound = actual_sc * 1.01
            assert (err <= bound).all(), (
                f"{label} dequant error at token {i}: "
                f"max={err.max():.6f}, "
                f"bound={bound.max():.6f}"
            )


# ===========================================================================
# 4. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_int8_per_token_negative_slot_skipped():
    """Tokens with slot_mapping=-1 should leave the cache unchanged."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    torch.set_default_device("cuda")
    num_tokens = 4
    num_heads = 2
    head_size = 64
    block_size = 16
    num_blocks = 2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.int8,
    )
    value_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        dtype=torch.int8,
    )
    k_scale_cache = torch.ones(
        num_blocks,
        block_size,
        dtype=torch.float32,
    )
    v_scale_cache = torch.ones(
        num_blocks,
        block_size,
        dtype=torch.float32,
    )

    # Mix valid and negative slot mappings
    slot_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    key_cache_before = key_cache.clone()
    val_cache_before = value_cache.clone()

    triton_reshape_and_cache_flash_per_token_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    # Slots 0 and 1 should have been written (tokens 0 and 2)
    assert not torch.equal(key_cache[0, 0], key_cache_before[0, 0])
    assert not torch.equal(key_cache[0, 1], key_cache_before[0, 1])
    assert not torch.equal(value_cache[0, 0], val_cache_before[0, 0])

    # All other slots should be unchanged
    assert torch.equal(key_cache[0, 2:], key_cache_before[0, 2:])
    assert torch.equal(key_cache[1], key_cache_before[1])
    assert torch.equal(value_cache[0, 2:], val_cache_before[0, 2:])


# ===========================================================================
# 5. process_weights_after_loading — INT8 early return
# ===========================================================================
class TestProcessWeightsAfterLoadingInt8:
    """Unit tests for kv_cache.py BaseKVCacheMethod.process_weights_after_loading
    when kv_cache_dtype='int8_per_token'.

    Per-token quant uses dynamic per-token scales computed in the kernel.
    Checkpoint scales are not used — the method sets placeholders and returns.
    """

    def _make_layer(self, kv_cache_dtype="int8_per_token"):
        """Create a mock attention layer with the required attributes."""
        layer = MagicMock()
        layer.kv_cache_dtype = kv_cache_dtype
        layer.calculate_kv_scales = False

        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        # Runtime buffers
        layer._k_scale = torch.tensor(0.0)
        layer._v_scale = torch.tensor(0.0)
        layer._k_scale_float = 0.0
        layer._v_scale_float = 0.0

        return layer

    def test_int8_sets_placeholder_scales(self):
        """INT8 should set _k_scale=1.0, _v_scale=1.0 and delete checkpoint attrs."""
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = self._make_layer("int8_per_token")
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()

        method.process_weights_after_loading(layer)

        assert layer._k_scale_float == 1.0
        assert layer._v_scale_float == 1.0
        assert not hasattr(layer, "k_scale")
        assert not hasattr(layer, "v_scale")
        assert not hasattr(layer, "q_scale")
        assert not hasattr(layer, "prob_scale")


# ===========================================================================
# 6. Triton unified_attention — INT8 KV round-trip (per-tensor scale)
# ===========================================================================
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 128), (1, 64)],
        [(1, 256)],
        [(4, 64), (1, 32)],
    ],
)
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_triton_unified_attention_int8_per_tensor(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: quantize KV to INT8 with per-tensor scale, run attention,
    compare to bf16 reference."""
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 2048

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
    )

    # Create bf16 KV cache for reference
    key_cache_bf16 = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    value_cache_bf16 = torch.randn_like(key_cache_bf16)

    # Quantize KV cache to INT8 with per-tensor scale
    k_absmax = key_cache_bf16.float().abs().max()
    v_absmax = value_cache_bf16.float().abs().max()
    k_scale = (k_absmax / 127.0).clamp(min=1e-6).float()
    v_scale = (v_absmax / 127.0).clamp(min=1e-6).float()

    key_cache_int8 = (
        (key_cache_bf16.float() / k_scale).round().clamp(-128, 127).to(torch.int8)
    )
    value_cache_int8 = (
        (value_cache_bf16.float() / v_scale).round().clamp(-128, 127).to(torch.int8)
    )

    # Dequantized reference (what the kernel should effectively use)
    key_cache_deq = key_cache_int8.float() * k_scale
    value_cache_deq = value_cache_int8.float() * v_scale

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    head_size_padded = next_power_of_2(head_size)
    seq_threshold_3D = 0
    num_par_softmax_segments = 16
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    descale_shape = (num_seqs, num_kv_heads)

    # INT8 output
    output_int8 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_int8,
        v=value_cache_int8,
        out=output_int8,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_scale.expand(descale_shape),
        v_descale=v_scale.expand(descale_shape),
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    # BF16 reference (using dequantized cache to account for quantization noise)
    output_ref = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_deq.to(torch.bfloat16),
        v=value_cache_deq.to(torch.bfloat16),
        out=output_ref,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    # Compare — generous tolerance due to int8 quantization noise
    torch.testing.assert_close(output_int8, output_ref, atol=5e-2, rtol=5e-2)


# ===========================================================================
# 7. Triton unified_attention — INT8 KV with per-token scale cache
# ===========================================================================
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 128)],
        [(1, 64), (1, 32)],
    ],
)
@pytest.mark.parametrize("num_heads", [(4, 4)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_triton_unified_attention_int8_per_token_scale(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: INT8 KV with per-(token,head) scale caches."""
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 2048

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
    )

    # Create bf16 KV cache
    key_cache_bf16 = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    value_cache_bf16 = torch.randn_like(key_cache_bf16)

    # Per-token quantization: one scale per (block, slot), shared across heads
    # absmax across heads and head_size dims → [num_blocks, block_size]
    k_absmax = key_cache_bf16.float().abs().amax(dim=(-2, -1))
    v_absmax = value_cache_bf16.float().abs().amax(dim=(-2, -1))
    k_scale_cache = (k_absmax / 127.0).clamp(min=1e-6).to(torch.float32)
    v_scale_cache = (v_absmax / 127.0).clamp(min=1e-6).to(torch.float32)

    # Broadcast scale [num_blocks, block_size] → [..., 1, 1] for quantization
    key_cache_int8 = (
        (key_cache_bf16.float() / k_scale_cache[:, :, None, None])
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    value_cache_int8 = (
        (value_cache_bf16.float() / v_scale_cache[:, :, None, None])
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )

    # Dequantized reference
    key_cache_deq = key_cache_int8.float() * k_scale_cache[:, :, None, None]
    value_cache_deq = value_cache_int8.float() * v_scale_cache[:, :, None, None]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    head_size_padded = next_power_of_2(head_size)
    seq_threshold_3D = 0
    num_par_softmax_segments = 16
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    # INT8 with per-token scale caches
    output_int8 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_int8,
        v=value_cache_int8,
        out=output_int8,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    # BF16 reference
    output_ref = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_deq.to(torch.bfloat16),
        v=value_cache_deq.to(torch.bfloat16),
        out=output_ref,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    torch.testing.assert_close(output_int8, output_ref, atol=5e-2, rtol=5e-2)
