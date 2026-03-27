# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 per-token KV cache quantization.

Covers:
- Per-token per-head Triton reshape-and-cache kernel with FP8 cache
- Round-trip quantize/dequantize accuracy
- process_weights_after_loading FP8 per-token early-return path
- End-to-end integration with Triton unified attention kernel

Run: pytest tests/quantization/test_fp8_per_token_kv_cache.py -v -s
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
        not (current_platform.is_cuda() or current_platform.is_rocm()),
        reason="FP8 per-token KV cache tests require CUDA or ROCm GPU.",
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

# Platform-dependent FP8 dtype
FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm() else torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX = FP8_INFO.max
FP8_MIN = FP8_INFO.min


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quantize_fp8_per_token_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token FP8 quantization (one scale per token).

    Returns (quantized_fp8, scales) where scales is [num_tokens].
    """
    absmax = data.float().abs().amax(dim=(1, 2))  # [num_tokens]
    scales = (absmax / FP8_MAX).clamp(min=1e-6)
    q = (data.float() / scales[:, None, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return q, scales


# ===========================================================================
# 1. is_quantized_kv_cache
# ===========================================================================
class TestIsQuantizedKvCacheFp8PerToken:
    def test_fp8_per_token(self):
        assert is_quantized_kv_cache("fp8_per_token")

    def test_kv_quant_mode(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert get_kv_quant_mode("fp8_per_token") == KVQuantMode.FP8_PER_TOKEN


# ===========================================================================
# 2. Triton per-token FP8 kernel
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_fp8_per_token(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_flash_per_token_quant with FP8 cache."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
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
    ref_k_quant, ref_k_scales = _quantize_fp8_per_token_ref(key)
    ref_v_quant, ref_v_scales = _quantize_fp8_per_token_ref(value)

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size

        # Check quantized values (compare as float, allow 1 ULP tolerance)
        torch.testing.assert_close(
            key_cache[blk, off].float(),
            ref_k_quant[i].float(),
            atol=2.0,
            rtol=0.1,
        )
        torch.testing.assert_close(
            value_cache[blk, off].float(),
            ref_v_quant[i].float(),
            atol=2.0,
            rtol=0.1,
        )

        # Check scales
        torch.testing.assert_close(
            k_scale_cache[blk, off], ref_k_scales[i], atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            v_scale_cache[blk, off], ref_v_scales[i], atol=1e-4, rtol=1e-3
        )


# ===========================================================================
# 3. Per-token FP8 round-trip accuracy (quantize → dequantize)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_fp8_per_token_round_trip_accuracy(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
):
    """Verify per-token FP8 round-trip: kernel dequant matches reference."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
    )
    k_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)

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

    for i in range(num_tokens):
        blk = i // block_size
        off = i % block_size

        for label, data, cache, sc in [
            ("key", key, key_cache, k_scale_cache),
            ("val", value, value_cache, v_scale_cache),
        ]:
            orig = data[i].float()
            absmax = orig.abs().amax()
            ref_scale = (absmax / FP8_MAX).clamp(min=1e-6)

            # FP8 quantization: clamp then cast
            ref_q = (orig / ref_scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
            ref_deq = ref_q.float() * ref_scale

            actual_q = cache[blk, off]
            actual_sc = sc[blk, off]
            actual_deq = actual_q.float() * actual_sc

            # Scales must match
            torch.testing.assert_close(
                actual_sc,
                ref_scale,
                atol=1e-5,
                rtol=1e-5,
            )
            # Dequantised values: relative tolerance for FP8 precision
            torch.testing.assert_close(
                actual_deq,
                ref_deq,
                atol=0.05,
                rtol=0.05,
            )


# ===========================================================================
# 4. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_fp8_per_token_negative_slot_skipped():
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
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=FP8_DTYPE
    )
    k_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, dtype=torch.float32)

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
# 5. process_weights_after_loading — FP8 per-token early return
# ===========================================================================
class TestProcessWeightsAfterLoadingFp8PerToken:
    """Unit tests for kv_cache.py BaseKVCacheMethod.process_weights_after_loading
    when kv_cache_dtype='fp8_per_token'.
    """

    def _make_layer(self, kv_cache_dtype="fp8_per_token"):
        layer = MagicMock()
        layer.kv_cache_dtype = kv_cache_dtype
        layer.calculate_kv_scales = False

        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        layer._k_scale = torch.tensor(0.0)
        layer._v_scale = torch.tensor(0.0)
        layer._k_scale_float = 0.0
        layer._v_scale_float = 0.0

        return layer

    def test_fp8_per_token_sets_placeholder_scales(self):
        """FP8 per-token should set _k_scale=1.0, _v_scale=1.0
        and delete checkpoint attrs."""
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod,
        )

        layer = self._make_layer("fp8_per_token")
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
# 6. Triton unified_attention — FP8 KV with per-token scale cache
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
def test_triton_unified_attention_fp8_per_token_scale(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: FP8 KV with per-token scale caches."""
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

    # Per-token quantization to FP8
    k_absmax = key_cache_bf16.float().abs().amax(dim=(-2, -1))
    v_absmax = value_cache_bf16.float().abs().amax(dim=(-2, -1))
    k_scale_cache = (k_absmax / FP8_MAX).clamp(min=1e-6).to(torch.float32)
    v_scale_cache = (v_absmax / FP8_MAX).clamp(min=1e-6).to(torch.float32)

    key_cache_fp8 = (
        (key_cache_bf16.float() / k_scale_cache[:, :, None, None])
        .clamp(FP8_MIN, FP8_MAX)
        .to(FP8_DTYPE)
    )
    value_cache_fp8 = (
        (value_cache_bf16.float() / v_scale_cache[:, :, None, None])
        .clamp(FP8_MIN, FP8_MAX)
        .to(FP8_DTYPE)
    )

    # Dequantized reference
    key_cache_deq = key_cache_fp8.float() * k_scale_cache[:, :, None, None]
    value_cache_deq = value_cache_fp8.float() * v_scale_cache[:, :, None, None]

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

    # FP8 with per-token scale caches
    output_fp8 = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_fp8,
        v=value_cache_fp8,
        out=output_fp8,
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
        kv_quant_mode=KVQuantMode.FP8_PER_TOKEN,
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

    torch.testing.assert_close(output_fp8, output_ref, atol=5e-2, rtol=5e-2)
