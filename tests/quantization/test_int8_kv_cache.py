# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for INT8 KV cache quantization.

Covers:
- Triton reshape-and-cache kernels (per-tensor, per-head, per-token)
- Round-trip quantize/dequantize accuracy
- process_weights_after_loading INT8 paths in kv_cache.py
- calc_kv_scales INT8 path in attention.py
- End-to-end integration with Triton attention backend

Run: pytest tests/quantization/test_int8_kv_cache.py -v -s
"""

import random
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import is_quantized_kv_cache

# Skip entire module if no CUDA/ROCm GPU available
pytestmark = [
    pytest.mark.skipif(
        not (current_platform.is_cuda() or current_platform.is_rocm()),
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
def _quantize_ref(
    data: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-tensor INT8 quantization."""
    q = (data.float() / scale.float()).round().clamp(-128, 127).to(torch.int8)
    return q


def _quantize_per_head_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
    scales: torch.Tensor,  # [num_heads]
) -> torch.Tensor:
    """Reference per-head INT8 quantization."""
    # scales: [num_heads] -> [1, num_heads, 1]
    s = scales.float().unsqueeze(0).unsqueeze(-1)
    q = (data.float() / s).round().clamp(-128, 127).to(torch.int8)
    return q


def _quantize_per_token_head_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-(token,head) INT8 quantization.

    Returns (quantized_int8, scales) where scales is [num_tokens, num_heads].
    """
    absmax = data.float().abs().amax(dim=-1)  # [num_tokens, num_heads]
    scales = (absmax / 127.0).clamp(min=1e-6)
    q = (data.float() / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return q, scales


# ===========================================================================
# 1. is_quantized_kv_cache
# ===========================================================================
class TestIsQuantizedKvCache:
    def test_fp8_variants(self):
        assert is_quantized_kv_cache("fp8")
        assert is_quantized_kv_cache("fp8_e4m3")
        assert is_quantized_kv_cache("fp8_e5m2")

    def test_int8(self):
        assert is_quantized_kv_cache("int8")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")


# ===========================================================================
# 2. Triton reshape_and_cache_flash — INT8 per-tensor scale
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_int8_per_tensor(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_flash with INT8 per-tensor scale."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    # Random key/value in bf16
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    # INT8 cache tensors (stored as int8 viewed as uint8 for the kernel)
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )

    # Slot mapping: assign each token to a unique slot
    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    # Per-tensor scales
    k_scale = (key.abs().max() / 127.0).clamp(min=1e-6).to(torch.float32)
    v_scale = (value.abs().max() / 127.0).clamp(min=1e-6).to(torch.float32)

    # Run kernel
    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype="int8",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Verify: read back from cache and compare to reference
    ref_k_quant = _quantize_ref(key, k_scale)
    ref_v_quant = _quantize_ref(value, v_scale)

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size
        actual_k = key_cache[blk, off]
        actual_v = value_cache[blk, off]
        expected_k = ref_k_quant[i]
        expected_v = ref_v_quant[i]
        # Allow +-1 for rounding differences between triton and pytorch
        torch.testing.assert_close(
            actual_k.float(), expected_k.float(), atol=1.0, rtol=0.0
        )
        torch.testing.assert_close(
            actual_v.float(), expected_v.float(), atol=1.0, rtol=0.0
        )


# ===========================================================================
# 3. Triton reshape_and_cache_flash — INT8 per-head scale
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_int8_per_head(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_flash with INT8 per-head scale [num_heads]."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
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

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    # Per-head scales: [num_heads]
    k_scales = (
        (key.float().abs().amax(dim=(0, 2)) / 127.0).clamp(min=1e-6).to(torch.float32)
    )
    v_scales = (
        (value.float().abs().amax(dim=(0, 2)) / 127.0).clamp(min=1e-6).to(torch.float32)
    )

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype="int8",
        k_scale=k_scales,
        v_scale=v_scales,
    )

    # Reference
    ref_k_quant = _quantize_per_head_ref(key, k_scales)
    ref_v_quant = _quantize_per_head_ref(value, v_scales)

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size
        torch.testing.assert_close(
            key_cache[blk, off].float(), ref_k_quant[i].float(), atol=1.0, rtol=0.0
        )
        torch.testing.assert_close(
            value_cache[blk, off].float(), ref_v_quant[i].float(), atol=1.0, rtol=0.0
        )


# ===========================================================================
# 4. Triton per-token per-head INT8 kernel
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
    """Test triton_reshape_and_cache_flash_int8_per_token kernel."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int8_per_token,
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
    k_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    triton_reshape_and_cache_flash_int8_per_token(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    # Reference
    ref_k_quant, ref_k_scales = _quantize_per_token_head_ref(key)
    ref_v_quant, ref_v_scales = _quantize_per_token_head_ref(value)

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
# 5. Per-token INT8 round-trip accuracy (quantize → dequantize)
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
    """Verify that quantize→dequantize round-trip has bounded error."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int8_per_token,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    # Use realistic attention-scale values (small magnitudes)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    k_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)

    # Sequential slot mapping for easy readback
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    triton_reshape_and_cache_flash_int8_per_token(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    # Dequantize
    for i in range(num_tokens):
        blk = i // block_size
        off = i % block_size
        k_deq = key_cache[blk, off].float() * k_scale_cache[blk, off].unsqueeze(-1)
        v_deq = value_cache[blk, off].float() * v_scale_cache[blk, off].unsqueeze(-1)

        # INT8 quantization error should be at most scale/2 per element
        # (rounding to nearest integer). Use a generous tolerance.
        k_err = (k_deq - key[i].float()).abs()
        v_err = (v_deq - value[i].float()).abs()

        k_max_expected_err = k_scale_cache[blk, off].unsqueeze(-1) * 0.6
        v_max_expected_err = v_scale_cache[blk, off].unsqueeze(-1) * 0.6

        assert (k_err <= k_max_expected_err).all(), (
            f"Key round-trip error too large at token {i}: "
            f"max_err={k_err.max():.6f}, max_expected={k_max_expected_err.max():.6f}"
        )
        assert (v_err <= v_max_expected_err).all(), (
            f"Value round-trip error too large at token {i}: "
            f"max_err={v_err.max():.6f}, max_expected={v_max_expected_err.max():.6f}"
        )


# ===========================================================================
# 6. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_int8_per_token_negative_slot_skipped():
    """Tokens with slot_mapping=-1 should leave the cache unchanged."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int8_per_token,
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
        num_heads,
        dtype=torch.float32,
    )
    v_scale_cache = torch.ones(
        num_blocks,
        block_size,
        num_heads,
        dtype=torch.float32,
    )

    # Mix valid and negative slot mappings
    slot_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    key_cache_before = key_cache.clone()
    val_cache_before = value_cache.clone()

    triton_reshape_and_cache_flash_int8_per_token(
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
# 7. process_weights_after_loading — INT8 paths
# ===========================================================================
class TestProcessWeightsAfterLoadingInt8:
    """Unit tests for kv_cache.py BaseKVCacheMethod.process_weights_after_loading
    when kv_cache_dtype='int8'."""

    def _make_layer(self, k_scale_val, v_scale_val, kv_cache_dtype="int8"):
        """Create a mock attention layer with the required attributes."""
        layer = MagicMock()
        layer.kv_cache_dtype = kv_cache_dtype
        layer.calculate_kv_scales = False

        if isinstance(k_scale_val, torch.Tensor):
            layer.k_scale = torch.nn.Parameter(k_scale_val, requires_grad=False)
            layer.v_scale = torch.nn.Parameter(v_scale_val, requires_grad=False)
        else:
            layer.k_scale = torch.nn.Parameter(
                torch.tensor(k_scale_val), requires_grad=False
            )
            layer.v_scale = torch.nn.Parameter(
                torch.tensor(v_scale_val), requires_grad=False
            )
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        # _k_scale, _v_scale, _q_scale, _prob_scale are the runtime buffers
        layer._k_scale = torch.tensor(0.0)
        layer._v_scale = torch.tensor(0.0)
        layer._q_scale = torch.tensor(0.0)
        layer._prob_scale = torch.tensor(0.0)
        layer._k_scale_float = 0.0
        layer._v_scale_float = 0.0
        layer._q_scale_float = 0.0

        return layer

    def test_int8_no_scales_enables_dynamic(self):
        """INT8 with no checkpoint scales (-1.0) should enable calculate_kv_scales."""
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = self._make_layer(-1.0, -1.0, "int8")
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()

        method.process_weights_after_loading(layer)

        assert layer.calculate_kv_scales is True

    def test_int8_per_tensor_positive_scales(self):
        """INT8 with positive scalar scales should store them correctly."""
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = self._make_layer(0.05, 0.03, "int8")
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()

        method.process_weights_after_loading(layer)

        assert abs(layer._k_scale_float - 0.05) < 1e-6
        assert abs(layer._v_scale_float - 0.03) < 1e-6

    def test_int8_per_head_positive_scales(self):
        """INT8 with per-head [num_kv_heads] scales should replace scalar buffer."""
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

        num_heads = 4
        k_scales = torch.rand(num_heads) * 0.1 + 0.01  # all positive
        v_scales = torch.rand(num_heads) * 0.1 + 0.01

        layer = self._make_layer(k_scales, v_scales, "int8")
        method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
        method.quant_config = MagicMock()

        method.process_weights_after_loading(layer)

        # Should be per-head tensors, not scalars
        assert layer._k_scale.numel() == num_heads
        assert layer._v_scale.numel() == num_heads
        # Float placeholders should be 1.0
        assert layer._k_scale_float == 1.0
        assert layer._v_scale_float == 1.0


# ===========================================================================
# 8. calc_kv_scales — INT8 path
# ===========================================================================
@torch.inference_mode()
def test_calc_kv_scales_int8():
    """Test that calc_kv_scales produces correct per-head scales for INT8."""
    num_tokens = 8
    num_kv_heads = 4
    head_size = 64

    # Create a mock attention layer
    layer = MagicMock()
    layer.kv_cache_dtype = "int8"
    layer.num_kv_heads = num_kv_heads
    layer._k_scale = torch.tensor(0.0, device="cuda")
    layer._v_scale = torch.tensor(0.0, device="cuda")
    layer._q_scale = torch.tensor(0.0, device="cuda")
    layer._k_scale_float = 0.0
    layer._v_scale_float = 0.0
    layer._q_scale_float = 0.0
    layer.q_range = 127.0
    layer.k_range = 127.0
    layer.v_range = 127.0
    layer.calculate_kv_scales = True

    query = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")
    key = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")
    value = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda")

    # Call the function directly (import the class method)
    from vllm.model_executor.layers.attention.attention import Attention

    Attention.calc_kv_scales(layer, query, key, value)

    # Verify per-head scales
    assert layer._k_scale.ndim == 1
    assert layer._k_scale.numel() == num_kv_heads
    assert layer._v_scale.ndim == 1
    assert layer._v_scale.numel() == num_kv_heads

    # Verify scale values match reference
    k_3d = key.reshape(num_tokens, num_kv_heads, head_size)
    v_3d = value.reshape(num_tokens, num_kv_heads, head_size)
    expected_k = (k_3d.abs().amax(dim=(0, 2)) / 127.0).clamp(min=1e-6)
    expected_v = (v_3d.abs().amax(dim=(0, 2)) / 127.0).clamp(min=1e-6)

    torch.testing.assert_close(layer._k_scale, expected_k, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(layer._v_scale, expected_v, atol=1e-5, rtol=1e-4)

    # Should have disabled further scale computation
    assert layer.calculate_kv_scales is False


# ===========================================================================
# 9. Triton unified_attention — INT8 KV round-trip (per-tensor scale)
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
# 10. Triton unified_attention — INT8 KV with per-token scale cache
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

    # Per-(token, head) quantization: scale per (block, slot, head)
    k_absmax = key_cache_bf16.float().abs().amax(dim=-1)
    v_absmax = value_cache_bf16.float().abs().amax(dim=-1)
    k_scale_cache = (k_absmax / 127.0).clamp(min=1e-6).to(torch.float32)
    v_scale_cache = (v_absmax / 127.0).clamp(min=1e-6).to(torch.float32)

    key_cache_int8 = (
        (key_cache_bf16.float() / k_scale_cache.unsqueeze(-1))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    value_cache_int8 = (
        (value_cache_bf16.float() / v_scale_cache.unsqueeze(-1))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )

    # Dequantized reference
    key_cache_deq = key_cache_int8.float() * k_scale_cache.unsqueeze(-1)
    value_cache_deq = value_cache_int8.float() * v_scale_cache.unsqueeze(-1)

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

    # INT8 with per-token scale caches — pass k_descale=None to trigger per-token path
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
