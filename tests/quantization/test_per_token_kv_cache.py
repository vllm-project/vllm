# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-token-head KV cache quantization (INT8 and FP8).

Covers:
- Per-token-head Triton reshape-and-cache kernel
- Round-trip quantize/dequantize accuracy
- process_weights_after_loading early-return path
- End-to-end integration with Triton unified attention kernel

Run: pytest tests/quantization/test_per_token_kv_cache.py -v -s
"""

import random
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVQuantMode, is_quantized_kv_cache

# Skip entire module if no CUDA/ROCm GPU available
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="Per-token-head KV cache tests require CUDA or ROCm GPU.",
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

# Platform-dependent FP8 dtype and range
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MIN, FP8_MAX = get_fp8_min_max()


# ---------------------------------------------------------------------------
# Per-dtype quantization config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantConfig:
    """Quantization parameters for a given cache dtype."""

    cache_dtype: torch.dtype  # torch.uint8, torch.int8, or FP8_DTYPE
    kv_cache_dtype_str: str  # "int4_per_token_head", "int8_per_token_head", or "fp8_per_token_head"
    quant_max: float
    quant_min: float
    kv_quant_mode: KVQuantMode
    # INT8 Triton stores truncate; FP8 hardware casts round.
    uses_trunc: bool


INT8_CONFIG = QuantConfig(
    cache_dtype=torch.int8,
    kv_cache_dtype_str="int8_per_token_head",
    quant_max=127.0,
    quant_min=-128.0,
    kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    uses_trunc=True,
)
FP8_CONFIG = QuantConfig(
    cache_dtype=FP8_DTYPE,
    kv_cache_dtype_str="fp8_per_token_head",
    quant_max=FP8_MAX,
    quant_min=FP8_MIN,
    kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
    uses_trunc=False,
)
INT4_CONFIG = QuantConfig(
    cache_dtype=torch.uint8,
    kv_cache_dtype_str="int4_per_token_head",
    quant_max=7.0,
    quant_min=-8.0,
    kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    uses_trunc=False,  # INT4 uses round-to-nearest via rint
)

QUANT_CONFIGS = [INT4_CONFIG, INT8_CONFIG, FP8_CONFIG]


@pytest.fixture(params=QUANT_CONFIGS, ids=["int4", "int8", "fp8"])
def qcfg(request) -> QuantConfig:
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quantize_per_token_head_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token-head quantization (one scale per token per head).

    Returns (quantized, scales) where scales is [num_tokens, num_heads].
    """
    absmax = data.float().abs().amax(dim=2)  # [num_tokens, num_heads]
    scales = (absmax / cfg.quant_max).clamp(min=1e-6)
    scaled = data.float() * (1.0 / scales[:, :, None])
    if cfg.uses_trunc:
        q = scaled.round().clamp(cfg.quant_min, cfg.quant_max).to(cfg.cache_dtype)
    else:
        q = scaled.clamp(cfg.quant_min, cfg.quant_max).to(cfg.cache_dtype)
    return q, scales


# ===========================================================================
# 1. is_quantized_kv_cache / get_kv_quant_mode
# ===========================================================================
class TestIsQuantizedKvCache:
    def test_fp8_variants(self):
        assert is_quantized_kv_cache("fp8")
        assert is_quantized_kv_cache("fp8_e4m3")
        assert is_quantized_kv_cache("fp8_e5m2")

    def test_int4_per_token_head(self):
        assert is_quantized_kv_cache("int4_per_token_head")

    def test_int8_per_token_head(self):
        assert is_quantized_kv_cache("int8_per_token_head")

    def test_fp8_per_token_head(self):
        assert is_quantized_kv_cache("fp8_per_token_head")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")

    def test_kv_quant_mode_int4(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert (
            get_kv_quant_mode("int4_per_token_head") == KVQuantMode.INT4_PER_TOKEN_HEAD
        )

    def test_kv_quant_mode_int8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert (
            get_kv_quant_mode("int8_per_token_head") == KVQuantMode.INT8_PER_TOKEN_HEAD
        )

    def test_kv_quant_mode_fp8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert get_kv_quant_mode("fp8_per_token_head") == KVQuantMode.FP8_PER_TOKEN_HEAD


# ===========================================================================
# 2. Triton per-token-head kernel (reshape-and-cache)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_per_token_head(
    qcfg: QuantConfig,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    seed: int,
):
    """Test triton_reshape_and_cache_flash_per_token_head_quant kernel."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_blocks = (num_tokens + block_size - 1) // block_size + 4
    is_int4 = qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
    cache_head_size = head_size // 2 if is_int4 else head_size

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    k_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        kv_quant_mode=qcfg.kv_quant_mode,
    )

    # INT4 uses asymmetric quantization with optimal clipping — no simple
    # PyTorch reference exists, so we only check dequantized round-trip.
    # INT8/FP8 use symmetric quantization: we verify both dequantized values
    # AND per-head scales against the PyTorch reference implementation.
    if not is_int4:
        # Reference
        ref_k_quant, ref_k_scales = _quantize_per_token_head_ref(key, qcfg)
        ref_v_quant, ref_v_scales = _quantize_per_token_head_ref(value, qcfg)

    # Compare dequantized values rather than raw quantized values.
    # Triton and PyTorch reductions can differ at FP8 rounding boundaries
    # (up to 32 in quantized domain for fp8_e4m3), but the dequantized
    # error is bounded by the scale.
    # INT4 has much coarser quantization + optimal clipping, so wider tolerance.
    deq_atol = 0.5 if is_int4 else 0.1
    deq_rtol = 0.5 if is_int4 else 0.1
    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size

        if is_int4:
            for label, data, cache, sc in [
                ("key", key, key_cache, k_scale_cache),
                ("val", value, value_cache, v_scale_cache),
            ]:
                packed_scale = sc[blk, off]  # [num_heads] float32
                # Extract zp and clean scale via steganography (zp in
                # low 4 mantissa bits of the float32 scale).
                scale_bits = packed_scale.view(torch.int32)
                zp = (scale_bits & 0xF).to(torch.float32)  # [num_heads]
                clean_scale = (scale_bits & -16).view(torch.float32)

                # Unpack uint8 → two unsigned int4 values per byte
                packed = cache[blk, off]  # [num_heads, head_size//2] uint8
                lo = (packed & 0xF).to(torch.float32)  # unsigned [0,15]
                hi = ((packed >> 4) & 0xF).to(torch.float32)
                # Interleave: even indices from lo, odd from hi
                full = torch.zeros(
                    num_heads, head_size, dtype=torch.float32, device=packed.device
                )
                full[:, 0::2] = lo
                full[:, 1::2] = hi
                # Asymmetric dequant: (q_uint - zp) * scale
                deq = (full - zp[:, None]) * clean_scale[:, None]
                ref_deq = data[i].float()
                torch.testing.assert_close(
                    deq, ref_deq, atol=deq_atol, rtol=deq_rtol
                )
        else:
            actual_k_scale = k_scale_cache[blk, off]  # [num_heads]
            k_deq = key_cache[blk, off].float() * actual_k_scale[:, None]
            k_ref_deq = key[i].float()
            torch.testing.assert_close(
                k_deq,
                k_ref_deq,
                atol=0.1,
                rtol=0.1,
            )
            actual_v_scale = v_scale_cache[blk, off]  # [num_heads]
            v_deq = value_cache[blk, off].float() * actual_v_scale[:, None]
            v_ref_deq = value[i].float()
            torch.testing.assert_close(
                v_deq,
                v_ref_deq,
                atol=0.1,
                rtol=0.1,
            )
            # Per-head scales: [num_heads]
            torch.testing.assert_close(
                k_scale_cache[blk, off], ref_k_scales[i], atol=1e-4, rtol=1e-3
            )
            torch.testing.assert_close(
                v_scale_cache[blk, off], ref_v_scales[i], atol=1e-4, rtol=1e-3
            )


# ===========================================================================
# 3. Per-token-head round-trip accuracy (quantize -> dequantize)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_per_token_head_round_trip_accuracy(
    qcfg: QuantConfig,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
):
    """Verify per-token-head round-trip: kernel dequant matches reference.

    INT8: Triton truncates on float->int8 store.
    FP8: hardware cast (clamp then cast).
    """
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    is_int4 = qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
    num_blocks = (num_tokens + block_size - 1) // block_size + 2
    cache_head_size = head_size // 2 if is_int4 else head_size

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    k_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)

    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        kv_quant_mode=qcfg.kv_quant_mode,
    )

    rt_atol = 0.5 if is_int4 else 0.1
    for i in range(num_tokens):
        blk = i // block_size
        off = i % block_size

        for label, data, cache, sc in [
            ("key", key, key_cache, k_scale_cache),
            ("val", value, value_cache, v_scale_cache),
        ]:
            for h in range(num_heads):
                orig = data[i, h].float()  # [head_size]
                actual_sc = sc[blk, off, h]

                if is_int4:
                    # Extract zp and clean scale via steganography
                    sc_bits = actual_sc.view(torch.int32)
                    zp = (sc_bits & 0xF).to(torch.float32)
                    clean_sc = (sc_bits & -16).view(torch.float32)

                    packed = cache[blk, off, h]  # [head_size//2] uint8
                    lo = (packed & 0xF).to(torch.float32)  # unsigned [0,15]
                    hi = ((packed >> 4) & 0xF).to(torch.float32)
                    full = torch.zeros(
                        head_size, dtype=torch.float32, device=packed.device
                    )
                    full[0::2] = lo
                    full[1::2] = hi
                    # Asymmetric dequant: (q_uint - zp) * scale
                    actual_deq = (full - zp) * clean_sc
                else:
                    actual_deq = cache[blk, off, h].float() * actual_sc

                torch.testing.assert_close(
                    actual_deq,
                    orig,
                    atol=rt_atol,
                    rtol=rt_atol,
                )


# ===========================================================================
# 4. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_per_token_head_negative_slot_skipped(qcfg: QuantConfig):
    """Tokens with slot_mapping=-1 should leave the cache unchanged."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    torch.set_default_device("cuda")
    num_tokens = 4
    num_heads = 2
    head_size = 64
    block_size = 16
    num_blocks = 2
    is_int4 = qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
    cache_head_size = head_size // 2 if is_int4 else head_size

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=qcfg.cache_dtype
    )
    k_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale_cache = torch.ones(num_blocks, block_size, num_heads, dtype=torch.float32)

    slot_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    key_cache_before = key_cache.clone()
    val_cache_before = value_cache.clone()

    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        kv_quant_mode=qcfg.kv_quant_mode,
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
# 5. process_weights_after_loading -- per-token-head early return
# ===========================================================================
@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["int4_per_token_head", "int8_per_token_head", "fp8_per_token_head"],
)
def test_process_weights_sets_placeholder_scales(kv_cache_dtype: str):
    """Per-token-head should set _k_scale=1.0, _v_scale=1.0
    and delete checkpoint attrs."""
    from vllm.model_executor.layers.quantization.kv_cache import (
        BaseKVCacheMethod,
    )

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
# 6. Triton unified_attention -- per-token-head scale cache (INT8 and FP8)
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
def test_triton_unified_attention_per_token_head_scale(
    qcfg: QuantConfig,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: quantized KV with per-token-head scale caches."""
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
        sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16
    )

    key_cache_bf16 = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache_bf16 = torch.randn_like(key_cache_bf16)

    is_int4 = qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD

    # Per-token-head quantization: one scale per (block, slot, head)
    k_absmax = key_cache_bf16.float().abs().amax(dim=-1)  # [..., num_kv_heads]
    v_absmax = value_cache_bf16.float().abs().amax(dim=-1)
    k_scale_cache = (k_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)
    v_scale_cache = (v_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)

    scaled_k = key_cache_bf16.float() / k_scale_cache[:, :, :, None]
    scaled_v = value_cache_bf16.float() / v_scale_cache[:, :, :, None]

    # Quantize
    key_cache_q_full = scaled_k.round().clamp(qcfg.quant_min, qcfg.quant_max)
    value_cache_q_full = scaled_v.round().clamp(qcfg.quant_min, qcfg.quant_max)

    # Dequantized reference (before packing, for the baseline comparison)
    key_cache_deq = key_cache_q_full * k_scale_cache[:, :, :, None]
    value_cache_deq = value_cache_q_full * v_scale_cache[:, :, :, None]

    if is_int4:
        # Pack two int4 values into one uint8: low_nibble=even, high_nibble=odd
        def _pack_int4(data_float):
            # data_float: [..., head_size] with values in [-8, 7]
            u = (data_float + 8.0).to(torch.uint8)  # [0, 15]
            lo = u[..., 0::2]  # even indices
            hi = u[..., 1::2]  # odd indices
            return (lo & 0xF) | ((hi & 0xF) << 4)  # uint8

        key_cache_q = _pack_int4(key_cache_q_full)
        value_cache_q = _pack_int4(value_cache_q_full)

        # Pack zp=8 into the low 4 mantissa bits of the float32 scale
        # (steganography), matching what the Triton kernel writes and
        # the attention kernel expects to read.
        zp = 8  # symmetric quant: offset = 8 maps [-8,7] → [0,15]
        k_bits = k_scale_cache.view(torch.int32)
        k_scale_cache = ((k_bits & -16) | (zp & 0xF)).view(torch.float32)
        v_bits = v_scale_cache.view(torch.int32)
        v_scale_cache = ((v_bits & -16) | (zp & 0xF)).view(torch.float32)
    elif qcfg.uses_trunc:
        key_cache_q = key_cache_q_full.to(qcfg.cache_dtype)
        value_cache_q = value_cache_q_full.to(qcfg.cache_dtype)
    else:
        key_cache_q = scaled_k.clamp(qcfg.quant_min, qcfg.quant_max).to(
            qcfg.cache_dtype
        )
        value_cache_q = scaled_v.clamp(qcfg.quant_min, qcfg.quant_max).to(
            qcfg.cache_dtype
        )

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

    output_q = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_q,
        v=value_cache_q,
        out=output_q,
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
        kv_quant_mode=qcfg.kv_quant_mode,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

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

    # INT4 has much coarser quantization (4 bits) so needs wider tolerance.
    atol = 0.5 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 5e-2
    rtol = 0.5 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 5e-2
    torch.testing.assert_close(output_q, output_ref, atol=atol, rtol=rtol)
