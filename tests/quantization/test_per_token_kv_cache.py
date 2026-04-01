# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-token KV cache quantization (INT8 and FP8).

Covers:
- Per-token Triton reshape-and-cache kernel
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
        reason="Per-token KV cache tests require CUDA or ROCm GPU.",
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

    cache_dtype: torch.dtype  # torch.int8 or FP8_DTYPE
    kv_cache_dtype_str: str  # "int8_per_token" or "fp8_per_token"
    quant_max: float
    quant_min: float
    kv_quant_mode: KVQuantMode
    # INT8 Triton stores truncate; FP8 hardware casts round.
    uses_trunc: bool


INT8_CONFIG = QuantConfig(
    cache_dtype=torch.int8,
    kv_cache_dtype_str="int8_per_token",
    quant_max=127.0,
    quant_min=-128.0,
    kv_quant_mode=KVQuantMode.INT8_PER_TOKEN,
    uses_trunc=True,
)
FP8_CONFIG = QuantConfig(
    cache_dtype=FP8_DTYPE,
    kv_cache_dtype_str="fp8_per_token",
    quant_max=FP8_MAX,
    quant_min=FP8_MIN,
    kv_quant_mode=KVQuantMode.FP8_PER_TOKEN,
    uses_trunc=False,
)

QUANT_CONFIGS = [INT8_CONFIG, FP8_CONFIG]


@pytest.fixture(params=QUANT_CONFIGS, ids=["int8", "fp8"])
def qcfg(request) -> QuantConfig:
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quantize_per_token_ref(
    data: torch.Tensor,  # [num_tokens, num_heads, head_size]
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token quantization (one scale per token).

    Returns (quantized, scales) where scales is [num_tokens].
    """
    absmax = data.float().abs().amax(dim=(1, 2))  # [num_tokens]
    scales = (absmax / cfg.quant_max).clamp(min=1e-6)
    scaled = data.float() / scales[:, None, None]
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

    def test_int8_per_token(self):
        assert is_quantized_kv_cache("int8_per_token")

    def test_fp8_per_token(self):
        assert is_quantized_kv_cache("fp8_per_token")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")

    def test_kv_quant_mode_int8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert get_kv_quant_mode("int8_per_token") == KVQuantMode.INT8_PER_TOKEN

    def test_kv_quant_mode_fp8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert get_kv_quant_mode("fp8_per_token") == KVQuantMode.FP8_PER_TOKEN


# ===========================================================================
# 2. Triton per-token kernel (reshape-and-cache)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_per_token(
    qcfg: QuantConfig,
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
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
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
    ref_k_quant, ref_k_scales = _quantize_per_token_ref(key, qcfg)
    ref_v_quant, ref_v_scales = _quantize_per_token_ref(value, qcfg)

    # FP8 has wider range so needs looser tolerance on quantized values
    data_atol = 2.0 if not qcfg.uses_trunc else 1.0
    data_rtol = 0.1 if not qcfg.uses_trunc else 0.0

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size

        torch.testing.assert_close(
            key_cache[blk, off].float(),
            ref_k_quant[i].float(),
            atol=data_atol,
            rtol=data_rtol,
        )
        torch.testing.assert_close(
            value_cache[blk, off].float(),
            ref_v_quant[i].float(),
            atol=data_atol,
            rtol=data_rtol,
        )
        torch.testing.assert_close(
            k_scale_cache[blk, off], ref_k_scales[i], atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            v_scale_cache[blk, off], ref_v_scales[i], atol=1e-4, rtol=1e-3
        )


# ===========================================================================
# 3. Per-token round-trip accuracy (quantize -> dequantize)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_per_token_round_trip_accuracy(
    qcfg: QuantConfig,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
):
    """Verify per-token round-trip: kernel dequant matches reference.

    INT8: Triton truncates on float->int8 store.
    FP8: hardware cast (clamp then cast).
    """
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_quant,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
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
            ref_scale = (absmax / qcfg.quant_max).clamp(min=1e-6)

            # Build reference matching kernel semantics
            scaled = orig / ref_scale
            if qcfg.uses_trunc:
                ref_q = (
                    scaled.clamp(qcfg.quant_min, qcfg.quant_max)
                    .trunc()
                    .to(qcfg.cache_dtype)
                )
            else:
                ref_q = scaled.clamp(qcfg.quant_min, qcfg.quant_max).to(
                    qcfg.cache_dtype
                )
            ref_deq = ref_q.float() * ref_scale

            actual_q = cache[blk, off]
            actual_sc = sc[blk, off]
            actual_deq = actual_q.float() * actual_sc

            # Scales must match
            torch.testing.assert_close(actual_sc, ref_scale, atol=1e-5, rtol=1e-5)

            if qcfg.uses_trunc:
                # INT8: allow +-1 for bf16->f32 differences
                torch.testing.assert_close(
                    actual_q.float(), ref_q.float(), atol=1.0, rtol=0.0
                )
                # Dequantised error bounded by 1 * scale
                err = (actual_deq - ref_deq).abs()
                bound = actual_sc * 1.01
                assert (err <= bound).all(), (
                    f"{label} dequant error at token {i}: "
                    f"max={err.max():.6f}, bound={bound.max():.6f}"
                )
            else:
                # FP8: wider tolerance
                torch.testing.assert_close(actual_deq, ref_deq, atol=0.05, rtol=0.05)


# ===========================================================================
# 4. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_per_token_negative_slot_skipped(qcfg: QuantConfig):
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
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=qcfg.cache_dtype
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
# 5. process_weights_after_loading -- per-token early return
# ===========================================================================
@pytest.mark.parametrize("kv_cache_dtype", ["int8_per_token", "fp8_per_token"])
def test_process_weights_sets_placeholder_scales(kv_cache_dtype: str):
    """Per-token should set _k_scale=1.0, _v_scale=1.0
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
# 6. Triton unified_attention -- per-token scale cache (INT8 and FP8)
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
def test_triton_unified_attention_per_token_scale(
    qcfg: QuantConfig,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
):
    """End-to-end: quantized KV with per-token scale caches."""
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

    # Per-token quantization: one scale per (block, slot)
    k_absmax = key_cache_bf16.float().abs().amax(dim=(-2, -1))
    v_absmax = value_cache_bf16.float().abs().amax(dim=(-2, -1))
    k_scale_cache = (k_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)
    v_scale_cache = (v_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)

    scaled_k = key_cache_bf16.float() / k_scale_cache[:, :, None, None]
    scaled_v = value_cache_bf16.float() / v_scale_cache[:, :, None, None]
    if qcfg.uses_trunc:
        key_cache_q = (
            scaled_k.round().clamp(qcfg.quant_min, qcfg.quant_max).to(qcfg.cache_dtype)
        )
        value_cache_q = (
            scaled_v.round().clamp(qcfg.quant_min, qcfg.quant_max).to(qcfg.cache_dtype)
        )
    else:
        key_cache_q = scaled_k.clamp(qcfg.quant_min, qcfg.quant_max).to(
            qcfg.cache_dtype
        )
        value_cache_q = scaled_v.clamp(qcfg.quant_min, qcfg.quant_max).to(
            qcfg.cache_dtype
        )

    # Dequantized reference
    key_cache_deq = key_cache_q.float() * k_scale_cache[:, :, None, None]
    value_cache_deq = value_cache_q.float() * v_scale_cache[:, :, None, None]

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

    torch.testing.assert_close(output_q, output_ref, atol=5e-2, rtol=5e-2)
