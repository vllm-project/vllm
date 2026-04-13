# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-token-head KV cache quantization (INT8, INT4 and FP8).

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
from compressed_tensors.transform import deterministic_hadamard_matrix

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    INT4_CODEBOOK_LEVELS,
    INT4_MAGNITUDE_LEVELS,
)
from vllm.v1.kv_cache_interface import (
    INT4_CHANNELS_PER_SCALE,
    KVQuantMode,
    get_int4_num_scale_groups,
    is_quantized_kv_cache,
)

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

    cache_dtype: torch.dtype
    kv_cache_dtype_str: str
    quant_max: float
    quant_min: float
    kv_quant_mode: KVQuantMode
    scale_dtype: torch.dtype
    # INT8 Triton stores truncate; FP8 casts round; INT4 uses codebook lookup.
    uses_trunc: bool


INT8_CONFIG = QuantConfig(
    cache_dtype=torch.int8,
    kv_cache_dtype_str="int8_per_token_head",
    quant_max=127.0,
    quant_min=-128.0,
    kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    scale_dtype=torch.float32,
    uses_trunc=True,
)
INT4_CONFIG = QuantConfig(
    cache_dtype=torch.uint8,
    kv_cache_dtype_str="int4_per_token_head",
    quant_max=1.0,
    quant_min=0.0,
    kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    scale_dtype=torch.float16,
    uses_trunc=False,
)
FP8_CONFIG = QuantConfig(
    cache_dtype=FP8_DTYPE,
    kv_cache_dtype_str="fp8_per_token_head",
    quant_max=FP8_MAX,
    quant_min=FP8_MIN,
    kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
    scale_dtype=torch.float32,
    uses_trunc=False,
)

QUANT_CONFIGS = [INT8_CONFIG, INT4_CONFIG, FP8_CONFIG]


@pytest.fixture(params=QUANT_CONFIGS, ids=["int8", "int4", "fp8"])
def qcfg(request) -> QuantConfig:
    cfg = request.param
    if (
        cfg.kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
        and not current_platform.supports_fp8()
    ):
        pytest.skip("FP8 per-token-head tests require FP8-capable hardware.")
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
INT4_MAGNITUDE_LEVELS_T = torch.tensor(INT4_MAGNITUDE_LEVELS, dtype=torch.float32)
INT4_CODEBOOK_LEVELS_T = torch.tensor(INT4_CODEBOOK_LEVELS, dtype=torch.float32)


def _packed_head_size(head_size: int, cfg: QuantConfig) -> int:
    if cfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        return (head_size + 1) // 2
    return head_size


def _allocate_cache_tensors(
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_head_size = _packed_head_size(head_size, cfg)
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, cache_head_size, dtype=cfg.cache_dtype
    )
    value_cache = torch.zeros_like(key_cache)
    scale_shape = (
        (num_blocks, block_size, num_heads, get_int4_num_scale_groups(head_size))
        if cfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
        else (num_blocks, block_size, num_heads)
    )
    k_scale_cache = torch.ones(scale_shape, dtype=cfg.scale_dtype)
    v_scale_cache = torch.ones_like(k_scale_cache)
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def _quantize_per_token_head_ref(
    data: torch.Tensor,
    cfg: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token-head quantization (one scale per token per head).

    Supports arbitrary leading dimensions with the last dimension treated as
    the head dimension.
    """
    absmax = data.float().abs().amax(dim=-1)
    if cfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        num_scale_groups = get_int4_num_scale_groups(data.shape[-1])
        scales = torch.empty(
            *data.shape[:-1],
            num_scale_groups,
            device=data.device,
            dtype=cfg.scale_dtype,
        )
        normalized = torch.empty_like(data, dtype=torch.float32)
        for group_idx in range(num_scale_groups):
            start = group_idx * INT4_CHANNELS_PER_SCALE
            end = min(start + INT4_CHANNELS_PER_SCALE, data.shape[-1])
            group = data[..., start:end].float()
            group_scale = group.abs().amax(dim=-1).clamp(min=1e-6)
            scales[..., group_idx] = group_scale.to(cfg.scale_dtype)
            normalized[..., start:end] = group / group_scale.unsqueeze(-1)
        level_dist = torch.abs(
            normalized.abs().unsqueeze(-1) - INT4_MAGNITUDE_LEVELS_T.to(data.device)
        )
        mag_idx = level_dist.argmin(dim=-1).to(torch.uint8)
        signed_idx = mag_idx | ((normalized < 0).to(torch.uint8) << 3)
        packed = torch.zeros(
            *data.shape[:-1],
            (data.shape[-1] + 1) // 2,
            dtype=torch.uint8,
            device=data.device,
        )
        packed.copy_(signed_idx[..., ::2])
        packed[..., : signed_idx[..., 1::2].shape[-1]] |= signed_idx[..., 1::2] << 4
        return packed, scales

    scales = (absmax / cfg.quant_max).clamp(min=1e-6).to(cfg.scale_dtype)
    scaled = data.float() * (1.0 / scales.unsqueeze(-1).float())
    if cfg.uses_trunc:
        q = scaled.round().clamp(cfg.quant_min, cfg.quant_max).to(cfg.cache_dtype)
    else:
        q = scaled.clamp(cfg.quant_min, cfg.quant_max).to(cfg.cache_dtype)
    return q, scales


def _dequantize_per_token_head_ref(
    q: torch.Tensor,
    scales: torch.Tensor,
    cfg: QuantConfig,
    head_size: int,
) -> torch.Tensor:
    if cfg.kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD:
        return q.float() * scales.unsqueeze(-1).float()

    lo = (q & 0x0F).to(torch.int32)
    hi = ((q >> 4) & 0x0F).to(torch.int32)
    packed = torch.stack((lo, hi), dim=-1).reshape(*q.shape[:-1], -1)[..., :head_size]
    levels = INT4_CODEBOOK_LEVELS_T.to(q.device)[packed]
    group_idx = (
        torch.arange(head_size, device=q.device) // INT4_CHANNELS_PER_SCALE
    ).view(*([1] * (levels.dim() - 1)), head_size)
    group_scales = torch.take_along_dim(
        scales.float(), group_idx.expand(*levels.shape[:-1], head_size), dim=-1
    )
    return levels * group_scales


def _dequantize_symmetric_linear_int4_ref(data: torch.Tensor) -> torch.Tensor:
    absmax = data.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = absmax / 7.0
    q = torch.round(data.float() / scale).clamp(-7, 7)
    return q * scale


def _apply_hadamard_ref(data: torch.Tensor) -> torch.Tensor:
    head_size = data.shape[-1]
    hadamard = (
        deterministic_hadamard_matrix(
            head_size, dtype=torch.float64, device=data.device
        )
        / head_size**0.5
    )
    return (data.to(torch.float64) @ hadamard.T).to(torch.float32)


# ===========================================================================
# 1. is_quantized_kv_cache / get_kv_quant_mode
# ===========================================================================
class TestIsQuantizedKvCache:
    def test_fp8_variants(self):
        assert is_quantized_kv_cache("fp8")
        assert is_quantized_kv_cache("fp8_e4m3")
        assert is_quantized_kv_cache("fp8_e5m2")

    def test_int8_per_token_head(self):
        assert is_quantized_kv_cache("int8_per_token_head")

    def test_int4_per_token_head(self):
        assert is_quantized_kv_cache("int4_per_token_head")

    def test_fp8_per_token_head(self):
        assert is_quantized_kv_cache("fp8_per_token_head")

    def test_auto(self):
        assert not is_quantized_kv_cache("auto")

    def test_bfloat16(self):
        assert not is_quantized_kv_cache("bfloat16")

    def test_kv_quant_mode_int8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert (
            get_kv_quant_mode("int8_per_token_head") == KVQuantMode.INT8_PER_TOKEN_HEAD
        )

    def test_kv_quant_mode_fp8(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert get_kv_quant_mode("fp8_per_token_head") == KVQuantMode.FP8_PER_TOKEN_HEAD

    def test_kv_quant_mode_int4(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        assert (
            get_kv_quant_mode("int4_per_token_head") == KVQuantMode.INT4_PER_TOKEN_HEAD
        )


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
    """Test per-token-head reshape-and-cache kernels."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int4_per_token_head,
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache, value_cache, k_scale_cache, v_scale_cache = _allocate_cache_tensors(
        num_blocks, block_size, num_heads, head_size, qcfg
    )

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )

    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        triton_reshape_and_cache_flash_int4_per_token_head(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
        )
    else:
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
        )

    ref_k_quant, ref_k_scales = _quantize_per_token_head_ref(key, qcfg)
    ref_v_quant, ref_v_scales = _quantize_per_token_head_ref(value, qcfg)
    ref_k_deq = _dequantize_per_token_head_ref(
        ref_k_quant, ref_k_scales, qcfg, head_size
    )
    ref_v_deq = _dequantize_per_token_head_ref(
        ref_v_quant, ref_v_scales, qcfg, head_size
    )

    for i, slot in enumerate(slot_mapping.tolist()):
        blk = slot // block_size
        off = slot % block_size

        actual_k_scale = k_scale_cache[blk, off]
        k_deq = _dequantize_per_token_head_ref(
            key_cache[blk, off][None], actual_k_scale[None], qcfg, head_size
        )[0]
        k_ref_deq = ref_k_deq[i].float()
        torch.testing.assert_close(
            k_deq,
            k_ref_deq,
            atol=0.12 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 0.1,
            rtol=0.12 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 0.1,
        )
        actual_v_scale = v_scale_cache[blk, off]
        v_deq = _dequantize_per_token_head_ref(
            value_cache[blk, off][None], actual_v_scale[None], qcfg, head_size
        )[0]
        v_ref_deq = ref_v_deq[i].float()
        torch.testing.assert_close(
            v_deq,
            v_ref_deq,
            atol=0.12 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 0.1,
            rtol=0.12 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 0.1,
        )
        torch.testing.assert_close(
            k_scale_cache[blk, off].float(),
            ref_k_scales[i].float(),
            atol=1e-4,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            v_scale_cache[blk, off].float(),
            ref_v_scales[i].float(),
            atol=1e-4,
            rtol=1e-3,
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
    """Verify per-token-head round-trip: kernel dequant matches reference."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int4_per_token_head,
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_blocks = (num_tokens + block_size - 1) // block_size + 2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16) * 0.5

    key_cache, value_cache, k_scale_cache, v_scale_cache = _allocate_cache_tensors(
        num_blocks, block_size, num_heads, head_size, qcfg
    )

    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        triton_reshape_and_cache_flash_int4_per_token_head(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
        )
    else:
        triton_reshape_and_cache_flash_per_token_head_quant(
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

        for data, cache, sc in [
            (key, key_cache, k_scale_cache),
            (value, value_cache, v_scale_cache),
        ]:
            for h in range(num_heads):
                orig = data[i, h].float()
                actual_q = cache[blk, off, h][None, None]
                actual_sc = sc[blk, off, h][None, None]
                actual_deq = _dequantize_per_token_head_ref(
                    actual_q, actual_sc, qcfg, head_size
                )[0, 0]
                torch.testing.assert_close(
                    actual_deq,
                    orig,
                    atol=0.12
                    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
                    else 0.1,
                    rtol=0.12
                    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
                    else 0.1,
                )


@pytest.mark.parametrize("head_size", [64, 128])
@torch.inference_mode()
def test_int4_gaussian_codebook_beats_linear_int4_mse(head_size: int):
    torch.set_default_device("cuda")
    set_random_seed(123)

    key = torch.randn(256, 8, head_size, dtype=torch.bfloat16)
    value = torch.randn(256, 8, head_size, dtype=torch.bfloat16)

    for label, data in [("key", key), ("value", value)]:
        quantized, scales = _quantize_per_token_head_ref(data, INT4_CONFIG)
        gaussian_deq = _dequantize_per_token_head_ref(
            quantized, scales, INT4_CONFIG, head_size
        )
        linear_deq = _dequantize_symmetric_linear_int4_ref(data)

        gaussian_mse = (data.float() - gaussian_deq.float()).square().mean()
        linear_mse = (data.float() - linear_deq.float()).square().mean()

        assert gaussian_mse < linear_mse, (
            f"{label} Gaussian-friendly int4 should beat symmetric linear int4 "
            f"for normal inputs, got {gaussian_mse.item():.6f} >= "
            f"{linear_mse.item():.6f}"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Hadamard-backed int4 path is currently CUDA-only.",
)
@torch.inference_mode()
def test_int4_hadamard_helper_preserves_fp16_signal_out_of_place():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    torch.set_default_device("cuda")
    set_random_seed(777)

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
    )

    x = torch.randn(8, 4, 128, dtype=torch.float16)
    x_before = x.clone()
    y = impl._maybe_hadamard_transform(x, inplace=False)

    assert y.dtype == x.dtype
    assert torch.count_nonzero(y).item() > 0
    torch.testing.assert_close(x, x_before)

    x_norm = x.float().norm(dim=-1)
    y_norm = y.float().norm(dim=-1)
    torch.testing.assert_close(y_norm, x_norm, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Hadamard-backed int4 path is currently CUDA-only.",
)
@torch.inference_mode()
def test_int4_hadamard_helper_reuses_workspace_out_of_place(workspace_init):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.worker.workspace import current_workspace_manager

    torch.set_default_device("cuda")
    set_random_seed(778)

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
    )

    x = torch.randn(8, 4, 128, dtype=torch.float16)
    y = impl._maybe_hadamard_transform(x, inplace=False)

    workspace = current_workspace_manager()._current_workspaces[0]
    assert workspace is not None
    assert y.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not hasattr(torch.ops._C, "hadacore_transform"),
    reason="Hadamard-backed int4 path requires CUDA hadacore.",
)
@torch.inference_mode()
def test_int4_hadamard_helper_inplace_non_contiguous_updates_source():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    torch.set_default_device("cuda")
    set_random_seed(779)

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
    )

    x = torch.randn(8, 4, 128, dtype=torch.float16).transpose(0, 1)
    x_before = x.clone()

    y = impl._maybe_hadamard_transform(x, inplace=True)

    assert y.data_ptr() == x.data_ptr()
    assert not torch.equal(x, x_before)

    ref = x_before.reshape(-1, 128).contiguous()
    ops.hadacore_transform(ref, inplace=True)
    ref = ref.view_as(x_before)
    torch.testing.assert_close(x, ref)


@torch.inference_mode()
def test_int4_hadamard_helper_uses_non_cuda_fallback(monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
    )

    x = torch.randn(2, 4, 128, dtype=torch.float32, device="cpu")
    expected = x + 1
    fallback = MagicMock(return_value=expected)

    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(triton_attn, "hadamard_transform", fallback)

    y = impl._maybe_hadamard_transform(x, inplace=False)

    fallback.assert_called_once_with(x, inplace=False)
    assert y is expected


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Hadamard-backed int4 path is currently CUDA-only.",
)
@torch.inference_mode()
def test_int4_encoder_attention_skips_hadamard_query_transform():
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    torch.set_default_device("cuda")

    impl = TritonAttentionImpl(
        num_heads=4,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=4,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_per_token_head",
        attn_type=AttentionType.ENCODER_ONLY,
    )
    impl._maybe_hadamard_transform = MagicMock(side_effect=AssertionError("unexpected"))

    attn_metadata = MagicMock(use_cascade=False, num_actual_tokens=2)
    query = torch.randn(2, 4, 128, dtype=torch.float16)
    key = torch.randn(2, 4, 128, dtype=torch.float16)
    value = torch.randn(2, 4, 128, dtype=torch.float16)
    kv_cache = torch.empty(0, dtype=torch.uint8)
    output = torch.empty_like(query)

    with pytest.raises(NotImplementedError, match="quantized KV cache"):
        impl.forward(
            layer=MagicMock(),
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

    impl._maybe_hadamard_transform.assert_not_called()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Hadamard-backed int4 path is currently CUDA-only.",
)
@pytest.mark.parametrize("head_size", [64, 128])
@torch.inference_mode()
def test_int4_hadamard_round_trip_preserves_attention_semantics(head_size: int):
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.set_default_device("cuda")
    set_random_seed(321)

    num_seqs = 2
    query_lens = [1, 1]
    kv_lens = [64, 96]
    num_query_heads = 4
    num_kv_heads = 4
    block_size = 16
    num_blocks = 512
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)

    query_h = ops.hadacore_transform(
        query.reshape(-1, head_size).contiguous(), inplace=False
    ).view_as(query)
    key_cache_h = ops.hadacore_transform(
        key_cache.reshape(-1, head_size).contiguous(), inplace=False
    ).view_as(key_cache)
    key_cache_q, k_scale_cache = _quantize_per_token_head_ref(key_cache_h, INT4_CONFIG)
    value_cache_q, v_scale_cache = _quantize_per_token_head_ref(
        value_cache, INT4_CONFIG
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
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

    output_h = torch.empty_like(query)
    unified_attention(
        q=query_h,
        k=key_cache_q,
        v=value_cache_q,
        out=output_h,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
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
        kv_quant_mode=INT4_CONFIG.kv_quant_mode,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )
    key_cache_h_deq = _dequantize_per_token_head_ref(
        key_cache_q, k_scale_cache, INT4_CONFIG, head_size
    ).to(torch.bfloat16)
    value_cache_h_deq = _dequantize_per_token_head_ref(
        value_cache_q, v_scale_cache, INT4_CONFIG, head_size
    ).to(torch.bfloat16)

    output_ref = torch.empty_like(query)
    unified_attention(
        q=query_h,
        k=key_cache_h_deq,
        v=value_cache_h_deq,
        out=output_ref,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
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
    # INT4 + Hadamard stays close to the dequantized reference, but the
    # smaller 64-d head case can be slightly noisier than the write/read-only
    # round-trip checks above.
    torch.testing.assert_close(output_h, output_ref, atol=2e-1, rtol=2e-1)


# ===========================================================================
# 4. Negative slot mapping (padding tokens should be skipped)
# ===========================================================================
@torch.inference_mode()
def test_per_token_head_negative_slot_skipped(qcfg: QuantConfig):
    """Tokens with slot_mapping=-1 should leave the cache unchanged."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_int4_per_token_head,
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    torch.set_default_device("cuda")
    num_tokens = 4
    num_heads = 2
    head_size = 64
    block_size = 16
    num_blocks = 2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)

    key_cache, value_cache, k_scale_cache, v_scale_cache = _allocate_cache_tensors(
        num_blocks, block_size, num_heads, head_size, qcfg
    )

    slot_mapping = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    key_cache_before = key_cache.clone()
    val_cache_before = value_cache.clone()

    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        triton_reshape_and_cache_flash_int4_per_token_head(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
        )
    else:
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
        )

    assert not torch.equal(key_cache[0, 0], key_cache_before[0, 0])
    assert not torch.equal(key_cache[0, 1], key_cache_before[0, 1])
    assert not torch.equal(value_cache[0, 0], val_cache_before[0, 0])

    assert torch.equal(key_cache[0, 2:], key_cache_before[0, 2:])
    assert torch.equal(key_cache[1], key_cache_before[1])
    assert torch.equal(value_cache[0, 2:], val_cache_before[0, 2:])


# ===========================================================================
# 5. process_weights_after_loading -- per-token-head early return
# ===========================================================================
@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["int8_per_token_head", "int4_per_token_head", "fp8_per_token_head"],
)
def test_process_weights_sets_placeholder_scales(kv_cache_dtype: str):
    """Per-token-head should set placeholder scales and delete checkpoint attrs."""
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
# 6. Triton unified_attention -- per-token-head scale cache
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

    query_for_attn = query
    if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        query_for_attn = ops.hadacore_transform(
            query.reshape(-1, head_size).contiguous(), inplace=False
        ).view_as(query)
        key_cache_for_quant = ops.hadacore_transform(
            key_cache_bf16.reshape(-1, head_size).contiguous(), inplace=False
        ).view_as(key_cache_bf16)
        key_cache_q, k_scale_cache = _quantize_per_token_head_ref(
            key_cache_for_quant, qcfg
        )
        value_cache_q, v_scale_cache = _quantize_per_token_head_ref(
            value_cache_bf16, qcfg
        )
        key_cache_deq = _dequantize_per_token_head_ref(
            key_cache_q, k_scale_cache, qcfg, head_size
        )
        value_cache_deq = _dequantize_per_token_head_ref(
            value_cache_q, v_scale_cache, qcfg, head_size
        )
    else:
        k_absmax = key_cache_bf16.float().abs().amax(dim=-1)
        v_absmax = value_cache_bf16.float().abs().amax(dim=-1)
        k_scale_cache = (k_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)
        v_scale_cache = (v_absmax / qcfg.quant_max).clamp(min=1e-6).to(torch.float32)

        scaled_k = key_cache_bf16.float() / k_scale_cache[:, :, :, None]
        scaled_v = value_cache_bf16.float() / v_scale_cache[:, :, :, None]
        if qcfg.uses_trunc:
            key_cache_q = (
                scaled_k.round()
                .clamp(qcfg.quant_min, qcfg.quant_max)
                .to(qcfg.cache_dtype)
            )
            value_cache_q = (
                scaled_v.round()
                .clamp(qcfg.quant_min, qcfg.quant_max)
                .to(qcfg.cache_dtype)
            )
        else:
            key_cache_q = scaled_k.clamp(qcfg.quant_min, qcfg.quant_max).to(
                qcfg.cache_dtype
            )
            value_cache_q = scaled_v.clamp(qcfg.quant_min, qcfg.quant_max).to(
                qcfg.cache_dtype
            )
        key_cache_deq = key_cache_q.float() * k_scale_cache[:, :, :, None]
        value_cache_deq = value_cache_q.float() * v_scale_cache[:, :, :, None]

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
        q=query_for_attn,
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
        q=query_for_attn,
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

    atol = 2e-1 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 5e-2
    rtol = 2e-1 if qcfg.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 5e-2
    torch.testing.assert_close(output_q, output_ref, atol=atol, rtol=rtol)


@torch.inference_mode()
def test_triton_unified_attention_int4_gqa_group_scales_regression():
    """INT4 GQA decode path must honor per-group scale-cache strides."""
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.set_default_device("cuda")
    set_random_seed(0)

    qcfg = INT4_CONFIG
    num_query_heads, num_kv_heads = 32, 8
    head_size = 64
    block_size = 16
    seq_lens = [(1, 16)]
    num_blocks = 64

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16
    )
    key_cache_bf16 = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache_bf16 = torch.randn_like(key_cache_bf16)

    query_for_attn = ops.hadacore_transform(
        query.reshape(-1, head_size).contiguous(), inplace=False
    ).view_as(query)
    key_cache_for_quant = ops.hadacore_transform(
        key_cache_bf16.reshape(-1, head_size).contiguous(), inplace=False
    ).view_as(key_cache_bf16)
    key_cache_q, k_scale_cache = _quantize_per_token_head_ref(key_cache_for_quant, qcfg)
    value_cache_q, v_scale_cache = _quantize_per_token_head_ref(value_cache_bf16, qcfg)
    key_cache_deq = _dequantize_per_token_head_ref(
        key_cache_q, k_scale_cache, qcfg, head_size
    )
    value_cache_deq = _dequantize_per_token_head_ref(
        value_cache_q, v_scale_cache, qcfg, head_size
    )

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    block_tables = torch.randint(0, num_blocks, (len(seq_lens), 1), dtype=torch.int32)

    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (0, num_query_heads, 16, head_size_padded), dtype=torch.float32
    )
    softmax_segm_max = torch.empty((0, num_query_heads, 16), dtype=torch.float32)
    softmax_segm_expsum = torch.empty((0, num_query_heads, 16), dtype=torch.float32)

    output_q = torch.empty_like(query)
    unified_attention(
        q=query_for_attn,
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
        seq_threshold_3D=0,
        num_par_softmax_segments=16,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=qcfg.kv_quant_mode,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    output_ref = torch.empty_like(query)
    unified_attention(
        q=query_for_attn,
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
        seq_threshold_3D=0,
        num_par_softmax_segments=16,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    torch.testing.assert_close(output_q, output_ref, atol=8e-2, rtol=8e-2)
