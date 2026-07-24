"""Tests for fused K RoPE + FP8 per-tensor KV cache write kernel."""

import pytest
import torch

from vllm._custom_ops import fused_rope_fp8_kvcache, rotary_embedding
from vllm.platforms import current_platform


def ref_rope_fp8_kvcache(
    key: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,
    key_cache: torch.Tensor,     # uint8 FP8
    value_cache: torch.Tensor,   # uint8 FP8
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor, # [max_pos, rot_dim] float32
    k_scale: float,
    v_scale: float,
    is_neox: bool,
    flash_layout: bool,
) -> None:
    """Reference implementation: unfused RoPE then FP8 quant + cache write."""
    from vllm._custom_ops import rotary_embedding

    key_rope = key.clone()
    # rotary_embedding applies RoPE to key in-place (no value rotation)
    rotary_embedding(positions, key_rope.view(key_rope.size(0), -1),
                     None, key_rope.size(2), cos_sin_cache, is_neox)
    key_rope = key_rope.float()
    value_f = value.float()

    def to_fp8(t: torch.Tensor, scale: float) -> torch.Tensor:
        scaled = t / scale
        # Clamp to fp8 range dynamically (224.0 on AMD FNUZ, 448.0 on NVIDIA/OCP)
        fp8_dtype = current_platform.fp8_dtype()
        if current_platform.is_rocm() and str(fp8_dtype).endswith("fnuz"):
            fp8_max = 224.0
        else:
            fp8_max = 448.0
            
        clamped = scaled.clamp(-fp8_max, fp8_max)
        # Simulate fp8 roundtrip via cast using platform-appropriate dtype
        return clamped.to(fp8_dtype).view(torch.uint8)

    num_tokens = key.size(0)
    num_kv_heads = key.size(1)

    for tok in range(num_tokens):
        slot = slot_mapping[tok].item()
        if slot < 0:
            continue
        block_size = key_cache.size(1) if flash_layout else key_cache.size(2)
        block_idx = slot // block_size
        block_off = slot % block_size

        k_fp8 = to_fp8(key_rope[tok], k_scale)  # [num_kv_heads, head_size]
        v_fp8 = to_fp8(value_f[tok], v_scale)

        for h in range(num_kv_heads):
            if flash_layout:
                key_cache[block_idx, block_off, h, :] = k_fp8[h]
                value_cache[block_idx, block_off, h, :] = v_fp8[h]
            else:
                key_cache[block_idx, h, block_off, :] = k_fp8[h]
                value_cache[block_idx, h, block_off, :] = v_fp8[h]


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("rot_dim", [64])   # rot_dim <= head_size
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("flash_layout", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rope_fp8_kvcache(
    num_tokens, num_kv_heads, head_size, rot_dim, block_size,
    is_neox, flash_layout, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = "cuda"

    num_blocks = (num_tokens + block_size - 1) // block_size + 2
    max_pos = 512
    k_scale = 0.01
    v_scale = 0.02

    key   = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(max_pos, rot_dim, device=device, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    k_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([v_scale], dtype=torch.float32, device=device)

    if flash_layout:
        key_cache   = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                                  dtype=torch.uint8, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                                  dtype=torch.uint8, device=device)
        key_cache_ref   = key_cache.clone()
        value_cache_ref = value_cache.clone()
    else:
        key_cache   = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                                  dtype=torch.uint8, device=device)
        value_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                                  dtype=torch.uint8, device=device)
        key_cache_ref   = key_cache.clone()
        value_cache_ref = value_cache.clone()

    # Run fused kernel
    fused_rope_fp8_kvcache(
        key, value,
        key_cache, value_cache,
        slot_mapping, positions,
        cos_sin_cache,
        k_scale_t, v_scale_t,
        is_neox, flash_layout,
    )

    # Reference
    ref_rope_fp8_kvcache(
        key, value,
        key_cache_ref, value_cache_ref,
        slot_mapping, positions,
        cos_sin_cache,
        k_scale, v_scale,
        is_neox, flash_layout,
    )

    # Compare (allow 1 ULP difference due to fp8 rounding)
    diff_k = (key_cache.int() - key_cache_ref.int()).abs()
    diff_v = (value_cache.int() - value_cache_ref.int()).abs()
    assert diff_k.max().item() <= 1, (
        f"K cache mismatch: max diff = {diff_k.max().item()}"
    )
    assert diff_v.max().item() <= 1, (
        f"V cache mismatch: max diff = {diff_v.max().item()}"
    )

    # Production Fallback (only applicable for flash_layout in V1)
    if flash_layout:
        key_cache_fallback = key_cache.clone().zero_()
        value_cache_fallback = value_cache.clone().zero_()
        
        # 1. Apply RoPE on key
        key_rope = key.clone()
        rotary_embedding(
            positions,
            key_rope.view(key_rope.size(0), -1),
            None,
            head_size,
            cos_sin_cache,
            is_neox,
        )
        
        # 2. Write to cache using production reshape_and_cache_flash
        from vllm._custom_ops import reshape_and_cache_flash
        reshape_and_cache_flash(
            key_rope, value,
            key_cache_fallback, value_cache_fallback,
            slot_mapping, "fp8",
            k_scale_t, v_scale_t,
        )

        # Compare fused vs fallback (allow 1 ULP difference)
        diff_k_fb = (key_cache.int() - key_cache_fallback.int()).abs()
        diff_v_fb = (value_cache.int() - value_cache_fallback.int()).abs()
        assert diff_k_fb.max().item() <= 1, (
            f"K cache fused vs fallback mismatch: "
            f"max diff = {diff_k_fb.max().item()}"
        )
        assert diff_v_fb.max().item() <= 1, (
            f"V cache fused vs fallback mismatch: "
            f"max diff = {diff_v_fb.max().item()}"
        )


@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_rope_fp8_kvcache_padded_slots(is_neox):
    """Verify that slot_mapping=-1 (padded tokens) are correctly skipped."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    num_tokens, num_kv_heads, head_size, block_size = 4, 2, 64, 16
    rot_dim = 64
    num_blocks = 4
    device = "cuda"

    key = torch.randn(
        num_tokens, num_kv_heads, head_size,
        dtype=torch.bfloat16, device=device
    )
    value = torch.randn(
        num_tokens, num_kv_heads, head_size,
        dtype=torch.bfloat16, device=device
    )
    cos_sin_cache = torch.randn(512, rot_dim, device=device, dtype=torch.float32)
    positions     = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    slot_mapping  = torch.tensor([0, -1, 1, -1], dtype=torch.int64, device=device)

    key_cache   = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=torch.uint8, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=torch.uint8, device=device)

    k_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    fused_rope_fp8_kvcache(
        key, value, key_cache, value_cache,
        slot_mapping, positions, cos_sin_cache,
        k_scale, v_scale, is_neox, flash_layout=True,
    )

    # Slots 0 and 1 (block 0, positions 0 and 1) should be written.
    # All other cache blocks must remain zero.
    assert key_cache[0, 0].any(), "slot 0 should be written"
    assert key_cache[0, 1].any(), "slot 1 should be written"
    assert not key_cache[1:].any(), "other blocks should be untouched"
