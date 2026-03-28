# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for the fused QK RMSNorm + RoPE + KV Cache + FP8-Quant CUDA kernel.

Run:
    python tests/kernels/test_fused_qk_norm_rope_cache_quant.py

The test JIT-compiles the .cu file and validates against a pure-PyTorch
reference implementation.
"""

import math
import os
import pathlib

import torch

# ──────────────────────────────────────────────────────────────────────
# JIT compile the CUDA extension
# ──────────────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).resolve().parent
_CSRC = _HERE.parent.parent / "csrc" / "fused_qk_norm_rope_cache_quant.cu"

assert _CSRC.exists(), f"CUDA source not found: {_CSRC}"


def _load_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="fused_qknrc",
        sources=[str(_CSRC)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Reference implementation (pure PyTorch)
# ──────────────────────────────────────────────────────────────────────


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """RMSNorm over the last dimension."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def rotary_embedding_neox_ref(
    q: torch.Tensor,  # [T, num_heads, head_dim]
    k: torch.Tensor,  # [T, num_kv_heads, head_dim]
    cos: torch.Tensor,  # [T, head_dim/2]
    sin: torch.Tensor,  # [T, head_dim/2]
):
    """Apply GPT-NeoX-style rotary embeddings."""
    embed_dim = q.shape[-1] // 2
    q_x, q_y = q[..., :embed_dim], q[..., embed_dim:]
    k_x, k_y = k[..., :embed_dim], k[..., embed_dim:]

    # Expand cos/sin for heads: [T, 1, embed_dim]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)

    q_out = torch.cat([q_x * c - q_y * s, q_y * c + q_x * s], dim=-1)
    k_out = torch.cat([k_x * c - k_y * s, k_y * c + k_x * s], dim=-1)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def reference_fused_qk_norm_rope_cache_quant(
    qkv: torch.Tensor,  # [T, (nq+2*nkv)*hd]
    q_weight: torch.Tensor,  # [head_dim]
    k_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rot_dim]
    positions: torch.Tensor,  # [T]
    slot_mapping: torch.Tensor,  # [T]
    k_cache: torch.Tensor,  # [num_blocks, block_size, nkv, hd]
    v_cache: torch.Tensor,  # same
    k_scale: float,
    v_scale: float,
    epsilon: float,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_size: int,
    is_fp8: bool,
):
    """Pure-PyTorch reference. Returns q_out and fills k_cache, v_cache."""
    T = qkv.shape[0]
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q_flat = qkv[:, :q_size]
    k_flat = qkv[:, q_size : q_size + kv_size]
    v_flat = qkv[:, q_size + kv_size :]

    # Reshape to per-head
    q_heads = q_flat.view(T, num_heads_q, head_dim)
    k_heads = k_flat.view(T, num_heads_kv, head_dim)
    v_heads = v_flat.view(T, num_heads_kv, head_dim)

    # RMSNorm per head
    q_normed = rms_norm_ref(q_heads.float(), q_weight.float(), epsilon).to(
        qkv.dtype
    )
    k_normed = rms_norm_ref(k_heads.float(), k_weight.float(), epsilon).to(
        qkv.dtype
    )

    # Gather cos/sin for each token's position
    embed_dim = head_dim // 2
    cos = cos_sin_cache[positions, :embed_dim].float()  # [T, embed_dim]
    sin = cos_sin_cache[positions, embed_dim:].float()  # [T, embed_dim]

    # RoPE
    q_rope, k_rope = rotary_embedding_neox_ref(
        q_normed.float(), k_normed.float(), cos, sin
    )
    q_rope = q_rope.to(qkv.dtype)
    k_rope = k_rope.to(qkv.dtype)

    # Q output: flatten back
    q_out = q_rope.view(T, q_size)

    # Write K and V to paged cache
    for t in range(T):
        slot = slot_mapping[t].item()
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size

        if is_fp8:
            # Quantise: fp8_val = float_val / scale
            k_quant = (k_rope[t].float() / k_scale).clamp(-448, 448)
            v_quant = (v_heads[t].float() / v_scale).clamp(-448, 448)
            k_cache[blk, off] = k_quant.to(torch.float8_e4m3fn)
            v_cache[blk, off] = v_quant.to(torch.float8_e4m3fn)
        else:
            k_cache[blk, off] = k_rope[t]
            v_cache[blk, off] = v_heads[t]

    return q_out


# ──────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────


def _make_cos_sin_cache(max_pos: int, head_dim: int, dtype: torch.dtype):
    """Create a cos/sin cache like vLLM's RotaryEmbedding."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    # vLLM layout: [max_pos, rot_dim] where first half = cos, second = sin
    return torch.cat([cos, sin], dim=-1).to(dtype).cuda()


def run_test(
    num_tokens: int = 8,
    num_heads_q: int = 32,
    num_heads_kv: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    num_blocks: int = 64,
    max_pos: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    is_fp8: bool = False,
    epsilon: float = 1e-6,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    ext=None,
):
    torch.manual_seed(42)
    device = "cuda"

    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    total_size = q_size + 2 * kv_size

    # Inputs
    qkv = torch.randn(num_tokens, total_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.1
    k_weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.1
    cos_sin_cache = _make_cos_sin_cache(max_pos, head_dim, dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.long, device=device)

    # Slot mapping: assign each token to a unique slot
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Allocate KV caches
    cache_dtype = torch.float8_e4m3fn if is_fp8 else dtype
    k_cache_ref = torch.zeros(
        num_blocks, block_size, num_heads_kv, head_dim,
        dtype=cache_dtype, device=device,
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)
    k_cache_test = torch.zeros_like(k_cache_ref)
    v_cache_test = torch.zeros_like(v_cache_ref)

    # ── Reference ──
    q_out_ref = reference_fused_qk_norm_rope_cache_quant(
        qkv, q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
        k_cache_ref, v_cache_ref, k_scale, v_scale, epsilon,
        num_heads_q, num_heads_kv, head_dim, block_size, is_fp8,
    )

    # ── CUDA kernel ──
    q_out_test = torch.empty_like(q_out_ref)
    ext.fused_qk_norm_rope_cache_quant(
        q_out_test, k_cache_test, v_cache_test,
        qkv, q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
        k_scale, v_scale, epsilon,
        num_heads_q, num_heads_kv, head_dim, block_size,
        True,  # is_neox
        is_fp8,
    )

    # ── Compare ──
    # Q output: should be very close (BF16 vs float ref → minor rounding)
    q_atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    q_close = torch.allclose(q_out_test, q_out_ref, atol=q_atol, rtol=1e-2)
    q_max_diff = (q_out_test.float() - q_out_ref.float()).abs().max().item()

    # KV cache
    if is_fp8:
        # FP8 has limited precision; compare as float
        k_test_f = k_cache_test.float()
        k_ref_f = k_cache_ref.float()
        v_test_f = v_cache_test.float()
        v_ref_f = v_cache_ref.float()
        kv_atol = 2.0  # FP8 E4M3 has ~1 ULP error around typical values
    else:
        k_test_f = k_cache_test.float()
        k_ref_f = k_cache_ref.float()
        v_test_f = v_cache_test.float()
        v_ref_f = v_cache_ref.float()
        kv_atol = q_atol

    k_close = torch.allclose(k_test_f, k_ref_f, atol=kv_atol, rtol=0.1)
    v_close = torch.allclose(v_test_f, v_ref_f, atol=kv_atol, rtol=0.1)
    k_max_diff = (k_test_f - k_ref_f).abs().max().item()
    v_max_diff = (v_test_f - v_ref_f).abs().max().item()

    tag = f"T={num_tokens} nq={num_heads_q} nkv={num_heads_kv} " \
          f"hd={head_dim} {'fp8' if is_fp8 else dtype}"
    status = "PASS" if (q_close and k_close and v_close) else "FAIL"
    print(f"[{status}] {tag}")
    print(f"  Q max diff: {q_max_diff:.6f}  (atol={q_atol})")
    print(f"  K max diff: {k_max_diff:.6f}  (atol={kv_atol})")
    print(f"  V max diff: {v_max_diff:.6f}  (atol={kv_atol})")

    if status == "FAIL":
        if not q_close:
            print("  >>> Q MISMATCH")
        if not k_close:
            print("  >>> K CACHE MISMATCH")
        if not v_close:
            print("  >>> V CACHE MISMATCH")

    return status == "PASS"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        exit(0)

    print("Compiling CUDA extension …")
    ext = _load_extension()
    print("Compilation done.\n")

    all_pass = True

    # ── BF16, no FP8 cache ──
    print("=" * 60)
    print("Test suite: BF16 model dtype, auto KV cache")
    print("=" * 60)
    for T in [1, 4, 16, 64]:
        ok = run_test(num_tokens=T, dtype=torch.bfloat16, is_fp8=False, ext=ext)
        all_pass &= ok

    # Different head configs
    print()
    ok = run_test(num_tokens=8, num_heads_q=8, num_heads_kv=2,
                  head_dim=64, dtype=torch.bfloat16, is_fp8=False, ext=ext)
    all_pass &= ok

    ok = run_test(num_tokens=8, num_heads_q=64, num_heads_kv=8,
                  head_dim=128, dtype=torch.bfloat16, is_fp8=False, ext=ext)
    all_pass &= ok

    # ── FP8 KV cache ──
    cap = torch.cuda.get_device_capability()
    if cap >= (8, 9):
        print()
        print("=" * 60)
        print("Test suite: BF16 model dtype, FP8 E4M3 KV cache")
        print("=" * 60)
        for T in [1, 4, 16]:
            ok = run_test(num_tokens=T, dtype=torch.bfloat16, is_fp8=True,
                          k_scale=1.0, v_scale=1.0, ext=ext)
            all_pass &= ok

        # Non-trivial scales
        ok = run_test(num_tokens=8, dtype=torch.bfloat16, is_fp8=True,
                      k_scale=0.5, v_scale=2.0, ext=ext)
        all_pass &= ok
    else:
        print(f"\nSkipping FP8 tests (GPU SM {cap[0]}.{cap[1]} < 8.9)")

    # ── FP16 ──
    print()
    print("=" * 60)
    print("Test suite: FP16 model dtype, auto KV cache")
    print("=" * 60)
    ok = run_test(num_tokens=8, dtype=torch.float16, is_fp8=False, ext=ext)
    all_pass &= ok

    print()
    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    exit(0 if all_pass else 1)
