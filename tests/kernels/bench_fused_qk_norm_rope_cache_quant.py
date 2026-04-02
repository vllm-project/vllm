# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for the fused QK RMSNorm + RoPE + KV Cache + FP8-Quant CUDA kernel.

Measures (kernel-only, no GEMM):
  1. v3 (warp-per-head): register-based, warp shuffle RMSNorm, vectorized I/O
  2. v2 (naive fusion):  block-per-token, smem RMSNorm, sequential heads
  3. Baseline: existing vLLM CUDA kernels chained (2 launches)
       fused_qk_norm_rope → reshape_and_cache_flash

Run (requires vllm to be installed):
    python tests/kernels/bench_fused_qk_norm_rope_cache_quant.py

Requires a CUDA GPU.
"""

import math

import torch

import vllm._custom_ops as ops  # registers torch.ops._C / _C_cache_ops


# ──────────────────────────────────────────────────────────────────────
# Baseline: existing vLLM CUDA kernels
#   Step 1: fused_qk_norm_rope  (QK RMSNorm + RoPE, 1 warp/head, in-place)
#   Step 2: reshape_and_cache_flash  (paged KV cache write ± FP8 quant)
# ──────────────────────────────────────────────────────────────────────


def baseline_cuda_ops(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_t: torch.Tensor,
    v_scale_t: torch.Tensor,
    epsilon: float,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_fp8: bool,
    k_buf: torch.Tensor | None = None,
    v_buf: torch.Tensor | None = None,
):
    """Baseline: chain two existing vLLM CUDA kernels.

    k_buf / v_buf are pre-allocated contiguous [T, nkv, hd] buffers.
    fused_qk_norm_rope works on packed QKV in-place; after it returns
    the K region is normed+roped and V is unchanged.  We copy them into
    the contiguous buffers before calling reshape_and_cache_flash.
    """
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    # ── Step 1: fused QK-Norm + RoPE (in-place) ──
    ops.fused_qk_norm_rope(
        qkv, num_heads_q, num_heads_kv, num_heads_kv,
        head_dim, epsilon, q_weight, k_weight,
        cos_sin_cache, True, positions,
    )

    # ── Step 2: copy K, V into contiguous buffers → cache write ──
    T = qkv.shape[0]
    k_buf.copy_(qkv[:, q_size:q_size + kv_size].view(T, num_heads_kv, head_dim))
    v_buf.copy_(qkv[:, q_size + kv_size:].view(T, num_heads_kv, head_dim))

    kv_cache_dtype = "fp8_e4m3" if is_fp8 else "auto"
    ops.reshape_and_cache_flash(
        k_buf, v_buf, k_cache, v_cache, slot_mapping,
        kv_cache_dtype, k_scale_t, v_scale_t,
    )


# ──────────────────────────────────────────────────────────────────────
# Benchmark harness
# ──────────────────────────────────────────────────────────────────────


def _make_cos_sin_cache(max_pos: int, head_dim: int, dtype: torch.dtype):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1).to(dtype).cuda()


def bench_one(
    num_tokens: int,
    hidden_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_size: int = 16,
    num_blocks: int = 2048,
    max_pos: int = 8192,
    dtype: torch.dtype = torch.bfloat16,
    is_fp8: bool = False,
    epsilon: float = 1e-6,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    warmup: int = 20,
    repeat: int = 100,
):
    device = "cuda"

    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    total_qkv_size = q_size + 2 * kv_size

    # ── Mock QKV projection weight (hidden_size → total_qkv_size) ──
    w_qkv = torch.randn(
        hidden_size, total_qkv_size, dtype=dtype, device=device
    ) * (1.0 / math.sqrt(hidden_size))

    # ── Inputs ──
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    q_weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.1
    k_weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.1
    cos_sin_cache = _make_cos_sin_cache(max_pos, head_dim, dtype)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device=device
    )
    slot_mapping = torch.arange(
        num_tokens, dtype=torch.long, device=device
    )

    cache_dtype = torch.float8_e4m3fn if is_fp8 else dtype
    k_cache = torch.zeros(
        num_blocks, block_size, num_heads_kv, head_dim,
        dtype=cache_dtype, device=device,
    )
    v_cache = torch.zeros_like(k_cache)

    # Scale tensors for reshape_and_cache_flash (expects Tensor, not float)
    k_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([v_scale], dtype=torch.float32, device=device)

    # Contiguous K, V buffers for baseline's reshape_and_cache_flash
    k_buf = torch.empty(
        num_tokens, num_heads_kv, head_dim, dtype=dtype, device=device
    )
    v_buf = torch.empty_like(k_buf)

    # Pre-allocate q_out for v2 (naive fusion needs separate output)
    q_out_v2 = torch.empty(
        num_tokens, q_size, dtype=dtype, device=device
    )

    # ==================================================================
    #  v3 (warp-per-head): Q in-place in qkv
    # ==================================================================
    def run_v3_only():
        torch.ops._C.fused_qk_norm_rope_cache_quant(
            qkv_static, k_cache, v_cache,
            q_weight, k_weight, cos_sin_cache,
            positions, slot_mapping,
            k_scale, v_scale, epsilon,
            num_heads_q, num_heads_kv, head_dim, block_size,
            True, is_fp8,
        )

    # ==================================================================
    #  v2 (naive fusion): block-per-token, smem reduce
    # ==================================================================
    def run_v2_only():
        torch.ops._C.fused_qk_norm_rope_cache_quant_v2(
            q_out_v2, k_cache, v_cache,
            qkv_static, q_weight, k_weight, cos_sin_cache,
            positions, slot_mapping,
            k_scale, v_scale, epsilon,
            num_heads_q, num_heads_kv, head_dim, block_size,
            True, is_fp8,
        )

    # ==================================================================
    #  Baseline: existing CUDA kernels (2 launches)
    #  fused_qk_norm_rope is in-place, so clone qkv each iteration.
    # ==================================================================
    qkv_static = torch.mm(hidden_states, w_qkv)

    def run_baseline_only():
        qkv_copy = qkv_static.clone()
        baseline_cuda_ops(
            qkv_copy, q_weight, k_weight, cos_sin_cache, positions,
            slot_mapping, k_cache, v_cache, k_scale_t, v_scale_t,
            epsilon, num_heads_q, num_heads_kv, head_dim, is_fp8,
            k_buf, v_buf,
        )

    # ── Timing helper with CUDA events ──
    def timed(fn, warmup_iters, repeat_iters):
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)
        ]

        for i in range(repeat_iters):
            start_events[i].record()
            fn()
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times.sort()
        # Trim outliers: use middle 80%
        trim = max(1, len(times) // 10)
        trimmed = times[trim:-trim] if trim < len(times) // 2 else times
        return sum(trimmed) / len(trimmed)  # ms

    # Run benchmarks
    t_v3 = timed(run_v3_only, warmup, repeat)
    t_v2 = timed(run_v2_only, warmup, repeat)
    t_baseline = timed(run_baseline_only, warmup, repeat)

    return t_v3, t_v2, t_baseline


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

_HEADER = (
    f"{'Batch':>6} | "
    f"{'v3 warp/hd':>12} | "
    f"{'v2 naive':>12} | "
    f"{'Baseline':>12} | "
    f"{'v3/Base':>10} | "
    f"{'v3/v2':>10} | "
    f"{'v2/Base':>10}"
)


def _print_row(bs, t_v3, t_v2, t_base):
    v3_vs_base = t_base / t_v3 if t_v3 > 0 else float("inf")
    v3_vs_v2 = t_v2 / t_v3 if t_v3 > 0 else float("inf")
    v2_vs_base = t_base / t_v2 if t_v2 > 0 else float("inf")
    print(
        f"{bs:>6} | "
        f"{t_v3 * 1000:>10.1f}us | "
        f"{t_v2 * 1000:>10.1f}us | "
        f"{t_base * 1000:>10.1f}us | "
        f"{v3_vs_base:>9.2f}x | "
        f"{v3_vs_v2:>9.2f}x | "
        f"{v2_vs_base:>9.2f}x"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {gpu_name} (SM {cap[0]}.{cap[1]})")
    print()

    # ── Model configs to benchmark ──
    # (name, hidden_size, num_heads_q, num_heads_kv, head_dim)
    configs = [
        ("Qwen3-8B",    4096, 32,  8, 128),
        ("Qwen3-32B",   5120, 40,  8, 128),
        ("Qwen3-72B",   8192, 64,  8, 128),
    ]

    # Batch sizes: from tiny decode to large prefill
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for config_name, hidden_size, nq, nkv, hd in configs:
        print("=" * 90)
        print(f"  {config_name}:  hidden={hidden_size}  "
              f"Q_heads={nq}  KV_heads={nkv}  head_dim={hd}")
        print("=" * 90)
        print(_HEADER)
        print("-" * 90)

        for bs in batch_sizes:
            try:
                t_v3, t_v2, t_base = bench_one(
                    num_tokens=bs,
                    hidden_size=hidden_size,
                    num_heads_q=nq,
                    num_heads_kv=nkv,
                    head_dim=hd,
                    dtype=torch.bfloat16,
                    is_fp8=False,
                )
                _print_row(bs, t_v3, t_v2, t_base)
            except Exception as e:
                print(f"{bs:>6} | ERROR: {e}")

        print()

    # ── FP8 KV cache benchmark ──
    if cap >= (8, 9):
        print("=" * 90)
        print("  FP8 KV Cache (Qwen3-8B config)")
        print("=" * 90)
        print(_HEADER)
        print("-" * 90)

        for bs in batch_sizes:
            try:
                t_v3, t_v2, t_base = bench_one(
                    num_tokens=bs,
                    hidden_size=4096,
                    num_heads_q=32,
                    num_heads_kv=8,
                    head_dim=128,
                    dtype=torch.bfloat16,
                    is_fp8=True,
                    k_scale=1.0,
                    v_scale=1.0,
                )
                _print_row(bs, t_v3, t_v2, t_base)
            except Exception as e:
                print(f"{bs:>6} | ERROR: {e}")

        print()

    print("Benchmark complete.")
