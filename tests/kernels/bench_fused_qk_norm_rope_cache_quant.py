# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for the fused QK RMSNorm + RoPE + KV Cache + FP8-Quant CUDA kernel.

Measures:
  1. Fused kernel (v2) latency
  2. Baseline: separate torch ops (matmul → RMSNorm → RoPE → cache write)
  3. Both with a preceding QKV GEMM to simulate realistic L2 cache state

Run:
    python tests/kernels/bench_fused_qk_norm_rope_cache_quant.py

Requires a CUDA GPU.
"""

import math
import pathlib
import time

import torch
import torch.nn.functional as F

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
        verbose=False,
    )


# ──────────────────────────────────────────────────────────────────────
# Baseline: separate PyTorch ops (simulates unfused path)
# ──────────────────────────────────────────────────────────────────────


def rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """RMSNorm over the last dim, applied per-head.
    x: [T, num_heads, head_dim], weight: [head_dim]"""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def rotary_embedding_neox(
    q: torch.Tensor,  # [T, num_heads, head_dim]
    k: torch.Tensor,  # [T, num_kv_heads, head_dim]
    cos: torch.Tensor,  # [T, head_dim/2]
    sin: torch.Tensor,  # [T, head_dim/2]
):
    embed_dim = q.shape[-1] // 2
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)

    q_x, q_y = q[..., :embed_dim], q[..., embed_dim:]
    k_x, k_y = k[..., :embed_dim], k[..., embed_dim:]

    q_out = torch.cat([q_x * c - q_y * s, q_y * c + q_x * s], dim=-1)
    k_out = torch.cat([k_x * c - k_y * s, k_y * c + k_x * s], dim=-1)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def baseline_separate_ops(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: float,
    v_scale: float,
    epsilon: float,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_size: int,
    is_fp8: bool,
):
    """Baseline: separate kernel calls matching the unfused code path."""
    T = qkv.shape[0]
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q = qkv[:, :q_size].view(T, num_heads_q, head_dim)
    k = qkv[:, q_size : q_size + kv_size].view(T, num_heads_kv, head_dim)
    v = qkv[:, q_size + kv_size :].view(T, num_heads_kv, head_dim)

    # RMSNorm
    q_normed = rms_norm_per_head(q, q_weight, epsilon)
    k_normed = rms_norm_per_head(k, k_weight, epsilon)

    # Flatten for RoPE
    q_flat = q_normed.view(T, q_size)
    k_flat = k_normed.view(T, kv_size)

    # RoPE
    embed_dim = head_dim // 2
    cos = cos_sin_cache[positions, :embed_dim].float()
    sin = cos_sin_cache[positions, embed_dim:].float()
    q_rope, k_rope = rotary_embedding_neox(
        q_normed.float(), k_normed.float(), cos, sin
    )
    q_out = q_rope.to(qkv.dtype).view(T, q_size)

    # Cache write (element-by-element to simulate reshape_and_cache_flash)
    k_rope_heads = k_rope.to(qkv.dtype)
    cache_dtype = torch.float8_e4m3fn if is_fp8 else qkv.dtype
    for t in range(T):
        slot = slot_mapping[t].item()
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        if is_fp8:
            k_cache[blk, off] = (k_rope_heads[t].float() / k_scale).clamp(
                -448, 448
            ).to(torch.float8_e4m3fn)
            v_cache[blk, off] = (v[t].float() / v_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn
            )
        else:
            k_cache[blk, off] = k_rope_heads[t]
            v_cache[blk, off] = v[t]

    return q_out


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
    ext=None,
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

    # ── Pre-allocate output ──
    q_out = torch.empty(
        num_tokens, q_size, dtype=dtype, device=device
    )

    # ==================================================================
    #  Benchmark 1: GEMM + Fused kernel
    # ==================================================================
    def run_fused():
        # QKV GEMM (simulates preceding linear projection)
        qkv = torch.mm(hidden_states, w_qkv)
        # Fused kernel
        ext.fused_qk_norm_rope_cache_quant(
            q_out, k_cache, v_cache,
            qkv, q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
            k_scale, v_scale, epsilon,
            num_heads_q, num_heads_kv, head_dim, block_size,
            True,  # is_neox
            is_fp8,
        )

    # ==================================================================
    #  Benchmark 2: GEMM + Separate ops (baseline)
    # ==================================================================
    def run_baseline():
        # QKV GEMM
        qkv = torch.mm(hidden_states, w_qkv)
        # Separate ops
        baseline_separate_ops(
            qkv, q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
            k_cache, v_cache, k_scale, v_scale, epsilon,
            num_heads_q, num_heads_kv, head_dim, block_size, is_fp8,
        )

    # ==================================================================
    #  Benchmark 3: Fused kernel only (no GEMM, isolate kernel perf)
    # ==================================================================
    qkv_static = torch.mm(hidden_states, w_qkv)

    def run_fused_only():
        ext.fused_qk_norm_rope_cache_quant(
            q_out, k_cache, v_cache,
            qkv_static, q_weight, k_weight, cos_sin_cache,
            positions, slot_mapping,
            k_scale, v_scale, epsilon,
            num_heads_q, num_heads_kv, head_dim, block_size,
            True,  # is_neox
            is_fp8,
        )

    # ── Timing helper with CUDA events ──
    def timed(fn, warmup_iters, repeat_iters):
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]

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
    t_gemm_fused = timed(run_fused, warmup, repeat)
    t_gemm_baseline = timed(run_baseline, warmup, repeat)
    t_fused_only = timed(run_fused_only, warmup, repeat)

    return t_gemm_fused, t_gemm_baseline, t_fused_only


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        exit(0)

    print("Compiling CUDA extension ...")
    ext = _load_extension()
    print("Done.\n")

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
        print("=" * 80)
        print(f"  {config_name}:  hidden={hidden_size}  "
              f"Q_heads={nq}  KV_heads={nkv}  head_dim={hd}")
        print("=" * 80)
        print(
            f"{'Batch':>6} | "
            f"{'GEMM+Fused':>12} | "
            f"{'GEMM+Baseline':>14} | "
            f"{'Fused Only':>12} | "
            f"{'Speedup':>8} | "
            f"{'Kernel us':>10}"
        )
        print("-" * 80)

        for bs in batch_sizes:
            try:
                t_gf, t_gb, t_fo = bench_one(
                    num_tokens=bs,
                    hidden_size=hidden_size,
                    num_heads_q=nq,
                    num_heads_kv=nkv,
                    head_dim=hd,
                    dtype=torch.bfloat16,
                    is_fp8=False,
                    ext=ext,
                )
                speedup = t_gb / t_gf if t_gf > 0 else float("inf")
                print(
                    f"{bs:>6} | "
                    f"{t_gf * 1000:>10.1f}us | "
                    f"{t_gb * 1000:>12.1f}us | "
                    f"{t_fo * 1000:>10.1f}us | "
                    f"{speedup:>7.2f}x | "
                    f"{t_fo * 1000:>8.1f}us"
                )
            except Exception as e:
                print(f"{bs:>6} | ERROR: {e}")

        print()

    # ── FP8 KV cache benchmark ──
    if cap >= (8, 9):
        print("=" * 80)
        print("  FP8 KV Cache (Qwen3-8B config)")
        print("=" * 80)
        print(
            f"{'Batch':>6} | "
            f"{'GEMM+Fused':>12} | "
            f"{'GEMM+Baseline':>14} | "
            f"{'Fused Only':>12} | "
            f"{'Speedup':>8} | "
            f"{'Kernel us':>10}"
        )
        print("-" * 80)

        for bs in batch_sizes:
            try:
                t_gf, t_gb, t_fo = bench_one(
                    num_tokens=bs,
                    hidden_size=4096,
                    num_heads_q=32,
                    num_heads_kv=8,
                    head_dim=128,
                    dtype=torch.bfloat16,
                    is_fp8=True,
                    k_scale=1.0,
                    v_scale=1.0,
                    ext=ext,
                )
                speedup = t_gb / t_gf if t_gf > 0 else float("inf")
                print(
                    f"{bs:>6} | "
                    f"{t_gf * 1000:>10.1f}us | "
                    f"{t_gb * 1000:>12.1f}us | "
                    f"{t_fo * 1000:>10.1f}us | "
                    f"{speedup:>7.2f}x | "
                    f"{t_fo * 1000:>8.1f}us"
                )
            except Exception as e:
                print(f"{bs:>6} | ERROR: {e}")

        print()

    print("Benchmark complete.")
