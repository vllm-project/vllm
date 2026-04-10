# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness + performance test: fused_qk_norm_rope_cache_quant_v2 vs v1 (v3/v4).

Sweeps token counts 1–4000:
  Part 1 – Correctness: checks outputs match within tolerance
  Part 2 – Performance: CUDA-event latency comparison with speedup ratios

Run:
    python tests/kernels/test_fused_qknrc_v2_vs_v1.py
"""

import sys
import time

import torch

import vllm._custom_ops as ops  # noqa: F401 – registers torch.ops._C

# ── Model params (Qwen3-8B-like) ──
NUM_HEADS_Q = 32
NUM_HEADS_KV = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 2048
MAX_POS = 8192
EPSILON = 1e-6
K_SCALE = 1.0
V_SCALE = 1.0
IS_NEOX = True
DTYPE = torch.bfloat16


def make_cos_sin_cache(max_pos: int, head_dim: int, dtype: torch.dtype):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype).cuda()


def run_v2(
    query, key, value, k_cache, v_cache,
    q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
    is_fp8=False,
):
    torch.ops._C.fused_qk_norm_rope_cache_quant_v2(
        query, key, value, k_cache, v_cache,
        q_weight, k_weight, cos_sin_cache,
        positions, slot_mapping,
        K_SCALE, V_SCALE, EPSILON,
        NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, BLOCK_SIZE,
        IS_NEOX, is_fp8,
    )


def run_v1(
    query, key, value, k_cache, v_cache,
    q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
    is_fp8=False,
):
    torch.ops._C.fused_qk_norm_rope_cache_quant(
        query, key, value, k_cache, v_cache,
        q_weight, k_weight, cos_sin_cache,
        positions, slot_mapping,
        K_SCALE, V_SCALE, EPSILON,
        NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, BLOCK_SIZE,
        IS_NEOX, is_fp8,
    )


def check_one(num_tokens: int, is_fp8: bool = False, verbose: bool = False):
    """Run both kernels with identical inputs, return max diffs.

    Comparison points:
      - Q output: both kernels write normed+roped Q back in-place
      - k_cache:  both kernels write normed+roped K to the paged cache
      - v_cache:  both kernels write V (± FP8 quant) to the paged cache

    Note: v1 does NOT write normed+roped K back to the qkv buffer (only to
    cache), so we compare k_cache contents rather than the K tensor directly.
    """
    q_size = NUM_HEADS_Q * HEAD_DIM
    kv_size = NUM_HEADS_KV * HEAD_DIM

    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda").abs() + 0.1
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda").abs() + 0.1
    cos_sin_cache = make_cos_sin_cache(MAX_POS, HEAD_DIM, DTYPE)
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.long, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    # v1 and v2 both use data_ptr<uint8_t>() when is_fp8, so cache must be uint8
    cache_dtype = torch.uint8 if is_fp8 else DTYPE
    k_cache_v2 = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS_KV, HEAD_DIM,
                             dtype=cache_dtype, device="cuda")
    v_cache_v2 = torch.zeros_like(k_cache_v2)
    k_cache_v1 = torch.zeros_like(k_cache_v2)
    v_cache_v1 = torch.zeros_like(k_cache_v2)

    # ── Shared source data ──
    qkv_data = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=DTYPE, device="cuda")

    # ── v2 path: split into separate q, k, v (contiguous copies) ──
    qkv_for_v2 = qkv_data.clone()
    q_v2 = qkv_for_v2[:, :q_size].reshape(num_tokens, NUM_HEADS_Q, HEAD_DIM).contiguous()
    k_v2 = qkv_for_v2[:, q_size:q_size + kv_size].reshape(num_tokens, NUM_HEADS_KV, HEAD_DIM).contiguous()
    v_v2 = qkv_for_v2[:, q_size + kv_size:].reshape(num_tokens, NUM_HEADS_KV, HEAD_DIM).contiguous()

    run_v2(q_v2, k_v2, v_v2, k_cache_v2, v_cache_v2,
           q_weight, k_weight, cos_sin_cache, positions, slot_mapping, is_fp8)

    # ── v1 path: separate q/k/v ──
    qkv_for_v1 = qkv_data.clone()
    q_v1 = qkv_for_v1[:, :q_size].reshape(num_tokens, NUM_HEADS_Q, HEAD_DIM).contiguous()
    k_v1_in = qkv_for_v1[:, q_size:q_size + kv_size].reshape(num_tokens, NUM_HEADS_KV, HEAD_DIM).contiguous()
    v_v1_in = qkv_for_v1[:, q_size + kv_size:].reshape(num_tokens, NUM_HEADS_KV, HEAD_DIM).contiguous()
    run_v1(q_v1, k_v1_in, v_v1_in, k_cache_v1, v_cache_v1,
           q_weight, k_weight, cos_sin_cache, positions, slot_mapping, is_fp8)

    torch.cuda.synchronize()

    # ── Compare Q outputs ──
    q_diff = (q_v2.float() - q_v1.float()).abs().max().item()

    # ── Compare cache contents (the authoritative K/V outputs) ──
    # Only check the slots that were actually written (0 .. num_tokens-1)
    # slot i maps to block i//BLOCK_SIZE, offset i%BLOCK_SIZE
    kc_v2_used = k_cache_v2.reshape(-1, NUM_HEADS_KV, HEAD_DIM)[:num_tokens]
    kc_v1_used = k_cache_v1.reshape(-1, NUM_HEADS_KV, HEAD_DIM)[:num_tokens]
    vc_v2_used = v_cache_v2.reshape(-1, NUM_HEADS_KV, HEAD_DIM)[:num_tokens]
    vc_v1_used = v_cache_v1.reshape(-1, NUM_HEADS_KV, HEAD_DIM)[:num_tokens]

    if is_fp8:
        # FP8 E4M3: 0x00 (+0) and 0x80 (-0) are semantically identical.
        # Mask out the sign-bit-only-zero case before comparing raw bytes.
        def fp8_abs_diff(a: torch.Tensor, b: torch.Tensor) -> int:
            ai = a.to(torch.int32)
            bi = b.to(torch.int32)
            # Treat -0 (0x80) as +0 (0x00)
            ai = torch.where(ai == 0x80, torch.zeros_like(ai), ai)
            bi = torch.where(bi == 0x80, torch.zeros_like(bi), bi)
            return (ai - bi).abs().max().item()
        kc_diff = fp8_abs_diff(kc_v2_used, kc_v1_used)
        vc_diff = fp8_abs_diff(vc_v2_used, vc_v1_used)
    else:
        kc_diff = (kc_v2_used.float() - kc_v1_used.float()).abs().max().item()
        vc_diff = (vc_v2_used.float() - vc_v1_used.float()).abs().max().item()

    if verbose:
        print(f"  T={num_tokens:>4}  q_diff={q_diff:.6f}  "
              f"kc_diff={kc_diff:.6f}  vc_diff={vc_diff:.6f}")

    return q_diff, kc_diff, vc_diff


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(0)

    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {gpu} (SM {cap[0]}.{cap[1]})")
    print(f"Config: Q_heads={NUM_HEADS_Q}  KV_heads={NUM_HEADS_KV}  "
          f"head_dim={HEAD_DIM}  dtype={DTYPE}  is_neox={IS_NEOX}")
    print()

    # bf16 tolerance: v2 uses smem block-reduce, v1 uses warp-reduce.
    # Both accumulate in float32 but write back bf16, so up to ~2 ULP
    # difference is expected (bf16 epsilon near 1.0 ≈ 0.0078, near 10.0 ≈ 0.0625).
    ATOL = 0.04
    FP8_ATOL = 2  # FP8 uint8 raw-byte comparison (allow ±1 bit rounding)
    RTOL_MSG = f"(atol={ATOL})"

    # ── Token counts: dense for 1–64, then sparser up to 4000 ──
    token_counts = list(range(1, 65))                    # 1..64
    token_counts += list(range(65, 257, 4))              # 65..256 step 4
    token_counts += list(range(260, 1025, 8))            # 260..1024 step 8
    token_counts += list(range(1032, 2049, 16))          # 1032..2048 step 16
    token_counts += list(range(2064, 4001, 32))          # 2064..4000 step 32
    token_counts = sorted(set(token_counts))

    def run_sweep(label, is_fp8, atol):
        print(f"=== {label}  (atol={atol}) ===")
        print(f"Testing {len(token_counts)} token counts in "
              f"[1, {token_counts[-1]}] ...")
        t0 = time.time()
        worst = [0.0, 0.0, 0.0]
        fail_count = 0

        for i, T in enumerate(token_counts):
            q_d, kc_d, vc_d = check_one(T, is_fp8=is_fp8, verbose=False)
            worst = [max(w, d) for w, d in zip(worst, [q_d, kc_d, vc_d])]
            ok = all(d <= atol for d in [q_d, kc_d, vc_d])
            if not ok:
                fail_count += 1
                print(f"  FAIL T={T:>4}  q={q_d:.6f}  "
                      f"kc={kc_d:.6f}  vc={vc_d:.6f}")
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(token_counts)} done")

        elapsed = time.time() - t0
        print(f"  Worst diffs: q={worst[0]:.6f}  "
              f"kc={worst[1]:.6f}  vc={worst[2]:.6f}")
        print(f"  Failures: {fail_count}/{len(token_counts)}  ({elapsed:.1f}s)")
        if fail_count == 0:
            print("  PASSED")
        else:
            print("  FAILED")
        print()

    run_sweep("BF16 KV cache", is_fp8=False, atol=ATOL)

    if cap >= (8, 9):
        run_sweep("FP8 KV cache", is_fp8=True, atol=FP8_ATOL)

    print("Correctness tests done.\n")

    # ==================================================================
    #  Part 2: Performance benchmark
    # ==================================================================
    bench(is_fp8=False)
    if cap >= (8, 9):
        bench(is_fp8=True)

    print("All done.")


# ──────────────────────────────────────────────────────────────────────
# Performance benchmark
# ──────────────────────────────────────────────────────────────────────

def timed(fn, warmup: int = 20, repeat: int = 100) -> float:
    """Return median kernel time in milliseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    return sum(trimmed) / len(trimmed)


def bench(is_fp8: bool = False):
    q_size = NUM_HEADS_Q * HEAD_DIM
    kv_size = NUM_HEADS_KV * HEAD_DIM

    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda").abs() + 0.1
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda").abs() + 0.1
    cos_sin_cache = make_cos_sin_cache(MAX_POS, HEAD_DIM, DTYPE)

    cache_dtype = torch.uint8 if is_fp8 else DTYPE
    k_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS_KV, HEAD_DIM,
                          dtype=cache_dtype, device="cuda")
    v_cache = torch.zeros_like(k_cache)

    label = "FP8" if is_fp8 else "BF16"
    print("=" * 80)
    print(f"  Performance: v1 (warp/head 2D) vs v2 (block/token smem)  "
          f"[{label} cache]")
    print("=" * 80)
    hdr = (f"{'Tokens':>7} | {'v1 (us)':>10} | {'v2 (us)':>10} | "
           f"{'v1/v2':>8} | {'winner':>6}")
    print(hdr)
    print("-" * 80)

    # Token counts for perf: powers of 2 + key points
    perf_tokens = sorted(set(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768,
         1024, 1536, 2048, 2560, 3072, 3584, 4000]
    ))

    for T in perf_tokens:
        positions = torch.randint(0, MAX_POS, (T,), dtype=torch.long,
                                  device="cuda")
        slot_mapping = torch.arange(T, dtype=torch.long, device="cuda")

        qkv = torch.randn(T, q_size + 2 * kv_size, dtype=DTYPE, device="cuda")

        # ── v1 closure (separate q/k/v, in-place) ──
        q_v1p = qkv[:, :q_size].reshape(T, NUM_HEADS_Q, HEAD_DIM).contiguous()
        k_v1p = qkv[:, q_size:q_size + kv_size].reshape(
            T, NUM_HEADS_KV, HEAD_DIM).contiguous()
        v_v1p = qkv[:, q_size + kv_size:].reshape(
            T, NUM_HEADS_KV, HEAD_DIM).contiguous()

        def fn_v1():
            torch.ops._C.fused_qk_norm_rope_cache_quant(
                q_v1p, k_v1p, v_v1p, k_cache, v_cache,
                q_weight, k_weight, cos_sin_cache,
                positions, slot_mapping,
                K_SCALE, V_SCALE, EPSILON,
                NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, BLOCK_SIZE,
                IS_NEOX, is_fp8,
            )

        # ── v2 closure (separate q/k/v, in-place) ──
        q_t = qkv[:, :q_size].reshape(T, NUM_HEADS_Q, HEAD_DIM).contiguous()
        k_t = qkv[:, q_size:q_size + kv_size].reshape(
            T, NUM_HEADS_KV, HEAD_DIM).contiguous()
        v_t = qkv[:, q_size + kv_size:].reshape(
            T, NUM_HEADS_KV, HEAD_DIM).contiguous()

        def fn_v2():
            torch.ops._C.fused_qk_norm_rope_cache_quant_v2(
                q_t, k_t, v_t, k_cache, v_cache,
                q_weight, k_weight, cos_sin_cache,
                positions, slot_mapping,
                K_SCALE, V_SCALE, EPSILON,
                NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, BLOCK_SIZE,
                IS_NEOX, is_fp8,
            )

        t_v1 = timed(fn_v1)
        t_v2 = timed(fn_v2)
        ratio = t_v1 / t_v2 if t_v2 > 0 else float("inf")
        winner = "v1" if t_v1 < t_v2 else "v2"

        print(f"{T:>7} | {t_v1 * 1000:>8.1f}us | {t_v2 * 1000:>8.1f}us | "
              f"{ratio:>7.2f}x | {winner:>6}")

    print()


if __name__ == "__main__":
    main()
