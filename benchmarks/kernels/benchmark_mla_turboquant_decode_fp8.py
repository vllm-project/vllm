# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 baseline microbench — FP8 (turboquant_k8v4) decode workspace path.

Times the current FP8 decode flow:
    cache_fp8 -> dequant to bf16 workspace (kv_c) + write k_pe ->
    decode_attention_fwd_grouped over bf16 workspace.

This is the path that `_forward_mqa_fp8_workspace` runs today. Once K3 has
a fused FP8 kernel, time it against these numbers (same B/ctx grid).

Usage:
    .venv/bin/python benchmarks/kernels/benchmark_mla_turboquant_decode_fp8.py
"""

from __future__ import annotations

import argparse
import math

import torch

from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
    decode_attention_fwd_grouped,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_tq_decode_stage1,
)

_FP8 = torch.float8_e4m3fn
_BF16 = torch.bfloat16


def _time_ms(fn, warmup=10, trials=30):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / trials


def _make_fp8_cache(n_active, page_size, L, R, device):
    """FP8 layout: [L bytes fp8 kv_c | 2*R bytes bf16 k_pe]."""
    kv_c_f = torch.randn(n_active, page_size, L, device=device, dtype=torch.float32)
    kv_c_f = kv_c_f.clamp(-448.0, 448.0).to(_FP8)
    k_pe = torch.randn(n_active, page_size, R, device=device, dtype=_BF16)
    packed = L + 2 * R
    cache = torch.empty(n_active, page_size, packed, device=device, dtype=torch.uint8)
    cache[..., :L] = kv_c_f.view(torch.uint8)
    cache[..., L:] = k_pe.view(torch.uint8).reshape(n_active, page_size, 2 * R)
    k_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    return cache, k_scale


def bench_one(B, ctx_len, H_q, L, R, page_size, num_kv_splits, device):
    num_pages_per_seq = ctx_len // page_size
    n_active = B * num_pages_per_seq
    sm_scale = 1.0 / math.sqrt(L + R)

    cache, k_scale = _make_fp8_cache(n_active, page_size, L, R, device)
    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), ctx_len, device=device, dtype=torch.int32)

    torch.manual_seed(0)
    q = torch.randn(B, H_q, L + R, device=device, dtype=_BF16)

    workspace = torch.empty(n_active, page_size, L + R, device=device, dtype=_BF16)
    o = torch.empty(B, H_q, L, device=device, dtype=_BF16)
    lse = torch.empty(B, H_q, device=device, dtype=_BF16)
    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )

    one_scale = torch.ones(1, device=device, dtype=torch.float32)

    def run_baseline():
        # Stage A: dequant fp8 → bf16 kv_c into workspace[..., :L]
        kv_c_fp8 = cache[..., :L].contiguous().view(_FP8)
        workspace[..., :L] = (kv_c_fp8.to(torch.float32) * k_scale).to(_BF16)
        # Stage B: copy k_pe (bf16 reinterpret) into workspace[..., L:]
        kpe = cache[..., L:].contiguous().view(_BF16).reshape(n_active, page_size, R)
        workspace[..., L:] = kpe
        # Stage C: bf16 grouped decode over workspace
        k_buffer = workspace.unsqueeze(2)  # (n_active, page_size, 1, L+R)
        v_buffer = k_buffer[..., :L]
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_tokens,
            b_seqlen,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            k_scale=one_scale,
            v_scale=one_scale,
            is_mla=True,
        )

    def run_staging():
        kv_c_fp8 = cache[..., :L].contiguous().view(_FP8)
        workspace[..., :L] = (kv_c_fp8.to(torch.float32) * k_scale).to(_BF16)
        kpe = cache[..., L:].contiguous().view(_BF16).reshape(n_active, page_size, R)
        workspace[..., L:] = kpe

    def run_decode_only():
        k_buffer = workspace.unsqueeze(2)
        v_buffer = k_buffer[..., :L]
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_tokens,
            b_seqlen,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            k_scale=one_scale,
            v_scale=one_scale,
            is_mla=True,
        )

    # K3 fused FP8 path: stage1 reads fp8 cache directly, no staging.
    centroids_unused = torch.empty(0, device=device, dtype=_BF16)
    o_fused = torch.empty(B, H_q, L, device=device, dtype=_BF16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=_BF16)
    v_holder = torch.empty((1, L), device=device, dtype=_BF16)
    attn_logits_f = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )

    def run_fused():
        fused_mla_tq_decode_stage1(
            q,
            cache,
            centroids_unused,
            attn_logits_f,
            req_to_tokens,
            b_seqlen,
            sm_scale=sm_scale,
            page_size=page_size,
            L=L,
            R=R,
            mse_bits=0,
            mse_bytes=0,
            kv_c_bytes=L,
            norm_correction=False,
            kpe_fp8=False,
            key_fp8=True,
            k_scale=float(k_scale.item()),
            num_kv_splits=num_kv_splits,
        )
        _decode_softmax_reducev_fwd(
            attn_logits_f,
            q,
            o_fused,
            lse_fused,
            v_holder,
            b_seqlen,
            num_kv_splits,
        )

    run_baseline()
    run_staging()
    run_decode_only()
    run_fused()
    torch.accelerator.synchronize()
    return (
        _time_ms(run_baseline),
        _time_ms(run_staging),
        _time_ms(run_decode_only),
        _time_ms(run_fused),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--H-q", type=int, default=128)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--num-kv-splits", type=int, default=4)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[1024, 4096, 16384])
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda")

    rows = []
    for B in args.batches:
        for ctx in args.ctx_lens:
            t_total, t_stage, t_dec, t_fused = bench_one(
                B=B,
                ctx_len=ctx,
                H_q=args.H_q,
                L=args.L,
                R=args.R,
                page_size=args.page_size,
                num_kv_splits=args.num_kv_splits,
                device=device,
            )
            speedup = t_total / t_fused if t_fused > 0 else 0.0
            rows.append((B, ctx, t_total, t_stage, t_dec, t_fused, speedup))
            print(
                f"FP8  B={B:>2} ctx={ctx:>5} | "
                f"total={t_total:7.3f}ms  staging={t_stage:7.3f}ms  "
                f"decode={t_dec:7.3f}ms  fused={t_fused:7.3f}ms  "
                f"speedup={speedup:.2f}x"
            )

    print("\n## Results (markdown)\n")
    print("| B | ctx_len | baseline ms | staging ms | decode ms | fused ms | speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for B, ctx, t_total, t_stage, t_dec, t_fused, sp in rows:
        print(
            f"| {B} | {ctx} | {t_total:.3f} | {t_stage:.3f} | "
            f"{t_dec:.3f} | {t_fused:.3f} | {sp:.2f}× |"
        )


if __name__ == "__main__":
    main()
