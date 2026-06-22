# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel microbench: fused MLA TurboQuant decode vs bf16 grouped MLA decode.

Times stage1 (+ stage2 reduce) for both paths over a small (batch, ctx_len)
grid at fixed num_q_heads/L/R, and prints a markdown table. Run on a single
GPU; intended for inclusion in PR #41803 evidence section.

Usage:
    .venv/bin/python benchmarks/kernels/benchmark_mla_turboquant_decode.py
"""

from __future__ import annotations

import argparse
import math

import torch

# Reuse synth helpers from the equivalence test (same package layout).
from tests.kernels.attention.test_mla_turboquant_dequant import (  # noqa: E402
    make_synth_cache,
    torch_ref_dequant,
)
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
    decode_attention_fwd_grouped,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_tq_decode_stage1,
)


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


def bench_one(B, ctx_len, H_q, L, R, mse_bits, page_size, num_kv_splits, device):
    num_pages_per_seq = ctx_len // page_size
    n_active = B * num_pages_per_seq
    sm_scale = 1.0 / math.sqrt(L + R)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    kpe_bytes = 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes

    cache, _, _, _ = make_synth_cache(
        n_active=n_active,
        block_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=False,
    )
    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)
    Pi = _build_hadamard(L, str(device)).to(torch.bfloat16)

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), ctx_len, device=device, dtype=torch.int32)

    torch.manual_seed(0)
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi).view(B, H_q, L)

    # Materialize bf16 cache once for the baseline (cost not counted —
    # baseline pretends an `auto` dtype that already stores bf16).
    kv_full = torch_ref_dequant(
        cache,
        centroids,
        Pi,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=True,
        kpe_fp8=False,
    )  # (n_active, page_size, L+R) bf16

    # ----- TurboQuant fused path -----
    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)

    def run_tq():
        fused_mla_tq_decode_stage1(
            q_rot,
            cache,
            centroids,
            attn_logits,
            req_to_tokens,
            b_seqlen,
            sm_scale=sm_scale,
            page_size=page_size,
            L=L,
            R=R,
            mse_bits=mse_bits,
            mse_bytes=mse_bytes,
            kv_c_bytes=kv_c_bytes,
            norm_correction=True,
            kpe_fp8=False,
            num_kv_splits=num_kv_splits,
        )
        _decode_softmax_reducev_fwd(
            attn_logits,
            q_rot,
            o_fused_rot,
            lse_fused,
            v_holder,
            b_seqlen,
            num_kv_splits,
        )

    # ----- bf16 baseline (decode_attention_fwd_grouped over bf16 cache) -----
    # kv_full: (n_active, page_size, L+R) → add KV-head dim of 1 for MLA layout.
    k_buffer = kv_full.unsqueeze(2)  # (n_active, page_size, 1, L+R)
    v_buffer = k_buffer[..., :L]
    o_ref = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_ref = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    attn_logits_ref = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    one_scale = torch.ones(1, device=device, dtype=torch.float32)

    def run_bf16():
        decode_attention_fwd_grouped(
            q_rot,
            k_buffer,
            v_buffer,
            o_ref,
            lse_ref,
            req_to_tokens,
            b_seqlen,
            attn_logits_ref,
            num_kv_splits,
            sm_scale,
            page_size,
            k_scale=one_scale,
            v_scale=one_scale,
            is_mla=True,
        )

    # Sanity warmup (also surfaces any signature mismatch up front).
    run_tq()
    run_bf16()
    torch.accelerator.synchronize()

    t_tq = _time_ms(run_tq)
    t_bf = _time_ms(run_bf16)
    return t_bf, t_tq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--H-q", type=int, default=128)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--num-kv-splits", type=int, default=4)
    p.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[1024, 4096, 16384])
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda")

    rows = []
    for bits in args.bits:
        for B in args.batches:
            for ctx in args.ctx_lens:
                t_bf, t_tq = bench_one(
                    B=B,
                    ctx_len=ctx,
                    H_q=args.H_q,
                    L=args.L,
                    R=args.R,
                    mse_bits=bits,
                    page_size=args.page_size,
                    num_kv_splits=args.num_kv_splits,
                    device=device,
                )
                speedup = t_bf / t_tq
                rows.append((bits, B, ctx, t_bf, t_tq, speedup))
                print(
                    f"bits={bits} B={B:>2} ctx={ctx:>5} | "
                    f"bf16={t_bf:7.3f}ms  tq={t_tq:7.3f}ms  "
                    f"speedup={speedup:5.2f}x"
                )

    print("\n## Results (markdown)\n")
    print("| bits | B | ctx_len | bf16 ms | TQ ms | TQ vs bf16 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for bits, B, ctx, t_bf, t_tq, sp in rows:
        print(f"| {bits} | {B} | {ctx} | {t_bf:.3f} | {t_tq:.3f} | {sp:.2f}× |")


if __name__ == "__main__":
    main()
