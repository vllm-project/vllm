#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TokenSpeed native extend against AITER unified attention.

This is the isolated kernel-author benchmark for the native MHA extend kernel.
It uses request-level extend metadata for both paths:

* TokenSpeed native extend: rocm_tokenspeed_mha_extend
* AITER unified attention: unified_attention

For sliding-window cases, --sliding-window is the vLLM semantic window. The
TokenSpeed kernel is called with sliding_window - 1, matching AITER's
window_size=(sliding_window - 1, 0).

The script does not modify tokenspeed_kernel_amd.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch
from aiter.ops.triton.unified_attention import unified_attention

from vllm.v1.attention.ops.rocm_tokenspeed_mha import rocm_tokenspeed_mha_extend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--query-len", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-sinks", action="store_true")
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    return parser.parse_args()


def tokenspeed_kernel_window(sliding_window: int) -> int:
    return -1 if sliding_window < 0 else sliding_window - 1


def aiter_window(sliding_window: int) -> tuple[int, int]:
    return (-1, -1) if sliding_window < 0 else (sliding_window - 1, 0)


def bench_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a ROCm/CUDA-visible GPU.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    total_q = args.requests * args.query_len
    pages_per_req = math.ceil(args.seq_len / args.block_size)
    ts_window = tokenspeed_kernel_window(args.sliding_window)
    fa_window = aiter_window(args.sliding_window)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    sinks = (
        None
        if args.no_sinks
        else torch.randn(args.heads, device=device, dtype=torch.float32)
    )
    query = torch.randn(total_q, args.heads, args.head_dim, device=device, dtype=dtype)
    key_cache = torch.randn(
        args.requests * pages_per_req,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(
        args.requests * pages_per_req, device=device, dtype=torch.int32
    ).reshape(args.requests, pages_per_req)
    query_start_loc_cpu = torch.arange(
        0,
        total_q + args.query_len,
        args.query_len,
        dtype=torch.int32,
    )
    query_start_loc = query_start_loc_cpu.to(device)
    final_seq_lens = torch.full(
        (args.requests,), args.seq_len, device=device, dtype=torch.int32
    )

    tokenspeed_output = torch.empty_like(query)
    aiter_output = torch.empty_like(query)

    def tokenspeed_native_extend() -> torch.Tensor:
        return rocm_tokenspeed_mha_extend(
            query=query,
            query_start_loc=query_start_loc,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=final_seq_lens,
            max_seq_len=args.seq_len,
            max_query_len=args.query_len,
            sliding_window=ts_window,
            sinks=sinks,
            output=tokenspeed_output,
        )

    def aiter_unified_extend() -> torch.Tensor:
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=aiter_output,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=args.query_len,
            seqused_k=final_seq_lens,
            max_seqlen_k=args.seq_len,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=fa_window,
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=sinks,
            output_scale=None,
        )
        return aiter_output

    print(
        "device="
        f"{torch.cuda.get_device_name(device)} "
        f"arch={torch.cuda.get_device_properties(device).gcnArchName} "
        f"dtype={dtype} requests={args.requests} query_len={args.query_len} "
        f"total_q={total_q} seq_len={args.seq_len} "
        f"vllm_sliding_window={args.sliding_window} "
        f"tokenspeed_kernel_sliding_window={ts_window} "
        f"aiter_window={fa_window}",
        flush=True,
    )

    tokenspeed_native_extend()
    aiter_unified_extend()
    diff = (tokenspeed_output.float() - aiter_output.float()).abs()
    print(
        "tokenspeed_native_extend_vs_aiter "
        f"allclose={torch.allclose(tokenspeed_output, aiter_output, atol=args.atol, rtol=args.rtol)} "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g}",
        flush=True,
    )

    print("backend,ms,rows_per_s,relative_to_aiter", flush=True)
    ts_ms = bench_ms(tokenspeed_native_extend, args.warmup, args.iters)
    aiter_ms = bench_ms(aiter_unified_extend, args.warmup, args.iters)
    for name, ms in [
        ("tokenspeed_native_extend", ts_ms),
        ("aiter_unified", aiter_ms),
    ]:
        print(
            f"{name},{ms:.4f},{total_q / (ms / 1000):.1f},{ms / aiter_ms:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
