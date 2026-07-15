#!/usr/bin/env python3
"""Benchmark TokenSpeed native extend vs decode-decomposed extend.

This isolates the main safe-path tradeoff in ROCM_TOKENSPEED_MHA:

* native extend is one TokenSpeed extend launch over request-level metadata;
* decode-decomposed extend expands request block tables to per-token rows and
  runs the TokenSpeed decode kernel with per-token sequence lengths.

The backend keeps native extend disabled by default because model-level GSM8K
accuracy regressed with native extend. This script measures the performance cost
of that accuracy-safe decode decomposition without modifying TokenSpeed kernels.

Recommended invocation:

    PYTHONPATH=/app/tokspd/tokspd-int CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/rocm_tokenspeed_mha/bench_extend_decomposition_perf.py
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch

from vllm.v1.attention.ops.rocm_tokenspeed_mha import (
    rocm_tokenspeed_mha_decode,
    rocm_tokenspeed_mha_extend,
)


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
    return parser.parse_args()


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

    sinks = (
        None
        if args.no_sinks
        else torch.randn(args.heads, device=device, dtype=torch.float32)
    )
    query = torch.randn(
        total_q, args.heads, args.head_dim, device=device, dtype=dtype
    )
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

    token_to_req = torch.repeat_interleave(
        torch.arange(args.requests, device=device, dtype=torch.int32),
        args.query_len,
    )
    decode_seq_lens_cpu = []
    start_seq_len = args.seq_len - args.query_len + 1
    for _ in range(args.requests):
        decode_seq_lens_cpu.extend(range(start_seq_len, args.seq_len + 1))
    decode_seq_lens = torch.tensor(
        decode_seq_lens_cpu, device=device, dtype=torch.int32
    )
    decode_block_table = block_table.index_select(0, token_to_req.long())
    pure_decode_seq_lens = final_seq_lens.repeat_interleave(args.query_len)

    native_output = torch.empty_like(query)
    decomposed_output = torch.empty_like(query)
    pure_decode_output = torch.empty_like(query)

    def native_extend() -> torch.Tensor:
        return rocm_tokenspeed_mha_extend(
            query=query,
            query_start_loc=query_start_loc,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=final_seq_lens,
            max_seq_len=args.seq_len,
            max_query_len=args.query_len,
            sliding_window=args.sliding_window,
            sinks=sinks,
            output=native_output,
        )

    def decode_decomposed_preexpanded() -> torch.Tensor:
        return rocm_tokenspeed_mha_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=decode_block_table,
            seq_lens=decode_seq_lens,
            max_seq_len=args.seq_len,
            max_query_len=1,
            sliding_window=args.sliding_window,
            sinks=sinks,
            output=decomposed_output,
        )

    def decode_decomposed_with_expand() -> torch.Tensor:
        expanded_block_table = block_table.index_select(0, token_to_req.long())
        return rocm_tokenspeed_mha_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=expanded_block_table,
            seq_lens=decode_seq_lens,
            max_seq_len=args.seq_len,
            max_query_len=1,
            sliding_window=args.sliding_window,
            sinks=sinks,
            output=decomposed_output,
        )

    def pure_decode() -> torch.Tensor:
        return rocm_tokenspeed_mha_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=decode_block_table,
            seq_lens=pure_decode_seq_lens,
            max_seq_len=args.seq_len,
            max_query_len=1,
            sliding_window=args.sliding_window,
            sinks=sinks,
            output=pure_decode_output,
        )

    print(
        "device="
        f"{torch.cuda.get_device_name(device)} "
        f"arch={torch.cuda.get_device_properties(device).gcnArchName} "
        f"dtype={dtype} requests={args.requests} query_len={args.query_len} "
        f"total_q={total_q} seq_len={args.seq_len} "
        f"sliding_window={args.sliding_window}",
        flush=True,
    )
    print("case,ms,rows_per_s,relative_to_native", flush=True)

    native_ms = bench_ms(native_extend, args.warmup, args.iters)
    print(
        f"native_extend,{native_ms:.4f},{total_q / (native_ms / 1000):.1f},1.000",
        flush=True,
    )

    for name, fn in [
        ("decode_decomposed_preexpanded", decode_decomposed_preexpanded),
        ("decode_decomposed_with_expand", decode_decomposed_with_expand),
        ("pure_decode_same_rows", pure_decode),
    ]:
        ms = bench_ms(fn, args.warmup, args.iters)
        print(
            f"{name},{ms:.4f},{total_q / (ms / 1000):.1f},{ms / native_ms:.3f}",
            flush=True,
        )

    torch.cuda.synchronize()
    diff = (native_output.float() - decomposed_output.float()).abs()
    print(
        "native_vs_decomposed_last_run "
        f"max_abs_diff={diff.max().item():.8g} "
        f"mean_abs_diff={diff.mean().item():.8g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
