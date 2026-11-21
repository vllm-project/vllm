#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TokenSpeed native-extend mixed attention against AITER unified.

This synthetic mixed-step benchmark compares only:

* tokenspeed_native_extend_all:
  * decode rows use TokenSpeed decode
  * extend rows use TokenSpeed native extend
  * no-prefix prefill rows use TokenSpeed native extend
* aiter_unified:
  * the same request-level mixed metadata in one AITER unified-attention call

For sliding-window cases, --sliding-window is the vLLM semantic window. The
TokenSpeed decode/extend kernels receive sliding_window - 1, matching AITER's
window_size=(sliding_window - 1, 0).

The script does not modify tokenspeed_kernel_amd.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch
from aiter.ops.triton.unified_attention import unified_attention

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
    parser.add_argument("--num-decodes", type=int, default=128)
    parser.add_argument("--decode-seq-len", type=int, default=1024)
    parser.add_argument("--num-extends", type=int, default=32)
    parser.add_argument("--extend-query-len", type=int, default=8)
    parser.add_argument("--extend-seq-len", type=int, default=1024)
    parser.add_argument("--num-prefills", type=int, default=16)
    parser.add_argument("--prefill-query-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sliding-window", type=int, default=-1)
    parser.add_argument("--no-sinks", action="store_true")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    return parser.parse_args()


def tokenspeed_decode_extend_window(sliding_window: int) -> int:
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
    ts_window = tokenspeed_decode_extend_window(args.sliding_window)
    fa_window = aiter_window(args.sliding_window)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    query_lens = (
        [1] * args.num_decodes
        + [args.extend_query_len] * args.num_extends
        + [args.prefill_query_len] * args.num_prefills
    )
    seq_lens_list = (
        [args.decode_seq_len] * args.num_decodes
        + [args.extend_seq_len] * args.num_extends
        + [args.prefill_query_len] * args.num_prefills
    )
    num_reqs = len(query_lens)
    total_tokens = sum(query_lens)
    max_query_len = max(query_lens) if query_lens else 1
    max_seq_len = max(seq_lens_list) if seq_lens_list else 1
    pages_per_req = math.ceil(max_seq_len / args.block_size)
    num_blocks = num_reqs * pages_per_req

    query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    query = torch.randn(
        total_tokens, args.heads, args.head_dim, device=device, dtype=dtype
    )
    key_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        num_reqs, pages_per_req
    )
    sinks = (
        None
        if args.no_sinks
        else torch.randn(args.heads, device=device, dtype=torch.float32)
    )

    decode_tokens = args.num_decodes
    extend_start = decode_tokens
    extend_end = extend_start + args.num_extends * args.extend_query_len
    prefill_start = extend_end
    prefill_req_start = args.num_decodes + args.num_extends

    extend_req_block_table = block_table[
        args.num_decodes : args.num_decodes + args.num_extends
    ]
    extend_query_start_loc = torch.arange(
        0,
        args.num_extends * args.extend_query_len + 1,
        args.extend_query_len,
        device=device,
        dtype=torch.int32,
    )
    prefill_req_block_table = block_table[prefill_req_start:]
    prefill_query_start_loc = torch.arange(
        0,
        args.num_prefills * args.prefill_query_len + 1,
        args.prefill_query_len,
        device=device,
        dtype=torch.int32,
    )

    output_tokenspeed = torch.empty_like(query)
    output_aiter = torch.empty_like(query)

    def tokenspeed_native_extend_all() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_window,
                sinks=sinks,
                output=output_tokenspeed[:decode_tokens],
            )
        if args.num_extends:
            rocm_tokenspeed_mha_extend(
                query=query[extend_start:extend_end],
                query_start_loc=extend_query_start_loc,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=extend_req_block_table,
                seq_lens=seq_lens[
                    args.num_decodes : args.num_decodes + args.num_extends
                ],
                max_seq_len=args.extend_seq_len,
                max_query_len=args.extend_query_len,
                sliding_window=ts_window,
                sinks=sinks,
                output=output_tokenspeed[extend_start:extend_end],
            )
        if args.num_prefills:
            rocm_tokenspeed_mha_extend(
                query=query[prefill_start:],
                query_start_loc=prefill_query_start_loc,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=prefill_req_block_table,
                seq_lens=seq_lens[prefill_req_start:],
                max_seq_len=args.prefill_query_len,
                max_query_len=args.prefill_query_len,
                sliding_window=ts_window,
                sinks=sinks,
                output=output_tokenspeed[prefill_start:],
            )
        return output_tokenspeed

    def aiter_unified() -> torch.Tensor:
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output_aiter,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            seqused_k=seq_lens,
            max_seqlen_k=max_seq_len,
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
        return output_aiter

    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"arch={torch.cuda.get_device_properties(device).gcnArchName} "
        f"dtype={dtype} decodes={args.num_decodes} "
        f"extends={args.num_extends}x{args.extend_query_len} "
        f"prefills={args.num_prefills}x{args.prefill_query_len} "
        f"total_tokens={total_tokens} max_seq_len={max_seq_len} "
        f"vllm_sliding_window={args.sliding_window} "
        f"tokenspeed_decode_extend_sliding_window={ts_window} "
        f"aiter_window={fa_window} sinks={sinks is not None}",
        flush=True,
    )

    tokenspeed_native_extend_all()
    aiter_unified()
    diff = (output_tokenspeed.float() - output_aiter.float()).abs()
    print(
        "tokenspeed_native_extend_all_vs_aiter "
        f"allclose={torch.allclose(output_tokenspeed, output_aiter, atol=args.atol, rtol=args.rtol)} "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g}",
        flush=True,
    )

    timings = [
        (
            "tokenspeed_native_extend_all",
            bench_ms(tokenspeed_native_extend_all, args.warmup, args.iters),
        ),
        ("aiter_unified", bench_ms(aiter_unified, args.warmup, args.iters)),
    ]
    aiter_ms = dict(timings)["aiter_unified"]
    print("backend,ms,tokens_per_s,relative_to_aiter", flush=True)
    for name, ms in timings:
        print(
            f"{name},{ms:.4f},{total_tokens / (ms / 1000):.1f},{ms / aiter_ms:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
