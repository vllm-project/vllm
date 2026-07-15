#!/usr/bin/env python3
"""Benchmark TokenSpeed's mixed safe path against AITER unified attention.

This isolates the main vLLM integration gap without running the full server:

* AITER unified consumes request-level decode/extend/prefill metadata in one op.
* ROCM_TOKENSPEED_MHA safe mode splits the same mixed batch into decode-shaped
  calls and expands extend/prefill request metadata to per-token decode rows.
* The native-extend-all diagnostic routes both extend rows and no-prefix
  prefill rows through TokenSpeed extend with request-level metadata.

The script does not modify TokenSpeed kernels.
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
    rocm_tokenspeed_mha_prefill,
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


def compare(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    torch.cuda.synchronize()
    diff = (lhs.float() - rhs.float()).abs()
    print(
        f"{name}: exact={torch.equal(lhs, rhs)} "
        f"max_abs={diff.max().item():.8g} mean_abs={diff.mean().item():.8g}",
        flush=True,
    )


def fill_prefill_cache_from_dense(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefill_req_start: int,
    prefill_token_start: int,
    num_prefills: int,
    prefill_query_len: int,
    block_size: int,
) -> None:
    for local_req in range(num_prefills):
        req = prefill_req_start + local_req
        token_base = prefill_token_start + local_req * prefill_query_len
        for pos in range(prefill_query_len):
            page = int(block_table[req, pos // block_size].item())
            offset = pos % block_size
            key_cache[page, offset].copy_(key[token_base + pos])
            value_cache[page, offset].copy_(value[token_base + pos])


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a ROCm/CUDA-visible GPU.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

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
    key = torch.randn(
        total_tokens, args.kv_heads, args.head_dim, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    key_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(num_reqs, pages_per_req)

    prefill_req_start = args.num_decodes + args.num_extends
    prefill_token_start = args.num_decodes + args.num_extends * args.extend_query_len
    fill_prefill_cache_from_dense(
        key_cache,
        value_cache,
        block_table,
        key,
        value,
        prefill_req_start,
        prefill_token_start,
        args.num_prefills,
        args.prefill_query_len,
        args.block_size,
    )

    output_ts_safe = torch.empty_like(query)
    output_ts_native_extend = torch.empty_like(query)
    output_ts_prefill = torch.empty_like(query)
    output_ts_safe_with_expand = torch.empty_like(query)
    output_ts_native_extend_prefill = torch.empty_like(query)
    output_ts_native_extend_all = torch.empty_like(query)
    output_aiter = torch.empty_like(query)
    sinks = (
        None
        if args.no_sinks
        else torch.randn(args.heads, device=device, dtype=torch.float32)
    )
    ts_decode_window = args.sliding_window
    ts_prefill_window = -1 if args.sliding_window < 0 else args.sliding_window - 1
    aiter_window = (
        (-1, -1)
        if args.sliding_window < 0
        else (args.sliding_window - 1, 0)
    )
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    decode_tokens = args.num_decodes
    extend_start = decode_tokens
    extend_end = extend_start + args.num_extends * args.extend_query_len
    prefill_start = extend_end

    extend_req_block_table = block_table[
        args.num_decodes : args.num_decodes + args.num_extends
    ]
    extend_token_to_req = torch.repeat_interleave(
        torch.arange(args.num_extends, device=device, dtype=torch.int32),
        args.extend_query_len,
    )
    extend_block_table = extend_req_block_table.index_select(
        0, extend_token_to_req.long()
    )
    extend_decode_seq_lens = []
    start_extend_seq = args.extend_seq_len - args.extend_query_len + 1
    for _ in range(args.num_extends):
        extend_decode_seq_lens.extend(range(start_extend_seq, args.extend_seq_len + 1))
    extend_decode_seq_lens_t = torch.tensor(
        extend_decode_seq_lens, device=device, dtype=torch.int32
    )
    extend_query_start_loc = torch.arange(
        0,
        args.num_extends * args.extend_query_len + 1,
        args.extend_query_len,
        device=device,
        dtype=torch.int32,
    )

    prefill_req_block_table = block_table[prefill_req_start:]
    prefill_token_to_req = torch.repeat_interleave(
        torch.arange(args.num_prefills, device=device, dtype=torch.int32),
        args.prefill_query_len,
    )
    prefill_block_table = prefill_req_block_table.index_select(
        0, prefill_token_to_req.long()
    )
    prefill_decode_seq_lens = []
    for _ in range(args.num_prefills):
        prefill_decode_seq_lens.extend(range(1, args.prefill_query_len + 1))
    prefill_decode_seq_lens_t = torch.tensor(
        prefill_decode_seq_lens, device=device, dtype=torch.int32
    )
    prefill_query_start_loc = torch.arange(
        0,
        args.num_prefills * args.prefill_query_len + 1,
        args.prefill_query_len,
        device=device,
        dtype=torch.int32,
    )
    prefill_query_start_loc_cpu = prefill_query_start_loc.cpu()

    def run_tokenspeed_safe() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe[:decode_tokens],
            )
        if args.num_extends:
            rocm_tokenspeed_mha_decode(
                query=query[extend_start:extend_end],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=extend_block_table,
                seq_lens=extend_decode_seq_lens_t,
                max_seq_len=args.extend_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe[extend_start:extend_end],
            )
        if args.num_prefills:
            rocm_tokenspeed_mha_decode(
                query=query[prefill_start:],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=prefill_block_table,
                seq_lens=prefill_decode_seq_lens_t,
                max_seq_len=args.prefill_query_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe[prefill_start:],
            )
        return output_ts_safe

    def run_tokenspeed_safe_with_expand() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe_with_expand[:decode_tokens],
            )
        if args.num_extends:
            expanded_extend_block_table = extend_req_block_table.index_select(
                0, extend_token_to_req.long()
            )
            rocm_tokenspeed_mha_decode(
                query=query[extend_start:extend_end],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=expanded_extend_block_table,
                seq_lens=extend_decode_seq_lens_t,
                max_seq_len=args.extend_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe_with_expand[extend_start:extend_end],
            )
        if args.num_prefills:
            expanded_prefill_block_table = prefill_req_block_table.index_select(
                0, prefill_token_to_req.long()
            )
            rocm_tokenspeed_mha_decode(
                query=query[prefill_start:],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=expanded_prefill_block_table,
                seq_lens=prefill_decode_seq_lens_t,
                max_seq_len=args.prefill_query_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_safe_with_expand[prefill_start:],
            )
        return output_ts_safe_with_expand

    def run_tokenspeed_native_extend() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend[:decode_tokens],
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
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend[extend_start:extend_end],
            )
        if args.num_prefills:
            rocm_tokenspeed_mha_decode(
                query=query[prefill_start:],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=prefill_block_table,
                seq_lens=prefill_decode_seq_lens_t,
                max_seq_len=args.prefill_query_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend[prefill_start:],
            )
        return output_ts_native_extend

    def run_tokenspeed_prefill_kernel() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_prefill[:decode_tokens],
            )
        if args.num_extends:
            rocm_tokenspeed_mha_decode(
                query=query[extend_start:extend_end],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=extend_block_table,
                seq_lens=extend_decode_seq_lens_t,
                max_seq_len=args.extend_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_prefill[extend_start:extend_end],
            )
        if args.num_prefills:
            rocm_tokenspeed_mha_prefill(
                query=query[prefill_start:],
                key=key[prefill_start:],
                value=value[prefill_start:],
                query_start_loc=prefill_query_start_loc,
                query_start_loc_cpu=prefill_query_start_loc_cpu,
                max_query_len=args.prefill_query_len,
                sliding_window=ts_prefill_window,
                sinks=sinks,
                output=output_ts_prefill[prefill_start:],
            )
        return output_ts_prefill

    def run_tokenspeed_native_extend_prefill_kernel() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend_prefill[:decode_tokens],
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
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend_prefill[extend_start:extend_end],
            )
        if args.num_prefills:
            rocm_tokenspeed_mha_prefill(
                query=query[prefill_start:],
                key=key[prefill_start:],
                value=value[prefill_start:],
                query_start_loc=prefill_query_start_loc,
                query_start_loc_cpu=prefill_query_start_loc_cpu,
                max_query_len=args.prefill_query_len,
                sliding_window=ts_prefill_window,
                sinks=sinks,
                output=output_ts_native_extend_prefill[prefill_start:],
            )
        return output_ts_native_extend_prefill

    def run_tokenspeed_native_extend_all() -> torch.Tensor:
        if args.num_decodes:
            rocm_tokenspeed_mha_decode(
                query=query[:decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[: args.num_decodes],
                seq_lens=seq_lens[: args.num_decodes],
                max_seq_len=max_seq_len,
                max_query_len=1,
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend_all[:decode_tokens],
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
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend_all[extend_start:extend_end],
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
                sliding_window=ts_decode_window,
                sinks=sinks,
                output=output_ts_native_extend_all[prefill_start:],
            )
        return output_ts_native_extend_all

    def run_aiter() -> torch.Tensor:
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
            window_size=aiter_window,
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
        f"sliding_window={args.sliding_window} sinks={sinks is not None}",
        flush=True,
    )

    run_tokenspeed_safe()
    run_tokenspeed_safe_with_expand()
    run_tokenspeed_native_extend()
    run_tokenspeed_prefill_kernel()
    run_tokenspeed_native_extend_prefill_kernel()
    run_tokenspeed_native_extend_all()
    run_aiter()
    compare("tokenspeed_safe_vs_aiter", output_ts_safe, output_aiter)
    compare("tokenspeed_safe_with_expand_vs_safe", output_ts_safe_with_expand,
            output_ts_safe)
    compare("tokenspeed_native_extend_vs_safe", output_ts_native_extend, output_ts_safe)
    compare("tokenspeed_prefill_kernel_vs_safe", output_ts_prefill, output_ts_safe)
    compare("tokenspeed_native_extend_prefill_vs_safe",
            output_ts_native_extend_prefill, output_ts_safe)
    compare("tokenspeed_native_extend_all_vs_safe",
            output_ts_native_extend_all, output_ts_safe)

    cases = [
        ("tokenspeed_safe", run_tokenspeed_safe),
        ("tokenspeed_safe_with_expand", run_tokenspeed_safe_with_expand),
        ("tokenspeed_native_extend", run_tokenspeed_native_extend),
        ("tokenspeed_prefill_kernel", run_tokenspeed_prefill_kernel),
        (
            "tokenspeed_native_extend_prefill_kernel",
            run_tokenspeed_native_extend_prefill_kernel,
        ),
        ("tokenspeed_native_extend_all", run_tokenspeed_native_extend_all),
        ("aiter_unified", run_aiter),
    ]
    timings = [(name, bench_ms(fn, args.warmup, args.iters)) for name, fn in cases]
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
