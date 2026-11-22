# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the GQA verify attention kernels against AITER unified attention.

One attention layer of speculative-decode verify (2 to 8 query tokens per request,
`--query-len`; GQA 8, head size 64, FP8 e4m3 KV cache in 16- or 32-token pages,
`--page`) on gfx950.
Each provider is captured into a HIP graph that rotates over several page tables,
query buffers and pool regions whose total footprint exceeds the last-level cache, so
every call reads its KV cache cold from HBM; the reported time is graph replay time
per call.

  python benchmarks/kernels/benchmark_rocm_gqa_verify.py --query-len 2 4 6 8
  python benchmarks/kernels/benchmark_rocm_gqa_verify.py --window 2048 --heads 32 \
      --kv-heads 4 --num-pages 874240

`--num-pages` sizes the pool; the default (7.2 GB for 4 KV heads) is a
production-sized pool and takes the 64-bit addressing path. The verify time excludes
the per-step work map of full-attention layers, which the backend builds once per
step for all of them.
"""

import argparse
from dataclasses import dataclass

import torch

from vllm.v1.attention.ops import rocm_gqa_verify

HD = rocm_gqa_verify.HEAD_SIZE
ROTATIONS = 8
POOL_TOKENS = 874240 * 16

# Name -> request lengths (context including the verify tokens).
BATCHES = {
    "D1 x 100k": [100000],
    "D4 x 100k": [100000] * 4,
    "D8 x 30k": [30000] * 8,
    "D8 x 100k": [100000] * 8,
    "D24 x 100k": [100000] * 24,
    "D40 x 100k": [100000] * 40,
    "D24 mixed": [200000] + [100000] * 7 + [30000] * 8 + [3000] * 8,
    "D40 mixed": [200000] * 2 + [100000] * 10 + [30000] * 14 + [3000] * 14,
}


@dataclass
class Layer:
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor


@dataclass
class Meta:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor


def graph_replay_time_us(fns, iters: int = 20) -> float:
    """Median time of one captured graph that runs every `fns` entry once, per
    entry, in microseconds."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in fns:
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in fns:
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / len(fns))
    return sorted(times)[len(times) // 2]


def make_pool(num_pages: int, kv_heads: int, page: int):
    pool = torch.empty(
        (num_pages, kv_heads, page, 2 * HD), dtype=torch.float8_e4m3fn, device="cuda"
    )
    # Finite FP8 values (uninitialized memory can hold NaN encodings).
    for start in range(0, num_pages, 16384):
        stop = min(num_pages, start + 16384)
        pool[start:stop] = (
            torch.randn_like(pool[start:stop], dtype=torch.float16) * 0.5
        ).to(torch.float8_e4m3fn)
    return pool[..., :HD].permute(0, 2, 1, 3), pool[..., HD:].permute(0, 2, 1, 3)


def make_tables(lens, num_pages: int, page: int):
    pages = [(n + page - 1) // page for n in lens]
    total = sum(pages)
    if total > num_pages:
        raise ValueError(f"batch needs {total} pages but the pool has {num_pages}")
    tables = []
    for _ in range(ROTATIONS):
        perm = torch.randperm(num_pages, device="cuda")[:total].to(torch.int32)
        table = torch.zeros((len(lens), max(pages)), dtype=torch.int32, device="cuda")
        offset = 0
        for i, p in enumerate(pages):
            table[i, :p] = perm[offset : offset + p]
            offset += p
        tables.append(table)
    return tables


def bench_batch(name, lens, ql, args, k, v, layer) -> None:
    from aiter.ops.triton.unified_attention import unified_attention

    n = len(lens)
    scale = HD**-0.5
    tables = make_tables(lens, args.num_pages, args.page)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cu = torch.arange(n + 1, dtype=torch.int32, device="cuda") * ql
    qs = [
        torch.randn((n * ql, args.heads, HD), device="cuda").to(torch.bfloat16)
        for _ in range(ROTATIONS)
    ]
    out_ref = torch.empty_like(qs[0])
    out_new = torch.empty_like(qs[0])
    window = (args.window - 1, 0) if args.window > 0 else (-1, -1)
    descale = (n, args.kv_heads)

    def baseline(i):
        return lambda: unified_attention(
            q=qs[i],
            k=k,
            v=v,
            out=out_ref,
            cu_seqlens_q=cu,
            max_seqlen_q=ql,
            seqused_k=seq_lens,
            max_seqlen_k=max(lens),
            softmax_scale=scale,
            causal=True,
            alibi_slopes=None,
            window_size=window,
            block_table=tables[i],
            softcap=0.0,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale),
            v_descale=layer._v_scale.expand(descale),
            sinks=None,
        )

    # Built once per step by the backend and shared by the full-attention layers.
    maps = (
        None if args.window > 0 else rocm_gqa_verify.verify_map(seq_lens, args.kv_heads)
    )

    def candidate(i):
        meta = Meta(cu, seq_lens, tables[i])
        return lambda: rocm_gqa_verify.fast_verify(
            qs[i], k, v, out_new, meta, n, n * ql, ql, scale, args.window, layer, maps
        )

    baseline(0)()
    assert candidate(0)(), "verify kernels did not run (layout / FlyDSL?)"
    torch.accelerator.synchronize()
    diff = (out_ref.float() - out_new.float()).abs().max().item()
    t_ref = graph_replay_time_us([baseline(i) for i in range(ROTATIONS)])
    t_new = graph_replay_time_us([candidate(i) for i in range(ROTATIONS)])
    speedup = t_ref / t_new
    print(
        f"{ql:5d} {name:14s} {t_ref:11.1f} {t_new:10.1f} {speedup:7.2f}x {diff:10.3f}"
    )


def run(args) -> None:
    args.num_pages = args.num_pages or POOL_TOKENS // args.page
    k, v = make_pool(args.num_pages, args.kv_heads, args.page)
    layer = Layer(torch.tensor(0.75, device="cuda"), torch.tensor(1.25, device="cuda"))
    print(
        f"heads={args.heads} kv_heads={args.kv_heads} window={args.window} "
        f"pool={args.num_pages} pages of {args.page}; {torch.cuda.get_device_name()}"
    )
    print(
        f"{'q_len':>5s} {'batch':14s} {'unified us':>11s} {'verify us':>10s} "
        f"{'speedup':>8s} {'max|diff|':>10s}"
    )
    for ql in args.query_len:
        for name, lens in BATCHES.items():
            bench_batch(name, lens, ql, args, k, v, layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument(
        "--window", type=int, default=0, help="sliding window in tokens, 0 = full"
    )
    parser.add_argument(
        "--page", type=int, default=16, choices=rocm_gqa_verify.BLOCK_SIZES
    )
    parser.add_argument("--num-pages", type=int, default=None)
    parser.add_argument(
        "--query-len",
        type=int,
        nargs="+",
        default=[rocm_gqa_verify.QUERY_LEN],
        help="query tokens per request (draft tokens + 1), 2 to 8",
    )
    run(parser.parse_args())
