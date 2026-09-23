# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm sparse-indexer decode top-k in isolation.

The end-to-end serving sweep cannot resolve this kernel, so measure it directly.
Widths are in *logit columns*: for a kpool indexer such as GLM-5.3-Flash
(index_topk=2048, index_kpool=4) a column is a pool, so a 1M context gives
--capacity 262144 and an 8K prompt gives --lens 2048, with k = 2048 // 4 = 512.
For a token-granular indexer such as DeepSeek-V4-Flash a column is a token.

    .venv/bin/python benchmarks/kernels/benchmark_top_k_per_row_decode.py \
        --rows 4 16 32 --lens 2048 8192 16384 65536 --capacity 262144

Run the same command on the base and candidate builds to compare.

The defaults cover low-concurrency long-context decode (256K-512K tokens at 1-8
concurrent requests), which is the band where both kernels fall back to a
per-row serial phase and neither has been calibrated.
"""

import argparse
import itertools
import json

import torch

import vllm._custom_ops  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.triton_utils import triton


def make_inputs(rows: int, live: int, capacity: int, k: int, seed: int):
    torch.manual_seed(seed)
    logits = torch.randn(rows, capacity, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((rows,), live, device="cuda", dtype=torch.int32)
    invalid = torch.arange(capacity, device="cuda")[None] >= seq_lens[:, None]
    logits.masked_fill_(invalid, -float("inf"))
    indices = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    return logits, seq_lens, indices


def run_native(logits, seq_lens, indices, k):
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k,
    )


def run_aiter(logits, seq_lens, indices, k):
    rocm_aiter_ops.indexer_top_k_decode(logits, 1, seq_lens, indices, k)


BACKENDS = {"native": run_native, "aiter": run_aiter}


def check(logits, indices, k):
    selected = logits.gather(1, indices.long()).sort(dim=1, descending=True).values
    expected = logits.topk(k, dim=1).values
    torch.testing.assert_close(selected, expected, atol=0, rtol=0)


def capture(run):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def benchmark(backend: str, rows: int, live: int, args) -> dict:
    logits, seq_lens, indices = make_inputs(
        rows, live, args.capacity, args.top_k, args.seed
    )
    fn = BACKENDS[backend]

    def run():
        fn(logits, seq_lens, indices, args.top_k)

    run()
    torch.accelerator.synchronize()
    check(logits, indices, args.top_k)
    timed = run
    if args.graph:
        graph = capture(run)
        indices.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        check(logits, indices, args.top_k)
        timed = graph.replay
    ms = triton.testing.do_bench(timed, warmup=args.warmup, rep=args.rep)
    return dict(
        backend=backend,
        mode="graph" if args.graph else "eager",
        rows=rows,
        live=live,
        capacity=args.capacity,
        k=args.top_k,
        us=ms * 1000,
        live_gbps=rows * live * 4 / (ms * 1e6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--lens",
        type=int,
        nargs="+",
        default=[32768, 65536, 98304, 131072],
        help="Valid columns per row; must be at least --top-k",
    )
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument(
        "--backends", nargs="+", choices=list(BACKENDS), default=["native", "aiter"]
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Time a captured graph replay, matching how decode actually runs",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), args=vars(args))))
    for backend, rows, live in itertools.product(args.backends, args.rows, args.lens):
        if live < args.top_k:
            continue
        print(json.dumps(benchmark(backend, rows, live, args)), flush=True)


if __name__ == "__main__":
    main()
