# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm bf16 sparse-MLA decode split-K kernel in isolation.

Example usage:
python3 benchmarks/kernels/benchmark_sparse_mla_decode_bf16.py \
    --rows 1 4 16 64 --lens 2048 --splits auto 4 16 32
"""

import argparse
import itertools
import json

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_decode_ragged_bf16_triton,
    _rocm_sparse_attn_prefill_ragged_triton,
    _sparse_decode_bf16_num_splits,
)

BLOCK_H = 16
BLOCK_K = 32


def make_inputs(rows: int, live: int, args):
    torch.manual_seed(args.seed)
    kv = (
        torch.randn(args.capacity, args.head_dim, device="cuda", dtype=torch.bfloat16)
        * 0.125
    )
    q = (
        torch.randn(
            rows, args.heads, args.head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.125
    )
    slots = torch.stack(
        [torch.randperm(args.capacity, device="cuda")[:live] for _ in range(rows)]
    )
    indices = slots.reshape(-1).to(torch.int32)
    indptr = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * live
    sink = (
        torch.randn(args.heads, device="cuda", dtype=torch.float32)
        if args.sink
        else None
    )
    return q, kv, indices, indptr, sink


def resolve_splits(spec: str, rows: int, live: int, args) -> int:
    if spec != "auto":
        return int(spec)
    return _sparse_decode_bf16_num_splits(
        rows, triton.cdiv(args.heads, BLOCK_H), rows * live, BLOCK_K
    )


def benchmark(backend: str, rows: int, live: int, spec: str, args) -> dict:
    q, kv, indices, indptr, sink = make_inputs(rows, live, args)
    nope, rope = args.head_dim, 0
    splits = resolve_splits(spec, rows, live, args)

    def ragged():
        return _rocm_sparse_attn_prefill_ragged_triton(
            q, kv, indices, indptr, args.scale, sink, nope, rope
        )

    def splitk():
        return _rocm_sparse_attn_decode_ragged_bf16_triton(
            q, kv, indices, indptr, args.scale, sink, nope, rope, splits
        )

    run = ragged if backend == "ragged" else splitk
    torch.testing.assert_close(run(), ragged(), atol=2e-2, rtol=2e-2)
    ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
    return dict(
        backend=backend,
        rows=rows,
        live=live,
        heads=args.heads,
        head_dim=args.head_dim,
        splits=splits if backend == "splitk" else 1,
        workgroups=rows * triton.cdiv(args.heads, BLOCK_H) * splits,
        us=ms * 1000,
        kv_gbps=rows * live * args.head_dim * 2 / (ms * 1e6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument(
        "--lens",
        type=int,
        nargs="+",
        default=[2048],
        help="Selected KV slots per row; GLM-5.3-Flash uses index_topk=2048",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["auto"],
        help="Split counts to sweep, or auto for the runtime heuristic",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["ragged", "splitk"],
        default=["ragged", "splitk"],
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Heads per rank; 64 at TP8 gives 8"
    )
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--scale", type=float, default=0.044194173824159216)
    parser.add_argument("--sink", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), args=vars(args))))
    for backend, rows, live in itertools.product(args.backends, args.rows, args.lens):
        specs = ["1"] if backend == "ragged" else args.splits
        for spec in specs:
            print(json.dumps(benchmark(backend, rows, live, spec, args)), flush=True)


if __name__ == "__main__":
    main()
