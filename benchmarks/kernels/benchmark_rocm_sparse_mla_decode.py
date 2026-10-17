# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode on gfx942: Triton split-K kernel vs the HIP kernel.

Shapes are DeepSeek-V4.1-Flash's per-rank decode at TP4: 16 heads, 448 nope +
64 rope dims, fp8_ds_mla caches with 64-token blocks, 128 SWA rows plus 512
top-k rows per query, and `--tokens-per-request` queries per request (1 + the
speculative drafts). Times are CUDA graph replay medians.

    python benchmarks/kernels/benchmark_rocm_sparse_mla_decode.py
"""

import functools
import json

import torch

from vllm import _custom_ops as ops
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sm

HEADS, NOPE, ROPE, BLOCK = 16, 448, 64, 64
ROW = NOPE + 2 * ROPE


def make_cache(num_rows: int, gen: torch.Generator) -> torch.Tensor:
    blocks = (num_rows + BLOCK - 1) // BLOCK
    raw = torch.empty(blocks, BLOCK * (ROW + 8), dtype=torch.uint8, device="cuda")
    data = raw[:, : BLOCK * ROW].view(blocks, BLOCK, ROW)
    nope = torch.randint(
        0, 0x70, (blocks, BLOCK, NOPE), dtype=torch.uint8, device="cuda", generator=gen
    )
    nope[nope == 0] = 1
    sign = torch.randint(
        0, 2, nope.shape, dtype=torch.uint8, device="cuda", generator=gen
    )
    data[..., :NOPE] = nope | (sign << 7)
    rope = torch.randn(blocks, BLOCK, ROPE, device="cuda", generator=gen) * 0.5
    data[..., NOPE:] = rope.to(torch.bfloat16).view(torch.uint8)
    raw[:, BLOCK * ROW :] = torch.randint(
        122, 128, (blocks, BLOCK * 8), dtype=torch.uint8, device="cuda", generator=gen
    )
    return raw.view(blocks, BLOCK, ROW + 8)


def make_inputs(
    requests: int, tokens_per_request: int, context: int, swa: int, topk: int
) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(0)
    num_q = requests * tokens_per_request
    q = torch.randn(num_q, HEADS, NOPE + ROPE, device="cuda", generator=gen)
    swa_span = swa + tokens_per_request
    main_rows, extra_rows = [], []
    for r in range(requests):
        for t in range(tokens_per_request):
            start = r * swa_span + t
            main_rows.append(torch.arange(start, start + swa, device="cuda"))
            sel = torch.randperm(context, device="cuda", generator=gen)[:topk]
            extra_rows.append(sel.sort().values + r * context)
    return dict(
        q=(q * 0.3).to(torch.bfloat16),
        main_cache=make_cache(requests * swa_span, gen),
        main_indices=torch.cat(main_rows).to(torch.int32),
        main_indptr=torch.arange(num_q + 1, device="cuda", dtype=torch.int32) * swa,
        extra_cache=make_cache(requests * context, gen),
        extra_indices=torch.cat(extra_rows).to(torch.int32),
        extra_indptr=torch.arange(num_q + 1, device="cuda", dtype=torch.int32) * topk,
        sink=torch.zeros(HEADS, dtype=torch.float32, device="cuda"),
    )


def run_triton(x: dict) -> torch.Tensor:
    return sm._rocm_sparse_attn_decode_ragged_triton(
        x["q"],
        x["main_cache"],
        x["main_indices"],
        x["main_indptr"],
        (NOPE + ROPE) ** -0.5,
        x["sink"],
        NOPE,
        ROPE,
        x["extra_cache"],
        x["extra_indices"],
        x["extra_indptr"],
    )


def run_hip(x: dict, out: torch.Tensor) -> torch.Tensor:
    ops.rocm_sparse_mla_decode_fp8_ds_mla(
        out,
        x["q"],
        x["main_cache"],
        x["main_indices"],
        x["main_indptr"],
        x["extra_cache"],
        x["extra_indices"],
        x["extra_indptr"],
        x["sink"],
        (NOPE + ROPE) ** -0.5,
        False,
        True,
        sm._hip_sparse_decode_num_splits(x["q"].shape[0]),
    )
    return out


def graph_time_us(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return sorted(times)[len(times) // 2]


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", default="1,2,3,4,6,8,10,12,14,16,24,32,64,128,256"
    )
    parser.add_argument("--tokens-per-request", type=int, default=6)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--swa", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--json", help="also write the rows to this file")
    args = parser.parse_args()
    assert sm._hip_sparse_decode_available(), "needs gfx942 and a _rocm_C with the op"

    rows = []
    print(
        f"{'requests':>8} {'tokens':>6} {'triton us':>10} {'hip us':>8} "
        f"{'speedup':>7} {'rel-L2':>8}"
    )
    for requests in [int(r) for r in args.requests.split(",")]:
        x = make_inputs(
            requests, args.tokens_per_request, args.context, args.swa, args.topk
        )
        out = torch.empty_like(x["q"])
        ref = run_triton(x).float()
        got = run_hip(x, out).float()
        rel = ((got - ref).norm() / ref.norm()).item()
        t_triton = graph_time_us(functools.partial(run_triton, x), args.iters)
        t_hip = graph_time_us(functools.partial(run_hip, x, out), args.iters)
        row = dict(
            requests=requests,
            tokens=x["q"].shape[0],
            triton_us=t_triton,
            hip_us=t_hip,
            speedup=t_triton / t_hip,
            rel_l2=rel,
        )
        rows.append(row)
        print(
            f"{requests:>8} {row['tokens']:>6} {t_triton:>10.1f} {t_hip:>8.1f} "
            f"{row['speedup']:>6.2f}x {rel:>8.1e}",
            flush=True,
        )
    faster = [r["tokens"] for r in rows if r["speedup"] > 1.0]
    print("HIP faster from", min(faster) if faster else None, "tokens")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
