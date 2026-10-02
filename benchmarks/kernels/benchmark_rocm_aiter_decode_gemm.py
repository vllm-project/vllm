# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B of the ROCm aiter decode-GEMM branch in `rocm_unquantized_gemm_impl`.

For every shape that has an aiter decode row on this GPU, times the unquantized
GEMM entry point with the aiter branch on (side A) and off (side B, the skinny
kernels) under CUDA-graph replay, in alternating pairs. Side B is also timed
against itself as a null control; pairs whose null gap exceeds 2% are dropped.
Each cell records which kernel ran on each side.

Usage:
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1 \
        python benchmarks/kernels/benchmark_rocm_aiter_decode_gemm.py --rounds 5
"""

import collections
import statistics

import torch

import vllm.model_executor.layers.utils as U
from vllm import _custom_ops as ops
from vllm.utils.argparse_utils import FlexibleArgumentParser

ROTATE, WARMUP, LAUNCHES, SAMPLES, NULL_LIMIT = 4, 20, 100, 5, 0.02


def decode_shapes() -> list[tuple[int, int, int]]:
    """(M, N, K) of every aiter decode row for this GPU, bias-free, BF16."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.tuned_gemm import get_GEMM_A16W16_config_, is_flydsl_decode_config

    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    return sorted(
        (int(key[2]), int(key[3]), int(key[4]))
        for key, row in get_GEMM_A16W16_config_().items()
        if key[0] == gfx
        and int(key[1]) == cu_num
        and key[5] is False
        and key[6] == "torch.bfloat16"
        and is_flydsl_decode_config(row)
    )


class AiterBranchOff:
    """Disable only the aiter decode branch; the rest of the ladder is unchanged."""

    def __enter__(self):
        self._orig = U.rocm_aiter_ops.has_tuned_decode_gemm
        U.rocm_aiter_ops.has_tuned_decode_gemm = lambda *args, **kwargs: False

    def __exit__(self, *exc):
        U.rocm_aiter_ops.has_tuned_decode_gemm = self._orig


def kernel_that_ran(fn) -> str:
    """Capture one call and report which GEMM kernel it launched.

    The warm-up call before capture runs eagerly and so always takes the skinny
    path; the captured call is the last one recorded.
    """
    from aiter.ops.flydsl import gemm_kernels

    seen = []
    patched = [
        (gemm_kernels, "gemm_decode_bf16", "aiter_decode"),
        (ops, "wvSplitK", "wvSplitK"),
        (ops, "wvSplitKrc", "wvSplitKrc"),
        (ops, "LLMM1", "LLMM1"),
    ]
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patched]
    for (obj, name, label), (_, _, orig) in zip(patched, originals):

        def spy(*args, _orig=orig, _label=label, **kwargs):
            seen.append(_label)
            return _orig(*args, **kwargs)

        setattr(obj, name, spy)
    try:
        capture(lambda _: fn(), 1)
    finally:
        for obj, name, orig in originals:
            setattr(obj, name, orig)
    return seen[-1] if seen else "other"


def capture(fn, launches: int) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(launches):
            fn(i)
    return graph


def replay_us(graph: torch.cuda.CUDAGraph) -> float:
    times = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / LAUNCHES)
    return statistics.median(times)


def bench_shape(m: int, n: int, k: int, pairs: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16) for _ in range(ROTATE)
    ]

    def call(i):
        return U.rocm_unquantized_gemm_impl(x, weights[i % ROTATE], None)

    kernel_a = kernel_that_ran(lambda: call(0))
    graph_a = capture(call, LAUNCHES)
    with AiterBranchOff():
        kernel_b = kernel_that_ran(lambda: call(0))
        graph_b = capture(call, LAUNCHES)
        graph_null = capture(call, LAUNCHES)
    for graph in (graph_a, graph_b, graph_null):
        for _ in range(WARMUP):
            graph.replay()
    torch.accelerator.synchronize()

    times_a, times_b = [], []
    for p in range(pairs):
        order = (graph_a, graph_b) if p % 2 == 0 else (graph_b, graph_a)
        first, second = replay_us(order[0]), replay_us(order[1])
        a, b = (first, second) if p % 2 == 0 else (second, first)
        null = replay_us(graph_null)
        if abs(null - b) / b <= NULL_LIMIT:
            times_a.append(a)
            times_b.append(b)
    return {
        "m": m,
        "shape": f"{n}x{k}",
        "kernel_a": kernel_a,
        "kernel_b": kernel_b,
        "times_a": times_a,
        "times_b": times_b,
    }


def main(args):
    shapes = decode_shapes()
    if args.m:
        shapes = [s for s in shapes if s[0] in args.m]
    if not shapes:
        print("No aiter decode rows for this GPU; nothing to compare.")
        return
    print(f"{len(shapes)} shapes, {args.rounds} rounds x {args.pairs} pairs")

    cells = {}
    for _ in range(args.rounds):
        for m, n, k in shapes:
            result = bench_shape(m, n, k, args.pairs)
            cell = cells.setdefault((m, n, k), result)
            if cell is not result:
                cell["times_a"] += result["times_a"]
                cell["times_b"] += result["times_b"]

    print(
        "\n| M | N x K | kernel A | kernel B | A, us | B, us | gain | A won |"
        "\n|---|---|---|---|---|---|---|---|"
    )
    total_a, total_b = collections.Counter(), collections.Counter()
    for (m, _, _), cell in sorted(cells.items()):
        if not cell["times_a"]:
            print(f"| {m} | {cell['shape']} | dropped by the null control |")
            continue
        a, b = statistics.median(cell["times_a"]), statistics.median(cell["times_b"])
        won = sum(x < y for x, y in zip(cell["times_a"], cell["times_b"]))
        total_a[m] += a
        total_b[m] += b
        print(
            f"| {m} | {cell['shape']} | {cell['kernel_a']} | {cell['kernel_b']} "
            f"| {a:.3f} | {b:.3f} | {(b - a) / b * 100:+.1f}% "
            f"| {won}/{len(cell['times_a'])} |"
        )
    print("\n| M | sum A, us | sum B, us | gain |\n|---|---|---|---|")
    for m in sorted(total_a):
        gain = (total_b[m] - total_a[m]) / total_b[m] * 100
        print(f"| {m} | {total_a[m]:.1f} | {total_b[m]:.1f} | {gain:+.1f}% |")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3, help="passes over all shapes")
    parser.add_argument(
        "--pairs", type=int, default=10, help="A/B pairs per shape per round"
    )
    parser.add_argument("--m", type=int, nargs="*", help="only these M values")
    main(parser.parse_args())
