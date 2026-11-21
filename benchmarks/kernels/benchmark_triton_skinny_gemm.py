# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark and tune the Triton skinny GEMM against cuBLAS (`F.linear`).

Default: time every planned Triton entry for this GPU against cuBLAS.
`--tune N,K [N,K ...]`: search tile configs for those weight shapes at every
CUDA graph size up to `--max-tokens`, printing the best config per size.

Each timing replays a CUDA graph that cycles through weight copies totalling
`--rotate-mb`, so weights are read from DRAM (as in decode, not from L2) and
launch overhead is excluded.
"""

import itertools

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import (
    TRITON_SKINNY_GEMM_PLANS_BY_CAPABILITY,
)
from vllm.model_executor.kernels.linear.triton_skinny_gemm import (
    TritonSkinnyGemmConfig,
    triton_skinny_gemm,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser


def graph_time_us(fn, weights, x, replays: int) -> float:
    for w in weights[:2]:
        fn(x, w)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in weights:
            fn(x, w)
    graph.replay()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * len(weights))


def config_grid(num_tokens: int):
    props = torch.cuda.get_device_properties()
    smem = getattr(props, "shared_memory_per_block_optin", 99 * 1024)
    for bm, bn, bk, warps, stages in itertools.product(
        (16, 32, 64, 128), (16, 32, 64, 128, 256), (64, 128), (4, 8), (2, 3, 4, 5)
    ):
        if bm > max(16, num_tokens) or stages * (bm + bn) * bk * 2 > smem:
            continue
        if (warps == 4 and bm * bn >= 128 * 128) or (warps == 8 and bm * bn < 2048):
            continue
        yield TritonSkinnyGemmConfig(bm, bn, bk, warps, stages)


def bench_shape(n, k, configs_for, token_counts, rotate_mb, replays):
    copies = max(2, -(-rotate_mb * 2**20 // (n * k * 2)))
    weights = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 64
        for _ in range(copies)
    ]
    for m in token_counts:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        reference = F.linear(x, weights[0]).float()
        cublas = graph_time_us(F.linear, weights, x, replays)
        best = None
        for config in configs_for(m):
            try:
                out = triton_skinny_gemm(x, weights[0], config).float()
            except Exception:  # Exceeds this GPU's registers or shared memory.
                continue
            err = (out - reference).abs().max() / reference.abs().max()
            assert err < 1e-2, f"{config} mismatch at {(n, k, m)}: {err:.3g}"
            t = graph_time_us(
                lambda a, b, c=config: triton_skinny_gemm(a, b, c), weights, x, replays
            )
            if best is None or t < best[0]:
                best = (t, config)
        if best is not None:
            t, config = best
            print(
                f"({n}, {k}) M={m:3d}  cuBLAS {cublas:8.1f} us  Triton {t:8.1f} us  "
                f"{cublas / t:.3f}x  {config}",
                flush=True,
            )


def main(args):
    torch.manual_seed(0)
    capability = current_platform.get_device_capability()
    print(f"{torch.cuda.get_device_name()} (capability {capability})")
    if args.tune:
        token_counts = list(range(8, args.max_tokens + 1, 8))
        for shape in args.tune:
            n, k = map(int, shape.split(","))
            bench_shape(
                n,
                k,
                lambda m: list(config_grid(m)),
                token_counts,
                args.rotate_mb,
                args.replays,
            )
        return
    plans = TRITON_SKINNY_GEMM_PLANS_BY_CAPABILITY.get(capability)
    if not plans:
        print("No Triton skinny GEMM plans for this GPU.")
        return
    for (n, k), plan in plans.items():
        bench_shape(
            n,
            k,
            lambda m, plan=plan: [plan[m]],
            sorted(plan),
            args.rotate_mb,
            args.replays,
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune", nargs="+", metavar="N,K", help="weight shapes to tune"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rotate-mb", type=int, default=512)
    parser.add_argument("--replays", type=int, default=5)
    main(parser.parse_args())
