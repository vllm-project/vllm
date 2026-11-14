# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the CuTe-DSL skinny GEMM against cuBLAS for the plans table.

`skinny_gemm_plans.py` keeps an entry only where the skinny GEMM beats cuBLAS
by more than 5%, so every entry needs a measurement on the target device. This
benchmark reproduces those measurements: both sides run through CUDA-graph
replay with rotating weight buffers larger than L2, so the comparison models a
decode step that streams the weights from HBM instead of reading a hot copy.

The default shapes are the unquantized BF16 projections GLM-5.3-Flash exposes
at TP=8 on the RTX PRO 6000 Blackwell family (SM120); `--shapes` overrides them
with any `NxK` list.

    python3 benchmarks/kernels/benchmark_skinny_gemm_sm120.py \
        --rows 1,2,4,8 --out plan_snippet.py
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# GLM-5.3-Flash, TP=8, local (N, K) of the unquantized BF16 linears.
GLM53_TP8_SHAPES = (
    (19360, 4096),  # LM head
    (3336, 4096),  # MLA fused q_a/kv_a
    (4096, 4096),  # KDA output projection
    (2560, 4096),  # KDA fused QKV/beta/f_a/g_a
    (2048, 4096),  # MLA q_b
    (160, 4096),
    (288, 4096),
    (512, 4096),
    (4096, 2048),
    (4096, 1536),
    (2048, 1536),
    (4096, 1280),
    (4096, 1024),
)

# Seed configs: the winners on this device family, plus the shapes the built-in
# heuristic and the other capability tables pick.
DEFAULT_CONFIGS = (
    SkinnyGemmConfig(0, 128, 1, k_unroll=4),
    SkinnyGemmConfig(0, 64, 1, k_unroll=4),
    SkinnyGemmConfig(0, 64, 2, k_unroll=4),
    SkinnyGemmConfig(0, 128, 2, k_unroll=4),
    SkinnyGemmConfig(0, 256, 1, k_unroll=4),
    SkinnyGemmConfig(0, 32, 1, k_unroll=4, vector_width=4),
)

KEEP_RATIO = 1.05


def parse_shape(value: str) -> tuple[int, int]:
    n, k = value.lower().split("x")
    return int(n), int(k)


def parse_config(value: str) -> SkinnyGemmConfig:
    parts = [int(part) for part in value.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("config must be BLOCK,OUTPUTS,UNROLL[,VEC]")
    block, outputs, unroll = parts[:3]
    vec = parts[3] if len(parts) == 4 else 8
    return SkinnyGemmConfig(0, block, outputs, k_unroll=unroll, vector_width=vec)


def satisfies_kernel_constraints(config: SkinnyGemmConfig, n: int, k: int) -> bool:
    """The constraints `test_skinny_gemm_plans_are_valid` enforces."""
    if config.block_size % 32 or n % config.outputs_per_block:
        return False
    k_tile = config.block_size * config.vector_width
    if k % k_tile:
        return False
    if config.static_k not in (None, k):
        return False
    return config.static_k is None or config.static_k >= 2 * k_tile


def candidates(n: int, k: int, seeds: tuple[SkinnyGemmConfig, ...]):
    """The seed configs, each with and without a compile-time K."""
    for seed in seeds:
        for static_k in (k, None):
            config = SkinnyGemmConfig(
                seed.num_rows,
                seed.block_size,
                seed.outputs_per_block,
                seed.k_unroll,
                seed.vector_width,
                static_k,
            )
            if satisfies_kernel_constraints(config, n, k):
                yield config


def time_graph(
    launch, repeats: int, replays: int
) -> tuple[float, float]:
    """Median and minimum microseconds per launch under CUDA-graph replay."""
    for _ in range(5):
        launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / replays)
    return statistics.median(samples), min(samples)


def measure_shape(
    n: int,
    k: int,
    rows: list[int],
    seeds: tuple[SkinnyGemmConfig, ...],
    args: argparse.Namespace,
    out_lines: list[str],
) -> None:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = properties.L2_cache_size
    for m in rows:
        bytes_per_buffer = (n * k + m * n) * 2
        wanted = math.ceil(args.cache_multiplier * l2_bytes / bytes_per_buffer)
        count = max(2, min(args.limit, wanted))
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weights = [
            torch.randn(n, k, dtype=torch.bfloat16, device="cuda") for _ in range(count)
        ]
        reference = F.linear(x, weights[0])
        cursor = [0]

        def cublas():
            index = cursor[0] % count
            cursor[0] += 1
            return F.linear(x, weights[index])

        base_median, _ = time_graph(cublas, args.repeats, args.replays)
        print(f"\n## {n}x{k} rows={m} buffers={count}\ncublas: {base_median:.2f}us")

        best: tuple[float, SkinnyGemmConfig] | None = None
        for config in candidates(n, k, seeds):
            config = SkinnyGemmConfig(
                m,
                config.block_size,
                config.outputs_per_block,
                config.k_unroll,
                config.vector_width,
                config.static_k,
            )
            try:
                output = shape_dynamic_skinny_gemm(x, weights[0], config, None)
            except Exception as error:  # noqa: BLE001 - a config may not compile
                print(f"  {config} unsupported: {type(error).__name__}")
                continue
            cosine = F.cosine_similarity(
                output.float().flatten(), reference.float().flatten(), dim=0
            ).item()
            if cosine < args.min_cosine:
                print(f"  {config} wrong, cosine={cosine:.6f}")
                continue
            local_cursor = [0]

            def kernel(config=config, local_cursor=local_cursor):
                index = local_cursor[0] % count
                local_cursor[0] += 1
                return shape_dynamic_skinny_gemm(x, weights[index], config, None)

            median, _ = time_graph(kernel, args.repeats, args.replays)
            print(f"  {config} {median:.2f}us speedup={base_median / median:.2f}x")
            if best is None or median < best[0]:
                best = (median, config)

        if best is not None and base_median / best[0] > KEEP_RATIO:
            median, config = best
            print(
                f"KEEP {n}x{k} rows={m}: {config} {median:.2f}us "
                f"speedup={base_median / median:.2f}x"
            )
            out_lines.append(f"        {m}: {config},")
        elif best is not None:
            print(f"drop {n}x{k} rows={m}: best is only {base_median / best[0]:.2f}x")
        del x, weights, reference
        torch.accelerator.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default=",".join(f"{n}x{k}" for n, k in GLM53_TP8_SHAPES),
        help="comma separated NxK weights to measure",
    )
    parser.add_argument("--rows", default="1,2,4,8", help="token counts to measure")
    parser.add_argument("--config", type=parse_config, action="append", default=None)
    parser.add_argument("--cache-multiplier", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=4, help="weight buffers to rotate")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--out", type=str, default="", help="write a plan snippet here")
    args = parser.parse_args()

    seeds = tuple(args.config) if args.config else DEFAULT_CONFIGS
    rows = [int(row) for row in args.rows.split(",")]
    out_lines: list[str] = []
    for shape in args.shapes.split(","):
        measure_shape(*parse_shape(shape), rows, seeds, args, out_lines)

    if args.out and out_lines:
        with open(args.out, "w") as handle:
            handle.write("\n".join(out_lines) + "\n")
        print(f"\nplan entries written to {args.out}")


if __name__ == "__main__":
    main()
