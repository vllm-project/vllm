# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark for the Qwen3.5 GDN block-FP8 projection."""

import argparse
import functools
import json
import statistics

import torch

from benchmarks.kernels.benchmark_w8a8_block_fp8 import w8a8_block_matmul
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import cutlass_scaled_mm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_block_fp8_configs,
)

M_VALUES = (1, 24, 128, 2048, 4096, 8192)
N = 32
K = 1024
BLOCK_SIZE = [128, 128]
GENERIC_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def bench(fn, *, warmup: int, repetitions: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    return statistics.median(samples)


def capture(fn):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.accelerator.synchronize()
    return graph.replay


def make_inputs(m: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    a = (
        ((torch.rand((m, K), generator=generator, device="cuda") - 0.5) * 2 * fp8_max)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    b = (
        ((torch.rand((N, K), generator=generator, device="cuda") - 0.5) * 2 * fp8_max)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    a_scales = torch.rand((m, K // 128), generator=generator, device="cuda") * 0.01
    a_scales_col_major = a_scales.t().contiguous().t()
    b_scales = torch.rand((1, K // 128), generator=generator, device="cuda") * 0.01
    return a, b, a_scales, a_scales_col_major, b_scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--skip-cutlass", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    configs = get_w8a8_block_fp8_configs(N, K, 128, 128)
    assert configs is not None
    bench_kwargs = {
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rounds": args.rounds,
    }
    rows = []
    for m in M_VALUES:
        a, b, a_scales, a_scales_col_major, b_scales = make_inputs(m)
        tuned_config = configs[min(configs, key=lambda size: abs(size - m))]
        tuned = functools.partial(
            w8a8_block_matmul,
            a,
            b,
            a_scales,
            b_scales,
            BLOCK_SIZE,
            tuned_config,
            torch.bfloat16,
        )
        generic = functools.partial(
            w8a8_block_matmul,
            a,
            b,
            a_scales,
            b_scales,
            BLOCK_SIZE,
            GENERIC_CONFIG,
            torch.bfloat16,
        )
        tuned_output = tuned()
        generic_output = generic()
        torch.testing.assert_close(tuned_output, generic_output, rtol=0, atol=0)
        tuned_bench = capture(tuned) if args.cuda_graph else tuned
        generic_bench = capture(generic) if args.cuda_graph else generic

        row = {
            "M": m,
            "tuned_us": bench(tuned_bench, **bench_kwargs),
            "generic_us": bench(generic_bench, **bench_kwargs),
        }
        row["tuned_over_generic"] = row["generic_us"] / row["tuned_us"]

        if not args.skip_cutlass:
            cutlass = functools.partial(
                cutlass_scaled_mm,
                a,
                b,
                a_scales_col_major,
                b_scales,
                BLOCK_SIZE,
                torch.bfloat16,
            )
            cutlass_output = cutlass()
            torch.testing.assert_close(
                tuned_output, cutlass_output, rtol=0.05, atol=0.5
            )
            cutlass_bench = capture(cutlass) if args.cuda_graph else cutlass
            row["cutlass_us"] = bench(cutlass_bench, **bench_kwargs)
            row["tuned_over_cutlass"] = row["cutlass_us"] / row["tuned_us"]
        rows.append(row)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
