# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark and tune quantized two-shot all-reduce kernels.

Benchmark mode (default):
    torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_quantized_allreduce.py

Tune mode (saves optimal params to JSON):
    torchrun --nproc_per_node=8 \
        benchmarks/kernels/benchmark_quantized_allreduce.py --tune
"""

import argparse
import functools
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.quantized_allreduce import (
    two_shot_quantized_allreduce,
)
from vllm.distributed.device_communicators.quantized_allreduce.two_shot_quantized_allreduce import (  # noqa: E501
    MAX_BLOCK_SIZE,
)

BLOCK_SIZES = [
    2**k for k in range(10, MAX_BLOCK_SIZE.bit_length()) if 2**k <= MAX_BLOCK_SIZE
]
NUM_WARPS = [4, 8, 16, 32]
SIZES = [2**k for k in range(20, 30)]
CUDA_GRAPH_CAPTURE_CYCLES = 10


def benchmark_fn(fn, warmup=5, iters=50):
    """Benchmark using CUDA graph capture."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(CUDA_GRAPH_CAPTURE_CYCLES):
                fn()

    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.accelerator.synchronize()

    return (time.perf_counter() - start) / iters / CUDA_GRAPH_CAPTURE_CYCLES * 1e6


def run_benchmark(args):
    """Run benchmark comparing quantized vs NCCL."""
    ws = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    for gs in args.group_size:
        if rank == 0:
            print(f"\nBenchmark: world_size={ws}, group_size={gs}")
            print(
                f"{'numel':>12} {'MiB':>8} {'nccl_us':>12}"
                f" {'int8_us':>12} {'speedup':>8}"
                f" {'fp8_us':>12} {'speedup':>8}"
            )
            print("-" * 80)

        for numel in args.sizes:
            msg = torch.randn(numel, dtype=torch.bfloat16, device=device)

            nccl_input = msg.clone()

            def nccl_fn(t=nccl_input):
                dist.all_reduce(t)

            def int8_fn(t=msg, _gs=gs):
                return two_shot_quantized_allreduce(t, group_size=_gs)

            def fp8_fn(t=msg, _gs=gs):
                return two_shot_quantized_allreduce(t, group_size=_gs, use_fp8=True)

            nccl_us = benchmark_fn(
                nccl_fn, warmup=args.num_warmup, iters=args.num_trials
            )
            can_run = numel >= BLOCK_SIZES[0] * ws
            int8_us = (
                benchmark_fn(int8_fn, warmup=args.num_warmup, iters=args.num_trials)
                if can_run
                else float("inf")
            )
            fp8_us = (
                benchmark_fn(fp8_fn, warmup=args.num_warmup, iters=args.num_trials)
                if can_run
                else float("inf")
            )

            if rank == 0:
                mib = numel * 2 / 1048576
                i8s = f"{nccl_us / int8_us:.2f}x" if int8_us < float("inf") else "N/A"
                fps = f"{nccl_us / fp8_us:.2f}x" if fp8_us < float("inf") else "N/A"
                i8 = f"{int8_us:.1f}" if int8_us < float("inf") else "N/A"
                fp = f"{fp8_us:.1f}" if fp8_us < float("inf") else "N/A"
                print(
                    f"{numel:>12} {mib:>8.1f}"
                    f" {nccl_us:>12.1f} {i8:>12}"
                    f" {i8s:>8} {fp:>12} {fps:>8}"
                )

            dist.barrier()


def run_tune(args):
    """Sweep block_size and num_warps, save optimal params."""
    ws = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    optimal = {}
    for use_fp8 in [False, True]:
        kernel_name = "fp8" if use_fp8 else "int8"
        for gs in args.group_size:
            key = f"{kernel_name}_gs{gs}"
            optimal[key] = {}

            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"Tuning {kernel_name} group_size={gs} world_size={ws}")
                print(f"{'=' * 60}")

            for numel in SIZES:
                msg = torch.randn(numel, dtype=torch.bfloat16, device=device)
                golden = msg.clone()
                dist.all_reduce(golden)

                results = {}
                for bs in BLOCK_SIZES:
                    if bs * ws > numel or bs % gs != 0:
                        for nw in NUM_WARPS:
                            results[(bs, nw)] = float("inf")
                        continue
                    for nw in NUM_WARPS:
                        try:
                            fn = functools.partial(
                                two_shot_quantized_allreduce,
                                msg.clone(),
                                torch.empty_like(msg),
                                block_size=bs,
                                group_size=gs,
                                num_warps=nw,
                                use_fp8=use_fp8,
                            )
                            out = fn()
                            torch.accelerator.synchronize()
                            diff = (out.float() - golden.float()).abs().max().item()
                            if diff > 10:
                                results[(bs, nw)] = float("inf")
                                continue
                            t = benchmark_fn(fn)
                            results[(bs, nw)] = t
                        except Exception:
                            results[(bs, nw)] = float("inf")

                best_key = min(results, key=results.get)
                best_time = results[best_key]
                if best_time < float("inf"):
                    optimal[key][str(numel)] = {
                        "BLOCK_SIZE": best_key[0],
                        "num_warps": best_key[1],
                        "time_us": round(best_time, 1),
                    }

                if rank == 0:
                    if best_time < float("inf"):
                        print(
                            f"  numel={numel:>12}  best:"
                            f" bs={best_key[0]},"
                            f" nw={best_key[1]},"
                            f" {best_time:.1f}us"
                        )
                    else:
                        print(f"  numel={numel:>12}  NO VALID CONFIG")

    if rank == 0:
        device_name = torch.cuda.get_device_name().replace(" ", "_")
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "vllm",
            "distributed",
            "device_communicators",
            "quantized_allreduce",
            "configs",
        )
        os.makedirs(out_dir, exist_ok=True)
        for key, params in optimal.items():
            parts = key.split("_")
            kernel = parts[0]
            gs = parts[1]
            filename = (
                f"dtype={kernel},device_name={device_name},world_size={ws},{gs}.json"
            )
            out_path = os.path.join(out_dir, filename)
            out_data = {
                "world_size": ws,
                "dtype": kernel,
                "device_name": device_name,
                "group_size": int(gs.replace("gs", "")),
                "params": params,
            }
            with open(out_path, "w") as fout:
                json.dump(out_data, fout, indent=2)
            print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark/tune quantized allreduce")
    parser.add_argument("--tune", action="store_true", help="Run tuning sweep")
    parser.add_argument("--group-size", type=int, nargs="+", default=[256])
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-trials", type=int, default=50, help="Number of benchmark trials"
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group("nccl", device_id=device)

    if args.tune:
        run_tune(args)
    else:
        run_benchmark(args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
