# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal multi-GPU GEMM + reduce-scatter microbenchmark.

Run on one node with, for example:

    torchrun --standalone --nproc-per-node=8 \
        .agents/skills/kernel-microbenchmark/benchmarks/multi_gpu_gemm_rs.py
"""

import argparse
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096, help="Per-rank K")
    parser.add_argument("--num-workspaces", type=int, default=4)
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    return parser.parse_args()


def make_gemm_rs(
    x: torch.Tensor,
    weight: torch.Tensor,
    partial: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    device_group: dist.ProcessGroup,
) -> Callable[[], None]:
    def run() -> None:
        torch.mm(x, weight.T, out=partial[:rows])
        dist.reduce_scatter_tensor(output, partial, group=device_group)

    return run


def check_correctness(
    run: Callable[[], None],
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    padded_rows: int,
    rank: int,
    device_group: dist.ProcessGroup,
) -> None:
    run()
    torch.accelerator.synchronize()
    actual = output.clone()

    expected_full = torch.zeros(
        (padded_rows, weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    torch.mm(x, weight.T, out=expected_full[: x.shape[0]])
    dist.all_reduce(expected_full, group=device_group)
    expected = expected_full.chunk(dist.get_world_size(device_group))[rank]
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=4.0)


def capture_graph(
    run: Callable[[], None],
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph, stream


def benchmark_graphs(
    graphs: list[torch.cuda.CUDAGraph],
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
) -> float:
    for index in range(warmup_replays):
        dist.barrier(group=device_group)
        graphs[index % len(graphs)].replay()
    torch.accelerator.synchronize()

    timings_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(samples):
        dist.barrier(group=device_group)
        start.record()
        graphs[index % len(graphs)].replay()
        end.record()
        end.synchronize()

        elapsed_us = torch.tensor(
            start.elapsed_time(end) * 1000,
            dtype=torch.float64,
            device=torch.accelerator.current_device_index(),
        )
        dist.all_reduce(elapsed_us, op=dist.ReduceOp.MAX, group=device_group)
        timings_us.append(elapsed_us.item())
    return statistics.median(timings_us)


def main() -> None:
    args = parse_args()
    assert min(args.m, args.n, args.k, args.num_workspaces, args.samples) > 0
    assert args.warmup_replays >= 0
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl")
    device_group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")

    rank = dist.get_rank(device_group)
    world_size = dist.get_world_size(device_group)
    padded_m = (args.m + world_size - 1) // world_size * world_size
    local_m = padded_m // world_size
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1000 + rank)
    workspaces = []
    runs = []
    for _ in range(args.num_workspaces):
        x = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
        weight = torch.randn(args.n, args.k, dtype=torch.bfloat16, device=device)
        partial = torch.zeros(padded_m, args.n, dtype=torch.bfloat16, device=device)
        output = torch.empty(local_m, args.n, dtype=torch.bfloat16, device=device)
        run = make_gemm_rs(
            x,
            weight,
            partial,
            output,
            args.m,
            device_group,
        )
        workspaces.append((x, weight, partial, output))
        runs.append(run)

    x, weight, _, output = workspaces[0]
    check_correctness(
        runs[0],
        output,
        x,
        weight,
        padded_m,
        rank,
        device_group,
    )

    graph_bundles = [capture_graph(run, cpu_group) for run in runs]
    graphs = [graph for graph, _ in graph_bundles]
    latency_us = benchmark_graphs(
        graphs,
        args.warmup_replays,
        args.samples,
        device_group,
    )
    global_tflops = 2 * args.m * args.n * args.k * world_size / (latency_us * 1e6)
    if rank == 0:
        print(
            {
                "world_size": world_size,
                "local_world_size": local_world_size,
                "num_nodes": world_size // local_world_size,
                "backend": dist.get_backend(device_group),
                "gpu": torch.cuda.get_device_name(local_rank),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "M": args.m,
                "N": args.n,
                "K_per_rank": args.k,
                "K_global": args.k * world_size,
                "latency_us": round(latency_us, 3),
                "global_tflops": round(global_tflops, 3),
            }
        )

    dist.barrier(group=cpu_group)
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
