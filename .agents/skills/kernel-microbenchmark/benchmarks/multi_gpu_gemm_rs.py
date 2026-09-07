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

import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4096],
        help="Per-rank K values",
    )
    parser.add_argument("--num-workspaces", type=int, default=10)
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
        dist.reduce_scatter_single(output, partial, group=device_group)

    return run


def check_correctness(
    runs: dict[str, Callable[[], None]],
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    padded_rows: int,
    rank: int,
    device_group: dist.ProcessGroup,
) -> None:
    expected_full = torch.zeros(
        (padded_rows, weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    torch.mm(x, weight.T, out=expected_full[: x.shape[0]])
    dist.all_reduce(expected_full, group=device_group)
    expected = expected_full.chunk(dist.get_world_size(device_group))[rank]
    for name, run in runs.items():
        run()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            outputs[name],
            expected,
            rtol=5e-2,
            atol=4.0,
        )


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
    candidate_graphs: dict[str, list[torch.cuda.CUDAGraph]],
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float]:
    candidate_names = list(candidate_graphs)
    for round_index in range(warmup_replays):
        for candidate_index in range(len(candidate_names)):
            candidate_id = (round_index + candidate_index) % len(candidate_names)
            name = candidate_names[candidate_id]
            graphs = candidate_graphs[name]
            device_barrier()
            graphs[round_index % len(graphs)].replay()
    torch.accelerator.synchronize()

    timings: dict[str, list[float]] = {name: [] for name in candidate_names}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample_index in range(samples):
        for candidate_index in range(len(candidate_names)):
            candidate_id = (sample_index + candidate_index) % len(candidate_names)
            name = candidate_names[candidate_id]
            graphs = candidate_graphs[name]
            device_barrier()
            start.record()
            graphs[sample_index % len(graphs)].replay()
            end.record()
            end.synchronize()

            elapsed_us = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed_us, op=dist.ReduceOp.MAX, group=device_group)
            timings[name].append(elapsed_us.item())
    return {name: statistics.median(values) for name, values in timings.items()}


def benchmark_shape(
    m: int,
    n: int,
    k: int,
    num_workspaces: int,
    warmup_replays: int,
    samples: int,
    device: torch.device,
    rank: int,
    world_size: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float | int]:
    padded_m = (m + world_size - 1) // world_size * world_size
    local_m = padded_m // world_size

    torch.manual_seed(1000 + rank * 10 + m + k)
    inputs = []
    weights = []
    for _ in range(num_workspaces):
        inputs.append(torch.randn(m, k, dtype=torch.bfloat16, device=device))
        weights.append(torch.randn(n, k, dtype=torch.bfloat16, device=device))

    ring_partial = torch.empty(padded_m, n, dtype=torch.bfloat16, device=device)
    ldmc_partial = symm_mem.empty(
        (padded_m, n),
        dtype=torch.bfloat16,
        device=device,
    )
    ldmc_handle = symm_mem.rendezvous(ldmc_partial, device_group)
    ring_output = torch.empty(local_m, n, dtype=torch.bfloat16, device=device)
    ldmc_output = torch.empty_like(ring_output)
    if padded_m > m:
        ring_partial[m:].zero_()
        ldmc_partial[m:].zero_()

    candidate_runs = {
        "ring_ll_us": [
            make_gemm_rs(
                x,
                weight,
                ring_partial,
                ring_output,
                m,
                device_group,
            )
            for x, weight in zip(inputs, weights)
        ],
        "ldmc_us": [
            make_gemm_rs(
                x,
                weight,
                ldmc_partial,
                ldmc_output,
                m,
                device_group,
            )
            for x, weight in zip(inputs, weights)
        ],
    }
    x = inputs[0]
    weight = weights[0]
    check_correctness(
        {name: runs[0] for name, runs in candidate_runs.items()},
        {"ring_ll_us": ring_output, "ldmc_us": ldmc_output},
        x,
        weight,
        padded_m,
        rank,
        device_group,
    )

    candidate_graphs = {}
    graph_keepalive: list[object] = [ldmc_handle]
    for name, runs in candidate_runs.items():
        bundles = [capture_graph(run, cpu_group) for run in runs]
        candidate_graphs[name] = [graph for graph, _ in bundles]
        graph_keepalive.extend(bundles)

    times = benchmark_graphs(
        candidate_graphs,
        warmup_replays,
        samples,
        device_group,
        device_barrier,
    )
    global_flops = 2 * m * n * k * world_size
    return {
        "M": m,
        "N": n,
        "K_per_rank": k,
        "K_global": k * world_size,
        **times,
        "ring_ll_tflops": global_flops / (times["ring_ll_us"] * 1e6),
        "ldmc_tflops": global_flops / (times["ldmc_us"] * 1e6),
        "ldmc_speedup": times["ring_ll_us"] / times["ldmc_us"],
    }


def main() -> None:
    args = parse_args()
    assert args.m and min(args.m) > 0
    assert args.k and min(args.k) > 0
    assert min(args.n, args.num_workspaces, args.samples) > 0
    assert args.warmup_replays >= 0
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    world_size = dist.get_world_size()
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    symm_mem.set_backend("NCCL")
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    device_group = tp_group.device_group
    cpu_group = tp_group.cpu_group
    rank = tp_group.rank_in_group
    device = torch.device("cuda", local_rank)
    group_warmup = torch.zeros(1, device=device)
    dist.all_reduce(group_warmup, group=device_group)
    pynccl_comm = tp_group.device_communicator.pynccl_comm
    assert pynccl_comm is not None
    sync_input = torch.zeros(1, device=device)
    sync_output = torch.empty_like(sync_input)

    def device_barrier() -> None:
        # Order the timed launch after a device-side rank rendezvous without
        # including the rendezvous itself in the measured event interval.
        pynccl_comm.all_reduce(sync_input, sync_output)

    results = [
        benchmark_shape(
            m,
            args.n,
            k,
            args.num_workspaces,
            args.warmup_replays,
            args.samples,
            device,
            rank,
            world_size,
            device_group,
            cpu_group,
            device_barrier,
        )
        for k in args.k
        for m in args.m
    ]

    if rank == 0:
        metadata = {
            "world_size": world_size,
            "local_world_size": local_world_size,
            "num_nodes": world_size // local_world_size,
            "backend": dist.get_backend(device_group),
            "gpu": torch.cuda.get_device_name(local_rank),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        print(pd.Series(metadata, name="value").to_string())
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    dist.barrier(group=cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
