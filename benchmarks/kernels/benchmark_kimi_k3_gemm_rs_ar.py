# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM100 Kimi-K3 GEMM-RS/AR kernel.

All ranks must belong to one NVLink domain. For example, run a TP8 sweep with:

    torchrun --nproc-per-node=8 \
        benchmarks/kernels/benchmark_kimi_k3_gemm_rs_ar.py
"""

import argparse
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass

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
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar import GemmRsAr

# Shared-expert down-proj and attention O-proj.
_KIMI_K3_PROJECTION_K = (6144, 12288)


@dataclass
class Candidate:
    name: str
    runs: list[Callable[[], torch.Tensor]]
    check_correctness: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("rs", "ar"),
        default="rs",
        help="Collective mode to benchmark.",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 8192, 32768],
        help="Global token counts to benchmark.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        help=(
            "Per-rank input dimensions. By default, derive the Kimi-K3 "
            "shared-expert down-proj and O-proj dimensions from the TP "
            "world size."
        ),
    )
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument(
        "--num-workspaces",
        type=int,
        default=10,
        help="Pointer-distinct inputs and CUDA graphs to rotate.",
    )
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    return parser.parse_args()


def capture_graph(
    op: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor | None]]:
    result: list[torch.Tensor | None] = [None]
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            result[0] = op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result[0] = op()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph, result


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

            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
            timings[name].append(elapsed.item())
    return {name: statistics.median(values) for name, values in timings.items()}


def valid_rows(M: int, local_M: int, rank: int) -> int:
    return min(max(M - rank * local_M, 0), local_M)


def benchmark_shape(
    gemm_rs_ar: GemmRsAr,
    mode: str,
    M: int,
    N: int,
    K: int,
    num_workspaces: int,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float | int | str]:
    all_reduce = mode == "ar"
    world_size = dist.get_world_size(device_group)
    rank = dist.get_rank(device_group)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    padded_M = (M + world_size - 1) // world_size * world_size
    local_M = padded_M // world_size

    rng = torch.Generator(device=device)
    rng.manual_seed(1000 + rank * 10 + M + K)
    inputs = [
        torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=rng)
        for _ in range(num_workspaces)
    ]
    weights = [
        torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=rng)
        for _ in range(num_workspaces)
    ]

    partial = torch.empty((padded_M, N), dtype=torch.bfloat16, device=device)
    symm_partial = symm_mem.empty((padded_M, N), dtype=torch.bfloat16, device=device)
    symm_partial_handle = symm_mem.rendezvous(symm_partial, device_group)
    collective_inputs = [torch.empty_like(partial) for _ in range(num_workspaces)]
    symm_collective_inputs = []
    symm_collective_handles = []
    for _ in range(num_workspaces):
        collective_input = symm_mem.empty(
            (padded_M, N),
            dtype=torch.bfloat16,
            device=device,
        )
        symm_collective_inputs.append(collective_input)
        symm_collective_handles.append(
            symm_mem.rendezvous(collective_input, device_group)
        )
    torch_output = torch.empty((local_M, N), dtype=torch.bfloat16, device=device)
    symm_output = torch.empty_like(torch_output)
    gemm_output = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    if padded_M > M:
        partial[M:].zero_()
        symm_partial[M:].zero_()

    def make_torch_ring_ll_gemm_collective(
        x: torch.Tensor, weight: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            torch.mm(x, weight.T, out=partial[:M])
            if all_reduce:
                dist.all_reduce(partial, group=device_group)
                return partial[:M]
            dist.reduce_scatter_single(torch_output, partial, group=device_group)
            return torch_output

        return run

    def make_torch_ldmc_gemm_collective(
        x: torch.Tensor, weight: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            torch.mm(x, weight.T, out=symm_partial[:M])
            if all_reduce:
                dist.all_reduce(symm_partial, group=device_group)
                return symm_partial[:M]
            dist.reduce_scatter_single(
                symm_output,
                symm_partial,
                group=device_group,
            )
            return symm_output

        return run

    def make_fused_gemm_collective(
        x: torch.Tensor, weight: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return gemm_rs_ar(x, weight)

        return run

    def make_torch_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return torch.mm(x, weight.T, out=gemm_output)

        return run

    def make_ring_ll_collective(
        collective_input: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            if all_reduce:
                dist.all_reduce(collective_input, group=device_group)
                return collective_input
            dist.reduce_scatter_single(
                torch_output, collective_input, group=device_group
            )
            return torch_output

        return run

    def make_ldmc_collective(
        collective_input: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            if all_reduce:
                dist.all_reduce(collective_input, group=device_group)
                return collective_input
            dist.reduce_scatter_single(
                symm_output, collective_input, group=device_group
            )
            return symm_output

        return run

    candidates = (
        Candidate(
            "ring_ll_us",
            [make_torch_ring_ll_gemm_collective(x, w) for x, w in zip(inputs, weights)],
        ),
        Candidate(
            "ldmc_us",
            [make_torch_ldmc_gemm_collective(x, w) for x, w in zip(inputs, weights)],
        ),
        Candidate(
            "gemm_rs_ar_us",
            [make_fused_gemm_collective(x, w) for x, w in zip(inputs, weights)],
        ),
        Candidate(
            "torch_gemm_us",
            [make_torch_gemm(x, w) for x, w in zip(inputs, weights)],
            check_correctness=False,
        ),
        Candidate(
            "ring_ll_collective_us",
            [make_ring_ll_collective(x) for x in collective_inputs],
            check_correctness=False,
        ),
        Candidate(
            "ldmc_collective_us",
            [make_ldmc_collective(x) for x in symm_collective_inputs],
            check_correctness=False,
        ),
    )

    expected = candidates[0].runs[0]()
    rows = M if all_reduce else valid_rows(M, local_M, rank)
    for candidate in candidates[1:]:
        if not candidate.check_correctness:
            continue
        actual = candidate.runs[0]()
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(
            actual[:rows],
            expected[:rows],
            rtol=5e-2,
            atol=4.0,
        )

    candidate_graphs = {}
    graph_keepalive: list[object] = [symm_partial_handle, *symm_collective_handles]
    for candidate in candidates:
        stream = torch.cuda.Stream()
        bundles = [capture_graph(run, stream, cpu_group) for run in candidate.runs]
        candidate_graphs[candidate.name] = [graph for graph, _ in bundles]
        graph_keepalive.extend(bundles)
        graph_keepalive.append(stream)

    times = benchmark_graphs(
        candidate_graphs,
        warmup_replays,
        samples,
        device_group,
        device_barrier,
    )

    best_nccl_us = min(times["ring_ll_collective_us"], times["ldmc_collective_us"])
    return {
        "mode": mode.upper(),
        "M": M,
        "N": N,
        "K": K,
        **times,
        "best_nccl_us": best_nccl_us,
        "speedup_vs_ring_ll": times["ring_ll_us"] / times["gemm_rs_ar_us"],
        "speedup_vs_ldmc": times["ldmc_us"] / times["gemm_rs_ar_us"],
    }


def print_results(results: list[dict[str, float | int | str]]) -> None:
    results_df = pd.DataFrame(results)
    collective = str(results_df["mode"].iloc[0])
    end_to_end = (
        results_df[
            [
                "M",
                "N",
                "K",
                "ring_ll_us",
                "ldmc_us",
                "gemm_rs_ar_us",
                "speedup_vs_ring_ll",
                "speedup_vs_ldmc",
            ]
        ]
        .rename(
            columns={
                "ring_ll_us": f"Torch GEMM + NCCL {collective} (RING_LL) (us)",
                "ldmc_us": f"Torch GEMM + NCCL {collective} (LDMC) (us)",
                "gemm_rs_ar_us": f"GEMM-{collective} (us)",
                "speedup_vs_ring_ll": "Speedup vs RING_LL",
                "speedup_vs_ldmc": "Speedup vs LDMC",
            }
        )
        .round(3)
    )
    components = (
        results_df[["M", "N", "K", "torch_gemm_us", "best_nccl_us", "gemm_rs_ar_us"]]
        .rename(
            columns={
                "torch_gemm_us": "Torch GEMM (us)",
                "best_nccl_us": f"NCCL {collective} (best) (us)",
                "gemm_rs_ar_us": f"GEMM-{collective} (us)",
            }
        )
        .round(2)
    )

    print(f"### {collective} end-to-end latency")
    print(end_to_end.to_markdown(index=False))
    print(f"\n### {collective} component latency")
    print(components.to_markdown(index=False))
    print(
        f"\nNCCL {collective} (best) is the faster of RING_LL and LDMC "
        "for each shape.\n"
    )


def main() -> None:
    args = parse_args()
    assert args.m and min(args.m) > 0
    assert args.n % 256 == 0
    assert args.num_workspaces > 0
    assert args.warmup_replays >= 0
    assert args.samples > 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    world_size = dist.get_world_size()
    if args.k is None:
        assert all(K % world_size == 0 for K in _KIMI_K3_PROJECTION_K)
        K_values = [K // world_size for K in _KIMI_K3_PROJECTION_K]
    else:
        K_values = args.k
    assert all(K % 64 == 0 for K in K_values)
    # Reserve symmetric memory for the NCCL-managed benchmark allocations.
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    # NCCL-managed symmetric allocations select the NVLS/LDMC collective path.
    symm_mem.set_backend("NCCL")
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    group_warmup = torch.zeros(1, device=torch.accelerator.current_device_index())
    dist.all_reduce(group_warmup, group=tp_group.device_group)
    pynccl_comm = tp_group.device_communicator.pynccl_comm
    assert pynccl_comm is not None
    sync_input = torch.zeros(1, device=torch.accelerator.current_device_index())
    sync_output = torch.empty_like(sync_input)

    def device_barrier() -> None:
        # Order the timed launch after a device-side rank rendezvous without
        # including the rendezvous itself in the measured event interval.
        pynccl_comm.all_reduce(sync_input, sync_output)

    gemm_rs_ar = GemmRsAr(
        max_M=max(args.m),
        N=args.n,
        all_reduce=args.mode == "ar",
    )
    results = [
        benchmark_shape(
            gemm_rs_ar,
            args.mode,
            M,
            args.n,
            K,
            args.num_workspaces,
            args.warmup_replays,
            args.samples,
            tp_group.device_group,
            tp_group.cpu_group,
            device_barrier,
        )
        for K in K_values
        for M in args.m
    ]
    del gemm_rs_ar

    if tp_group.rank_in_group == 0:
        print_results(results)

    dist.barrier(group=tp_group.cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
