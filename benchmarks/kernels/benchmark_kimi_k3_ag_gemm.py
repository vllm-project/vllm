# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Kimi-K3 AG-GEMM path against explicit AG followed by GEMM.

The default shapes are the Kimi-K3 KDA and MLA input projections for the
selected TP size. All ranks must belong to one NVLink domain. For example:

    torchrun --nproc-per-node=8 \
        benchmarks/kernels/benchmark_kimi_k3_ag_gemm.py
"""

import argparse
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.kimi_k3.nvidia.ops.ag_gemm import AgGemm

_KIMI_K3_HIDDEN_SIZE = 7168
_KIMI_K3_NUM_HEADS = 96
_KIMI_K3_HEAD_DIM = 128
_KIMI_K3_Q_LORA_RANK = 1536
_KIMI_K3_KV_LORA_RANK = 512
_KIMI_K3_QK_ROPE_HEAD_DIM = 64
_KIMI_K3_V_HEAD_DIM = 128
_PROJECTIONS = ("kda", "mla")
_LOCAL_M_SWEEP = [256, 384, 448, 512, 1024, 2048, 4096, 8192]
_MAX_DEFAULT_M = 32768


@dataclass
class Candidate:
    name: str
    runs: list[Callable[[], torch.Tensor]]
    check_correctness: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        help=(
            "Global token counts to benchmark. By default, sweep per-rank "
            "counts densely around the crossover, up to 32768 global tokens."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=_PROJECTIONS,
        nargs="+",
        default=list(_PROJECTIONS),
    )
    parser.add_argument("--hidden-size", type=int, default=_KIMI_K3_HIDDEN_SIZE)
    parser.add_argument(
        "--num-workspaces",
        type=int,
        default=5,
        help="Pointer-distinct inputs and CUDA graphs to rotate.",
    )
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    return parser.parse_args()


def projection_n(projection: str, world_size: int) -> int:
    assert _KIMI_K3_NUM_HEADS % world_size == 0
    if projection == "kda":
        projection_size = _KIMI_K3_NUM_HEADS * _KIMI_K3_HEAD_DIM
        local_projection_size = projection_size // world_size
        local_num_heads = _KIMI_K3_NUM_HEADS // world_size
        output_size = 4 * local_projection_size + _KIMI_K3_HEAD_DIM + local_num_heads
        return output_size + (-output_size % 16)
    assert projection == "mla"
    return (
        _KIMI_K3_Q_LORA_RANK
        + _KIMI_K3_KV_LORA_RANK
        + _KIMI_K3_QK_ROPE_HEAD_DIM
        + _KIMI_K3_NUM_HEADS * _KIMI_K3_V_HEAD_DIM // world_size
    )


def capture_graph(
    op: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor | None]]:
    result: list[torch.Tensor | None] = [None]
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        device_barrier()
        with torch.cuda.stream(stream):
            result[0] = op()
        stream.synchronize()
    device_barrier()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result[0] = op()
    stream.synchronize()
    device_barrier()
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


def benchmark_shape(
    ag_gemm: AgGemm,
    projection: str,
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
    rank = dist.get_rank(device_group)
    world_size = dist.get_world_size(device_group)
    local_M = (M + world_size - 1) // world_size
    padded_M = local_M * world_size
    device = torch.device("cuda", torch.accelerator.current_device_index())
    rng = torch.Generator(device=device)
    rng.manual_seed(1000 + rank * 10 + local_M + N)
    inputs = [
        torch.randn(local_M, K, dtype=torch.bfloat16, device=device, generator=rng)
        for _ in range(num_workspaces)
    ]
    gemm_inputs = [
        torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=rng)
        for _ in range(num_workspaces)
    ]
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=rng)

    def make_explicit(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            gathered = sp_all_gather(x)[:M]
            return torch.mm(gathered, weight.T)

        return run

    def make_fused(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return ag_gemm(x, weight)[:M]

        return run

    def make_torch_gemm(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return torch.mm(x, weight.T)

        return run

    def make_all_gather(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            return sp_all_gather(x)[:M]

        return run

    candidates = (
        Candidate("explicit_ag_gemm_us", [make_explicit(x) for x in inputs]),
        Candidate("fused_ag_gemm_us", [make_fused(x) for x in inputs]),
        Candidate(
            "torch_gemm_us",
            [make_torch_gemm(x) for x in gemm_inputs],
            check_correctness=False,
        ),
        Candidate(
            "all_gather_us",
            [make_all_gather(x) for x in inputs],
            check_correctness=False,
        ),
    )

    expected = candidates[0].runs[0]()
    for candidate in candidates[1:]:
        if not candidate.check_correctness:
            continue
        torch.accelerator.synchronize()
        device_barrier()
        actual = candidate.runs[0]()
        torch.accelerator.synchronize()
        device_barrier()
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=4.0)

    candidate_graphs = {}
    graph_keepalive: list[object] = []
    for candidate in candidates:
        stream = torch.cuda.Stream()
        bundles = [
            capture_graph(run, stream, cpu_group, device_barrier)
            for run in candidate.runs
        ]
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
    return {
        "projection": projection,
        "M": M,
        "padded_M": padded_M,
        "local_M": local_M,
        "N": N,
        "K": K,
        **times,
        "speedup": times["explicit_ag_gemm_us"] / times["fused_ag_gemm_us"],
    }


def print_results(results: list[dict[str, float | int | str]]) -> None:
    results_df = pd.DataFrame(results)
    end_to_end = results_df[
        [
            "projection",
            "M",
            "padded_M",
            "local_M",
            "N",
            "K",
            "explicit_ag_gemm_us",
            "fused_ag_gemm_us",
            "speedup",
        ]
    ].rename(
        columns={
            "explicit_ag_gemm_us": "Explicit AG + GEMM (us)",
            "fused_ag_gemm_us": "Fused AG-GEMM (us)",
            "speedup": "Fused speedup",
        }
    )
    end_to_end = end_to_end.round(
        {
            "Explicit AG + GEMM (us)": 2,
            "Fused AG-GEMM (us)": 2,
            "Fused speedup": 3,
        }
    )

    print("### End-to-end latency")
    print(end_to_end.to_markdown(index=False))

    components = results_df[
        [
            "projection",
            "M",
            "local_M",
            "N",
            "K",
            "torch_gemm_us",
            "all_gather_us",
            "fused_ag_gemm_us",
        ]
    ].rename(
        columns={
            "torch_gemm_us": "Torch GEMM (us)",
            "all_gather_us": "AG (us)",
            "fused_ag_gemm_us": "AG-GEMM (us)",
        }
    )
    components = components.round(2)

    print("\n### Component latency")
    print(components.to_markdown(index=False))


def main() -> None:
    args = parse_args()
    assert args.m is None or (args.m and min(args.m) > 0)
    assert args.hidden_size % 64 == 0
    assert args.num_workspaces > 0
    assert args.warmup_replays >= 0
    assert args.samples > 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    world_size = dist.get_world_size()
    M_values = args.m or [
        local_M * world_size
        for local_M in _LOCAL_M_SWEEP
        if local_M * world_size <= _MAX_DEFAULT_M
    ]
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
        pynccl_comm.all_reduce(sync_input, sync_output)

    ag_gemm = AgGemm(
        max_global_tokens=max(M_values),
        hidden_size=args.hidden_size,
    )
    results = [
        benchmark_shape(
            ag_gemm,
            projection,
            M,
            projection_n(projection, world_size),
            args.hidden_size,
            args.num_workspaces,
            args.warmup_replays,
            args.samples,
            tp_group.device_group,
            tp_group.cpu_group,
            device_barrier,
        )
        for projection in args.projection
        for M in M_values
    ]

    if tp_group.rank_in_group == 0:
        print_results(results)

    dist.barrier(group=tp_group.cpu_group)
    del ag_gemm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
