# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the Kimi-K3 GEMM-AR with FlashInfer's two-shot GEMM-AR.

Times are measured from pointer-distinct CUDA graphs and reported as the median
of the maximum latency across TP ranks. Candidate order is alternated, with a
device-side barrier before every replay.
Run on one NVLink domain, for example:

    torchrun --nproc-per-node=4 workspace/benchmark_gemm_ar_compare.py
"""

import argparse
import os
import statistics
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack
from flashinfer.cute_dsl.gemm_allreduce_two_shot import PersistentDenseGemmKernel

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar import GemmRsAr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m", type=int, nargs="+", default=[128, 256, 512, 1024, 4096, 16384]
    )
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--num-workspaces", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    return parser.parse_args()


def as_cute_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    leading_dim: int | None = 1,
) -> cute.Tensor:
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = dtype
    if leading_dim is None:
        return result.mark_layout_dynamic()
    return result.mark_layout_dynamic(leading_dim=leading_dim)


class FlashInferGemmAR:
    """Minimal direct wrapper around FlashInfer's two-shot GEMM-AR kernel."""

    def __init__(self, M: int, N: int, K: int, group: dist.ProcessGroup) -> None:
        self.M = M
        self.N = N
        self.K = K
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.group = group
        self.input = torch.empty((M, K), dtype=torch.bfloat16, device=self.device)
        self.input_cute = as_cute_tensor(self.input.unsqueeze(-1), cutlass.BFloat16)

        self.output = symm_mem.empty(
            (M, N, 1), dtype=torch.bfloat16, device=self.device
        )
        self.output_handle = symm_mem.rendezvous(self.output, group)
        assert self.output_handle.multicast_ptr != 0
        output_mc = cutlass_torch.as_tensor(
            self.output_handle.multicast_ptr, self.output.shape, self.output.dtype
        )

        use_2cta = M % 256 == 0
        mma_tiler = (256, 256) if use_2cta else (128, 256)
        cluster_shape = (2, 1) if use_2cta else (1, 1)
        cta_tile_m = mma_tiler[0] // (2 if use_2cta else 1)
        num_clusters_m = cute.ceil_div(M, cta_tile_m * cluster_shape[0])
        num_clusters_n = cute.ceil_div(N, mma_tiler[1] * cluster_shape[1])
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        num_flags = (
            num_clusters_m * num_clusters_n * cluster_shape[0] * cluster_shape[1]
            + num_sms
        )
        self.flags = symm_mem.empty(num_flags, dtype=torch.int32, device=self.device)
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, group)
        assert self.flags_handle.multicast_ptr != 0
        flags_mc = cutlass_torch.as_tensor(
            self.flags_handle.multicast_ptr, self.flags.shape, self.flags.dtype
        )

        self.output_cute = as_cute_tensor(self.output, cutlass.BFloat16)
        self.output_mc_cute = as_cute_tensor(output_mc, cutlass.BFloat16)
        self.generation_barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.flags_cute = as_cute_tensor(self.flags, cutlass.Int32, leading_dim=None)
        self.flags_mc_cute = as_cute_tensor(flags_mc, cutlass.Int32, leading_dim=None)
        self.kernel = PersistentDenseGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster_shape,
            True,
            all_reduce="two_shot",
            sm_version="sm_103",
        )
        self.max_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            cluster_shape[0] * cluster_shape[1]
        )
        self.compiled = None

    def compile(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled = cute.compile(
            self.kernel,
            self.input_cute,
            as_cute_tensor(weight.unsqueeze(-1), cutlass.BFloat16),
            self.output_cute,
            self.max_clusters,
            stream,
            c_mc=self.output_mc_cute,
            barrier_flag=self.flags_cute,
            barrier_flag_mc=self.flags_mc_cute,
        )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        assert self.compiled is not None
        self.input.copy_(x)
        self.flags.zero_()
        dist.all_reduce(self.generation_barrier, group=self.group)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled(
            self.input_cute,
            as_cute_tensor(weight.unsqueeze(-1), cutlass.BFloat16),
            self.output_cute,
            stream,
            c_mc=self.output_mc_cute,
            barrier_flag=self.flags_cute,
            barrier_flag_mc=self.flags_mc_cute,
        )
        return self.output[:, :, 0].clone()


def capture_graph(
    op: Callable[[], torch.Tensor],
    prepare: Callable[[], None],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor | None]]:
    result: list[torch.Tensor | None] = [None]
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            prepare()
            result[0] = op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    with torch.cuda.stream(stream):
        prepare()
    stream.synchronize()
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result[0] = op()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph, result


def benchmark_graphs(
    candidates: dict[str, list[torch.cuda.CUDAGraph]],
    prepare: dict[str, Callable[[], None]],
    warmups: int,
    samples: int,
    group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float]:
    names = list(candidates)
    for round_index in range(warmups):
        for candidate_index in range(len(names)):
            name = names[(round_index + candidate_index) % len(names)]
            prepare[name]()
            device_barrier()
            candidates[name][round_index % len(candidates[name])].replay()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: dict[str, list[float]] = {name: [] for name in names}
    for sample_index in range(samples):
        for candidate_index in range(len(names)):
            name = names[(sample_index + candidate_index) % len(names)]
            graph = candidates[name][sample_index % len(candidates[name])]
            prepare[name]()
            device_barrier()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
            timings[name].append(elapsed.item())
    return {name: statistics.median(values) for name, values in timings.items()}


def benchmark_shape(
    gemm_ar: GemmRsAr,
    M: int,
    N: int,
    K: int,
    num_workspaces: int,
    warmups: int,
    samples: int,
    group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float | int]:
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + rank)
    inputs = [
        torch.randn(
            M, K, dtype=torch.bfloat16, device=device, generator=generator
        ).mul_(0.25)
        for _ in range(num_workspaces)
    ]
    weights = [
        torch.randn(
            N, K, dtype=torch.bfloat16, device=device, generator=generator
        ).mul_(0.25)
        for _ in range(num_workspaces)
    ]

    flashinfer_ar = FlashInferGemmAR(M, N, K, group)
    if rank == 0:
        print(f"Compiling kernels for M={M}...", flush=True)
    flashinfer_ar.compile(inputs[0], weights[0])
    gemm_ar(inputs[0], weights[0])
    flashinfer_ar(inputs[0], weights[0])
    torch.accelerator.synchronize(device)

    expected = torch.mm(inputs[0], weights[0].T)
    dist.all_reduce(expected, group=group)
    actual_vllm = gemm_ar(inputs[0], weights[0])
    actual_flashinfer = flashinfer_ar(inputs[0], weights[0])
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(actual_vllm, expected, rtol=5e-2, atol=1.0)
    torch.testing.assert_close(actual_flashinfer, expected, rtol=5e-2, atol=1.0)

    candidate_runs = {
        "vllm_us": [
            lambda x=x, weight=weight: gemm_ar(x, weight)
            for x, weight in zip(inputs, weights)
        ],
        "flashinfer_us": [
            lambda x=x, weight=weight: flashinfer_ar(x, weight)
            for x, weight in zip(inputs, weights)
        ],
    }
    candidate_prepare = {
        "vllm_us": lambda: None,
        "flashinfer_us": lambda: None,
    }
    candidate_graphs = {}
    graph_keepalive: list[object] = []
    for name, runs in candidate_runs.items():
        stream = torch.cuda.Stream()
        bundles = [
            capture_graph(run, candidate_prepare[name], stream, cpu_group)
            for run in runs
        ]
        candidate_graphs[name] = [graph for graph, _ in bundles]
        graph_keepalive.extend(bundles)
        graph_keepalive.append(stream)

    timings = benchmark_graphs(
        candidate_graphs,
        candidate_prepare,
        warmups,
        samples,
        group,
        device_barrier,
    )
    return {
        "M": M,
        **timings,
        "vllm_speedup": timings["flashinfer_us"] / timings["vllm_us"],
    }


def main() -> None:
    args = parse_args()
    assert args.m and min(args.m) >= 128
    assert all(M % 128 == 0 for M in args.m)
    assert args.n % 256 == 0 and args.k % 64 == 0
    assert args.num_workspaces > 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    symm_mem.set_backend("NCCL")
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    tp_group = get_tp_group()
    group_warmup = torch.zeros(1, device=torch.accelerator.current_device_index())
    dist.all_reduce(group_warmup, group=tp_group.device_group)
    pynccl_comm = tp_group.device_communicator.pynccl_comm
    assert pynccl_comm is not None
    sync_input = torch.zeros(1, device=torch.accelerator.current_device_index())
    sync_output = torch.empty_like(sync_input)

    def device_barrier() -> None:
        pynccl_comm.all_reduce(sync_input, sync_output)

    gemm_ar = GemmRsAr(max_M=max(args.m), N=args.n, all_reduce=True)
    results = [
        benchmark_shape(
            gemm_ar,
            M,
            args.n,
            args.k,
            args.num_workspaces,
            args.warmups,
            args.samples,
            tp_group.device_group,
            tp_group.cpu_group,
            device_barrier,
        )
        for M in args.m
    ]
    if tp_group.rank_in_group == 0:
        table = pd.DataFrame(results).rename(
            columns={
                "M": "M",
                "vllm_us": "vLLM (us)",
                "flashinfer_us": "FlashInfer (us)",
                "vllm_speedup": "vLLM speedup",
            }
        )
        print(f"TP={dist.get_world_size()}, N={args.n}, K/rank={args.k}")
        print(
            table.round(
                {"vLLM (us)": 2, "FlashInfer (us)": 2, "vLLM speedup": 3}
            ).to_markdown(index=False)
        )

    dist.barrier(group=tp_group.cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
