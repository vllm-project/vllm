# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import weakref
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Tuple

import torch
import torch.distributed as dist
from vllm.platforms import current_platform
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.utils import get_canonical_gpu_name
from vllm.distributed.parallel_state import _groups, GroupCoordinator
from vllm.kernels.helion.distributed.all_gather_gemm_fp8 import helion_all_gather_fp8_gemm
from torch.profiler import profile, ProfilerActivity

FP8_DTYPE = current_platform.fp8_dtype()

try:
    config_manager = ConfigManager.get_instance()
except RuntimeError:
    config_manager = ConfigManager()

platform = get_canonical_gpu_name()
configs = config_manager.get_platform_configs("helion_matmul_w_progress_fp8", platform)
if len(configs) == 0:
    raise RuntimeError(f"Current GPU platform {platform} is not supported for Helion kernel")

@dataclass
class Row:
    shape: str
    baseline_ms: float
    kernel_ms: float
    speedup_x: float
    baseline_peak_mb: float
    kernel_peak_mb: float
    mem_improve_x: float

def save_rows_json(rows, rank=0, out_dir="bench_results"):
    import json, os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"results_rank{rank}.json"), "w") as f:
        json.dump([r.__dict__ for r in rows], f, indent=2)

def print_table(rows: List[Row]) -> None:
    headers = ["shape", "baseline_ms", "kernel_ms", "speedup(x)", "baseline_peak(MB)", "kernel_peak(MB)", "mem_improve(x)"]
    data = [
        [
            r.shape,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
            f"{r.baseline_peak_mb:.2f}",
            f"{r.kernel_peak_mb:.2f}",
            f"{r.mem_improve_x:.3f}",
        ]
        for r in rows
    ]
    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row):
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))

# NOTE: triton.testing.do_bench() is not safe for distributed collectives like
# torch.ops.symm_mem.fused_all_gather_scaled_matmul because it calls local
# torch.cuda.synchronize() inside the timing loop. Use events + pre-iteration barrier.
def do_bench_distributed(
    fn: Callable,
    repeat: int = 50,
    device: Optional[Union[torch.device, int]] = None,
    dist_group: Optional[dist.ProcessGroup] = None,
    return_mode: str = "mean",
    warmup: int = 5,
    post_iteration_barrier: bool = False,
) -> Union[float, List[float]]:
    """
    Distributed-safe benchmark for CUDA kernels.

    - Pre-iteration dist.barrier() aligns ranks before launching collectives.
    - Record start_event, call fn(), record end_event immediately after fn() returns.
    - Call local torch.cuda.synchronize(device) to wait for the GPU work to complete,
      then measure elapsed_time (ms).
    """

    if device is None:
        device = torch.device("cuda")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")

    torch.cuda.set_device(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup runs
    for _ in range(warmup):
        if dist_group is not None:
            dist.barrier(group=dist_group)
        fn()
        torch.cuda.synchronize(device)

    times: List[float] = []


    for _ in range(repeat):
        if dist_group is not None:
            dist.barrier(group=dist_group)

        start_event.record()
        fn()
        end_event.record()

        torch.cuda.synchronize(device)
        if post_iteration_barrier and dist_group is not None:
            dist.barrier(group=dist_group)
        # elapsed_time returns ms
        times.append(start_event.elapsed_time(end_event))
        # Optionally ensure all ranks finished this iteration before next iteration


    if return_mode == "mean":
        return sum(times) / len(times)
    elif return_mode == "median":
        s = sorted(times)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "all":
        return times
    else:
        raise ValueError(f"Unknown return_mode: {return_mode}")


def do_bench_distributed_graph(
    fn: Callable,
    repeat: int = 50,
    device: Optional[Union[torch.device, int]] = None,
    dist_group: Optional[dist.ProcessGroup] = None,
    return_mode: str = "mean",
    warmup: int = 5,
    post_iteration_barrier: bool = False,
) -> Union[float, List[float]]:
    
    if device is None:
        device = torch.device("cuda")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    torch.cuda.set_device(device)

    # 1. Warmup
    warmup_count = max(warmup, 11)
    for _ in range(warmup_count):
        if dist_group is not None:
            dist.barrier(group=dist_group)
        fn()
    torch.cuda.synchronize()

    # 2. Capture Phase
    g = torch.cuda.CUDAGraph()
    # Note: fn() must not contain standard NCCL collectives
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    # 3. Benchmark Phase
    times: List[float] = [] # FIX: Was missing in your snippet
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(repeat):
        if dist_group is not None:
            dist.barrier(group=dist_group)

        start_event.record()
        g.replay()
        end_event.record()

        torch.cuda.synchronize(device)

        if post_iteration_barrier and dist_group is not None:
            dist.barrier(group=dist_group)
        
        times.append(start_event.elapsed_time(end_event))

    # Aggregation
    if return_mode == "mean":
        return sum(times) / len(times)
    elif return_mode == "median":
        s = sorted(times)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    elif return_mode in ["min", "max", "all"]:
        results = {"min": min(times), "max": max(times), "all": times}
        return results[return_mode]
    else:
        raise ValueError(f"Unknown return_mode: {return_mode}")

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device
        )

    # minimal GroupCoordinator wrapping WORLD
    world_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="world",
    )
    dist_group = dist.group.WORLD
    assert dist_group is not None
    _groups[getattr(dist_group, "group_name", "world")] = weakref.ref(world_group)
    return rank, local_rank, world_size, device, dist_group, world_group

def benchmark_all_gather_gemm_fp8(TEST_SHAPES: List[Tuple[int, int, int]], rank: int, local_rank: int, world_size: int, device: torch.device, dist_group: dist.ProcessGroup, world_group: GroupCoordinator, repeat: int = 50):
    MB = 1024 ** 2
    rows: List[Row] = []

    group_name = getattr(dist_group, "group_name", "world")

    for M, N, K in TEST_SHAPES:
        M_per_rank = M // world_size

        # inputs
        a_shared = (torch.rand(M_per_rank, K, device=device, dtype=torch.float32) * 0.05).to(FP8_DTYPE)
        b = (torch.rand(K, N, device=device, dtype=torch.float32) * 0.05).T.contiguous().T.to(FP8_DTYPE)
        scale_a = torch.rand((M_per_rank , 1), device=device, dtype=torch.float32) * 0.05 + 0.01
        scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

        #adding clamping to avoid nan, inf (overflow)
        min_val=1e-3 
        max_val = 0.02 * (1024 / max(K, N))

        scale_a = scale_a.clamp(min=min_val, max=max_val)
        scale_b = scale_b.clamp(min=min_val, max=max_val)
        # preallocation for cuda graph capture
        
        a_shared_symm = dist._symmetric_memory.empty(
            a_shared.shape,
            dtype=a_shared.dtype,
            device=a_shared.device
        )
        a_shared_symm.copy_(a_shared)
        
        candidate_splits = [1, 2, 4]  

        for sp in candidate_splits:
            if M_per_rank % sp != 0:
                continue  # skip invalid splits

            helion_kernel = lambda: torch.ops.vllm.helion_all_gather_fp8_gemm(
                a_shared_symm,
                b,
                scale_a,
                scale_b,
                world_size,
                group_name,
                SPLITS_PER_RANK=sp,
            )
            baseline_kernel = lambda: torch.ops.symm_mem.fused_all_gather_scaled_matmul(
                a_shared_symm,
                [b],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[torch.bfloat16],
                use_fast_accum=[False],
                group_name=group_name,
            )
            # if rank == 0:
            #     print(f"[Rank:{rank}] Sanity check Testing shape M={M},N={N},K={K} with split {sp} (tokens per rank: {M_per_rank})")
            a_out, c = helion_kernel()
            ag_golden, mm_golden = baseline_kernel()
            torch.testing.assert_close(a_out, ag_golden), "All-gather outputs do not match"
            torch.testing.assert_close(c, mm_golden[0].to(torch.bfloat16), rtol=1e-1, atol=1e-1), "Matmul outputs do not match"
            if os.getenv("PROFILING") == "1":
                # ---- Prepare the kernel for profiling ----
                for _ in range(3):
                    helion_kernel()
                torch.cuda.synchronize()

                # ---- PROFILE the helion kernel (only on rank 0) ----
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logdir/helion_M{M}_N{N}_K{K}_sp{sp}_RANK{rank}')
                ) as prof:
                    helion_kernel()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
                print(f"Profiling trace saved. To view: tensorboard --logdir=./logdir")
                # ---- Prepare the kernel for profiling (warmup) ----
                for _ in range(3):
                    baseline_kernel()
                    torch.cuda.synchronize()
                # ---- PROFILE the baseline kernel (only on rank 0) ----
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logdir/baseline_M{M}_N{N}_K{K}_sp{sp}_RANK{rank}')
                ) as prof:
                    baseline_kernel()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
                print(f"Profiling trace saved. To view: tensorboard --logdir=./logdir")
                
            # benchmark Helion kernel
            torch.cuda.reset_peak_memory_stats(device)
            # if rank == 0:
            #     print(f" Rank:{rank}] Benchmarking Helion M={M},N={N},K={K} ")
            helion_latency = do_bench_distributed(helion_kernel, repeat=repeat, return_mode='mean', device=device, dist_group=dist_group)
            helion_peak_mem = torch.cuda.max_memory_allocated(device) / MB

            # benchmark baseline kernel
            torch.cuda.reset_peak_memory_stats(device)
            # if rank == 0:
            #     print(f"[Rank:{rank}] Benchmarking baseline M={M},N={N},K={K}")            
            baseline_latency = do_bench_distributed(baseline_kernel, repeat=repeat, return_mode='mean', device=device, dist_group=dist_group)
            baseline_peak_mem = torch.cuda.max_memory_allocated(device) / MB

            # compute speedup and memory improvement (guard against zero)
            speedup_x = baseline_latency / helion_latency if helion_latency > 0 else float("inf")
            mem_improve_x = baseline_peak_mem / helion_peak_mem if helion_peak_mem > 0 else float("inf")

            # if rank == 0:
            #     print(f"Rank:{rank}] Finished Benchmarking on shape M={M},N={N},K={K}")            

            rows.append(
                Row(
                    shape=f"M={M},N={N},K={K}splits={sp}",
                    baseline_ms=baseline_latency,
                    kernel_ms=helion_latency,
                    speedup_x=speedup_x,
                    baseline_peak_mb=baseline_peak_mem,
                    kernel_peak_mb=helion_peak_mem,
                    mem_improve_x=mem_improve_x,
                )
            )
    save_rows_json(rows, rank=rank)
    if rank == 0:
        print("\n=== Benchmark Results ===")
        print_table(rows)

    dist.barrier()  # ensure all ranks finished
    dist.destroy_process_group()

if __name__ == "__main__":
    """
    example how to run it:
        VLLM_USE_HELION_BACKEND=1  torchrun --nproc_per_node=4   benchmarks/kernels/helion/benchmark_all_gather_gemm_fp8.py
    """
    # list of shapes to benchmark
    TEST_SHAPES = [
        #(128, 32, 64),
        #(128, 128, 128),
        #(256, 1024, 1024),
        #medium shapes
        #(2048, 1024, 2048), 
        (2048, 4096, 4096),
        #(4096, 2048, 4096),
        #large shapes
        #(4096, 5120, 5120), # this fails to do_bench_distributed_graph
        #(8192, 8192, 8192), this fails to benchmark (might be OOM) for split_per_rank=1,2,4
    ]
    rank, local_rank, world_size, device, dist_group, world_group = setup_distributed()
    try:
        benchmark_all_gather_gemm_fp8(TEST_SHAPES, rank, local_rank, world_size, device, dist_group, world_group, repeat=10)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()