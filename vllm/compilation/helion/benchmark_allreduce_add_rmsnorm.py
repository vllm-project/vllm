#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for Helion AllReduce + Add + RMSNorm fusion.

Run with:
    python vllm/compilation/helion/benchmark_allreduce_add_rmsnorm.py --mode quick
    python vllm/compilation/helion/benchmark_allreduce_add_rmsnorm.py --mode full
"""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

try:
    import flashinfer.comm as flashinfer_comm

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    flashinfer_comm = None

from vllm.compilation.helion.allreduce_add_rmsnorm import (
    helion_allreduce_add_rmsnorm,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single configuration."""

    M: int
    K: int
    dtype: str
    splits_per_rank: int
    baseline_time_ms: float
    helion_time_ms: float
    speedup: float

    def to_dict(self):
        return {
            "M": self.M,
            "K": self.K,
            "dtype": self.dtype,
            "splits_per_rank": self.splits_per_rank,
            "baseline_time_ms": self.baseline_time_ms,
            "helion_time_ms": self.helion_time_ms,
            "speedup": self.speedup,
        }


def benchmark_worker(
    local_rank: int,
    world_size: int,
    configs: list,
    num_iterations: int,
    warmup: int,
    use_cudagraph: bool,
    verify: bool,
    atol: float,
    rtol: float,
):
    """Worker function - copied from test file structure."""
    print(f"[Rank {local_rank}] Starting worker", flush=True)

    current_platform.seed_everything(0)
    print(f"[Rank {local_rank}] Seeded RNG", flush=True)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print(f"[Rank {local_rank}] Set CUDA device", flush=True)

    # Initialize distributed environment - EXACTLY like the test
    print(f"[Rank {local_rank}] Updating environment variables", flush=True)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
        }
    )

    print(f"[Rank {local_rank}] Initializing distributed environment", flush=True)
    init_distributed_environment()
    print(f"[Rank {local_rank}] Initializing model parallel", flush=True)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    print(f"[Rank {local_rank}] Distributed initialization complete", flush=True)

    # Print header (rank 0 only)
    if local_rank == 0:
        print("\nBenchmark: Helion AllReduce + Add + RMSNorm vs FlashInfer")
        print(f"Running on: {torch.cuda.get_device_name()}")
        print(f"World size: {world_size}")
        print(f"Iterations: {num_iterations}, Warmup: {warmup}")
        print(f"CUDA graphs: {'ENABLED' if use_cudagraph else 'DISABLED'}")
        print(f"Correctness verification: {'ENABLED' if verify else 'DISABLED'}\n")

    results = []

    for config_idx, (M, K, dtype, splits_per_rank) in enumerate(configs):
        print(
            f"[Rank {local_rank}] Config {config_idx + 1}/{len(configs)}: M={M}, K={K}, dtype={dtype}, splits={splits_per_rank}",
            flush=True,
        )
        rms_eps = 1e-6

        # Setup FlashInfer workspace - EXACTLY like the test
        print(
            f"[Rank {local_rank}] Creating FlashInfer workspace for M={M}, K={K}",
            flush=True,
        )
        flashinfer_ipc_handles, flashinfer_workspace = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=local_rank,
                tp_size=world_size,
                max_token_num=M,
                hidden_dim=K,
                group=dist.group.WORLD,
                use_fp32_lamport=False,
            )
        )
        print(f"[Rank {local_rank}] FlashInfer workspace created", flush=True)

        # ========== Numerical Correctness Verification ==========
        if verify:
            print(f"[Rank {local_rank}] Starting correctness verification", flush=True)
            torch.manual_seed(42 + local_rank)
            input_data = torch.randn(M, K, dtype=dtype, device=device)
            residual_data = torch.randn(M, K, dtype=dtype, device=device)
            rms_gamma = torch.ones(K, dtype=dtype, device=device)
            print(f"[Rank {local_rank}] Created test data", flush=True)

            # Run FlashInfer baseline - EXACTLY like the test
            print(
                f"[Rank {local_rank}] Creating symmetric memory for baseline",
                flush=True,
            )
            input_baseline = symm_mem.empty(M, K, dtype=dtype, device=device)
            input_baseline.copy_(input_data)
            residual_baseline = residual_data.clone()

            norm_out_baseline = input_baseline
            residual_out_baseline = residual_baseline

            print(f"[Rank {local_rank}] Running FlashInfer baseline", flush=True)
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=input_baseline,
                token_num=M,
                residual_in=residual_baseline,
                residual_out=residual_out_baseline,
                norm_out=norm_out_baseline,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                hidden_dim=K,
                workspace_ptrs=flashinfer_workspace,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None,
                quant_out=None,
                scale_out=None,
                layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
                scale_factor=None,
                use_oneshot=False,
                world_rank=local_rank,
                world_size=world_size,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                fp32_acc=True,
            )
            print(
                f"[Rank {local_rank}] FlashInfer baseline complete, synchronizing",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[Rank {local_rank}] Synchronization complete", flush=True)

            # Run Helion - EXACTLY like the test
            print(
                f"[Rank {local_rank}] Creating symmetric memory for Helion", flush=True
            )
            input_helion = symm_mem.empty(M, K, dtype=dtype, device=device)
            input_helion.copy_(input_data)
            residual_helion = residual_data.clone()

            print(f"[Rank {local_rank}] Running Helion", flush=True)
            norm_out_helion, residual_out_helion = helion_allreduce_add_rmsnorm(
                input_helion, residual_helion, rms_gamma, rms_eps, splits_per_rank
            )
            print(f"[Rank {local_rank}] Helion complete, synchronizing", flush=True)
            torch.cuda.synchronize()
            print(f"[Rank {local_rank}] Synchronization complete", flush=True)

            # Check correctness
            print(f"[Rank {local_rank}] Checking correctness", flush=True)
            if local_rank == 0:
                print(
                    f"Verifying M={M} K={K} dtype={dtype} splits={splits_per_rank}... ",
                    end="",
                    flush=True,
                )

            try:
                torch.testing.assert_close(
                    norm_out_helion,
                    norm_out_baseline,
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    residual_out_helion,
                    residual_out_baseline,
                    rtol=rtol,
                    atol=atol,
                )
                print(f"[Rank {local_rank}] Correctness check passed", flush=True)
                if local_rank == 0:
                    print("✓ PASSED")
            except AssertionError as e:
                print(f"[Rank {local_rank}] Correctness check failed: {e}", flush=True)
                if local_rank == 0:
                    print(f"✗ FAILED: {e}")
                # Cleanup and skip benchmarking
                print(
                    f"[Rank {local_rank}] Destroying FlashInfer workspace (after failure)",
                    flush=True,
                )
                try:
                    flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                        flashinfer_ipc_handles
                    )
                except:
                    pass
                print(
                    f"[Rank {local_rank}] Cleanup complete, continuing to next config",
                    flush=True,
                )
                continue

        # ========== Performance Benchmarking ==========
        print(f"[Rank {local_rank}] Starting performance benchmarking", flush=True)
        torch.manual_seed(42 + local_rank)
        rms_gamma = torch.ones(K, dtype=dtype, device=device)

        # Time kernel function - with distributed-aware cudagraph support
        def time_kernel(kernel_fn):
            print(
                f"[Rank {local_rank}] time_kernel called, use_cudagraph={use_cudagraph}",
                flush=True,
            )
            if use_cudagraph:
                print(
                    f"[Rank {local_rank}] Using manual CUDA graph for distributed ops",
                    flush=True,
                )

                # Warmup before graph capture
                print(
                    f"[Rank {local_rank}] Running {warmup} warmup iterations before graph capture",
                    flush=True,
                )
                for i in range(warmup):
                    kernel_fn()
                    dist.barrier()
                torch.cuda.synchronize()
                print(f"[Rank {local_rank}] Warmup complete", flush=True)

                # Capture CUDA graph
                print(f"[Rank {local_rank}] Capturing CUDA graph", flush=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    kernel_fn()
                print(f"[Rank {local_rank}] CUDA graph captured", flush=True)

                # Synchronize all ranks after graph capture
                print(f"[Rank {local_rank}] Barrier after graph capture", flush=True)
                dist.barrier()
                print(f"[Rank {local_rank}] Starting timed graph replays", flush=True)

                # Time graph replays
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for i in range(num_iterations):
                    print(
                        f"[Rank {local_rank}] Graph replay {i + 1}/{num_iterations}",
                        flush=True,
                    )
                    graph.replay()
                    # CRITICAL: Synchronize between replays for distributed ops
                    dist.barrier()
                end_event.record()

                torch.cuda.synchronize()
                print(f"[Rank {local_rank}] All graph replays complete", flush=True)

                return start_event.elapsed_time(end_event) / num_iterations

            # Fallback to standard timing
            print(
                f"[Rank {local_rank}] Using standard timing (non-cudagraph)", flush=True
            )
            print(f"[Rank {local_rank}] Running {warmup} warmup iterations", flush=True)
            for i in range(warmup):
                print(
                    f"[Rank {local_rank}] Warmup iteration {i + 1}/{warmup}", flush=True
                )
                kernel_fn()
                # CRITICAL: Synchronize between iterations for distributed ops
                dist.barrier()
            print(f"[Rank {local_rank}] Warmup complete, synchronizing", flush=True)
            torch.cuda.synchronize()
            print(f"[Rank {local_rank}] Starting timed iterations", flush=True)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for i in range(num_iterations):
                print(
                    f"[Rank {local_rank}] Timed iteration {i + 1}/{num_iterations}",
                    flush=True,
                )
                kernel_fn()
                # CRITICAL: Synchronize between iterations for distributed ops
                dist.barrier()
            end_event.record()
            print(
                f"[Rank {local_rank}] Timed iterations complete, synchronizing",
                flush=True,
            )
            torch.cuda.synchronize()

            return start_event.elapsed_time(end_event) / num_iterations

        # Benchmark FlashInfer - EXACTLY like the test
        print(f"[Rank {local_rank}] Preparing FlashInfer benchmark tensors", flush=True)
        input_baseline_perf = symm_mem.empty(M, K, dtype=dtype, device=device)
        residual_baseline_perf = torch.empty(M, K, dtype=dtype, device=device)
        input_data_perf = torch.randn(M, K, dtype=dtype, device=device)
        residual_data_perf = torch.randn(M, K, dtype=dtype, device=device)

        def baseline_fn():
            input_baseline_perf.copy_(input_data_perf)
            residual_baseline_perf.copy_(residual_data_perf)

            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=input_baseline_perf,
                token_num=M,
                residual_in=residual_baseline_perf,
                residual_out=residual_baseline_perf,
                norm_out=input_baseline_perf,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                hidden_dim=K,
                workspace_ptrs=flashinfer_workspace,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None,
                quant_out=None,
                scale_out=None,
                layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
                scale_factor=None,
                use_oneshot=False,
                world_rank=local_rank,
                world_size=world_size,
                launch_with_pdl=True,
                trigger_completion_at_end=True,
                fp32_acc=True,
            )

        print(
            f"[Rank {local_rank}] Entering barrier before FlashInfer benchmark",
            flush=True,
        )
        dist.barrier()
        print(
            f"[Rank {local_rank}] Barrier passed, benchmarking FlashInfer", flush=True
        )
        baseline_time = time_kernel(baseline_fn)
        print(
            f"[Rank {local_rank}] FlashInfer benchmark complete, entering barrier",
            flush=True,
        )
        dist.barrier()
        print(f"[Rank {local_rank}] Barrier passed", flush=True)

        # Benchmark Helion - EXACTLY like the test
        print(f"[Rank {local_rank}] Preparing Helion benchmark tensors", flush=True)
        input_helion_perf = symm_mem.empty(M, K, dtype=dtype, device=device)
        residual_helion_perf = torch.empty(M, K, dtype=dtype, device=device)

        def helion_fn():
            input_helion_perf.copy_(input_data_perf)
            residual_helion_perf.copy_(residual_data_perf)

            helion_allreduce_add_rmsnorm(
                input_helion_perf,
                residual_helion_perf,
                rms_gamma,
                rms_eps,
                splits_per_rank,
            )

        print(
            f"[Rank {local_rank}] Entering barrier before Helion benchmark", flush=True
        )
        dist.barrier()
        print(f"[Rank {local_rank}] Barrier passed, benchmarking Helion", flush=True)
        helion_time = time_kernel(helion_fn)
        print(
            f"[Rank {local_rank}] Helion benchmark complete, entering barrier",
            flush=True,
        )
        dist.barrier()
        print(f"[Rank {local_rank}] Barrier passed", flush=True)

        if local_rank == 0:
            speedup = baseline_time / helion_time
            result = BenchmarkResult(
                M=M,
                K=K,
                dtype=str(dtype),
                splits_per_rank=splits_per_rank,
                baseline_time_ms=baseline_time,
                helion_time_ms=helion_time,
                speedup=speedup,
            )
            results.append(result)
            print(
                f"  FlashInfer: {baseline_time:.4f} ms, Helion: {helion_time:.4f} ms, Speedup: {speedup:.2f}x"
            )

        # Cleanup
        print(f"[Rank {local_rank}] Destroying FlashInfer workspace", flush=True)
        try:
            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                flashinfer_ipc_handles
            )
            print(f"[Rank {local_rank}] FlashInfer workspace destroyed", flush=True)
        except Exception as e:
            print(f"[Rank {local_rank}] Failed to destroy workspace: {e}", flush=True)

        print(
            f"[Rank {local_rank}] Completed config {config_idx + 1}/{len(configs)}",
            flush=True,
        )

    print(f"[Rank {local_rank}] All configs complete, exiting worker", flush=True)
    return results


def print_results(results):
    """Print benchmark results."""
    if not results:
        return

    print("\n" + "=" * 110)
    print(
        f"{'M':<8} {'K':<8} {'DType':<12} {'Splits':<8} "
        f"{'Baseline (ms)':<15} {'Helion (ms)':<15} {'Speedup':<10}"
    )
    print("=" * 110)

    for result in results:
        print(
            f"{result.M:<8} "
            f"{result.K:<8} "
            f"{result.dtype:<12} "
            f"{result.splits_per_rank:<8} "
            f"{result.baseline_time_ms:<15.4f} "
            f"{result.helion_time_ms:<15.4f} "
            f"{result.speedup:<10.2f}x"
        )
    print("=" * 110 + "\n")


def print_summary_statistics(results):
    """Print summary statistics."""
    if not results:
        return

    speedups = [r.speedup for r in results]
    baseline_times = [r.baseline_time_ms for r in results]
    helion_times = [r.helion_time_ms for r in results]

    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total configurations: {len(results)}")
    print("\nSpeedup:")
    print(f"  Average: {statistics.mean(speedups):.2f}x")
    print(f"  Median:  {statistics.median(speedups):.2f}x")
    print(f"  Min:     {min(speedups):.2f}x")
    print(f"  Max:     {max(speedups):.2f}x")
    print("\nLatency (ms):")
    print(
        f"  Baseline - Avg: {statistics.mean(baseline_times):.4f}, "
        f"Min: {min(baseline_times):.4f}, Max: {max(baseline_times):.4f}"
    )
    print(
        f"  Helion   - Avg: {statistics.mean(helion_times):.4f}, "
        f"Min: {min(helion_times):.4f}, Max: {max(helion_times):.4f}"
    )
    print("=" * 60)


def save_results_csv(results, filename):
    """Save results to CSV."""
    if not results:
        return

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows([r.to_dict() for r in results])

    print(f"\n✓ Results saved to {filepath}")


def save_results_json(results, filename):
    """Save results to JSON."""
    if not results:
        return

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "device": torch.cuda.get_device_name(),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
        )

    print(f"✓ Results saved to {filepath}")


# Global container for results (needed for multiprocessing)
_results_container = [None]


def worker_wrapper(
    local_rank: int,
    world_size: int,
    configs: list,
    num_iterations: int,
    warmup: int,
    use_cudagraph: bool,
    verify: bool,
    atol: float,
    rtol: float,
):
    """Wrapper to call benchmark_worker and collect results."""
    print(f"[Rank {local_rank}] worker_wrapper started", flush=True)
    worker_results = benchmark_worker(
        local_rank,
        world_size,
        configs,
        num_iterations,
        warmup,
        use_cudagraph,
        verify,
        atol,
        rtol,
    )
    print(f"[Rank {local_rank}] benchmark_worker returned", flush=True)
    if local_rank == 0:
        _results_container[0] = worker_results
        print(f"[Rank {local_rank}] Results saved to container", flush=True)
    print(f"[Rank {local_rank}] worker_wrapper exiting", flush=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Helion AllReduce + Add + RMSNorm"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark mode",
    )
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--num-iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--no-cudagraph", action="store_true", help="Disable cudagraph")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    if torch.cuda.device_count() < args.num_gpus:
        print(
            f"ERROR: Need {args.num_gpus} GPUs, only {torch.cuda.device_count()} available"
        )
        return

    if not FLASHINFER_AVAILABLE:
        print("ERROR: FlashInfer not available")
        return

    # Test configurations
    if args.mode == "quick":
        configs = [
            (128, 2048, torch.bfloat16, 4),
            (512, 4096, torch.bfloat16, 4),
            (1024, 4096, torch.bfloat16, 4),
        ]
    else:
        configs = []
        shapes = [
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
            (1024, 4096),
            (2048, 4096),
            (512, 2048),
            (512, 8192),
            (512, 16384),
            (128, 8192),
            (256, 8192),
            (1024, 8192),
        ]

        for M, K in shapes:
            for dtype in [torch.bfloat16, torch.float16]:
                for splits in [2, 4, 8]:
                    configs.append((M, K, dtype, splits))

    print(f"Testing {len(configs)} configurations")

    # Info about cudagraph mode
    if not args.no_cudagraph:
        print("\nUsing manual CUDA graph capture with distributed synchronization")
    else:
        print("\nCUDA graphs disabled, using standard timing")

    # Spawn workers - EXACTLY like the test
    print("Spawning workers...", flush=True)
    torch.multiprocessing.spawn(
        worker_wrapper,
        args=(
            args.num_gpus,
            configs,
            args.num_iterations,
            args.warmup,
            not args.no_cudagraph,
            not args.no_verify,
            args.atol,
            args.rtol,
        ),
        nprocs=args.num_gpus,
    )
    print("Workers completed", flush=True)

    # Print and save results (from rank 0)
    if _results_container[0]:
        print_results(_results_container[0])
        print_summary_statistics(_results_container[0])

        if args.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_file = output_dir / f"benchmark_{args.mode}_{timestamp}.csv"
            json_file = output_dir / f"benchmark_{args.mode}_{timestamp}.json"

            save_results_csv(_results_container[0], str(csv_file))
            save_results_json(_results_container[0], str(json_file))


if __name__ == "__main__":
    main()
