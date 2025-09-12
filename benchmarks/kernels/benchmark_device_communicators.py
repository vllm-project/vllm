#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for device communicators:
CustomAllreduce (oneshot, twoshot), PyNcclCommunicator,
and SymmMemCommunicator (multimem, two-shot).

Usage:
    torchrun --nproc_per_node=<N> benchmark_device_communicators.py [options]

Example:
    torchrun --nproc_per_node=2 benchmark_device_communicators.py
    --sequence-lengths 512 1024 2048 --num-warmup 10 --num-trials 100
"""

import json
import os
import time
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)

# Default sequence lengths to benchmark
DEFAULT_SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]

# Fixed hidden size and dtype for all benchmarks
HIDDEN_SIZE = 8192
BENCHMARK_DTYPE = torch.bfloat16

# CUDA graph settings
CUDA_GRAPH_CAPTURE_CYCLES = 10


class CommunicatorBenchmark:
    """Benchmark class for testing device communicators."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        cpu_group: ProcessGroup,
        sequence_lengths: list[int],
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.cpu_group = cpu_group

        # Calculate max_size_override based on largest sequence length
        max_seq_len = max(sequence_lengths)
        max_tensor_elements = max_seq_len * HIDDEN_SIZE
        self.max_size_override = max_tensor_elements * BENCHMARK_DTYPE.itemsize + 1

        # Initialize communicators
        self.custom_allreduce = None
        self.pynccl_comm = None
        self.symm_mem_comm = None
        self.symm_mem_comm_multimem = None
        self.symm_mem_comm_two_shot = None

        self._init_communicators()

    def _init_communicators(self):
        """Initialize all available communicators."""
        try:
            self.custom_allreduce = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                max_size=self.max_size_override,
            )
            if not self.custom_allreduce.disabled:
                logger.info("Rank %s: CustomAllreduce initialized", self.rank)
            else:
                logger.info("Rank %s: CustomAllreduce disabled", self.rank)
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize CustomAllreduce: %s", self.rank, e
            )
            self.custom_allreduce = None

        try:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group, device=self.device
            )
            if not self.pynccl_comm.disabled:
                logger.info("Rank %s: PyNcclCommunicator initialized", self.rank)
            else:
                logger.info("Rank %s: PyNcclCommunicator disabled", self.rank)
                self.pynccl_comm = None
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize PyNcclCommunicator: %s", self.rank, e
            )
            self.pynccl_comm = None

        # Initialize variants for SymmMemCommunicator
        try:
            self.symm_mem_comm_multimem = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
                force_multimem=True,
                max_size_override=self.max_size_override,
            )
            if not self.symm_mem_comm_multimem.disabled:
                logger.info(
                    "Rank %s: SymmMemCommunicator (multimem) initialized", self.rank
                )
            else:
                self.symm_mem_comm_multimem = None
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize SymmMemCommunicator (multimem): %s",
                self.rank,
                e,
            )
            self.symm_mem_comm_multimem = None

        try:
            self.symm_mem_comm_two_shot = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
                force_multimem=False,
                max_size_override=self.max_size_override,
            )
            if not self.symm_mem_comm_two_shot.disabled:
                logger.info(
                    "Rank %s: SymmMemCommunicator (two_shot) initialized", self.rank
                )
            else:
                self.symm_mem_comm_two_shot = None
        except Exception as e:
            logger.warning(
                "Rank %s: Failed to initialize SymmMemCommunicator (two_shot): %s",
                self.rank,
                e,
            )
            self.symm_mem_comm_two_shot = None

    def benchmark_allreduce(
        self, sequence_length: int, num_warmup: int, num_trials: int
    ) -> dict[str, float]:
        """Benchmark allreduce operations for all available communicators."""

        results = {}

        # Define communicators with their benchmark functions
        communicators = []

        if self.custom_allreduce is not None:
            comm = self.custom_allreduce
            # CustomAllreduce one-shot
            communicators.append(
                (
                    "ca_1stage",
                    lambda t, c=comm: c.custom_all_reduce(t),
                    lambda t, c=comm: c.should_custom_ar(t),
                    comm.capture(),
                    "1stage",  # env variable value
                )
            )
            # CustomAllreduce two-shot
            communicators.append(
                (
                    "ca_2stage",
                    lambda t, c=comm: c.custom_all_reduce(t),
                    lambda t, c=comm: c.should_custom_ar(t),
                    comm.capture(),
                    "2stage",  # env variable value
                )
            )

        if self.pynccl_comm is not None:
            comm = self.pynccl_comm
            communicators.append(
                (
                    "pynccl",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t: True,  # Always available if initialized
                    nullcontext(),
                    None,  # no env variable needed
                )
            )

        if self.symm_mem_comm_multimem is not None:
            comm = self.symm_mem_comm_multimem
            communicators.append(
                (
                    "symm_mem_multimem",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_symm_mem(t),
                    nullcontext(),
                    None,  # no env variable needed
                )
            )

        if self.symm_mem_comm_two_shot is not None:
            comm = self.symm_mem_comm_two_shot
            communicators.append(
                (
                    "symm_mem_two_shot",
                    lambda t, c=comm: c.all_reduce(t),
                    lambda t, c=comm: c.should_use_symm_mem(t),
                    nullcontext(),
                    None,  # no env variable needed
                )
            )

        # Benchmark each communicator
        for name, allreduce_fn, should_use_fn, context, env_var in communicators:
            # Set environment variable if needed
            if env_var is not None:
                os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = env_var
            else:
                # Clear the environment variable to avoid interference
                os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)

            latency = self.benchmark_allreduce_single(
                sequence_length,
                allreduce_fn,
                should_use_fn,
                context,
                num_warmup,
                num_trials,
            )
            if latency is not None:
                results[name] = latency

        return results

    def benchmark_allreduce_single(
        self,
        sequence_length: int,
        allreduce_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
        should_use_fn: Callable[[torch.Tensor], bool],
        context,
        num_warmup: int,
        num_trials: int,
    ) -> Optional[float]:
        """Benchmark method with CUDA graph optimization."""
        try:
            # Create test tensor (2D: sequence_length x hidden_size)
            tensor = torch.randn(
                sequence_length, HIDDEN_SIZE, dtype=BENCHMARK_DTYPE, device=self.device
            )
            if not should_use_fn(tensor):
                return None

            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph_input = tensor.clone()

                # Warmup before capture
                for _ in range(3):
                    allreduce_fn(graph_input)

                # Capture the graph using context manager
                with context:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(CUDA_GRAPH_CAPTURE_CYCLES):
                            allreduce_fn(graph_input)

            torch.cuda.synchronize()
            for _ in range(num_warmup):
                graph.replay()
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_trials):
                graph.replay()
            torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Convert to ms and divide by CUDA_GRAPH_CAPTURE_CYCLES
            return (
                (end_time - start_time) / num_trials / CUDA_GRAPH_CAPTURE_CYCLES * 1000
            )

        except Exception as e:
            logger.error("CUDA graph benchmark failed: %s", e)
            raise RuntimeError(
                f"CUDA graph benchmark failed for communicator: {e}"
            ) from e


def _calculate_speedup_info(comm_results: dict[str, float]) -> str:
    """Calculate speedup information for a single tensor size."""
    if not comm_results:
        return "N/A"

    # Find the fastest communicator
    fastest_comm = min(comm_results.keys(), key=lambda k: comm_results[k])
    fastest_time = comm_results[fastest_comm]

    # Calculate speedup vs PyNccl if available
    if "pynccl" in comm_results:
        pynccl_time = comm_results["pynccl"]
        speedup = pynccl_time / fastest_time
        return f"{fastest_comm} ({speedup:.2f}x)"
    else:
        return f"{fastest_comm} (N/A)"


def print_results(
    results: dict[str, dict[str, float]], sequence_lengths: list[int], world_size: int
):
    """Print benchmark results in a formatted table."""

    print(f"\n{'=' * 130}")
    print("Device Communicator Benchmark Results")
    print(
        f"World Size: {world_size}, Data Type: {BENCHMARK_DTYPE}, "
        f"Hidden Size: {HIDDEN_SIZE}"
    )
    print(f"{'=' * 130}")

    # Get all communicator names
    all_comms = set()
    for size_results in results.values():
        all_comms.update(size_results.keys())

    all_comms = sorted(list(all_comms))

    # Print header
    header = f"{'Tensor Shape':<20}{'Tensor Size':<15}"
    for comm in all_comms:
        header += f"{comm:<20}"
    header += f"{'Best (Speedup vs PyNccl)':<30}"
    print(header)
    print("-" * len(header))

    # Print results for each sequence length
    for seq_len in sequence_lengths:
        if seq_len in results:
            # Calculate tensor size in elements and bytes
            tensor_elements = seq_len * HIDDEN_SIZE
            tensor_bytes = tensor_elements * BENCHMARK_DTYPE.itemsize

            # Format tensor size (MB)
            tensor_size_mb = tensor_bytes / (1024 * 1024)
            tensor_size_str = f"{tensor_size_mb:.2f} MB"

            # Format tensor shape
            tensor_shape = f"({seq_len}, {HIDDEN_SIZE})"

            row = f"{tensor_shape:<20}{tensor_size_str:<15}"
            for comm in all_comms:
                if comm in results[seq_len]:
                    row += f"{results[seq_len][comm]:<20.3f}"
                else:
                    row += f"{'N/A':<20}"

            # Calculate speedup information
            speedup_info = _calculate_speedup_info(results[seq_len])
            row += f"{speedup_info:<30}"

            print(row)

    print(f"{'=' * 130}")
    print("All times are in milliseconds (ms) per allreduce operation")
    print("Speedup column shows: fastest_algorithm (speedup_vs_pynccl)")


def main():
    parser = FlexibleArgumentParser(description="Benchmark device communicators")

    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Sequence lengths to benchmark (tensor shape: seq_len x hidden_size)",
    )

    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup iterations"
    )

    parser.add_argument(
        "--num-trials", type=int, default=50, help="Number of benchmark trials"
    )

    parser.add_argument("--output-json", type=str, help="Output results to JSON file")

    args = parser.parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Get CPU process group
    cpu_group = dist.new_group(backend="gloo")

    # Disable USE_SYMM_MEM to avoid affecting the max_sizes
    # in symm_mem and custom_all_reduce for benchmark
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

    # Initialize benchmark
    benchmark = CommunicatorBenchmark(
        rank, world_size, device, cpu_group, args.sequence_lengths
    )

    # Run benchmarks
    all_results = {}

    for seq_len in args.sequence_lengths:
        if rank == 0:
            logger.info(
                "Benchmarking sequence length: %s (tensor shape: %s x %s)",
                seq_len,
                seq_len,
                HIDDEN_SIZE,
            )

        results = benchmark.benchmark_allreduce(
            sequence_length=seq_len,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )

        all_results[seq_len] = results

        # Synchronize between ranks
        dist.barrier()

    # Print results (only rank 0)
    if rank == 0:
        print_results(all_results, args.sequence_lengths, world_size)

        # Save to JSON if requested
        if args.output_json:
            # Add speedup information to results
            enhanced_results = {}
            for seq_len, comm_results in all_results.items():
                enhanced_results[seq_len] = {
                    "timings": comm_results,
                    "speedup_info": _calculate_speedup_info(comm_results),
                }

            output_data = {
                "world_size": world_size,
                "dtype": str(BENCHMARK_DTYPE),
                "hidden_size": HIDDEN_SIZE,
                "sequence_lengths": args.sequence_lengths,
                "num_warmup": args.num_warmup,
                "num_trials": args.num_trials,
                "cuda_graph_capture_cycles": CUDA_GRAPH_CAPTURE_CYCLES,
                "results": enhanced_results,
            }

            with open(args.output_json, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info("Results saved to %s", args.output_json)

    # Cleanup
    if cpu_group != dist.group.WORLD:
        dist.destroy_process_group(cpu_group)


if __name__ == "__main__":
    main()
