# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generic benchmarking framework for Helion kernels.

This module provides a flexible framework to benchmark Helion kernels against
their CUDA reference implementations, measuring both correctness and performance.
"""

import csv
import json
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Registry for KernelBenchmark classes
_benchmark_registry: dict[str, type["KernelBenchmark"]] = {}

# Global container for distributed benchmark result collection
_distributed_results_container = [None]


def verify_correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    compare_dtype: torch.dtype = None,
) -> bool:
    """
    Verify that kernel output matches reference implementation.

    Args:
        output: Kernel output tensor
        reference: Reference output tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        compare_dtype: Optional dtype to convert both tensors to before comparison

    Returns:
        True if outputs match within tolerance, False otherwise
    """
    try:
        if compare_dtype is not None:
            output = output.to(dtype=compare_dtype)
            reference = reference.to(dtype=compare_dtype)

        torch.testing.assert_close(
            output,
            reference,
            atol=atol,
            rtol=rtol,
        )
        return True
    except AssertionError as e:
        print(f"Correctness check failed: {e}")
        return False


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    shape_params: dict[str, Any]
    dtype: torch.dtype
    num_iterations: int = 1000
    warmup: int = 10
    use_cudagraph: bool = True
    verify: bool = True
    atol: float = 1e-5
    rtol: float = 1e-3


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    shape_params: dict[str, Any]
    shape_desc: str
    dtype: str
    baseline_time_ms: float
    helion_time_ms: float
    speedup: float
    correctness_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            **self.shape_params,  # Unpack shape params into the dict
            "shape_desc": self.shape_desc,
            "dtype": self.dtype,
            "baseline_time_ms": self.baseline_time_ms,
            "helion_time_ms": self.helion_time_ms,
            "speedup": self.speedup,
            "correctness_passed": self.correctness_passed,
        }
        return result


class KernelBenchmark(ABC):
    """
    Base class for kernel benchmarking.

    Subclasses should implement kernel-specific logic for creating inputs,
    running reference and Helion kernels, and providing test configurations.

    Subclasses are automatically registered using their benchmark_name attribute.

    Example:
        class SiluMulFp8Benchmark(KernelBenchmark):
            benchmark_name = "silu_mul_fp8"
            ...
    """

    benchmark_name: str = ""

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses with non-empty benchmark_name."""
        super().__init_subclass__(**kwargs)
        # Only register classes that set a benchmark_name
        # This allows abstract base classes to have empty names
        if cls.benchmark_name != "":
            _benchmark_registry[cls.benchmark_name] = cls

    @staticmethod
    def get_benchmark_class(name: str) -> type["KernelBenchmark"]:
        """
        Get a registered benchmark class by name.

        Args:
            name: Registered name of the benchmark

        Returns:
            The benchmark class

        Raises:
            ValueError: If benchmark is not found
        """
        if name not in _benchmark_registry:
            raise ValueError(
                f"Unknown benchmark '{name}'. "
                f"Available: {', '.join(_benchmark_registry.keys())}"
            )
        return _benchmark_registry[name]

    @staticmethod
    def list_benchmarks() -> list[str]:
        """
        List all registered benchmarks.

        Returns:
            List of registered benchmark names
        """
        return list(_benchmark_registry.keys())

    @staticmethod
    def time_kernel(
        kernel_fn: Callable,
        num_iterations: int = 1000,
        warmup: int = 10,
        use_cudagraph: bool = False,
        distributed: bool = False,
    ) -> float:
        """
        Time a kernel function and return average execution time.

        Uses manual CUDA graph capture for compatibility with distributed operations.

        Args:
            kernel_fn: Callable that executes the kernel
            num_iterations: Number of timing iterations
            warmup: Number of warmup iterations
            use_cudagraph: Whether to use CUDAGraph for timing
            distributed: Whether to use distributed barriers for multi-GPU synchronization

        Returns:
            Average execution time in milliseconds
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(warmup):
            kernel_fn()
            if distributed:
                dist.barrier()
        torch.cuda.synchronize()

        if use_cudagraph:
            # Capture CUDA graph - fail fast if capture fails
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel_fn()

            # Synchronize all ranks after graph capture (if distributed)
            if distributed:
                dist.barrier()

            # Time graph replays
            start_event.record()
            for _ in range(num_iterations):
                graph.replay()
                if distributed:
                    dist.barrier()
            end_event.record()
            torch.cuda.synchronize()

        else:
            # Time kernel calls
            start_event.record()
            for _ in range(num_iterations):
                kernel_fn()
                if distributed:
                    dist.barrier()
            end_event.record()
            torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iterations

        return avg_time_ms

    @abstractmethod
    def get_quick_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list[Any]]]]:
        """
        Get test configurations for quick smoke testing.

        Quick tests should use a small number of representative configurations
        to verify correctness and basic functionality.

        Returns:
            List of (shapes, dtype, extra_params) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)
            - extra_params: Dict mapping parameter names to lists of values
                           to test (e.g., {"splits_per_rank": [4]}). Use empty
                           dict if no extra parameters.

        Example:
            [
                ([(1, 8192), (256, 8192), (1024, 16384)], torch.bfloat16, {}),
                ([(1, 8192), (256, 8192)], torch.float16, {"splits": [2, 4]}),
            ]
        """
        raise NotImplementedError

    @abstractmethod
    def get_full_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list[Any]]]]:
        """
        Get test configurations for comprehensive benchmarking.

        Full tests should cover a wide range of configurations
        to thoroughly benchmark performance across different scenarios.

        Returns:
            List of (shapes, dtype, extra_params) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)
            - extra_params: Dict mapping parameter names to lists of values
                           to test (e.g., {"splits_per_rank": [2, 4, 8]}). Use empty
                           dict if no extra parameters.

        Example:
            [
                ([(1, 1024), (256, 4096), (1024, 8192)], torch.bfloat16, {}),
                ([(1, 1024), (256, 4096), (1024, 8192)], torch.float16, {"splits": [2, 4, 8]}),
            ]
        """
        raise NotImplementedError

    @abstractmethod
    def create_inputs(self, dtype: torch.dtype, **shape_params) -> tuple[Any, ...]:
        """
        Create input tensors for the kernel.

        Args:
            dtype: Data type for inputs
            **shape_params: Kernel-specific shape parameters
                (e.g., shape=(1024, 8192))

        Returns:
            Tuple of input tensors (or other input types)
        """
        raise NotImplementedError

    @abstractmethod
    def run_baseline(self, *args, **kwargs) -> Any:
        """
        Run the baseline reference kernel.

        Args:
            *args: Positional arguments for the kernel
            **kwargs: Keyword arguments for the kernel

        Returns:
            Output from baseline kernel (could be tensor, tuple, None, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def run_helion(self, *args, **kwargs) -> Any:
        """
        Run the Helion kernel.

        Args:
            *args: Positional arguments for the kernel
            **kwargs: Keyword arguments for the kernel

        Returns:
            Output from Helion kernel (could be tensor, tuple, None, etc.)
        """
        raise NotImplementedError

    def get_shape_description(self, **shape_params) -> str:
        """
        Get a human-readable description for a shape configuration.
        Can be overridden by subclasses to provide more meaningful names.

        Args:
            **shape_params: Kernel-specific shape parameters

        Returns:
            Description string (default: "key1=val1_key2=val2_...")
        """
        return "_".join(f"{k}={v}" for k, v in sorted(shape_params.items()))

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult | None:
        """
        Run a single benchmark configuration.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult or None if verification fails
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create inputs
        inputs = self.create_inputs(config.dtype, **config.shape_params)

        # Get shape description
        shape_desc = self.get_shape_description(**config.shape_params)

        # Verify correctness if requested
        if config.verify:
            print(
                f"Verifying correctness for {shape_desc} {config.dtype}... ",
                end="",
            )
            baseline_output = self.run_baseline(*inputs)
            helion_output = self.run_helion(*inputs)

            passed = verify_correctness(
                helion_output,
                baseline_output,
                atol=config.atol,
                rtol=config.rtol,
                compare_dtype=config.dtype,
            )

            if passed:
                print("✓ PASSED")
            else:
                print("✗ FAILED")
                return None

        # Reset random seed for benchmark
        torch.manual_seed(42)
        inputs = self.create_inputs(config.dtype, **config.shape_params)

        # Benchmark baseline kernel
        def baseline_fn():
            return self.run_baseline(*inputs)

        baseline_time = self.time_kernel(
            baseline_fn,
            num_iterations=config.num_iterations,
            warmup=config.warmup,
            use_cudagraph=config.use_cudagraph,
        )

        # Benchmark Helion kernel
        def helion_fn():
            return self.run_helion(*inputs)

        helion_time = self.time_kernel(
            helion_fn,
            num_iterations=config.num_iterations,
            warmup=config.warmup,
            use_cudagraph=config.use_cudagraph,
        )

        return BenchmarkResult(
            shape_params=config.shape_params,
            shape_desc=shape_desc,
            dtype=str(config.dtype),
            baseline_time_ms=baseline_time,
            helion_time_ms=helion_time,
            speedup=baseline_time / helion_time,
            correctness_passed=True,
        )

    def run(
        self,
        mode: str = "quick",
        num_iterations: int = 1000,
        warmup: int = 10,
        use_cudagraph: bool = True,
        verify: bool = True,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> list[BenchmarkResult]:
        """
        Run all benchmark configurations for the given mode.

        This method iterates over all test configurations and runs
        a benchmark for each one.

        Args:
            mode: Benchmark mode - either "quick" or "full"
            num_iterations: Number of iterations for each benchmark
            warmup: Number of warmup iterations
            use_cudagraph: Whether to use CUDA graphs
            verify: Whether to verify correctness
            atol: Absolute tolerance for verification
            rtol: Relative tolerance for verification

        Returns:
            List of BenchmarkResult objects (excluding failed verifications)

        Raises:
            ValueError: If mode is not "quick" or "full"
        """
        # Get test configurations based on mode
        if mode == "quick":
            test_configs = self.get_quick_test_shapes()
        elif mode == "full":
            test_configs = self.get_full_test_shapes()
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'quick' or 'full'.")

        # Run benchmarks for all configurations
        results = []
        for shapes, dtype, extra_params in test_configs:
            for shape in shapes:
                config = BenchmarkConfig(
                    shape_params={"shape": shape},
                    dtype=dtype,
                    num_iterations=num_iterations,
                    warmup=warmup,
                    use_cudagraph=use_cudagraph,
                    verify=verify,
                    atol=atol,
                    rtol=rtol,
                )
                result = self.run_benchmark(config)
                if result is not None:
                    results.append(result)

        return results


class DistributedKernelBenchmark(KernelBenchmark):
    """
    Base class for distributed kernel benchmarking.

    This class extends KernelBenchmark to support multi-GPU benchmarks
    that require torch.multiprocessing.spawn and distributed synchronization.

    Subclasses should implement the same interface as KernelBenchmark but
    with the understanding that create_inputs(), run_baseline(), and run_helion()
    will be called within distributed worker processes.

    Example:
        class MyDistributedBenchmark(DistributedKernelBenchmark):
            benchmark_name = "my_distributed_kernel"

            def __init__(self, num_gpus: int = 2):
                super().__init__(num_gpus=num_gpus, master_port=12348)

            def get_quick_test_shapes(self):
                return [([(128, 4096), (256, 4096)], torch.bfloat16)]

            def create_inputs(self, dtype, **shape_params):
                # Create inputs in distributed context
                M, K = shape_params["shape"]
                input_tensor = symm_mem.empty(M, K, dtype=dtype, device="cuda")
                return (input_tensor,)

            def run_baseline(self, input_tensor):
                # Run baseline kernel
                return baseline_kernel(input_tensor)

            def run_helion(self, input_tensor):
                # Run Helion kernel
                return helion_kernel(input_tensor)
    """

    # Don't register this base class
    benchmark_name = ""

    def __init__(
        self,
        num_gpus: int = 2,
        master_port: int = 12348,
        init_fn: Callable[[int, int], None] | None = None,
    ):
        """
        Args:
            num_gpus: Number of GPUs to use for distributed benchmark
            master_port: Port for distributed initialization
            init_fn: Optional custom initialization function(local_rank, world_size)
                    If None, uses default vLLM distributed initialization
        """
        self.num_gpus = num_gpus
        self.master_port = master_port
        self.init_fn = init_fn

    def run(
        self,
        mode: str = "quick",
        num_iterations: int = 1000,
        warmup: int = 10,
        use_cudagraph: bool = True,
        verify: bool = True,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> list[BenchmarkResult]:
        """
        Run distributed benchmark using torch.multiprocessing.spawn.

        This overrides the base class run() method to handle the distributed
        nature of multi-GPU operations.

        Args:
            mode: Benchmark mode - either "quick" or "full"
            num_iterations: Number of iterations for each benchmark
            warmup: Number of warmup iterations
            use_cudagraph: Whether to use CUDA graphs
            verify: Whether to verify correctness
            atol: Absolute tolerance for verification
            rtol: Relative tolerance for verification

        Returns:
            List of BenchmarkResult objects
        """
        # Get test configurations
        if mode == "quick":
            test_configs = self.get_quick_test_shapes()
        elif mode == "full":
            test_configs = self.get_full_test_shapes()
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'quick' or 'full'.")

        # Build configuration list from test shapes
        configs = []
        for shapes, dtype, extra_params in test_configs:
            for shape in shapes:
                # If there are extra parameters, create all combinations
                if extra_params:
                    # Generate all combinations of extra parameters
                    import itertools

                    param_names = list(extra_params.keys())
                    param_values = list(extra_params.values())

                    for param_combo in itertools.product(*param_values):
                        shape_params = {"shape": shape}
                        # Add each extra parameter
                        for param_name, param_value in zip(param_names, param_combo):
                            shape_params[param_name] = param_value

                        configs.append(
                            BenchmarkConfig(
                                shape_params=shape_params,
                                dtype=dtype,
                                num_iterations=num_iterations,
                                warmup=warmup,
                                use_cudagraph=use_cudagraph,
                                verify=verify,
                                atol=atol,
                                rtol=rtol,
                            )
                        )
                else:
                    # No extra parameters, just use shape
                    configs.append(
                        BenchmarkConfig(
                            shape_params={"shape": shape},
                            dtype=dtype,
                            num_iterations=num_iterations,
                            warmup=warmup,
                            use_cudagraph=use_cudagraph,
                            verify=verify,
                            atol=atol,
                            rtol=rtol,
                        )
                    )

        # Launch distributed workers with proper result collection
        # Use a manager for inter-process communication
        import torch.multiprocessing as mp

        manager = mp.Manager()
        results_dict = manager.dict()
        results_dict['data'] = None

        torch.multiprocessing.spawn(
            _distributed_worker_wrapper,
            args=(self, configs, results_dict),
            nprocs=self.num_gpus,
        )

        # Collect and return results
        results = results_dict.get('data')

        if results is None:
            raise RuntimeError(
                "Distributed benchmark worker failed to return results. "
                "Check worker logs for errors."
            )

        # Check if all configs failed verification
        if not results:
            raise RuntimeError(
                "All benchmark configurations failed correctness verification. "
                "No performance results available. Try running with --no-verify to skip "
                "correctness checks, or increase tolerances with --atol and --rtol."
            )

        return results

    def _init_distributed(self, local_rank: int, world_size: int):
        """
        Initialize distributed environment for worker process.

        Args:
            local_rank: Local rank of this worker
            world_size: Total number of workers
        """
        if self.init_fn is not None:
            # Use custom initialization
            self.init_fn(local_rank, world_size)
        else:
            # Default vLLM distributed initialization
            from vllm.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            from vllm.platforms import current_platform
            from vllm.utils.system_utils import update_environment_variables

            current_platform.seed_everything(0)
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)

            update_environment_variables({
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(self.master_port),
            })

            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=world_size)

    def setup_config(self, local_rank: int, world_size: int, config: BenchmarkConfig):
        """
        Hook for subclasses to perform per-config setup.

        Called before each config is benchmarked. Useful for recreating
        resources that need to be fresh for each config (e.g., FlashInfer workspace).

        Args:
            local_rank: Local rank of this worker
            world_size: Total number of workers
            config: Benchmark configuration that will be run
        """
        pass

    def teardown_config(self):
        """
        Hook for subclasses to perform per-config cleanup.

        Called after each config completes, even if errors occur during that config.
        """
        pass

    def _clone_inputs(self, inputs: tuple) -> tuple:
        """
        Clone input tensors to prevent mutation between baseline and helion runs.

        Args:
            inputs: Tuple of inputs from create_inputs()

        Returns:
            Cloned tuple where tensors are cloned, other types are preserved
        """
        cloned = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                cloned.append(inp.clone())
            else:
                # Non-tensor inputs (floats, ints, etc.) are immutable or copied
                cloned.append(inp)
        return tuple(cloned)

    def _benchmark_worker(
        self,
        local_rank: int,
        world_size: int,
        configs: list[BenchmarkConfig],
    ) -> list[BenchmarkResult]:
        """
        Worker process for distributed benchmarking.

        This runs on each GPU and performs the actual benchmarking work.
        Only rank 0 returns results; other ranks return empty list.

        Args:
            local_rank: Local rank of this worker
            world_size: Total number of workers
            configs: List of benchmark configurations to run

        Returns:
            List of BenchmarkResult objects (only from rank 0)
        """
        # Initialize distributed environment
        self._init_distributed(local_rank, world_size)

        results = []

        for config_idx, config in enumerate(configs):
            if local_rank == 0:
                shape_desc = self.get_shape_description(**config.shape_params)
                print(
                    f"\n[Config {config_idx + 1}/{len(configs)}] "
                    f"{shape_desc}, dtype={config.dtype}"
                )

            # Per-config setup hook (e.g., recreate FlashInfer workspace)
            self.setup_config(local_rank, world_size, config)

            try:
                # Create inputs
                inputs = self.create_inputs(config.dtype, **config.shape_params)

                # Verify correctness if requested
                if config.verify:
                    # Ensure all ranks have created inputs before proceeding
                    dist.barrier()

                    if local_rank == 0:
                        print("  Verifying correctness... ", end="", flush=True)

                    # Clone inputs for baseline to prevent mutation
                    baseline_inputs = self._clone_inputs(inputs)
                    baseline_output = self.run_baseline(*baseline_inputs)

                    # Ensure baseline kernel completes
                    torch.cuda.synchronize()
                    dist.barrier()

                    # Clone inputs for Helion to ensure same starting state
                    helion_inputs = self._clone_inputs(inputs)
                    helion_output = self.run_helion(*helion_inputs)

                    # Ensure Helion kernel completes
                    torch.cuda.synchronize()
                    dist.barrier()

                    # Compare outputs (only rank 0 checks)
                    verification_passed = True
                    if local_rank == 0:
                        # Handle tuple outputs - compare all elements
                        if isinstance(baseline_output, tuple) and isinstance(helion_output, tuple):
                            if len(baseline_output) != len(helion_output):
                                print(f"✗ FAILED: Output tuple length mismatch: baseline={len(baseline_output)}, helion={len(helion_output)}")
                                verification_passed = False
                            else:
                                try:
                                    for i, (base_elem, helion_elem) in enumerate(zip(baseline_output, helion_output)):
                                        torch.testing.assert_close(
                                            helion_elem,
                                            base_elem,
                                            atol=config.atol,
                                            rtol=config.rtol,
                                        )
                                    print("✓ PASSED")
                                except AssertionError as e:
                                    print(f"✗ FAILED at output element {i}: {str(e)[:200]}")
                                    verification_passed = False
                        else:
                            # Single output or convert tuples to single elements
                            if isinstance(baseline_output, tuple):
                                baseline_output = baseline_output[0]
                            if isinstance(helion_output, tuple):
                                helion_output = helion_output[0]

                            try:
                                torch.testing.assert_close(
                                    helion_output,
                                    baseline_output,
                                    atol=config.atol,
                                    rtol=config.rtol,
                                )
                                print("✓ PASSED")
                            except AssertionError as e:
                                print(f"✗ FAILED: {e}")
                                verification_passed = False

                    # Broadcast verification result to all ranks
                    if dist.is_initialized():
                        verification_tensor = torch.tensor(
                            [1 if verification_passed else 0],
                            dtype=torch.int,
                            device="cuda"
                        )
                        dist.broadcast(verification_tensor, src=0)
                        verification_passed = bool(verification_tensor.item())

                    # All ranks must agree to skip if verification failed
                    if not verification_passed:
                        dist.barrier()  # Sync before continuing to next config
                        continue

                # Performance benchmark - create fresh inputs once
                # Note: run_baseline and run_helion internally copy data,
                # so we don't need to clone on each iteration
                inputs_perf = self.create_inputs(config.dtype, **config.shape_params)

                dist.barrier()

                # Benchmark baseline
                if local_rank == 0:
                    print(f"  Benchmarking baseline... ", flush=True)

                def baseline_fn():
                    return self.run_baseline(*inputs_perf)

                baseline_time = self._time_kernel_distributed(
                    baseline_fn,
                    config.num_iterations,
                    config.warmup,
                    config.use_cudagraph,
                )

                if local_rank == 0:
                    print(f"    Baseline: {baseline_time:.4f} ms")

                dist.barrier()

                # Benchmark Helion
                if local_rank == 0:
                    print(f"  Benchmarking Helion... ", flush=True)

                def helion_fn():
                    return self.run_helion(*inputs_perf)

                # Check if Helion supports CUDA graphs
                use_cudagraph_helion = config.use_cudagraph and self.supports_cudagraph()

                # Warn if CUDA graphs are disabled for Helion
                if config.use_cudagraph and not use_cudagraph_helion and local_rank == 0:
                    print(f"    (CUDA graphs disabled for Helion: cross-stream synchronization)")

                helion_time = self._time_kernel_distributed(
                    helion_fn,
                    config.num_iterations,
                    config.warmup,
                    use_cudagraph_helion,
                )

                if local_rank == 0:
                    print(f"    Helion: {helion_time:.4f} ms")

                dist.barrier()

                # Only rank 0 collects results
                if local_rank == 0:
                    shape_desc = self.get_shape_description(**config.shape_params)
                    result = BenchmarkResult(
                        shape_params=config.shape_params,
                        shape_desc=shape_desc,
                        dtype=str(config.dtype),
                        baseline_time_ms=baseline_time,
                        helion_time_ms=helion_time,
                        speedup=baseline_time / helion_time,
                        correctness_passed=True,
                    )
                    results.append(result)
                    print(f"    Speedup: {result.speedup:.2f}x")

                # Barrier at end of iteration to ensure all ranks finish this config
                # before moving to the next one
                dist.barrier()

            finally:
                # Per-config teardown hook (e.g., cleanup FlashInfer workspace)
                self.teardown_config()

        return results if local_rank == 0 else []

    def supports_cudagraph(self) -> bool:
        """
        Check if this benchmark supports CUDA graphs.

        Subclasses can override this to disable CUDA graphs if their kernels
        use cross-stream synchronization or other features incompatible with
        CUDA graph capture.

        Returns:
            True if CUDA graphs are supported, False otherwise
        """
        return True

    def _time_kernel_distributed(
        self,
        kernel_fn: Callable,
        num_iterations: int,
        warmup: int,
        use_cudagraph: bool,
    ) -> float:
        """
        Time a distributed kernel function.

        This uses manual CUDA graph capture/replay with proper barriers
        for distributed synchronization.

        Args:
            kernel_fn: Callable that executes the kernel
            num_iterations: Number of timing iterations
            warmup: Number of warmup iterations
            use_cudagraph: Whether to use CUDA graphs

        Returns:
            Average execution time in milliseconds
        """
        return self.time_kernel(
            kernel_fn=kernel_fn,
            num_iterations=num_iterations,
            warmup=warmup,
            use_cudagraph=use_cudagraph,
            distributed=True,
        )


def _distributed_worker_wrapper(
    local_rank: int,
    benchmark: DistributedKernelBenchmark,
    configs: list[BenchmarkConfig],
    results_dict: dict,
):
    """
    Module-level wrapper for multiprocessing.spawn.

    This must be at module level to be picklable.

    Args:
        local_rank: Local rank of this worker
        benchmark: The DistributedKernelBenchmark instance
        configs: List of benchmark configurations
        results_dict: Shared dictionary for storing results
    """
    world_size = benchmark.num_gpus

    try:
        worker_results = benchmark._benchmark_worker(local_rank, world_size, configs)

        # Rank 0 stores results in shared dictionary
        if local_rank == 0:
            results_dict['data'] = worker_results
            print(f"\n[Rank {local_rank}] Stored {len(worker_results)} results in container")
    except Exception as e:
        print(f"\n[Rank {local_rank}] ERROR in worker: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_results(results: list[BenchmarkResult]):
    """Pretty print benchmark results."""
    if not results:
        return

    print("\n" + "=" * 100)
    print(
        f"{'Shape':<30} {'DType':<12} "
        f"{'Baseline ms':<15} {'Helion ms':<15} {'Speedup':<10}"
    )
    print("=" * 100)

    for result in results:
        print(
            f"{result.shape_desc:<30} "
            f"{result.dtype:<12} "
            f"{result.baseline_time_ms:<15.4f} "
            f"{result.helion_time_ms:<15.4f} "
            f"{result.speedup:<10.2f}x"
        )
    print("=" * 100 + "\n")


def print_summary_statistics(results: list[BenchmarkResult]):
    """Print comprehensive summary statistics."""
    if not results:
        return

    speedups = [r.speedup for r in results]
    baseline_times = [r.baseline_time_ms for r in results]
    helion_times = [r.helion_time_ms for r in results]

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total configurations tested: {len(results)}")
    print()

    print("Speedup:")
    print(f"  Average: {statistics.mean(speedups):.2f}x")
    print(f"  Median:  {statistics.median(speedups):.2f}x")
    print(f"  Min:     {min(speedups):.2f}x")
    print(f"  Max:     {max(speedups):.2f}x")
    print()

    print("Latency (ms):")
    print(
        f"  Baseline - Avg: {statistics.mean(baseline_times):.4f}, "
        f"Min: {min(baseline_times):.4f}, "
        f"Max: {max(baseline_times):.4f}"
    )
    print(
        f"  Helion   - Avg: {statistics.mean(helion_times):.4f}, "
        f"Min: {min(helion_times):.4f}, "
        f"Max: {max(helion_times):.4f}"
    )
    print("=" * 60 + "\n")


def save_results_csv(results: list[BenchmarkResult], filename: str):
    """Save results to CSV file."""
    if not results:
        return

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows([r.to_dict() for r in results])

    print(f"\n✓ Results saved to {filepath}")


def save_results_json(results: list[BenchmarkResult], filename: str):
    """Save results to JSON file."""
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
