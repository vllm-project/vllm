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

# Registry for KernelBenchmark classes
_benchmark_registry: dict[str, type["KernelBenchmark"]] = {}


def time_kernel(
    kernel_fn: Callable,
    num_iterations: int = 1000,
    warmup: int = 10,
    use_cudagraph: bool = False,
) -> float:
    """
    Time a kernel function and return average execution time.

    Args:
        kernel_fn: Callable that executes the kernel
        num_iterations: Number of timing iterations
        warmup: Number of warmup iterations
        use_cudagraph: Whether to use CUDAGraph for timing

    Returns:
        Average execution time in milliseconds
    """
    if use_cudagraph:
        from vllm.triton_utils import triton

        # do_bench_cudagraph returns time in milliseconds
        avg_time_ms = triton.testing.do_bench_cudagraph(kernel_fn, rep=num_iterations)
    else:
        # Warmup
        for _ in range(warmup):
            kernel_fn()
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iterations

    return avg_time_ms


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
    atol: float = 1e-3
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
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        assert cls.benchmark_name != "", "subclass must set benchmark_name"
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

    @abstractmethod
    def get_quick_test_shapes(self) -> list[tuple[list[tuple], torch.dtype]]:
        """
        Get test configurations for quick smoke testing.

        Quick tests should use a small number of representative configurations
        to verify correctness and basic functionality.

        Returns:
            List of (shapes, dtype) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)

        Example:
            [
                ([(1, 8192), (256, 8192), (1024, 16384)], torch.bfloat16),
                ([(1, 8192), (256, 8192)], torch.float16),
            ]
        """
        raise NotImplementedError

    @abstractmethod
    def get_full_test_shapes(self) -> list[tuple[list[tuple], torch.dtype]]:
        """
        Get test configurations for comprehensive benchmarking.

        Full tests should cover a wide range of configurations
        to thoroughly benchmark performance across different scenarios.

        Returns:
            List of (shapes, dtype) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)

        Example:
            [
                ([(1, 1024), (256, 4096), (1024, 8192)], torch.bfloat16),
                ([(1, 1024), (256, 4096), (1024, 8192)], torch.float16),
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

        baseline_time = time_kernel(
            baseline_fn,
            num_iterations=config.num_iterations,
            warmup=config.warmup,
            use_cudagraph=config.use_cudagraph,
        )

        # Benchmark Helion kernel
        def helion_fn():
            return self.run_helion(*inputs)

        helion_time = time_kernel(
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
        for shapes, dtype in test_configs:
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
