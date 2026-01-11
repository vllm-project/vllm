#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive Helion kernel benchmarking script for multi-GPU architecture analysis.

This script performs a complete benchmarking workflow:
1. Forces autotuning of all registered Helion kernels
2. Runs full benchmarks on all kernels
3. Generates detailed performance reports
4. Creates comprehensive summaries for cross-architecture comparison

Usage:
    # Run complete benchmark suite
    python scripts/comprehensive_helion_benchmark.py

    # Custom output directory
    python scripts/comprehensive_helion_benchmark.py --output-dir ./benchmark_results

    # Skip autotuning (use existing configs)
    python scripts/comprehensive_helion_benchmark.py --skip-autotune

    # Custom benchmark parameters
    python scripts/comprehensive_helion_benchmark.py --iterations 10000 --warmup 100


Requirements:
    - CUDA GPU available
    - Helion package installed
    - vLLM environment setup
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add vLLM to path if not already available
try:
    from vllm.compilation.helion.config_manager import ConfigManager
    from vllm.compilation.helion.register import (
        HELION_AVAILABLE,
        get_registered_kernels,
    )
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.logger import init_logger
except ImportError as e:
    print(f"Error importing vLLM: {e}")
    print("Please ensure vLLM is installed and in your Python path")
    sys.exit(1)

logger = init_logger("vllm.scripts.comprehensive_helion_benchmark")


def cleanup_gpu_resources():
    """
    Clean up GPU resources between kernel benchmarks to prevent conflicts.

    This helps prevent hangs when transitioning between different kernel types
    by clearing CUDA contexts, memory, and compilation caches.
    """
    import gc

    try:
        if torch.cuda.is_available():
            # Clear GPU memory cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear torch compilation cache
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()

            # Synchronize all CUDA streams
            torch.cuda.synchronize()

            # Reset peak memory stats for clean measurements
            torch.cuda.reset_peak_memory_stats()

            logger.debug("GPU resources cleaned up successfully")

    except Exception as e:
        logger.warning("Failed to cleanup GPU resources: %s", e)


def get_system_info() -> dict:
    """Get comprehensive system and GPU information."""
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        system_info.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [],
            }
        )

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "device_id": i,
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
            system_info["gpus"].append(gpu_info)

    return system_info


def check_requirements() -> bool:
    """Check if all requirements are met for benchmarking."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Helion benchmarking requires GPU.")
        return False

    if not HELION_AVAILABLE:
        logger.error("Helion is not installed. Please install Helion package.")
        return False

    return True


def run_autotune() -> bool:
    """
    Run autotuning for all Helion kernels with force option.
    Uses the default configuration location.

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting forced autotuning of all Helion kernels...")

        # Construct autotune command
        autotune_script = Path(__file__).parent / "autotune_helion_kernels.py"
        cmd = [
            sys.executable,
            str(autotune_script),
            "--kernels",
            "all",
            "--force",  # Force re-autotuning
            "--verbose",
        ]

        logger.info("Running command: %s", " ".join(cmd))

        # Run autotuning (no timeout - let it finish naturally)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Autotuning completed successfully")
            return True
        else:
            logger.error("Autotuning failed with return code %d", result.returncode)
            logger.error("STDOUT: %s", result.stdout)
            logger.error("STDERR: %s", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("Autotuning timed out unexpectedly")
        return False
    except Exception as e:
        logger.error("Failed to run autotuning: %s", e)
        return False


def run_kernel_benchmark(
    kernel_name: str,
    output_dir: str,
    hidden_size: int,
    iterations: int = 5000,
    warmup: int = 50,
    no_capture: bool = True,
) -> tuple[bool, dict | None]:
    """
    Run full benchmark for a specific kernel with a specific hidden_size.

    Args:
        kernel_name: Name of the kernel to benchmark
        output_dir: Directory to save benchmark results
        hidden_size: Hidden size to filter shapes for
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (success, benchmark_results)
    """
    try:
        logger.info(
            "Benchmarking kernel: %s (hidden_size=%d)", kernel_name, hidden_size
        )

        # Map kernel name to benchmark name (kernels have _helion suffix in benchmarks)
        benchmark_name = f"{kernel_name}_helion"

        # Create kernel-specific output directory with hidden size
        kernel_output_dir = (
            Path(output_dir) / f"{kernel_name}_hidden{hidden_size}_benchmark"
        )
        kernel_output_dir.mkdir(parents=True, exist_ok=True)

        # Construct benchmark command
        benchmark_script = (
            Path(__file__).parent.parent / "benchmarks" / "benchmark_helion.py"
        )

        cmd = [
            sys.executable,
            str(benchmark_script),
            "--benchmark",
            benchmark_name,
            "--mode",
            "full",
            "--num-iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--hidden-size",
            str(hidden_size),
            "--output-dir",
            str(kernel_output_dir),
        ]

        logger.info("Running benchmark command: %s", " ".join(cmd))

        # Debug: Print working directory and environment info
        import os

        current_dir = os.getcwd()
        logger.info("Current working directory: %s", current_dir)
        logger.info("Python executable: %s", sys.executable)
        logger.info(
            "CUDA_VISIBLE_DEVICES: %s",
            os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        )

        # Run benchmark with real-time progress display
        start_time = time.time()

        # Prepare environment - inherit all current environment variables
        env = os.environ.copy()

        # Ensure proper CUDA environment
        env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering

        # Add Triton-specific environment variables to help with subprocess execution
        env["TRITON_CACHE_DIR"] = str(
            Path.home() / ".triton"
        )  # Explicit cache directory
        env["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow asynchronous CUDA launches
        env["TORCH_CUDA_ARCH_LIST"] = ""  # Let it auto-detect

        # Debug: Print Triton-related environment
        logger.info("Triton cache dir: %s", env.get("TRITON_CACHE_DIR"))
        logger.info("CUDA_LAUNCH_BLOCKING: %s", env.get("CUDA_LAUNCH_BLOCKING"))

        # Check if we're in a subprocess already (for debugging nested subprocess issues)
        if "SUBPROCESS_LEVEL" in os.environ:
            env["SUBPROCESS_LEVEL"] = str(int(os.environ["SUBPROCESS_LEVEL"]) + 1)
        else:
            env["SUBPROCESS_LEVEL"] = "1"
        logger.info("Subprocess level: %s", env["SUBPROCESS_LEVEL"])

        if no_capture:
            # Run without output capture (like direct execution) for debugging
            logger.info("Running in no-capture mode (direct execution)")
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=current_dir,
                    timeout=7200,  # 2 hour timeout
                )
                end_time = time.time()

                # Create a mock result for consistency
                class MockResult:
                    def __init__(self, returncode):
                        self.returncode = returncode
                        self.stdout = "No output captured (no-capture mode)"
                        self.stderr = ""

                result = MockResult(result.returncode)

            except subprocess.TimeoutExpired:
                raise

        else:
            # Use Popen for real-time output streaming with proper environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=0,  # Unbuffered for immediate output
                universal_newlines=True,
                env=env,  # Explicitly pass environment
                cwd=current_dir,  # Ensure same working directory
            )

            # Capture output while displaying progress
            stdout_lines = []
            stderr_lines = []

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] 🚀 Starting benchmark for {kernel_name} (hidden_size={hidden_size})..."
            )
            print("=" * 60)

            try:
                # Stream output in real-time
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        # Add timestamp to all progress output
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        # Display progress indicators with timestamps
                        if any(
                            keyword in line.lower()
                            for keyword in [
                                "iteration",
                                "warmup",
                                "benchmark",
                                "ms",
                                "speedup",
                                "verifying",
                                "correctness",
                                "shape",
                            ]
                        ):
                            print(f"[{timestamp}] 📊 {line}")
                        elif any(
                            keyword in line.lower()
                            for keyword in ["error", "failed", "exception"]
                        ):
                            print(f"[{timestamp}] ❌ {line}")
                        elif any(
                            keyword in line.lower()
                            for keyword in ["completed", "success", "done"]
                        ):
                            print(f"[{timestamp}] ✅ {line}")
                        else:
                            print(f"[{timestamp}]    {line}")

                        stdout_lines.append(line)

                # Wait for process to complete
                return_code = process.wait(timeout=7200)  # 2 hour timeout per kernel

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise

            # Create result object similar to subprocess.run
            class MockResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            result = MockResult(
                returncode=return_code,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )

        end_time = time.time()

        duration = end_time - start_time

        if result.returncode == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print("=" * 60)
            print(
                f"[{timestamp}] ✅ Benchmark COMPLETED for {kernel_name} (hidden_size={hidden_size}) in {duration:.2f}s"
            )
            print("=" * 60)
            logger.info("✓ Benchmark completed for %s (%.2fs)", kernel_name, duration)

            # Parse benchmark results from stdout and save structured data
            benchmark_data = {
                "kernel_name": kernel_name,
                "hidden_size": hidden_size,
                "duration_seconds": duration,
                "iterations": iterations,
                "warmup": warmup,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_dir": str(kernel_output_dir),
                "timestamp": datetime.now().isoformat(),
            }

            # Try to extract performance metrics from stdout
            try:
                import re
                import statistics

                perf_data = {}

                # Method 1: Parse from the generated JSON file (most reliable)
                json_files = list(kernel_output_dir.glob("*_helion_full_*.json"))

                if json_files:
                    try:
                        with open(json_files[0]) as f:
                            json_data = json.load(f)

                        results = json_data.get("results", [])
                        if results:
                            # Calculate performance statistics from JSON data
                            baseline_times = [
                                r["baseline_time_ms"]
                                for r in results
                                if "baseline_time_ms" in r
                            ]
                            helion_times = [
                                r["helion_time_ms"]
                                for r in results
                                if "helion_time_ms" in r
                            ]
                            speedups = [r["speedup"] for r in results if "speedup" in r]

                            if baseline_times and helion_times and speedups:
                                perf_data = {
                                    "baseline_avg_time_ms": statistics.mean(
                                        baseline_times
                                    ),
                                    "helion_avg_time_ms": statistics.mean(helion_times),
                                    "speedup_average": statistics.mean(speedups),
                                    "speedup_median": statistics.median(speedups),
                                    "speedup_min": min(speedups),
                                    "speedup_max": max(speedups),
                                    "total_configurations": len(results),
                                }
                                logger.debug(
                                    "Parsed performance data from JSON: %d configurations",
                                    len(results),
                                )

                    except Exception as e:
                        logger.warning("Failed to parse JSON performance data: %s", e)

                # Method 2: Fallback to stdout parsing with timestamp handling
                if not perf_data:
                    # Look for performance summary in the output
                    lines = result.stdout.split("\n")

                    # Parse performance summary from benchmark output
                    summary_section = False
                    for line in lines:
                        # Clean line: remove timestamps and emoji prefixes
                        clean_line = line.strip()

                        # Remove timestamp pattern [HH:MM:SS]
                        clean_line = re.sub(
                            r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", clean_line
                        )

                        # Remove emoji prefixes (📊, ❌, ✅, etc.)
                        clean_line = re.sub(r"^[📊❌✅⏰🚀]\s*", "", clean_line)

                        # Remove extra whitespace
                        clean_line = clean_line.strip()

                        if not clean_line:
                            continue

                        # Look for summary statistics section
                        if "Summary Statistics" in clean_line:
                            summary_section = True
                            continue

                        if summary_section:
                            # Parse speedup statistics
                            if "Average:" in clean_line and "x" in clean_line:
                                avg_match = re.search(
                                    r"Average:\s*(\d+\.?\d*)x", clean_line
                                )
                                if avg_match:
                                    perf_data["speedup_average"] = float(
                                        avg_match.group(1)
                                    )
                            elif "Median:" in clean_line and "x" in clean_line:
                                med_match = re.search(
                                    r"Median:\s*(\d+\.?\d*)x", clean_line
                                )
                                if med_match:
                                    perf_data["speedup_median"] = float(
                                        med_match.group(1)
                                    )
                            elif "Min:" in clean_line and "x" in clean_line:
                                min_match = re.search(
                                    r"Min:\s*(\d+\.?\d*)x", clean_line
                                )
                                if min_match:
                                    perf_data["speedup_min"] = float(min_match.group(1))
                            elif "Max:" in clean_line and "x" in clean_line:
                                max_match = re.search(
                                    r"Max:\s*(\d+\.?\d*)x", clean_line
                                )
                                if max_match:
                                    perf_data["speedup_max"] = float(max_match.group(1))

                            # Parse latency statistics
                            elif "Baseline - Avg:" in clean_line:
                                baseline_match = re.search(
                                    r"Baseline - Avg:\s*(\d+\.?\d*)", clean_line
                                )
                                if baseline_match:
                                    perf_data["baseline_avg_time_ms"] = float(
                                        baseline_match.group(1)
                                    )
                            elif "Helion   - Avg:" in clean_line:
                                helion_match = re.search(
                                    r"Helion\s+- Avg:\s*(\d+\.?\d*)", clean_line
                                )
                                if helion_match:
                                    perf_data["helion_avg_time_ms"] = float(
                                        helion_match.group(1)
                                    )

                        # Also parse timing outside summary section (fallback)
                        if (
                            "Baseline - Avg:" in clean_line
                            and "baseline_avg_time_ms" not in perf_data
                        ):
                            baseline_match = re.search(
                                r"Baseline - Avg:\s*(\d+\.?\d*)", clean_line
                            )
                            if baseline_match:
                                perf_data["baseline_avg_time_ms"] = float(
                                    baseline_match.group(1)
                                )
                        elif (
                            "Helion   - Avg:" in clean_line
                            and "helion_avg_time_ms" not in perf_data
                        ):
                            helion_match = re.search(
                                r"Helion\s+- Avg:\s*(\d+\.?\d*)", clean_line
                            )
                            if helion_match:
                                perf_data["helion_avg_time_ms"] = float(
                                    helion_match.group(1)
                                )

                        # Parse total configurations tested
                        if "Total configurations tested:" in clean_line:
                            config_match = re.search(
                                r"Total configurations tested:\s*(\d+)", clean_line
                            )
                            if config_match:
                                perf_data["total_configurations"] = int(
                                    config_match.group(1)
                                )

                if perf_data:
                    benchmark_data["performance_metrics"] = perf_data

            except Exception as e:
                logger.warning("Failed to parse performance metrics from output: %s", e)

            # Save benchmark data to file
            result_file = kernel_output_dir / "benchmark_results.json"
            with open(result_file, "w") as f:
                json.dump(benchmark_data, f, indent=2)

            return True, benchmark_data

        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print("=" * 60)
            print(
                f"[{timestamp}] ❌ Benchmark FAILED for {kernel_name} (hidden_size={hidden_size}) after {duration:.2f}s"
            )
            print(f"[{timestamp}]    Return code: {result.returncode}")
            print("=" * 60)
            logger.error(
                "Benchmark failed for %s (return code %d)",
                kernel_name,
                result.returncode,
            )
            logger.error("STDOUT: %s", result.stdout[-500:])  # Last 500 chars
            logger.error("STDERR: %s", result.stderr[-500:])

            # Save error information
            error_data = {
                "kernel_name": kernel_name,
                "hidden_size": hidden_size,
                "duration_seconds": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

            error_file = (
                Path(output_dir) / f"{kernel_name}_hidden{hidden_size}_error.json"
            )
            with open(error_file, "w") as f:
                json.dump(error_data, f, indent=2)

            return False, None

    except subprocess.TimeoutExpired:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("=" * 60)
        print(
            f"[{timestamp}] ⏰ Benchmark TIMED OUT for {kernel_name} (hidden_size={hidden_size}) after 2 hours"
        )
        print("=" * 60)
        logger.error("Benchmark timed out for %s after 2 hours", kernel_name)
        return False, None
    except Exception as e:
        logger.error("Failed to benchmark %s: %s", kernel_name, e)
        return False, None


def generate_summary_report(
    system_info: dict,
    benchmark_results: dict[str, dict[int, dict]],
    output_dir: str,
    total_duration: float,
) -> str:
    """
    Generate a comprehensive summary report.

    Args:
        system_info: System and GPU information
        benchmark_results: Dictionary of kernel_name -> {hidden_size -> benchmark_data}
        output_dir: Output directory
        total_duration: Total benchmarking duration

    Returns:
        Path to the summary report file
    """
    # Count successful and failed benchmarks across all combinations
    total_benchmarks = 0
    successful_benchmarks = 0
    failed_benchmarks = 0

    for kernel_name, hidden_size_results in benchmark_results.items():
        for hidden_size, result in hidden_size_results.items():
            total_benchmarks += 1
            if result is not None:
                successful_benchmarks += 1
            else:
                failed_benchmarks += 1

    summary = {
        "system_info": system_info,
        "benchmark_summary": {
            "total_kernels": len(benchmark_results),
            "total_benchmarks": total_benchmarks,
            "successful_benchmarks": successful_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "total_duration_seconds": total_duration,
        },
        "kernel_performance": {},
        "performance_analysis": {},
    }

    # Extract performance metrics for each kernel-hidden_size combination
    successful_results = {}
    for kernel_name, hidden_size_results in benchmark_results.items():
        kernel_perf = {}

        for hidden_size, result in hidden_size_results.items():
            if result is not None:
                combination_key = f"{kernel_name}_hidden{hidden_size}"
                successful_results[combination_key] = result

                # Extract key performance metrics from parsed data
                perf_metrics = {}
                if "performance_metrics" in result:
                    parsed_perf = result["performance_metrics"]
                    perf_metrics.update(parsed_perf)

                # Add duration and other metadata
                perf_metrics["benchmark_duration_seconds"] = result.get(
                    "duration_seconds", 0
                )
                perf_metrics["iterations"] = result.get("iterations", 0)
                perf_metrics["warmup"] = result.get("warmup", 0)
                perf_metrics["hidden_size"] = hidden_size

                kernel_perf[hidden_size] = perf_metrics

        if kernel_perf:
            summary["kernel_performance"][kernel_name] = kernel_perf

    # Generate performance analysis
    if successful_results:
        # Collect all speedup averages from all kernel-hidden_size combinations
        speedup_averages = []
        all_speedups_max = []
        all_speedups_min = []

        for kernel_name, kernel_perf in summary["kernel_performance"].items():
            for hidden_size, metrics in kernel_perf.items():
                if metrics.get("speedup_average", 0) > 0:
                    speedup_averages.append(metrics["speedup_average"])
                if metrics.get("speedup_max", 0) > 0:
                    all_speedups_max.append(metrics["speedup_max"])
                if metrics.get("speedup_min", 0) > 0:
                    all_speedups_min.append(metrics["speedup_min"])

        if speedup_averages:
            summary["performance_analysis"] = {
                "average_speedup": sum(speedup_averages) / len(speedup_averages),
                "max_speedup": max(all_speedups_max) if all_speedups_max else 0,
                "min_speedup": min(all_speedups_min) if all_speedups_min else 0,
                "combinations_with_speedup": len(
                    [s for s in speedup_averages if s > 1.0]
                ),
                "combinations_with_slowdown": len(
                    [s for s in speedup_averages if s < 1.0]
                ),
                "total_combinations_tested": len(speedup_averages),
            }

    # Save summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = "unknown"
    if system_info.get("gpus"):
        gpu_name = system_info["gpus"][0]["name"].replace(" ", "_")

    summary_file = (
        Path(output_dir) / f"helion_benchmark_summary_{gpu_name}_{timestamp}.json"
    )

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary report saved to: %s", summary_file)

    # Also create a human-readable text summary
    text_summary_file = (
        Path(output_dir) / f"helion_benchmark_summary_{gpu_name}_{timestamp}.txt"
    )

    with open(text_summary_file, "w") as f:
        f.write("HELION KERNEL BENCHMARK SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # System info
        f.write("SYSTEM INFORMATION:\n")
        f.write(f"Timestamp: {system_info['timestamp']}\n")
        f.write(f"Python: {system_info['python_version'].split()[0]}\n")
        f.write(f"PyTorch: {system_info['torch_version']}\n")
        f.write(f"CUDA: {system_info.get('cuda_version', 'N/A')}\n\n")

        # GPU info
        if system_info.get("gpus"):
            f.write("GPU INFORMATION:\n")
            for gpu in system_info["gpus"]:
                f.write(f"  {gpu['name']} (SM {gpu['major']}.{gpu['minor']})\n")
                f.write(f"    Memory: {gpu['total_memory'] / 1024**3:.1f} GB\n")
                f.write(f"    SMs: {gpu['multi_processor_count']}\n")
        f.write("\n")

        # Benchmark summary
        bench_summary = summary["benchmark_summary"]
        f.write("BENCHMARK SUMMARY:\n")
        f.write(f"  Total kernels: {bench_summary['total_kernels']}\n")
        f.write(f"  Total combinations: {bench_summary['total_benchmarks']}\n")
        f.write(f"  Successful: {bench_summary['successful_benchmarks']}\n")
        f.write(f"  Failed: {bench_summary['failed_benchmarks']}\n")
        f.write(f"  Total duration: {bench_summary['total_duration_seconds']:.1f}s\n\n")

        # Performance analysis
        if "performance_analysis" in summary and summary["performance_analysis"]:
            perf = summary["performance_analysis"]
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write(f"  Average speedup: {perf.get('average_speedup', 0):.2f}x\n")
            f.write(f"  Max speedup: {perf.get('max_speedup', 0):.2f}x\n")
            f.write(f"  Min speedup: {perf.get('min_speedup', 0):.2f}x\n")
            f.write(
                f"  Combinations with speedup (>1x): {perf.get('combinations_with_speedup', 0)}\n"
            )
            f.write(
                f"  Combinations with slowdown (<1x): {perf.get('combinations_with_slowdown', 0)}\n"
            )
            f.write(
                f"  Total combinations tested: {perf.get('total_combinations_tested', 0)}\n\n"
            )

        # Individual kernel performance by hidden size
        f.write("KERNEL PERFORMANCE BY HIDDEN SIZE:\n")
        for kernel_name, kernel_perf in summary["kernel_performance"].items():
            f.write(f"  {kernel_name}:\n")
            for hidden_size, metrics in kernel_perf.items():
                f.write(f"    Hidden size {hidden_size}:\n")
                f.write(
                    f"      Helion avg time: {metrics.get('helion_avg_time_ms', 0):.4f}ms\n"
                )
                f.write(
                    f"      Baseline avg time: {metrics.get('baseline_avg_time_ms', 0):.4f}ms\n"
                )
                f.write(
                    f"      Speedup (avg): {metrics.get('speedup_average', 0):.2f}x\n"
                )
                f.write(
                    f"      Speedup (median): {metrics.get('speedup_median', 0):.2f}x\n"
                )
                f.write(
                    f"      Speedup (min-max): {metrics.get('speedup_min', 0):.2f}x - {metrics.get('speedup_max', 0):.2f}x\n"
                )
                f.write(
                    f"      Total configurations: {metrics.get('total_configurations', 0)}\n"
                )
                f.write(
                    f"      Benchmark duration: {metrics.get('benchmark_duration_seconds', 0):.1f}s\n"
                )
                f.write(f"      Iterations: {metrics.get('iterations', 0)}\n")
                f.write("\n")
            f.write("\n")

    logger.info("Text summary saved to: %s", text_summary_file)

    return str(summary_file)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Helion kernel benchmarking suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for all results (default: ./helion_benchmark_results_<timestamp>)",
    )

    parser.add_argument(
        "--skip-autotune",
        action="store_true",
        help="Skip autotuning phase and use existing configs",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of benchmark iterations per kernel (default: 1000)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations per kernel (default: 50)",
    )

    parser.add_argument(
        "--kernels",
        nargs="+",
        help="Specific kernels to benchmark (default: all kernels)",
    )

    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[2048, 4096, 5120, 8192],
        help="Hidden sizes to test for each kernel (default: [2048, 4096, 5120, 8192])",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture benchmark output (default: no-capture for debugging hangs)",
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging

        logging.getLogger("vllm").setLevel(logging.DEBUG)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Configure vLLM to enable all custom ops for benchmarking
    vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    set_current_vllm_config(vllm_config)
    logger.info("Enabled all custom ops for benchmarking")

    # Get system information
    system_info = get_system_info()
    logger.info("System info: %d GPU(s) detected", len(system_info.get("gpus", [])))

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_name = "unknown"
        if system_info.get("gpus"):
            gpu_name = system_info["gpus"][0]["name"].replace(" ", "_")
        output_dir = Path(f"helion_benchmark_results_{gpu_name}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Save system info
    with open(output_dir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    total_start_time = time.time()

    try:
        # Phase 1: Autotuning
        if not args.skip_autotune:
            logger.info("=" * 60)
            logger.info("PHASE 1: AUTOTUNING ALL HELION KERNELS")
            logger.info("=" * 60)

            autotune_start = time.time()
            if not run_autotune():
                logger.error("Autotuning failed, aborting benchmark")
                sys.exit(1)
            autotune_duration = time.time() - autotune_start
            logger.info("Autotuning completed in %.1fs", autotune_duration)
        else:
            logger.info("Skipping autotuning phase")

        # Phase 2: Get kernels to benchmark
        helion_kernels = get_registered_kernels()
        if not helion_kernels:
            logger.error("No Helion kernels found in registry")
            sys.exit(1)

        # Filter kernels if specified
        if args.kernels:
            if len(args.kernels) == 1 and args.kernels[0].lower() == "all":
                pass  # Keep all kernels
            else:
                filtered_kernels = {}
                missing_kernels = []
                for kernel_name in args.kernels:
                    if kernel_name in helion_kernels:
                        filtered_kernels[kernel_name] = helion_kernels[kernel_name]
                    else:
                        missing_kernels.append(kernel_name)

                if missing_kernels:
                    logger.error("Kernel(s) not found: %s", missing_kernels)
                    logger.error("Available kernels: %s", list(helion_kernels.keys()))
                    sys.exit(1)

                helion_kernels = filtered_kernels

        logger.info(
            "Will benchmark %d kernels: %s",
            len(helion_kernels),
            list(helion_kernels.keys()),
        )
        logger.info("Hidden sizes to test: %s", args.hidden_sizes)

        # Phase 3: Benchmarking
        logger.info("=" * 60)
        logger.info("PHASE 2: BENCHMARKING ALL KERNELS ACROSS HIDDEN SIZES")
        logger.info("=" * 60)

        benchmark_results = {}
        successful_benchmarks = 0
        failed_benchmarks = 0
        total_combinations = len(helion_kernels) * len(args.hidden_sizes)

        benchmark_start = time.time()

        # Initial cleanup before starting benchmarks
        logger.info("Performing initial GPU cleanup before benchmarking...")
        cleanup_gpu_resources()

        combination_count = 0
        for kernel_name in helion_kernels.keys():
            kernel_results = {}
            for hidden_size in args.hidden_sizes:
                combination_count += 1
                logger.info(
                    "Benchmarking %d/%d: %s (hidden_size=%d)",
                    combination_count,
                    total_combinations,
                    kernel_name,
                    hidden_size,
                )

                success, result = run_kernel_benchmark(
                    kernel_name,
                    str(output_dir),
                    hidden_size,
                    args.iterations,
                    args.warmup,
                    not args.capture,
                )

                kernel_results[hidden_size] = result

                if success:
                    successful_benchmarks += 1
                else:
                    failed_benchmarks += 1
                    logger.warning(
                        "Failed to benchmark %s (hidden_size=%d)",
                        kernel_name,
                        hidden_size,
                    )

                # Light cleanup after each configuration to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            benchmark_results[kernel_name] = kernel_results

            # Cleanup between kernels to prevent resource conflicts
            logger.info("Cleaning up GPU resources before next kernel...")
            cleanup_gpu_resources()

        benchmark_duration = time.time() - benchmark_start
        total_duration = time.time() - total_start_time

        # Phase 4: Generate comprehensive report
        logger.info("=" * 60)
        logger.info("PHASE 3: GENERATING REPORTS")
        logger.info("=" * 60)

        summary_file = generate_summary_report(
            system_info, benchmark_results, str(output_dir), total_duration
        )

        # Final summary
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info("Total kernels: %d", len(helion_kernels))
        logger.info("Hidden sizes tested: %d", len(args.hidden_sizes))
        logger.info("Total combinations: %d", total_combinations)
        logger.info("Successful benchmarks: %d", successful_benchmarks)
        logger.info("Failed benchmarks: %d", failed_benchmarks)
        logger.info("Total duration: %.1fs", total_duration)
        logger.info("Results saved to: %s", output_dir)
        logger.info("Summary report: %s", summary_file)

        if failed_benchmarks > 0:
            logger.warning(
                "Some benchmarks failed. Check individual kernel logs for details."
            )
            sys.exit(1)
        else:
            logger.info("All benchmarks completed successfully!")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        raise


if __name__ == "__main__":
    main()
