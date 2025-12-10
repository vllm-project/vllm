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
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def get_system_info() -> Dict:
    """Get comprehensive system and GPU information."""
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        system_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": []
        })

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
            sys.executable, str(autotune_script),
            "--kernels", "all",
            "--force",  # Force re-autotuning
            "--verbose"
        ]

        logger.info("Running command: %s", " ".join(cmd))

        # Run autotuning (no timeout - let it finish naturally)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

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
    iterations: int = 5000,
    warmup: int = 50
) -> Tuple[bool, Optional[Dict]]:
    """
    Run full benchmark for a specific kernel.

    Args:
        kernel_name: Name of the kernel to benchmark
        output_dir: Directory to save benchmark results
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (success, benchmark_results)
    """
    try:
        logger.info("Benchmarking kernel: %s", kernel_name)

        # Map kernel name to benchmark name (kernels have _helion suffix in benchmarks)
        benchmark_name = f"{kernel_name}_helion"

        # Create kernel-specific output directory
        kernel_output_dir = Path(output_dir) / f"{kernel_name}_benchmark"
        kernel_output_dir.mkdir(parents=True, exist_ok=True)

        # Construct benchmark command
        benchmark_script = Path(__file__).parent.parent / "benchmarks" / "benchmark_helion.py"

        cmd = [
            sys.executable, str(benchmark_script),
            "--benchmark", benchmark_name,
            "--mode", "full",
            "--num-iterations", str(iterations),
            "--warmup", str(warmup),
            "--output-dir", str(kernel_output_dir)
        ]

        logger.info("Running benchmark command: %s", " ".join(cmd))

        # Run benchmark
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per kernel
        )
        end_time = time.time()

        duration = end_time - start_time

        if result.returncode == 0:
            logger.info("✓ Benchmark completed for %s (%.2fs)", kernel_name, duration)

            # Parse benchmark results from stdout and save structured data
            benchmark_data = {
                "kernel_name": kernel_name,
                "duration_seconds": duration,
                "iterations": iterations,
                "warmup": warmup,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_dir": str(kernel_output_dir),
                "timestamp": datetime.now().isoformat()
            }

            # Try to extract performance metrics from stdout
            try:
                import re
                # Look for performance summary in the output
                lines = result.stdout.split('\n')
                perf_data = {}

                # Parse performance summary from benchmark output
                summary_section = False
                for line in lines:
                    line = line.strip()

                    # Look for summary statistics section
                    if "Summary Statistics" in line:
                        summary_section = True
                        continue

                    if summary_section:
                        # Parse speedup statistics
                        if "Average:" in line and "x" in line:
                            avg_match = re.search(r'Average:\s*(\d+\.?\d*)x', line)
                            if avg_match:
                                perf_data["speedup_average"] = float(avg_match.group(1))
                        elif "Median:" in line and "x" in line:
                            med_match = re.search(r'Median:\s*(\d+\.?\d*)x', line)
                            if med_match:
                                perf_data["speedup_median"] = float(med_match.group(1))
                        elif "Min:" in line and "x" in line:
                            min_match = re.search(r'Min:\s*(\d+\.?\d*)x', line)
                            if min_match:
                                perf_data["speedup_min"] = float(min_match.group(1))
                        elif "Max:" in line and "x" in line:
                            max_match = re.search(r'Max:\s*(\d+\.?\d*)x', line)
                            if max_match:
                                perf_data["speedup_max"] = float(max_match.group(1))

                        # Parse latency statistics
                        elif "Baseline - Avg:" in line:
                            baseline_match = re.search(r'Baseline - Avg:\s*(\d+\.?\d*)', line)
                            if baseline_match:
                                perf_data["baseline_avg_time_ms"] = float(baseline_match.group(1))
                        elif "Helion   - Avg:" in line:
                            helion_match = re.search(r'Helion   - Avg:\s*(\d+\.?\d*)', line)
                            if helion_match:
                                perf_data["helion_avg_time_ms"] = float(helion_match.group(1))

                    # Parse total configurations tested
                    if "Total configurations tested:" in line:
                        config_match = re.search(r'Total configurations tested:\s*(\d+)', line)
                        if config_match:
                            perf_data["total_configurations"] = int(config_match.group(1))

                if perf_data:
                    benchmark_data["performance_metrics"] = perf_data

            except Exception as e:
                logger.warning("Failed to parse performance metrics from output: %s", e)

            # Save benchmark data to file
            result_file = kernel_output_dir / "benchmark_results.json"
            with open(result_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)

            return True, benchmark_data

        else:
            logger.error("Benchmark failed for %s (return code %d)", kernel_name, result.returncode)
            logger.error("STDOUT: %s", result.stdout[-500:])  # Last 500 chars
            logger.error("STDERR: %s", result.stderr[-500:])

            # Save error information
            error_data = {
                "kernel_name": kernel_name,
                "duration_seconds": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }

            error_file = Path(output_dir) / f"{kernel_name}_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2)

            return False, None

    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out for %s after 30 minutes", kernel_name)
        return False, None
    except Exception as e:
        logger.error("Failed to benchmark %s: %s", kernel_name, e)
        return False, None


def generate_summary_report(
    system_info: Dict,
    benchmark_results: Dict[str, Dict],
    output_dir: str,
    total_duration: float
) -> str:
    """
    Generate a comprehensive summary report.

    Args:
        system_info: System and GPU information
        benchmark_results: Dictionary of kernel_name -> benchmark_data
        output_dir: Output directory
        total_duration: Total benchmarking duration

    Returns:
        Path to the summary report file
    """
    summary = {
        "system_info": system_info,
        "benchmark_summary": {
            "total_kernels": len(benchmark_results),
            "successful_benchmarks": len([r for r in benchmark_results.values() if r is not None]),
            "failed_benchmarks": len([r for r in benchmark_results.values() if r is None]),
            "total_duration_seconds": total_duration,
        },
        "kernel_performance": {},
        "performance_analysis": {}
    }

    # Extract performance metrics for each kernel
    successful_results = {}
    for kernel_name, result in benchmark_results.items():
        if result is not None:
            successful_results[kernel_name] = result

            # Extract key performance metrics from parsed data
            perf_metrics = {}
            if 'performance_metrics' in result:
                parsed_perf = result['performance_metrics']
                perf_metrics.update(parsed_perf)

            # Add duration and other metadata
            perf_metrics['benchmark_duration_seconds'] = result.get('duration_seconds', 0)
            perf_metrics['iterations'] = result.get('iterations', 0)
            perf_metrics['warmup'] = result.get('warmup', 0)

            summary['kernel_performance'][kernel_name] = perf_metrics

    # Generate performance analysis
    if successful_results:
        speedup_averages = [
            metrics.get('speedup_average', 0)
            for metrics in summary['kernel_performance'].values()
            if metrics.get('speedup_average', 0) > 0
        ]

        if speedup_averages:
            summary['performance_analysis'] = {
                "average_speedup": sum(speedup_averages) / len(speedup_averages),
                "max_speedup": max([metrics.get('speedup_max', 0) for metrics in summary['kernel_performance'].values()]),
                "min_speedup": min([metrics.get('speedup_min', 0) for metrics in summary['kernel_performance'].values() if metrics.get('speedup_min', 0) > 0]),
                "kernels_with_speedup": len([s for s in speedup_averages if s > 1.0]),
                "kernels_with_slowdown": len([s for s in speedup_averages if s < 1.0]),
            }

    # Save summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = "unknown"
    if system_info.get("gpus"):
        gpu_name = system_info["gpus"][0]["name"].replace(" ", "_")

    summary_file = Path(output_dir) / f"helion_benchmark_summary_{gpu_name}_{timestamp}.json"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary report saved to: %s", summary_file)

    # Also create a human-readable text summary
    text_summary_file = Path(output_dir) / f"helion_benchmark_summary_{gpu_name}_{timestamp}.txt"

    with open(text_summary_file, 'w') as f:
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
        bench_summary = summary['benchmark_summary']
        f.write("BENCHMARK SUMMARY:\n")
        f.write(f"  Total kernels: {bench_summary['total_kernels']}\n")
        f.write(f"  Successful: {bench_summary['successful_benchmarks']}\n")
        f.write(f"  Failed: {bench_summary['failed_benchmarks']}\n")
        f.write(f"  Total duration: {bench_summary['total_duration_seconds']:.1f}s\n\n")

        # Performance analysis
        if 'performance_analysis' in summary and summary['performance_analysis']:
            perf = summary['performance_analysis']
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write(f"  Average speedup: {perf.get('average_speedup', 0):.2f}x\n")
            f.write(f"  Max speedup: {perf.get('max_speedup', 0):.2f}x\n")
            f.write(f"  Min speedup: {perf.get('min_speedup', 0):.2f}x\n")
            f.write(f"  Kernels with speedup (>1x): {perf.get('kernels_with_speedup', 0)}\n")
            f.write(f"  Kernels with slowdown (<1x): {perf.get('kernels_with_slowdown', 0)}\n\n")

        # Individual kernel performance
        f.write("KERNEL PERFORMANCE:\n")
        for kernel_name, metrics in summary['kernel_performance'].items():
            f.write(f"  {kernel_name}:\n")
            f.write(f"    Helion avg time: {metrics.get('helion_avg_time_ms', 0):.4f}ms\n")
            f.write(f"    Baseline avg time: {metrics.get('baseline_avg_time_ms', 0):.4f}ms\n")
            f.write(f"    Speedup (avg): {metrics.get('speedup_average', 0):.2f}x\n")
            f.write(f"    Speedup (median): {metrics.get('speedup_median', 0):.2f}x\n")
            f.write(f"    Speedup (min-max): {metrics.get('speedup_min', 0):.2f}x - {metrics.get('speedup_max', 0):.2f}x\n")
            f.write(f"    Total configurations: {metrics.get('total_configurations', 0)}\n")
            f.write(f"    Benchmark duration: {metrics.get('benchmark_duration_seconds', 0):.1f}s\n")
            f.write(f"    Iterations: {metrics.get('iterations', 0)}\n")
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
        help="Output directory for all results (default: ./helion_benchmark_results_<timestamp>)"
    )

    parser.add_argument(
        "--skip-autotune",
        action="store_true",
        help="Skip autotuning phase and use existing configs"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of benchmark iterations per kernel (default: 5000)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations per kernel (default: 50)"
    )

    parser.add_argument(
        "--kernels",
        nargs="+",
        help="Specific kernels to benchmark (default: all kernels)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging
        logging.getLogger("vllm").setLevel(logging.DEBUG)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Configure vLLM to enable all custom ops for benchmarking
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["all"]
        )
    )
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
    with open(output_dir / "system_info.json", 'w') as f:
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
            if len(args.kernels) == 1 and args.kernels[0].lower() == 'all':
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

        logger.info("Will benchmark %d kernels: %s", len(helion_kernels), list(helion_kernels.keys()))

        # Phase 3: Benchmarking
        logger.info("=" * 60)
        logger.info("PHASE 2: BENCHMARKING ALL KERNELS")
        logger.info("=" * 60)

        benchmark_results = {}
        successful_benchmarks = 0
        failed_benchmarks = 0

        benchmark_start = time.time()

        for i, kernel_name in enumerate(helion_kernels.keys(), 1):
            logger.info("Benchmarking %d/%d: %s", i, len(helion_kernels), kernel_name)

            success, result = run_kernel_benchmark(
                kernel_name,
                str(output_dir),
                args.iterations,
                args.warmup
            )

            benchmark_results[kernel_name] = result

            if success:
                successful_benchmarks += 1
            else:
                failed_benchmarks += 1
                logger.warning("Failed to benchmark %s", kernel_name)

        benchmark_duration = time.time() - benchmark_start
        total_duration = time.time() - total_start_time

        # Phase 4: Generate comprehensive report
        logger.info("=" * 60)
        logger.info("PHASE 3: GENERATING REPORTS")
        logger.info("=" * 60)

        summary_file = generate_summary_report(
            system_info,
            benchmark_results,
            str(output_dir),
            total_duration
        )

        # Final summary
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info("Total kernels: %d", len(helion_kernels))
        logger.info("Successful benchmarks: %d", successful_benchmarks)
        logger.info("Failed benchmarks: %d", failed_benchmarks)
        logger.info("Total duration: %.1fs", total_duration)
        logger.info("Results saved to: %s", output_dir)
        logger.info("Summary report: %s", summary_file)

        if failed_benchmarks > 0:
            logger.warning("Some benchmarks failed. Check individual kernel logs for details.")
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