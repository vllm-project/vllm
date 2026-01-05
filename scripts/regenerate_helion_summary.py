#!/usr/bin/env python3
"""
Regenerate Helion benchmark summary from existing benchmark results.

This script parses existing benchmark result JSON files and creates a comprehensive
summary report with correct performance metrics.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics

def parse_kernel_results(benchmark_dir: Path) -> Dict[str, Any]:
    """
    Parse kernel benchmark results from a benchmark directory.

    Args:
        benchmark_dir: Path to kernel benchmark directory (e.g., silu_mul_fp8_hidden2048_benchmark)

    Returns:
        Dictionary containing parsed performance data
    """
    # Find the main JSON result file
    json_files = list(benchmark_dir.glob("*_helion_full_*.json"))

    if not json_files:
        print(f"Warning: No result JSON files found in {benchmark_dir}")
        return {}

    # Use the first (and likely only) JSON file
    result_file = json_files[0]

    try:
        with open(result_file, 'r') as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            print(f"Warning: No results found in {result_file}")
            return {}

        # Extract performance metrics
        baseline_times = [r["baseline_time_ms"] for r in results if "baseline_time_ms" in r]
        helion_times = [r["helion_time_ms"] for r in results if "helion_time_ms" in r]
        speedups = [r["speedup"] for r in results if "speedup" in r]

        if not baseline_times or not helion_times or not speedups:
            print(f"Warning: Missing timing data in {result_file}")
            return {}

        # Calculate statistics
        perf_data = {
            "baseline_avg_time_ms": statistics.mean(baseline_times),
            "helion_avg_time_ms": statistics.mean(helion_times),
            "speedup_avg": statistics.mean(speedups),
            "speedup_median": statistics.median(speedups),
            "speedup_min": min(speedups),
            "speedup_max": max(speedups),
            "total_configurations": len(results),
            "timestamp": data.get("timestamp"),
            "device": data.get("device"),
            "all_passed_correctness": all(r.get("correctness_passed", False) for r in results)
        }

        return perf_data

    except Exception as e:
        print(f"Error parsing {result_file}: {e}")
        return {}

def extract_kernel_and_hidden_size(dirname: str) -> tuple[str, int]:
    """
    Extract kernel name and hidden size from directory name.

    Args:
        dirname: Directory name like "silu_mul_fp8_hidden2048_benchmark"

    Returns:
        Tuple of (kernel_name, hidden_size)
    """
    # Pattern: {kernel_name}_hidden{size}_benchmark
    match = re.match(r"(.+)_hidden(\d+)_benchmark", dirname)
    if match:
        kernel_name = match.group(1)
        hidden_size = int(match.group(2))
        return kernel_name, hidden_size

    raise ValueError(f"Cannot parse directory name: {dirname}")

def collect_all_results(results_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Collect all benchmark results from a results directory.

    Args:
        results_dir: Path to directory containing all benchmark results

    Returns:
        Nested dict: {kernel_name: {hidden_size: performance_data}}
    """
    all_results = {}

    # Find all benchmark directories
    benchmark_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith('_benchmark')]

    for benchmark_dir in benchmark_dirs:
        try:
            kernel_name, hidden_size = extract_kernel_and_hidden_size(benchmark_dir.name)

            # Parse performance data
            perf_data = parse_kernel_results(benchmark_dir)
            if not perf_data:
                continue

            # Store in nested structure
            if kernel_name not in all_results:
                all_results[kernel_name] = {}
            all_results[kernel_name][hidden_size] = perf_data

        except Exception as e:
            print(f"Error processing {benchmark_dir}: {e}")
            continue

    return all_results

def load_system_info(results_dir: Path) -> Dict[str, Any]:
    """Load system information from system_info.json"""
    system_file = results_dir / "system_info.json"

    try:
        with open(system_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load system info: {e}")
        return {}

def generate_summary_report(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    system_info: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    Generate comprehensive summary report.

    Args:
        all_results: Collected benchmark results
        system_info: System information
        output_dir: Directory to save report

    Returns:
        Path to generated summary file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = "unknown"

    # Extract GPU name from system info or results
    if system_info.get("gpus"):
        gpu_name = system_info["gpus"][0]["name"].replace(" ", "_")

    summary_filename = f"helion_benchmark_summary_{gpu_name}_{timestamp}.txt"
    summary_path = output_dir / summary_filename

    # Calculate totals
    total_kernels = len(all_results)
    total_combinations = sum(len(hidden_sizes) for hidden_sizes in all_results.values())
    successful_combinations = sum(
        len([hs for hs, data in hidden_sizes.items() if data.get("total_configurations", 0) > 0])
        for hidden_sizes in all_results.values()
    )
    failed_combinations = total_combinations - successful_combinations

    # Generate report content
    report_lines = [
        "HELION KERNEL BENCHMARK SUMMARY",
        "=" * 50,
        "",
        "SYSTEM INFORMATION:",
        f"Timestamp: {system_info.get('timestamp', 'Unknown')}",
        f"Python: {system_info.get('python_version', 'Unknown').split()[0] if system_info.get('python_version') else 'Unknown'}",
        f"PyTorch: {system_info.get('torch_version', 'Unknown')}",
        f"CUDA: {system_info.get('cuda_version', 'Unknown')}",
        "",
        "GPU INFORMATION:"
    ]

    # Add GPU info
    for gpu in system_info.get("gpus", []):
        memory_gb = gpu.get("memory_gb", 0)
        report_lines.extend([
            f"  {gpu.get('name', 'Unknown GPU')} (SM {gpu.get('compute_capability', 'Unknown')})",
            f"    Memory: {memory_gb:.1f} GB",
            f"    SMs: {gpu.get('multiprocessor_count', 'Unknown')}"
        ])

    report_lines.extend([
        "",
        "BENCHMARK SUMMARY:",
        f"  Total kernels: {total_kernels}",
        f"  Total combinations: {total_combinations}",
        f"  Successful: {successful_combinations}",
        f"  Failed: {failed_combinations}",
        "",
        "KERNEL PERFORMANCE BY HIDDEN SIZE:"
    ])

    # Add kernel performance details
    for kernel_name in sorted(all_results.keys()):
        report_lines.append(f"  {kernel_name}:")

        hidden_sizes = all_results[kernel_name]
        for hidden_size in sorted(hidden_sizes.keys()):
            data = hidden_sizes[hidden_size]

            report_lines.extend([
                f"    Hidden size {hidden_size}:",
                f"      Helion avg time: {data.get('helion_avg_time_ms', 0):.4f}ms",
                f"      Baseline avg time: {data.get('baseline_avg_time_ms', 0):.4f}ms",
                f"      Speedup (avg): {data.get('speedup_avg', 0):.2f}x",
                f"      Speedup (median): {data.get('speedup_median', 0):.2f}x",
                f"      Speedup (min-max): {data.get('speedup_min', 0):.2f}x - {data.get('speedup_max', 0):.2f}x",
                f"      Total configurations: {data.get('total_configurations', 0)}",
                f"      Correctness: {'✓ Passed' if data.get('all_passed_correctness') else '✗ Failed'}",
                ""
            ])

        report_lines.append("")

    # Write report
    with open(summary_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return str(summary_path)

def generate_json_summary(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    system_info: Dict[str, Any],
    output_dir: Path
) -> str:
    """Generate JSON summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = "unknown"

    if system_info.get("gpus"):
        gpu_name = system_info["gpus"][0]["name"].replace(" ", "_")

    json_filename = f"helion_benchmark_summary_{gpu_name}_{timestamp}.json"
    json_path = output_dir / json_filename

    # Create JSON structure
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "summary_stats": {
            "total_kernels": len(all_results),
            "total_combinations": sum(len(hidden_sizes) for hidden_sizes in all_results.values()),
            "successful_combinations": sum(
                len([hs for hs, data in hidden_sizes.items() if data.get("total_configurations", 0) > 0])
                for hidden_sizes in all_results.values()
            )
        },
        "kernel_results": all_results
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    return str(json_path)

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Helion benchmark summary from existing results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing benchmark results (e.g., ~/bench/latest_b200)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for summary (default: same as results_dir)"
    )

    args = parser.parse_args()

    # Resolve paths
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return 1

    print(f"Processing benchmark results from: {results_dir}")

    # Load system information
    system_info = load_system_info(results_dir)

    # Collect all benchmark results
    all_results = collect_all_results(results_dir)

    if not all_results:
        print("Error: No benchmark results found!")
        return 1

    print(f"Found results for {len(all_results)} kernels:")
    for kernel_name, hidden_sizes in all_results.items():
        print(f"  {kernel_name}: {len(hidden_sizes)} hidden sizes")

    # Generate reports
    print(f"\nGenerating summary reports in: {output_dir}")

    txt_path = generate_summary_report(all_results, system_info, output_dir)
    json_path = generate_json_summary(all_results, system_info, output_dir)

    print(f"✓ Text summary: {txt_path}")
    print(f"✓ JSON summary: {json_path}")

    return 0

if __name__ == "__main__":
    exit(main())