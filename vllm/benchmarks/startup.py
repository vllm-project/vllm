# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the cold and warm startup time of vLLM models.

This script measures total startup time (including model loading, compilation,
and cache operations) for both cold and warm scenarios:
- Cold startup: Fresh start with no caches (temporary cache directories)
- Warm startup: Using cached compilation and model info
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from tqdm import tqdm

from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from vllm.engine.arg_utils import EngineArgs


@contextmanager
def cold_startup():
    """
    Context manager to measure cold startup time:
    1. Uses a temporary directory for vLLM cache to avoid any pollution
       between cold startup iterations.
    2. Uses inductor's fresh_cache to clear torch.compile caches.
    """
    from torch._inductor.utils import fresh_cache

    # Use temporary directory for caching to avoid any pollution between cold startups
    original_cache_root = os.environ.get("VLLM_CACHE_ROOT")
    temp_cache_dir = tempfile.mkdtemp(prefix="vllm_startup_bench_cold_")
    try:
        os.environ["VLLM_CACHE_ROOT"] = temp_cache_dir
        with fresh_cache():
            yield
    finally:
        # Clean up temporary cache directory
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        if original_cache_root:
            os.environ["VLLM_CACHE_ROOT"] = original_cache_root
        else:
            os.environ.pop("VLLM_CACHE_ROOT", None)


def run_startup_in_subprocess(engine_args, result_queue):
    """
    Run LLM startup in a subprocess and return timing metrics via a queue.
    This ensures complete isolation between iterations.
    """
    try:
        # Import inside the subprocess to avoid issues with forking
        from vllm import LLM

        # Measure total startup time
        start_time = time.perf_counter()

        llm = LLM(**dataclasses.asdict(engine_args))

        total_startup_time = time.perf_counter() - start_time

        # Extract compilation time if available
        compilation_time = 0.0
        if hasattr(llm.llm_engine, "vllm_config"):
            vllm_config = llm.llm_engine.vllm_config
            if (
                hasattr(vllm_config, "compilation_config")
                and vllm_config.compilation_config is not None
            ):
                compilation_time = vllm_config.compilation_config.compilation_time

        result_queue.put(
            {
                "total_startup_time": total_startup_time,
                "compilation_time": compilation_time,
            }
        )

    except Exception as e:
        result_queue.put(None)
        result_queue.put(str(e))


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    base_name = os.path.splitext(args.output_json)[0]

    cold_startup_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "avg_cold_startup_time": results["avg_cold_startup_time"],
        },
        extra_info={
            "cold_startup_times": results["cold_startup_times"],
            "cold_startup_percentiles": results["cold_startup_percentiles"],
        },
    )
    if cold_startup_records:
        write_to_json(f"{base_name}.cold_startup.pytorch.json", cold_startup_records)

    cold_compilation_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "avg_cold_compilation_time": results["avg_cold_compilation_time"],
        },
        extra_info={
            "cold_compilation_times": results["cold_compilation_times"],
            "cold_compilation_percentiles": results["cold_compilation_percentiles"],
        },
    )
    if cold_compilation_records:
        write_to_json(
            f"{base_name}.cold_compilation.pytorch.json", cold_compilation_records
        )

    warm_startup_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "avg_warm_startup_time": results["avg_warm_startup_time"],
        },
        extra_info={
            "warm_startup_times": results["warm_startup_times"],
            "warm_startup_percentiles": results["warm_startup_percentiles"],
        },
    )
    if warm_startup_records:
        write_to_json(f"{base_name}.warm_startup.pytorch.json", warm_startup_records)

    warm_compilation_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "avg_warm_compilation_time": results["avg_warm_compilation_time"],
        },
        extra_info={
            "warm_compilation_times": results["warm_compilation_times"],
            "warm_compilation_percentiles": results["warm_compilation_percentiles"],
        },
    )
    if warm_compilation_records:
        write_to_json(
            f"{base_name}.warm_compilation.pytorch.json", warm_compilation_records
        )


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-iters-cold",
        type=int,
        default=3,
        help="Number of cold startup iterations.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations before benchmarking warm startups.",
    )
    parser.add_argument(
        "--num-iters-warm",
        type=int,
        default=3,
        help="Number of warm startup iterations.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the startup time results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


def main(args: argparse.Namespace):
    # Set multiprocessing start method to 'spawn' for clean process isolation
    # This ensures each subprocess starts fresh without inheriting state
    multiprocessing.set_start_method("spawn", force=True)

    engine_args = EngineArgs.from_cli_args(args)

    def create_llm_and_measure_startup():
        """
        Create LLM instance in a subprocess and measure startup time.
        Returns timing metrics, using subprocess for complete isolation.
        """

        # Create a queue for inter-process communication
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_startup_in_subprocess,
            args=(
                engine_args,
                result_queue,
            ),
        )
        process.start()
        process.join()

        if not result_queue.empty():
            result = result_queue.get()
            if result is None:
                if not result_queue.empty():
                    error_msg = result_queue.get()
                    raise RuntimeError(f"Subprocess failed: {error_msg}")
                else:
                    raise RuntimeError("Subprocess failed with unknown error")
            return result
        else:
            raise RuntimeError("Subprocess did not return a result")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 to collect startup metrics.\n")

    print("Measuring cold startup time...\n")
    cold_startup_times = []
    cold_compilation_times = []
    for i in tqdm(range(args.num_iters_cold), desc="Cold startup iterations"):
        with cold_startup():
            metrics = create_llm_and_measure_startup()
            cold_startup_times.append(metrics["total_startup_time"])
            cold_compilation_times.append(metrics["compilation_time"])

    # Warmup for warm startup
    print("\nWarming up for warm startup measurement...\n")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        create_llm_and_measure_startup()

    print("\nMeasuring warm startup time...\n")
    warm_startup_times = []
    warm_compilation_times = []
    for i in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):
        metrics = create_llm_and_measure_startup()
        warm_startup_times.append(metrics["total_startup_time"])
        warm_compilation_times.append(metrics["compilation_time"])

    # Calculate statistics
    cold_startup_array = np.array(cold_startup_times)
    cold_compilation_array = np.array(cold_compilation_times)
    warm_startup_array = np.array(warm_startup_times)
    warm_compilation_array = np.array(warm_compilation_times)

    avg_cold_startup = np.mean(cold_startup_array)
    avg_cold_compilation = np.mean(cold_compilation_array)
    avg_warm_startup = np.mean(warm_startup_array)
    avg_warm_compilation = np.mean(warm_compilation_array)

    percentages = [10, 25, 50, 75, 90, 99]
    cold_startup_percentiles = np.percentile(cold_startup_array, percentages)
    cold_compilation_percentiles = np.percentile(cold_compilation_array, percentages)
    warm_startup_percentiles = np.percentile(warm_startup_array, percentages)
    warm_compilation_percentiles = np.percentile(warm_compilation_array, percentages)

    print("\n" + "=" * 60)
    print("STARTUP TIME BENCHMARK RESULTS")
    print("=" * 60)

    # Cold startup statistics
    print("\nCOLD STARTUP:")
    print(f"Avg total startup time: {avg_cold_startup:.2f} seconds")
    print(f"Avg compilation time:   {avg_cold_compilation:.2f} seconds")
    print("Startup time percentiles:")
    for percentage, percentile in zip(percentages, cold_startup_percentiles):
        print(f"  {percentage}%: {percentile:.2f} seconds")
    print("Compilation time percentiles:")
    for percentage, percentile in zip(percentages, cold_compilation_percentiles):
        print(f"  {percentage}%: {percentile:.2f} seconds")

    # Warm startup statistics
    print("\nWARM STARTUP:")
    print(f"Avg total startup time: {avg_warm_startup:.2f} seconds")
    print(f"Avg compilation time:   {avg_warm_compilation:.2f} seconds")
    print("Startup time percentiles:")
    for percentage, percentile in zip(percentages, warm_startup_percentiles):
        print(f"  {percentage}%: {percentile:.2f} seconds")
    print("Compilation time percentiles:")
    for percentage, percentile in zip(percentages, warm_compilation_percentiles):
        print(f"  {percentage}%: {percentile:.2f} seconds")

    print("=" * 60)

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_cold_startup_time": float(avg_cold_startup),
            "avg_cold_compilation_time": float(avg_cold_compilation),
            "cold_startup_times": cold_startup_times,
            "cold_compilation_times": cold_compilation_times,
            "cold_startup_percentiles": dict(
                zip(percentages, cold_startup_percentiles.tolist())
            ),
            "cold_compilation_percentiles": dict(
                zip(percentages, cold_compilation_percentiles.tolist())
            ),
            "avg_warm_startup_time": float(avg_warm_startup),
            "avg_warm_compilation_time": float(avg_warm_compilation),
            "warm_startup_times": warm_startup_times,
            "warm_compilation_times": warm_compilation_times,
            "warm_startup_percentiles": dict(
                zip(percentages, warm_startup_percentiles.tolist())
            ),
            "warm_compilation_percentiles": dict(
                zip(percentages, warm_compilation_percentiles.tolist())
            ),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
