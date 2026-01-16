# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the cold and warm startup time of vLLM models.

This script measures total startup time (including model loading, compilation,
and cache operations) for both cold and warm scenarios:
- Cold startup: Fresh start with no caches (temporary cache directories)
- Warm startup: Using cached compilation and model info
- AOT compile mode: Measures serialization of compiled artifacts to disk
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


@contextmanager
def aot_warm_startup(cache_dir: str, mega_aot_artifact: bool = False):
    """
    Context manager for warm startup with AOT compile artifacts.
    Uses the provided cache directory (which should contain pre-compiled artifacts).
    """
    original_cache_root = os.environ.get("VLLM_CACHE_ROOT")
    original_aot_compile = os.environ.get("VLLM_USE_AOT_COMPILE")
    original_standalone = os.environ.get("VLLM_USE_STANDALONE_COMPILE")
    original_mega_aot = os.environ.get("VLLM_USE_MEGA_AOT_ARTIFACT")

    try:
        os.environ["VLLM_CACHE_ROOT"] = cache_dir
        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
        if mega_aot_artifact:
            os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"] = "1"
        yield
    finally:
        if original_cache_root:
            os.environ["VLLM_CACHE_ROOT"] = original_cache_root
        else:
            os.environ.pop("VLLM_CACHE_ROOT", None)
        if original_aot_compile:
            os.environ["VLLM_USE_AOT_COMPILE"] = original_aot_compile
        else:
            os.environ.pop("VLLM_USE_AOT_COMPILE", None)
        if original_standalone:
            os.environ["VLLM_USE_STANDALONE_COMPILE"] = original_standalone
        else:
            os.environ.pop("VLLM_USE_STANDALONE_COMPILE", None)
        if original_mega_aot:
            os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"] = original_mega_aot
        else:
            os.environ.pop("VLLM_USE_MEGA_AOT_ARTIFACT", None)


def get_directory_size(path: str) -> int:
    """Calculate total size of all files in a directory recursively."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def run_startup_in_subprocess(
    engine_args, result_queue, measure_artifact_size=False, measure_inference=False
):
    """
    Run LLM startup in a subprocess and return timing metrics via a queue.
    This ensures complete isolation between iterations.
    """
    import traceback

    try:
        # Import inside the subprocess to avoid issues with forking
        from vllm import LLM, SamplingParams

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

        artifact_size = 0
        if measure_artifact_size:
            cache_root = os.environ.get("VLLM_CACHE_ROOT", "")
            aot_dir = os.path.join(cache_root, "torch_aot_compile")
            if os.path.exists(aot_dir):
                artifact_size = get_directory_size(aot_dir)

        inference_tokens_per_sec = 0.0
        if measure_inference:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
            prompt = "The quick brown fox jumps over the lazy dog."

            _ = llm.generate([prompt], sampling_params)

            inference_start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            inference_time = time.perf_counter() - inference_start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            if inference_time > 0:
                inference_tokens_per_sec = total_tokens / inference_time

        result_queue.put(
            {
                "total_startup_time": total_startup_time,
                "compilation_time": compilation_time,
                "artifact_size": artifact_size,
                "inference_tokens_per_sec": inference_tokens_per_sec,
            }
        )

    except Exception as e:
        result_queue.put(None)
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result_queue.put(error_msg)


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

    if "avg_aot_cold_startup_time" in results:
        aot_cold_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={
                "avg_aot_cold_startup_time": results["avg_aot_cold_startup_time"],
            },
            extra_info={
                "aot_cold_startup_times": results["aot_cold_startup_times"],
                "aot_cold_startup_percentiles": results["aot_cold_startup_percentiles"],
            },
        )
        if aot_cold_records:
            write_to_json(
                f"{base_name}.aot_cold_startup.pytorch.json", aot_cold_records
            )

        aot_warm_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={
                "avg_aot_warm_startup_time": results["avg_aot_warm_startup_time"],
            },
            extra_info={
                "aot_warm_startup_times": results["aot_warm_startup_times"],
                "aot_warm_startup_percentiles": results["aot_warm_startup_percentiles"],
            },
        )
        if aot_warm_records:
            write_to_json(
                f"{base_name}.aot_warm_startup.pytorch.json", aot_warm_records
            )


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-iters-cold",
        type=int,
        default=5,
        help="Number of cold startup iterations.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before benchmarking warm startups.",
    )
    parser.add_argument(
        "--num-iters-warm",
        type=int,
        default=5,
        help="Number of warm startup iterations.",
    )
    parser.add_argument(
        "--aot-compile",
        action="store_true",
        help="Enable AOT compile mode to benchmark serialization of "
        "compiled artifacts.",
    )
    parser.add_argument(
        "--mega-aot-artifact",
        action="store_true",
        help="Enable mega AOT artifact mode (bundles all compiled artifacts). "
        "Requires --aot-compile.",
    )
    parser.add_argument(
        "--measure-inference",
        action="store_true",
        help="Measure inference performance (tokens/sec) after startup.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the startup time results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def main(args: argparse.Namespace):
    if args.mega_aot_artifact and not args.aot_compile:
        raise ValueError("--mega-aot-artifact requires --aot-compile")

    # Set multiprocessing start method to 'spawn' for clean process isolation
    # This ensures each subprocess starts fresh without inheriting state
    multiprocessing.set_start_method("spawn", force=True)

    engine_args = EngineArgs.from_cli_args(args)

    def create_llm_and_measure_startup(
        measure_artifact_size=False, measure_inference=False
    ):
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
                measure_artifact_size,
                measure_inference,
            ),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            error_msg = "Unknown error"
            if not result_queue.empty():
                result = result_queue.get()
                if result is None and not result_queue.empty():
                    error_msg = result_queue.get()
            raise RuntimeError(
                f"Subprocess failed with exit code {process.exitcode}: {error_msg}"
            )

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
            raise RuntimeError(
                f"Subprocess did not return a result (exit code: {process.exitcode})"
            )

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
    warm_inference_speeds = []
    for i in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):
        metrics = create_llm_and_measure_startup(
            measure_inference=args.measure_inference
        )
        warm_startup_times.append(metrics["total_startup_time"])
        warm_compilation_times.append(metrics["compilation_time"])
        if args.measure_inference:
            warm_inference_speeds.append(metrics["inference_tokens_per_sec"])

    aot_cold_startup_times = []
    aot_cold_compilation_times = []
    aot_warm_startup_times = []
    aot_warm_compilation_times = []
    aot_warm_inference_speeds = []
    aot_artifact_sizes = []

    if args.aot_compile:
        print("\n" + "=" * 60)
        print("AOT COMPILE BENCHMARKING")
        if args.mega_aot_artifact:
            print("(with MEGA AOT ARTIFACT enabled)")
        print("=" * 60)

        aot_cache_dir = tempfile.mkdtemp(prefix="vllm_startup_bench_aot_")

        try:
            print("\nMeasuring AOT cold startup (compile + serialize)...\n")
            for i in tqdm(
                range(args.num_iters_cold), desc="AOT cold startup iterations"
            ):
                if os.path.exists(aot_cache_dir):
                    shutil.rmtree(aot_cache_dir)
                os.makedirs(aot_cache_dir, exist_ok=True)

                with aot_warm_startup(aot_cache_dir, args.mega_aot_artifact):
                    from torch._inductor.utils import fresh_cache

                    with fresh_cache():
                        metrics = create_llm_and_measure_startup(
                            measure_artifact_size=True
                        )
                        aot_cold_startup_times.append(metrics["total_startup_time"])
                        aot_cold_compilation_times.append(metrics["compilation_time"])
                        aot_artifact_sizes.append(metrics["artifact_size"])

            print("\nPopulating AOT cache for warm measurements...\n")
            if os.path.exists(aot_cache_dir):
                shutil.rmtree(aot_cache_dir)
            os.makedirs(aot_cache_dir, exist_ok=True)

            with aot_warm_startup(aot_cache_dir, args.mega_aot_artifact):
                from torch._inductor.utils import fresh_cache

                with fresh_cache():
                    create_llm_and_measure_startup(measure_artifact_size=True)

            print("\nMeasuring AOT warm startup (load from cache)...\n")
            for i in tqdm(
                range(args.num_iters_warm), desc="AOT warm startup iterations"
            ):
                with aot_warm_startup(aot_cache_dir, args.mega_aot_artifact):
                    metrics = create_llm_and_measure_startup(
                        measure_inference=args.measure_inference
                    )
                    aot_warm_startup_times.append(metrics["total_startup_time"])
                    aot_warm_compilation_times.append(metrics["compilation_time"])
                    if args.measure_inference:
                        aot_warm_inference_speeds.append(
                            metrics["inference_tokens_per_sec"]
                        )

        finally:
            shutil.rmtree(aot_cache_dir, ignore_errors=True)

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

    # Prepare results dict
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

    if args.aot_compile and aot_cold_startup_times:
        aot_cold_startup_array = np.array(aot_cold_startup_times)
        aot_cold_compilation_array = np.array(aot_cold_compilation_times)
        aot_warm_startup_array = np.array(aot_warm_startup_times)
        aot_warm_compilation_array = np.array(aot_warm_compilation_times)
        aot_artifact_array = np.array(aot_artifact_sizes)

        avg_aot_cold_startup = np.mean(aot_cold_startup_array)
        avg_aot_cold_compilation = np.mean(aot_cold_compilation_array)
        avg_aot_warm_startup = np.mean(aot_warm_startup_array)
        avg_aot_warm_compilation = np.mean(aot_warm_compilation_array)
        avg_artifact_size = np.mean(aot_artifact_array)

        aot_cold_startup_percentiles = np.percentile(
            aot_cold_startup_array, percentages
        )
        aot_cold_compilation_percentiles = np.percentile(
            aot_cold_compilation_array, percentages
        )
        aot_warm_startup_percentiles = np.percentile(
            aot_warm_startup_array, percentages
        )
        aot_warm_compilation_percentiles = np.percentile(
            aot_warm_compilation_array, percentages
        )

        print("\n" + "-" * 60)
        print("AOT COMPILE RESULTS")
        print("-" * 60)

        print("\nAOT COLD STARTUP (compile + serialize):")
        print(f"Avg total startup time: {avg_aot_cold_startup:.2f} seconds")
        print(f"Avg compilation time:   {avg_aot_cold_compilation:.2f} seconds")
        print(f"Avg artifact size:      {format_size(avg_artifact_size)}")
        print("Startup time percentiles:")
        for percentage, percentile in zip(percentages, aot_cold_startup_percentiles):
            print(f"  {percentage}%: {percentile:.2f} seconds")

        print("\nAOT WARM STARTUP (load from cache):")
        print(f"Avg total startup time: {avg_aot_warm_startup:.2f} seconds")
        print(f"Avg compilation time:   {avg_aot_warm_compilation:.2f} seconds")
        print("Startup time percentiles:")
        for percentage, percentile in zip(percentages, aot_warm_startup_percentiles):
            print(f"  {percentage}%: {percentile:.2f} seconds")

        speedup = avg_aot_cold_startup / avg_aot_warm_startup
        time_saved = avg_aot_cold_startup - avg_aot_warm_startup
        print(f"\nAOT Speedup: {speedup:.2f}x ({time_saved:.2f}s saved)")

        print("\n" + "-" * 60)
        print("COMPARISON: AOT Warm vs Regular Warm")
        print("-" * 60)
        warm_vs_aot_improvement = avg_warm_startup - avg_aot_warm_startup
        warm_vs_aot_speedup = avg_warm_startup / avg_aot_warm_startup
        print(f"Regular warm startup: {avg_warm_startup:.2f}s")
        print(f"AOT warm startup:     {avg_aot_warm_startup:.2f}s")
        if warm_vs_aot_improvement > 0:
            print(
                f"Improvement:          {warm_vs_aot_improvement:.2f}s faster "
                f"({warm_vs_aot_speedup:.2f}x)"
            )
        else:
            slowdown = 1 / warm_vs_aot_speedup
            print(
                f"Regression:           {-warm_vs_aot_improvement:.2f}s slower "
                f"({slowdown:.2f}x)"
            )

        avg_warm_inference = 0.0
        avg_aot_warm_inference = 0.0
        if (
            args.measure_inference
            and warm_inference_speeds
            and aot_warm_inference_speeds
        ):
            avg_warm_inference = np.mean(warm_inference_speeds)
            avg_aot_warm_inference = np.mean(aot_warm_inference_speeds)
            inference_diff_pct = (
                (avg_aot_warm_inference - avg_warm_inference) / avg_warm_inference * 100
                if avg_warm_inference > 0
                else 0
            )
            print("\nInference Performance:")
            print(f"Regular warm: {avg_warm_inference:.1f} tokens/sec")
            print(f"AOT warm:     {avg_aot_warm_inference:.1f} tokens/sec")
            if abs(inference_diff_pct) < 5:
                print(f"Difference:   {inference_diff_pct:+.1f}% (within noise)")
            elif inference_diff_pct > 0:
                print(f"Improvement:  {inference_diff_pct:+.1f}%")
            else:
                print(f"Regression:   {inference_diff_pct:+.1f}%")

        results.update(
            {
                "avg_aot_cold_startup_time": float(avg_aot_cold_startup),
                "avg_aot_cold_compilation_time": float(avg_aot_cold_compilation),
                "aot_cold_startup_times": aot_cold_startup_times,
                "aot_cold_compilation_times": aot_cold_compilation_times,
                "aot_cold_startup_percentiles": dict(
                    zip(percentages, aot_cold_startup_percentiles.tolist())
                ),
                "aot_cold_compilation_percentiles": dict(
                    zip(percentages, aot_cold_compilation_percentiles.tolist())
                ),
                "avg_aot_warm_startup_time": float(avg_aot_warm_startup),
                "avg_aot_warm_compilation_time": float(avg_aot_warm_compilation),
                "aot_warm_startup_times": aot_warm_startup_times,
                "aot_warm_compilation_times": aot_warm_compilation_times,
                "aot_warm_startup_percentiles": dict(
                    zip(percentages, aot_warm_startup_percentiles.tolist())
                ),
                "aot_warm_compilation_percentiles": dict(
                    zip(percentages, aot_warm_compilation_percentiles.tolist())
                ),
                "avg_artifact_size_bytes": float(avg_artifact_size),
                "artifact_sizes": aot_artifact_sizes,
                "aot_speedup": float(speedup),
                "warm_vs_aot_improvement_seconds": float(warm_vs_aot_improvement),
                "warm_vs_aot_speedup": float(warm_vs_aot_speedup),
            }
        )

        if (
            args.measure_inference
            and warm_inference_speeds
            and aot_warm_inference_speeds
        ):
            results.update(
                {
                    "avg_warm_inference_tokens_per_sec": float(avg_warm_inference),
                    "avg_aot_warm_inference_tokens_per_sec": float(
                        avg_aot_warm_inference
                    ),
                    "warm_inference_tokens_per_sec": warm_inference_speeds,
                    "aot_warm_inference_tokens_per_sec": aot_warm_inference_speeds,
                }
            )

    print("=" * 60)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
