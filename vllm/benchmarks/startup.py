# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the cold and warm startup time of vLLM models.

This script measures total startup time (including model loading, compilation,
and cache operations) for both cold and warm scenarios:
- Cold startup: Fresh start with no caches (temporary cache directories)
- Warm startup: Using cached compilation and model info
"""

import argparse
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from vllm.engine.arg_utils import EngineArgs

PERCENTAGES = [10, 25, 50, 75, 90, 99]


class MetricDesc(NamedTuple):
    """Descriptor for a metric to collect from each iteration."""

    iter_key: str  # key in the iteration result dict
    suffix: str  # result key suffix, e.g. "startup", "compilation"
    display_name: str


class MetricStats(NamedTuple):
    """Aggregated statistics for a single benchmark metric."""

    key: str  # e.g. "cold_startup", "warm_encoder_compilation"
    display_name: str
    values: list[float]
    avg: float
    percentiles: dict[int, float]


_BASE_METRICS = [
    MetricDesc("total_startup_time", "startup", "Startup time"),
    MetricDesc("compilation_time", "compilation", "Compilation time"),
]
_ENCODER_METRIC = MetricDesc(
    "encoder_compilation_time",
    "encoder_compilation",
    "Encoder compilation time",
)


def _compute_metric(
    phase: str,
    desc: MetricDesc,
    iterations: list[dict[str, float]],
) -> MetricStats:
    values = [m[desc.iter_key] for m in iterations]
    arr = np.array(values)
    return MetricStats(
        key=f"{phase}_{desc.suffix}",
        display_name=desc.display_name,
        values=values,
        avg=float(np.mean(arr)),
        percentiles=dict(zip(PERCENTAGES, np.percentile(arr, PERCENTAGES).tolist())),
    )


def _collect_phase_metrics(
    phase: str,
    iterations: list[dict[str, float]],
    has_encoder: bool,
) -> list[MetricStats]:
    metrics = [_compute_metric(phase, desc, iterations) for desc in _BASE_METRICS]
    if has_encoder:
        metrics.append(_compute_metric(phase, _ENCODER_METRIC, iterations))
    return metrics


def _print_phase(phase_name: str, metrics: list[MetricStats]) -> None:
    print(f"\n{phase_name}:")
    for m in metrics:
        print(f"Avg {m.display_name.lower()}: {m.avg:.2f} seconds")
    for m in metrics:
        print(f"{m.display_name} percentiles:")
        for pct, val in m.percentiles.items():
            print(f"  {pct}%: {val:.2f} seconds")


def _metric_to_json(m: MetricStats) -> dict[str, Any]:
    return {
        f"avg_{m.key}_time": m.avg,
        f"{m.key}_times": m.values,
        f"{m.key}_percentiles": m.percentiles,
    }


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

        llm = LLM.from_engine_args(engine_args)

        total_startup_time = time.perf_counter() - start_time

        # Extract compilation time if available
        compilation_time = 0.0
        encoder_compilation_time = 0.0
        if hasattr(llm.llm_engine, "vllm_config"):
            vllm_config = llm.llm_engine.vllm_config
            if (
                hasattr(vllm_config, "compilation_config")
                and vllm_config.compilation_config is not None
            ):
                compilation_time = vllm_config.compilation_config.compilation_time
                encoder_compilation_time = (
                    vllm_config.compilation_config.encoder_compilation_time
                )

        result_queue.put(
            {
                "total_startup_time": total_startup_time,
                "compilation_time": compilation_time,
                "encoder_compilation_time": encoder_compilation_time,
            }
        )

    except Exception as e:
        result_queue.put(None)
        result_queue.put(str(e))


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: list[MetricStats]
) -> None:
    base_name = os.path.splitext(args.output_json)[0]
    for m in metrics:
        records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={f"avg_{m.key}_time": [m.avg]},
            extra_info={
                f"{m.key}_times": m.values,
                f"{m.key}_percentiles": m.percentiles,
            },
        )
        if records:
            write_to_json(f"{base_name}.{m.key}.pytorch.json", records)


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

    # Collect cold startup iterations
    print("Measuring cold startup time...\n")
    cold_iterations = []
    for i in tqdm(range(args.num_iters_cold), desc="Cold startup iterations"):
        with cold_startup():
            cold_iterations.append(create_llm_and_measure_startup())

    # Warmup for warm startup
    print("\nWarming up for warm startup measurement...\n")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        create_llm_and_measure_startup()

    # Collect warm startup iterations
    print("\nMeasuring warm startup time...\n")
    warm_iterations = []
    for i in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):
        warm_iterations.append(create_llm_and_measure_startup())

    # Determine if encoder compilation occurred in any iteration
    has_encoder = any(
        m["encoder_compilation_time"] > 0 for m in cold_iterations + warm_iterations
    )

    cold_metrics = _collect_phase_metrics("cold", cold_iterations, has_encoder)
    warm_metrics = _collect_phase_metrics("warm", warm_iterations, has_encoder)
    all_metrics = cold_metrics + warm_metrics

    # Print results
    print("\n" + "=" * 60)
    print("STARTUP TIME BENCHMARK RESULTS")
    print("=" * 60)
    _print_phase("COLD STARTUP", cold_metrics)
    _print_phase("WARM STARTUP", warm_metrics)
    print("=" * 60)

    # Output JSON results if specified
    if args.output_json:
        results: dict[str, Any] = {}
        for m in all_metrics:
            results.update(_metric_to_json(m))
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, all_metrics)
