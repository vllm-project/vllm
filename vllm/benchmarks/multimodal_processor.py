# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark multimodal processor latency.

This benchmark measures the latency of the multimodal processor module
across different configurations (data parallel, multiple instances, high concurrency).

On the server side, run:
    VLLM_ENABLE_MM_PROCESSOR_STATS=1 vllm serve <your_model> <engine arguments>

On the client side, run:
    vllm bench multimodal-processor \
        --model <your_model> \
        --dataset-name <dataset_name> \
        --dataset-path <dataset_path> \
        --data-parallel-size <dp_size> \
        --num-instances <num_instances> \
        --max-concurrency <concurrency>
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import vllm.envs as envs
from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.benchmarks.serve import (
    TaskType,
    benchmark,
    calculate_metrics,
    calculate_metrics_for_embeddings,
    get_request,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.multimodal.processing import (
    MultiModalProcessorTimingStats,
    clear_mm_processor_timing_stats,
    get_mm_processor_timing_stats,
)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import join_host_port


@dataclass
class MultimodalProcessorBenchmarkMetrics:
    """Metrics for multimodal processor benchmark."""

    # Standard serving metrics
    completed: int
    failed: int
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]

    # Multimodal processor timing metrics
    mm_processor_stats: dict[str, dict[str, float]]
    """Per-stage timing stats: mean, median, std, percentiles for each stage."""


def _parse_prometheus_histogram(metrics_text: str, metric_name: str, stage: str) -> dict[str, float]:
    """
    Parse Prometheus histogram from /metrics endpoint text.
    
    Returns a dict with 'count', 'sum', and bucket values.
    """
    import re
    
    # Map stage names to Prometheus label values
    stage_map = {
        "hf_processor": "hf_processor",
        "hashing": "hashing",
        "cache_lookup": "cache_lookup",
        "prompt_update": "prompt_update",
    }
    prometheus_stage = stage_map.get(stage, stage)
    
    result = {"count": 0.0, "sum": 0.0, "buckets": {}}
    
    # Pattern to match histogram bucket lines: metric_name_bucket{labels} value
    bucket_pattern = re.compile(
        rf"{re.escape(metric_name)}_bucket\{{[^}}]*stage="{re.escape(prometheus_stage)}"[^}}]*\}}\s+([\d\.e\+\-]+)"
    )
    count_pattern = re.compile(
        rf"{re.escape(metric_name)}_count\{{[^}}]*stage="{re.escape(prometheus_stage)}"[^}}]*\}}\s+([\d\.e\+\-]+)"
    )
    sum_pattern = re.compile(
        rf"{re.escape(metric_name)}_sum\{{[^}}]*stage="{re.escape(prometheus_stage)}"[^}}]*\}}\s+([\d\.e\+\-]+)"
    )
    
    # Also match bucket lines with le label
    bucket_with_le_pattern = re.compile(
        rf"{re.escape(metric_name)}_bucket\{{[^}}]*stage="{re.escape(prometheus_stage)}"[^}}]*le="([^"]+)"[^}}]*\}}\s+([\d\.e\+\-]+)"
    )
    
    for line in metrics_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Match bucket with le label
        match = bucket_with_le_pattern.search(line)
        if match:
            le_value = match.group(1)
            bucket_count = float(match.group(2))
            result["buckets"][le_value] = bucket_count
            continue
        
        # Match count
        match = count_pattern.search(line)
        if match:
            result["count"] = float(match.group(1))
            continue
        
        # Match sum
        match = sum_pattern.search(line)
        if match:
            result["sum"] = float(match.group(1))
            continue
    
    return result


def _calculate_percentile_from_histogram(histogram: dict[str, float], percentile: float) -> float:
    """
    Calculate a percentile from a Prometheus histogram.
    
    Uses linear interpolation between buckets.
    """
    count = histogram.get("count", 0.0)
    if count == 0:
        return 0.0
    
    target_count = count * (percentile / 100.0)
    buckets = histogram.get("buckets", {})
    
    # Sort buckets by le value (convert +Inf to a large number)
    sorted_buckets = []
    for le_str, bucket_count in buckets.items():
        if le_str == "+Inf":
            le_value = float("inf")
        else:
            try:
                le_value = float(le_str)
            except ValueError:
                continue
        sorted_buckets.append((le_value, bucket_count))
    
    sorted_buckets.sort(key=lambda x: x[0])
    
    # Find the bucket containing the target count
    cumulative = 0.0
    for i, (le_value, bucket_count) in enumerate(sorted_buckets):
        cumulative = bucket_count
        if cumulative >= target_count:
            if i == 0:
                return le_value
            # Linear interpolation
            prev_le, prev_count = sorted_buckets[i - 1]
            if cumulative == prev_count:
                return le_value
            # Interpolate between prev_le and le_value
            ratio = (target_count - prev_count) / (cumulative - prev_count)
            return prev_le + (le_value - prev_le) * ratio
    
    # If we didn't find it, return the max bucket
    if sorted_buckets:
        return sorted_buckets[-1][0]
    return 0.0


async def collect_mm_processor_stats(
    outputs: list[RequestFuncOutput],
    base_url: str,
    session: Any,
    input_requests: list[SampleRequest],
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    """
    Collect multimodal processor timing stats by querying the /metrics endpoint.

    This queries the Prometheus /metrics endpoint and parses histogram data
    for multimodal processor timing.
    """
    import aiohttp

    stats_by_stage = {
        "hf_processor_time": [],
        "hashing_time": [],
        "cache_lookup_time": [],
        "prompt_update_time": [],
        "total_time": [],
    }

    # Query /metrics endpoint
    metrics_url = f"{base_url}/metrics"
    try:
        async with session.get(metrics_url) as resp:
            if resp.status != 200:
                return stats_by_stage
            metrics_text = await resp.text()
    except Exception:
        # If metrics endpoint is unavailable, return empty stats
        return stats_by_stage

    # Parse histograms for each stage
    metric_name = "vllm:mm_processor_time_seconds"
    stages = ["hf_processor", "hashing", "cache_lookup", "prompt_update"]
    histograms = {}
    
    for stage in stages:
        histogram = _parse_prometheus_histogram(metrics_text, metric_name, stage)
        count = histogram.get("count", 0.0)
        
        if count > 0:
            stage_key = f"{stage}_time" if stage != "hf_processor" else "hf_processor_time"
            histograms[stage_key] = histogram
    
    # Return histograms dict for percentile calculation
    # The stats_by_stage dict is kept for compatibility but won't be used
    # when histograms are available
    return stats_by_stage, histograms if histograms else {}


def calculate_mm_processor_metrics(
    stats_by_stage: dict[str, list[float]],
    selected_percentiles: list[float],
    histograms: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Calculate aggregate metrics for each processor stage.
    
    If histograms are provided, calculates percentiles from histogram buckets.
    Otherwise, calculates from individual observations.
    """
    metrics = {}
    
    if histograms:
        # Calculate metrics from Prometheus histograms
        for stage_name, histogram in histograms.items():
            count = histogram.get("count", 0.0)
            sum_value = histogram.get("sum", 0.0)
            
            if count == 0:
                metrics[stage_name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    **{f"p{p}": 0.0 for p in selected_percentiles},
                }
                continue
            
            mean = (sum_value / count) * 1000  # Convert to milliseconds
            median = _calculate_percentile_from_histogram(histogram, 50.0) * 1000
            
            # Calculate percentiles from histogram
            percentile_values = {
                f"p{p}": _calculate_percentile_from_histogram(histogram, p) * 1000
                for p in selected_percentiles
            }
            
            # Estimate std from histogram (rough approximation)
            # For now, use a simple approximation based on percentiles
            p75 = _calculate_percentile_from_histogram(histogram, 75.0) * 1000
            p25 = _calculate_percentile_from_histogram(histogram, 25.0) * 1000
            std_approx = (p75 - p25) / 1.35  # IQR-based std approximation
            
            metrics[stage_name] = {
                "mean": mean,
                "median": median,
                "std": std_approx,
                **percentile_values,
            }
    else:
        # Calculate from individual observations
        for stage_name, times in stats_by_stage.items():
            if not times:
                metrics[stage_name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    **{f"p{p}": 0.0 for p in selected_percentiles},
                }
                continue

            times_ms = [t * 1000 for t in times]  # Convert to milliseconds
            metrics[stage_name] = {
                "mean": float(np.mean(times_ms)),
                "median": float(np.median(times_ms)),
                "std": float(np.std(times_ms)),
                **{
                    f"p{p}": float(np.percentile(times_ms, p))
                    for p in selected_percentiles
                },
            }
    
    return metrics


async def benchmark_multimodal_processor(
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str | None,
    tokenizer: Any,
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    max_concurrency: int | None,
    lora_modules: list[str] | None,
    extra_headers: dict[str, str] | None,
    extra_body: dict[str, Any],
    ready_check_timeout_sec: int,
) -> dict[str, Any]:
    """
    Run the multimodal processor benchmark.

    This is similar to the serve benchmark but also collects and reports
    multimodal processor timing statistics.
    """
    from tqdm.asyncio import tqdm

    import aiohttp

    # Ensure stats collection is enabled
    if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
        print(
            "Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled. "
            "MM processor timing stats will not be collected."
        )

    task_type = (
        TaskType.POOLING
        if "embeddings" in endpoint_type or "rerank" in endpoint_type
        else TaskType.GENERATION
    )

    request_func = ASYNC_REQUEST_FUNCS[endpoint_type]

    async with aiohttp.ClientSession() as session:
        # Run the standard benchmark to get request latencies
        benchmark_result = await benchmark(
            task_type=task_type,
            endpoint_type=endpoint_type,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=None,
            request_rate=request_rate,
            burstiness=burstiness,
            disable_tqdm=disable_tqdm,
            num_warmups=num_warmups,
            profile=False,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            ignore_eos=ignore_eos,
            goodput_config_dict={},
            max_concurrency=max_concurrency,
            lora_modules=lora_modules,
            extra_headers=extra_headers,
            extra_body=extra_body,
            ramp_up_strategy=None,
            ramp_up_start_rps=None,
            ramp_up_end_rps=None,
            ready_check_timeout_sec=ready_check_timeout_sec,
        )

        # Collect MM processor stats by querying the /metrics endpoint
        outputs = benchmark_result.get("outputs", [])
        mm_stats_by_stage, histograms = await collect_mm_processor_stats(
            outputs, base_url, session, input_requests
        )

        mm_processor_metrics = calculate_mm_processor_metrics(
            mm_stats_by_stage, selected_percentiles, histograms=histograms
        )

        # Merge MM processor metrics into benchmark result
        benchmark_result["mm_processor_stats"] = mm_processor_metrics

    return benchmark_result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the multimodal processor benchmark."""
    # Reuse dataset arguments
    add_dataset_parser(parser)

    # Engine arguments
    parser = AsyncEngineArgs.add_cli_args(parser)

    # Benchmark-specific arguments
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallel size for the benchmark.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of vLLM instances to run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (requests per second). Use 'inf' for maximum rate.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request generation.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=10,
        help="Number of warmup requests.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,95,99",
        help="Comma-separated list of percentiles to calculate.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results in JSON format.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address of the vLLM server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the vLLM server.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint to benchmark.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai-chat",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend/endpoint type.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token during generation.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=60,
        help="Timeout for endpoint ready check.",
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for the multimodal processor benchmark."""
    import asyncio
    from datetime import datetime

    # Validate arguments
    if args.dataset_name is None:
        raise ValueError("--dataset-name is required")

    # Ensure multimodal dataset
    if args.dataset_name not in ("hf", "vision-arena", "mmvu"):
        print(
            f"Warning: Dataset '{args.dataset_name}' may not be multimodal. "
            "Results may not include MM processor timing."
        )

    # Setup
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    host_port = join_host_port(args.host, args.port)
    api_url = f"http://{host_port}{args.endpoint}"
    base_url = f"http://{host_port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    # Load dataset
    input_requests = get_samples(args, tokenizer)

    # Check if stats collection is enabled
    if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
        print(
            "\n⚠️  Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled.\n"
            "   MM processor timing stats will not be collected.\n"
            "   Set VLLM_ENABLE_MM_PROCESSOR_STATS=1 when starting the server.\n"
        )

    # Sampling parameters
    sampling_params = {}
    if args.backend in OPENAI_COMPATIBLE_BACKENDS:
        for param in ["temperature", "top_p", "top_k"]:
            if hasattr(args, param):
                val = getattr(args, param, None)
                if val is not None:
                    sampling_params[param] = val

    extra_body = sampling_params
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    # Avoid GC overhead
    freeze_gc_heap()

    # Run benchmark
    print("Starting multimodal processor benchmark...")
    result = asyncio.run(
        benchmark_multimodal_processor(
            endpoint_type=args.backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=None,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            num_warmups=args.num_warmups,
            selected_percentile_metrics=["e2el"],
            selected_percentiles=selected_percentiles,
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            lora_modules=None,
            extra_headers=None,
            extra_body=extra_body,
            ready_check_timeout_sec=args.ready_check_timeout_sec,
        )
    )

    # Print results
    print("\n" + "=" * 80)
    print("Multimodal Processor Benchmark Results")
    print("=" * 80)

    if "mm_processor_stats" in result:
        print("\nMM Processor Timing (ms):")
        for stage, metrics in result["mm_processor_stats"].items():
            print(f"  {stage}:")
            print(f"    Mean: {metrics['mean']:.2f}")
            print(f"    Median: {metrics['median']:.2f}")
            print(f"    Std: {metrics['std']:.2f}")
            for p in selected_percentiles:
                print(f"    P{p}: {metrics.get(f'p{p}', 0.0):.2f}")

    if "mean_e2el_ms" in result:
        print("\nEnd-to-End Latency (ms):")
        print(f"  Mean: {result['mean_e2el_ms']:.2f}")
        print(f"  Median: {result['median_e2el_ms']:.2f}")
        print(f"  Std: {result['std_e2el_ms']:.2f}")

    # Save results
    if args.output_json:
        result["config"] = {
            "model": model_id,
            "data_parallel_size": args.data_parallel_size,
            "num_instances": args.num_instances,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
        }
        result["timestamp"] = datetime.now().isoformat()

        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark multimodal processor latency"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

