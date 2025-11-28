# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark multimodal processor latency.

This benchmark measures the latency of the multimodal processor module.

On the server side, run:
    VLLM_ENABLE_MM_PROCESSOR_STATS=1 vllm serve <your_model> <engine arguments>

On the client side, run:
    vllm bench multimodal-processor \
        --model <your_model> \
        --dataset-name <dataset_name> \
        --dataset-path <dataset_path> \
        --max-concurrency <concurrency>
"""

import argparse
import asyncio
import aiohttp
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

import vllm.envs as envs
from vllm.benchmarks.datasets import SampleRequest, get_samples
from vllm.benchmarks.lib.endpoint_request_func import OPENAI_COMPATIBLE_BACKENDS
from vllm.benchmarks.serve import TaskType, benchmark
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import join_host_port


@dataclass
class MultimodalProcessorBenchmarkMetrics:
    """Metrics for multimodal processor benchmark."""

    completed: int
    failed: int
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]

    """Per-stage timing stats: mean, median, std, percentiles for each stage."""
    mm_processor_stats: dict[str, dict[str, float]]


def _get_instance_urls(
    args: argparse.Namespace | None,
    base_url: str | None,
) -> list[str]:
    """
    Get the instance URL for stats collection.
    
    Args:
        args: Command-line arguments (may be None)
        base_url: Base URL for the server (may be None)
    
    Returns:
        List containing a single instance URL
    """
    if base_url:
        return [base_url]
    
    host = getattr(args, "host", "127.0.0.1") if args else "127.0.0.1"
    port = getattr(args, "port", 8000) if args else 8000
    return [f"http://{host}:{port}"]


async def _clear_mm_processor_stats_registry(
    instance_urls: list[str],
    debug: bool = False,
) -> int:
    """
    Clear MM processor stats registry from all instances via HTTP. 
    Makes POST requests on the client-side across all instances to the /clear_mm_processor_stats endpoint.
    
    Args:
        instance_urls: List of base URLs for vLLM instances
        debug: Enable debug logging
    
    Returns:
        Total number of stats cleared across all instances
    """
    
    
    clear_session = aiohttp.ClientSession()
    try:
        clear_tasks = []
        for instance_url in instance_urls:
            clear_url = f"{instance_url}/clear_mm_processor_stats"
            clear_tasks.append(
                clear_session.post(clear_url, timeout=aiohttp.ClientTimeout(total=10))
            )
        
        clear_responses = await asyncio.gather(*clear_tasks, return_exceptions=True)
        total_cleared = 0
        for i, response in enumerate(clear_responses):
            if isinstance(response, Exception):
                if debug:
                    print(f"Warning: Failed to clear stats from {instance_urls[i]}: {response}")
            else:
                try:
                    result = await response.json()
                    count = result.get("count", 0)
                    total_cleared += count
                except Exception:
                    pass  # Ignore parsing errors
        
        if total_cleared > 0 and not debug:
            print(f"Cleared {total_cleared} MM processor stats from registry (removed test run stats)")
        
        return total_cleared
    finally:
        await clear_session.close()


async def collect_mm_processor_stats(
    instance_urls: list[str],
    session: Any | None = None,
    debug: bool = False,
) -> dict[str, list[float]]:
    """
    Collect multimodal processor timing stats from vLLM server(s). 
    Queries the /mm-processor-stats endpoint from each instance and collects stats.

    Args:
        instance_urls: List of base URLs for vLLM instances (required)
        session: aiohttp session (required)
        debug: Enable minimal debug logging

    Returns:
        Dictionary mapping stage names to lists of timing values (in seconds)
    """
    if session is None:
        raise ValueError("session is required for collecting MM processor stats")
    
    if not instance_urls:
        raise ValueError("instance_urls must be provided")

    stats_by_stage = {
        "hf_processor_time": [],
        "hashing_time": [],
        "cache_lookup_time": [],
        "prompt_update_time": [],
        "total_time": [],
    }

    all_stats = {}
    for instance_url in instance_urls:
        stats_url = f"{instance_url}/mm-processor-stats"
        try:
            async with session.get(stats_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "stats" in data:
                            all_stats.update(data["stats"])
        except Exception as e:
            print(f"Warning: Failed to query stats from {instance_url}: {e}")

    for stats_dict in all_stats.values():
        stats_by_stage["hf_processor_time"].append(
            stats_dict.get("hf_processor_time", 0.0)
        )
        stats_by_stage["hashing_time"].append(stats_dict.get("hashing_time", 0.0))
        stats_by_stage["cache_lookup_time"].append(
            stats_dict.get("cache_lookup_time", 0.0)
        )
        stats_by_stage["prompt_update_time"].append(
            stats_dict.get("prompt_update_time", 0.0)
        )
        stats_by_stage["total_time"].append(stats_dict.get("total_time", 0.0))

    if debug and not any(stats_by_stage.values()):
        print("Warning: No MM processor stats found. Ensure VLLM_ENABLE_MM_PROCESSOR_STATS=1 on server.")

    return stats_by_stage


def calculate_mm_processor_metrics(
    stats_by_stage: dict[str, list[float]],
    selected_percentiles: list[float],
) -> dict[str, dict[str, float]]:
    """
    Calculate aggregate metrics from stats by stage.
    """
    metrics = {}

    for stage_name, times in stats_by_stage.items():
        if not times:
            metrics[stage_name] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                **{f"p{p}": 0.0 for p in selected_percentiles},
            }
            continue

        times_ms = [t * 1000 for t in times]
        metrics[stage_name] = {
            "mean": float(np.mean(times_ms)),
            "median": float(np.median(times_ms)),
            "std": float(np.std(times_ms)),
            **{
                f"p{p}": float(np.percentile(times_ms, p)) for p in selected_percentiles
            },
        }

    return metrics


async def benchmark_multimodal_processor(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Run the multimodal processor benchmark.
    """

    model_id = args.model
    model_name = getattr(args, "served_model_name", None)
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = getattr(args, "tokenizer_mode", "auto")

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = args.base_url
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    input_requests = get_samples(args, tokenizer)

    task_type = (
        TaskType.POOLING
        if "embeddings" in args.backend or "rerank" in args.backend
        else TaskType.GENERATION
    )

    if task_type == TaskType.GENERATION:
        sampling_params = {
            k: v
            for k, v in {
                "top_p": getattr(args, "top_p", None),
                "top_k": getattr(args, "top_k", None),
                "min_p": getattr(args, "min_p", None),
                "temperature": getattr(args, "temperature", None),
                "frequency_penalty": getattr(args, "frequency_penalty", None),
                "presence_penalty": getattr(args, "presence_penalty", None),
                "repetition_penalty": getattr(args, "repetition_penalty", None),
            }.items()
            if v is not None
        }

        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError(
                "Sampling parameters are only supported by openai-compatible backends."
            )

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 0.0
    else:
        sampling_params = {}

    extra_body = getattr(args, "extra_body", {}) or {}
    extra_body = {**sampling_params, **extra_body}

    headers = None
    if hasattr(args, "header") and args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")

    percentile_metrics = getattr(args, "percentile_metrics", None) or "e2el"
    selected_percentile_metrics = percentile_metrics.split(",")
    selected_percentiles = [
        float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
    ]

    freeze_gc_heap()

    if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
        print(
            "Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled. "
            "MM processor timing stats will not be collected."
        )

    debug = getattr(args, "debug_mm_stats", False)
    instance_urls = _get_instance_urls(args, base_url)
    should_clear_registry = not getattr(args, "no_clear_mm_stats_registry", False)
    if should_clear_registry:
        await _clear_mm_processor_stats_registry(instance_urls, debug=debug)

    benchmark_result = await benchmark(
        task_type=task_type,
        endpoint_type=args.backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=None,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=False,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        ignore_eos=args.ignore_eos,
        goodput_config_dict={},
        max_concurrency=args.max_concurrency,
        lora_modules=getattr(args, "lora_modules", None),
        extra_headers=headers,
        extra_body=extra_body,
        ramp_up_strategy=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=getattr(args, "ready_check_timeout_sec", 600),
    )

    session = aiohttp.ClientSession()
    try:
        mm_stats_by_stage = await collect_mm_processor_stats(
            instance_urls=instance_urls,
            session=session,
            debug=debug,
        )
    finally:
        await session.close()

    if not any(mm_stats_by_stage.values()):
        if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
            print(
                "\n⚠️  Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled.\n"
                "   MM processor timing stats will not be collected.\n"
                "   Set VLLM_ENABLE_MM_PROCESSOR_STATS=1 when starting the server.\n"
            )
        else:
            print(
                "\n⚠️  Warning: No MM processor stats found in registry.\n"
                "   This may indicate that:\n"
                "   - No multimodal requests were processed\n"
                "   - Stats were already retrieved (registry is cleared after retrieval)\n"
            )

    mm_processor_metrics = calculate_mm_processor_metrics(
        mm_stats_by_stage, selected_percentiles
    )
    benchmark_result["mm_processor_stats"] = mm_processor_metrics

    return benchmark_result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the multimodal processor benchmark."""
    from vllm.benchmarks import serve

    serve.add_cli_args(parser)

    parser.set_defaults(endpoint="/v1/chat/completions")
    parser.set_defaults(backend="openai-chat")

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the benchmark results in JSON format.",
    )
    parser.add_argument(
        "--no-clear-mm-stats-registry",
        action="store_true",
        default=False,
        help="Don't clear MM processor stats registry before collecting stats (keeps test run stats)",
    )
    parser.add_argument(
        "--debug-mm-stats",
        action="store_true",
        help="Enable debug logging for MM processor stats collection.",
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for the multimodal processor benchmark."""
    import asyncio
    from datetime import datetime

    if args.dataset_name is None:
        raise ValueError("--dataset-name is required")

    if args.dataset_name not in ("hf", "vision-arena", "mmvu"):
        print(
            f"Warning: Dataset '{args.dataset_name}' may not be multimodal. "
            "Results may not include MM processor timing."
        )

    print("Starting multimodal processor benchmark...")
    result = asyncio.run(benchmark_multimodal_processor(args))

    print("\n" + "=" * 80)
    print("Multimodal Processor Benchmark Results")
    print("=" * 80)

    if "mm_processor_stats" in result:
        print("\nMM Processor Timing (ms):")
        selected_percentiles = [
            float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
        ]
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

    if args.output_json:
        result["config"] = {
            "model": args.model,
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
