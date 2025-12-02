# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark multimodal processor latency.

This benchmark measures the latency of the multimodal processor module.

Run:
    vllm bench multimodal-processor \
        --model <your_model> \
        --dataset-name <dataset_name> \
        --dataset-path <dataset_path> \
        --enable-mm-processor-stats
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import vllm.envs as envs
from vllm.benchmarks.datasets import SampleRequest, get_samples
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.processing import (
    clear_timing_stats_from_engine_client,
    get_timing_stats_from_engine_client,
)
from vllm.utils.gc_utils import freeze_gc_heap


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


def clear_mm_processor_stats(
    llm_engine: Any,
    debug: bool = False,
) -> int:
    """
    Clear MM processor stats registry.
    Returns the number of stats cleared.
    """
    count = clear_timing_stats_from_engine_client(llm_engine)
    if count > 0 and not debug:
        print(
            f"Cleared {count} MM processor stats from registry (removed test run stats)"
        )
    return count


def collect_mm_processor_stats(
    llm_engine: Any,
    debug: bool = False,
) -> dict[str, list[float]]:
    """
    Collect multimodal processor timing stats.
    Returns a dictionary mapping stage names to lists of timing values (in seconds).
    """
    all_stats = get_timing_stats_from_engine_client(llm_engine)

    stats_by_stage = {
        "hf_processor_time": [],
        "hashing_time": [],
        "cache_lookup_time": [],
        "prompt_update_time": [],
        "total_time": [],
    }

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
        print(
            "Warning: No MM processor stats found. Ensure --enable-mm-processor-stats is set."
        )

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


def benchmark_multimodal_processor(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Run the multimodal processor benchmark.
    """
    from vllm import LLM, SamplingParams

    engine_args = EngineArgs.from_cli_args(args)

    llm = LLM(**dataclasses.asdict(engine_args))

    tokenizer = llm.get_tokenizer()
    input_requests = get_samples(args, tokenizer)

    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in input_requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = []
    sampling_params: list[SamplingParams] = []

    for request in input_requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=getattr(args, "temperature", 0.0),
                top_p=getattr(args, "top_p", None),
                top_k=getattr(args, "top_k", None),
                min_p=getattr(args, "min_p", None),
                frequency_penalty=getattr(args, "frequency_penalty", None),
                presence_penalty=getattr(args, "presence_penalty", None),
                repetition_penalty=getattr(args, "repetition_penalty", None),
                ignore_eos=getattr(args, "ignore_eos", False),
                max_tokens=request.expected_output_len,
                detokenize=True,
            )
        )

    selected_percentiles = [
        float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
    ]

    freeze_gc_heap()

    if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
        print(
            "Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled. "
            "MM processor timing stats will not be collected. "
            "Set --enable-mm-processor-stats to enable."
        )

    debug = getattr(args, "debug_mm_stats", False)
    should_clear_registry = not getattr(args, "no_clear_mm_stats_registry", False)

    if should_clear_registry:
        clear_mm_processor_stats(llm.llm_engine, debug=debug)

    print(f"Processing {len(prompts)} requests...")
    start_time = time.perf_counter()

    outputs = llm.chat(
        prompts, sampling_params, use_tqdm=not getattr(args, "disable_tqdm", False)
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    mm_stats_by_stage = collect_mm_processor_stats(
        llm.llm_engine,
        debug=debug,
    )

    if not any(mm_stats_by_stage.values()):
        if not envs.VLLM_ENABLE_MM_PROCESSOR_STATS:
            print(
                "\n⚠️  Warning: VLLM_ENABLE_MM_PROCESSOR_STATS is not enabled.\n"
                "   MM processor timing stats will not be collected.\n"
                "   Set --enable-mm-processor-stats to enable.\n"
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

    completed = len([o for o in outputs if o.finished])
    failed = len(outputs) - completed

    e2el_times = []
    for output in outputs:
        if output.finished and output.metrics is not None:
            if hasattr(output.metrics, "finished_time") and hasattr(
                output.metrics, "arrival_time"
            ):
                if (
                    output.metrics.finished_time is not None
                    and output.metrics.arrival_time is not None
                ):
                    e2el_times.append(
                        (output.metrics.finished_time - output.metrics.arrival_time)
                        * 1000
                    )
            elif hasattr(output.metrics, "last_token_time") and hasattr(
                output.metrics, "arrival_time"
            ):
                if (
                    output.metrics.last_token_time is not None
                    and output.metrics.arrival_time is not None
                ):
                    e2el_times.append(
                        (output.metrics.last_token_time - output.metrics.arrival_time)
                        * 1000
                    )

    if not e2el_times and completed > 0:
        avg_time_per_request = total_time / completed
        e2el_times = [avg_time_per_request * 1000] * completed

    if e2el_times:
        mean_e2el_ms = float(np.mean(e2el_times))
        median_e2el_ms = float(np.median(e2el_times))
        std_e2el_ms = float(np.std(e2el_times))
        percentiles_e2el_ms = [
            (p, float(np.percentile(e2el_times, p))) for p in selected_percentiles
        ]
    else:
        mean_e2el_ms = 0.0
        median_e2el_ms = 0.0
        std_e2el_ms = 0.0
        percentiles_e2el_ms = [(p, 0.0) for p in selected_percentiles]

    benchmark_result = {
        "completed": completed,
        "failed": failed,
        "mean_e2el_ms": mean_e2el_ms,
        "median_e2el_ms": median_e2el_ms,
        "std_e2el_ms": std_e2el_ms,
        "percentiles_e2el_ms": percentiles_e2el_ms,
        "mm_processor_stats": mm_processor_metrics,
    }

    return benchmark_result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the multimodal processor benchmark."""
    from vllm.benchmarks.datasets import add_dataset_parser
    from vllm.engine.arg_utils import EngineArgs

    add_dataset_parser(parser)

    EngineArgs.add_cli_args(parser)

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
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles to calculate (e.g., '50,90,99').",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    # Sampling parameters
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature sampling parameter.",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter.",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty sampling parameter.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty sampling parameter.",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty sampling parameter.",
    )
    sampling_group.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token during generation.",
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for the multimodal processor benchmark."""
    from datetime import datetime

    if args.dataset_name is None:
        raise ValueError("--dataset-name is required")

    if args.dataset_name not in ("hf", "vision-arena", "mmvu"):
        print(
            f"Warning: Dataset '{args.dataset_name}' may not be multimodal. "
            "Results may not include MM processor timing."
        )

    print("Starting multimodal processor benchmark...")
    result = benchmark_multimodal_processor(args)

    print("\n" + "=" * 80)
    print("Multimodal Processor Benchmark Results")
    print("=" * 80)

    if "mm_processor_stats" in result:
        print("\nMM Processor Timing (ms):")
        selected_percentiles = [
            float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
        ]
        mm_data = []
        for stage, metrics in result["mm_processor_stats"].items():
            row = {
                "Stage": stage,
                "Mean": f"{metrics['mean']:.2f}",
                "Median": f"{metrics['median']:.2f}",
                "Std": f"{metrics['std']:.2f}",
            }
            for p in selected_percentiles:
                row[f"P{p}"] = f"{metrics.get(f'p{p}', 0.0):.2f}"
            mm_data.append(row)

        mm_df = pd.DataFrame(mm_data)
        print(mm_df.to_string(index=False))

    if "mean_e2el_ms" in result:
        print("\nEnd-to-End Latency (ms):")
        selected_percentiles = [
            float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
        ]

        e2el_data = [
            {"Metric": "Mean", "Value (ms)": f"{result['mean_e2el_ms']:.2f}"},
            {"Metric": "Median", "Value (ms)": f"{result['median_e2el_ms']:.2f}"},
            {"Metric": "Std", "Value (ms)": f"{result['std_e2el_ms']:.2f}"},
        ]

        for p in selected_percentiles:
            percentile_value = next(
                (val for pct, val in result["percentiles_e2el_ms"] if pct == p),
                0.0,
            )
            e2el_data.append(
                {
                    "Metric": f"P{p}",
                    "Value (ms)": f"{percentile_value:.2f}",
                }
            )

        e2el_df = pd.DataFrame(e2el_data)
        print(e2el_df.to_string(index=False))

    if args.output_json:
        result["config"] = {
            "model": args.model,
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
