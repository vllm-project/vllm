# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark multimodal processor latency.

This benchmark measures the latency of the mm processor module
using multimodal prompts from datasets.
MM processor stats are automatically enabled.

Run:
    vllm bench mm-processor \
        --model <your_model> \
        --dataset-name random-mm \
        --num-prompts 10 \
"""

import argparse
import dataclasses
import json
import time
from datetime import datetime
from typing import Any

import numpy as np

from vllm.benchmarks.throughput import get_requests
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.processing.context import (
    get_timing_stats_from_engine_client,
)
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.import_utils import PlaceholderModule

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")


def collect_mm_processor_stats(
    llm_engine: Any,
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


def validate_args(args):
    """
    Validate command-line arguments for mm_processor benchmark.
    """
    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model
    if not hasattr(args, "dataset_path"):
        args.dataset_path = None
    if not hasattr(args, "lora_path"):
        args.lora_path = None
    if not hasattr(args, "max_loras"):
        args.max_loras = None


def benchmark_multimodal_processor(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Run the multimodal processor benchmark.
    """
    from vllm import LLM, SamplingParams

    validate_args(args)

    if args.seed is None:
        args.seed = 0

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    tokenizer = llm.get_tokenizer()
    requests = get_requests(args, tokenizer)

    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = [request.prompt for request in requests]
    expected_output_lens = [request.expected_output_len for request in requests]

    sampling_params = [
        SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=output_len,
            detokenize=True,
        )
        for output_len in expected_output_lens
    ]

    selected_percentiles = [
        float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
    ]

    freeze_gc_heap()

    print(f"Processing {len(prompts)} requests...")
    start_time = time.perf_counter()

    outputs = llm.chat(
        prompts, sampling_params, use_tqdm=not getattr(args, "disable_tqdm", False)
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    mm_stats_by_stage = collect_mm_processor_stats(
        llm.llm_engine,
    )

    if not any(mm_stats_by_stage.values()):
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
        if not output.finished or output.metrics is None:
            continue
        metrics = output.metrics
        for attr in ("finished_time", "last_token_time"):
            if (
                getattr(metrics, attr, None) is not None
                and getattr(metrics, "arrival_time", None) is not None
            ):
                e2el_times.append(
                    (getattr(metrics, attr) - metrics.arrival_time) * 1000
                )
                break

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
    from vllm.engine.arg_utils import EngineArgs

    EngineArgs.add_cli_args(parser)

    parser.set_defaults(enable_mm_processor_stats=True)

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random-mm",
        choices=["random-mm", "random-rerank"],
        help="Name of the dataset to benchmark on. Defaults to 'random-mm'.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to process.",
    )

    from vllm.benchmarks.datasets import (
        add_random_dataset_base_args,
        add_random_multimodal_dataset_args,
    )

    add_random_dataset_base_args(parser)
    add_random_multimodal_dataset_args(parser)

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the benchmark results in JSON format.",
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


def main(args: argparse.Namespace) -> None:
    """Main entry point for the multimodal processor benchmark."""

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
            "num_prompts": args.num_prompts,
            "input_len": getattr(args, "random_input_len", None),
            "output_len": getattr(args, "random_output_len", None),
        }
        result["timestamp"] = datetime.now().isoformat()

        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mm processor latency")
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)
