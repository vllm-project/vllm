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
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from vllm.benchmarks.datasets import (
    MultiModalConversationDataset,
    VisionArenaDataset,
)
from vllm.benchmarks.throughput import get_requests
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.import_utils import PlaceholderModule

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

if TYPE_CHECKING:  # Avoid having to mock during docs build
    from vllm.v1.engine.llm_engine import LLMEngine
else:
    LLMEngine = object


def get_timing_stats_from_engine(llm_engine: LLMEngine) -> dict[str, dict[str, float]]:
    """
    Get all multimodal timing stats from the LLM engine.

    Collects both preprocessing stats (HF processor, hashing, cache lookup,
    prompt update) and encoder forward pass timing, merged by request_id.

    Args:
        llm_engine: The LLM engine (has input_processor and workers).

    Returns:
        Dictionary mapping request_id to merged stats dict containing
        both preprocessing and encoder timing metrics.

    Example:
        {
            'request-123': {
                'get_mm_hashes_secs': 0.02,
                'get_cache_missing_items_secs': 0.01,
                'apply_hf_processor_secs': 0.45,
                'merge_mm_kwargs_secs': 0.01,
                'apply_prompt_updates_secs': 0.03,
                'preprocessor_total_secs': 0.51,
                'encoder_forward_secs': 0.23,
                'num_encoder_calls': 1
            }
        }
    """
    observability_config = llm_engine.vllm_config.observability_config
    if not observability_config or not observability_config.enable_mm_processor_stats:
        return {}

    renderer = llm_engine.renderer
    mm_processor_stats = renderer._mm_timing_registry.stat()

    encoder_stats = dict[str, dict[str, float]]()
    for worker_stats in llm_engine.collective_rpc("get_encoder_timing_stats"):
        if not worker_stats:
            continue

        for request_id, stats_dict in worker_stats.items():
            if request_id not in encoder_stats:
                encoder_stats[request_id] = dict(stats_dict)
            else:
                # Aggregate timing metrics across workers
                current_time = encoder_stats[request_id].get(
                    "encoder_forward_secs", 0.0
                )
                new_time = stats_dict.get("encoder_forward_secs", 0.0)
                encoder_stats[request_id]["encoder_forward_secs"] = max(
                    current_time, new_time
                )

                current_calls = encoder_stats[request_id].get("num_encoder_calls", 0)
                new_calls = stats_dict.get("num_encoder_calls", 0)
                encoder_stats[request_id]["num_encoder_calls"] = max(
                    current_calls, new_calls
                )

    merged_stats = dict[str, dict[str, float]]()

    for request_id, prep_dict in mm_processor_stats.items():
        merged_stats[request_id] = dict(prep_dict)

    for request_id, enc_dict in encoder_stats.items():
        if request_id in merged_stats:
            merged_stats[request_id].update(enc_dict)
            continue

        # In V1 engine, the request_id in encoder_stats has a suffix
        # appended to the original request_id (which is used in
        # preprocessing_stats).
        # We try to strip the suffix to find the matching request.
        possible_original_id = request_id.rpartition("-")[0]
        if possible_original_id and possible_original_id in merged_stats:
            merged_stats[possible_original_id].update(enc_dict)
        else:
            merged_stats[request_id] = dict(enc_dict)

    return merged_stats


def collect_mm_processor_stats(llm_engine: LLMEngine) -> dict[str, list[float]]:
    """
    Collect multimodal processor timing stats.
    Returns a dictionary mapping stage names to lists of timing values (in seconds).
    """
    all_stats = get_timing_stats_from_engine(llm_engine)

    stats_by_stage = defaultdict[str, list[float]](list)

    for stats_dict in all_stats.values():
        for stat_key, stat_val in stats_dict.items():
            stats_by_stage[stat_key].append(stat_val)

    return stats_by_stage


def calculate_mm_processor_metrics(
    stats_by_stage: dict[str, list[float]],
    selected_percentiles: list[float],
    *,
    unit: Literal["us", "ms", "s"] = "ms",
) -> dict[str, dict[str, float]]:
    """
    Calculate aggregate metrics from stats by stage.
    """
    unit2mult = {"us": 1000000, "ms": 1000, "s": 1}
    unit_mult = unit2mult[unit]

    metrics = {}

    for stage, times in stats_by_stage.items():
        stage_name = stage.replace("_secs", "_" + unit)

        if not times:
            metrics[stage_name] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                **{f"p{p}": 0.0 for p in selected_percentiles},
            }
            continue

        is_count_metric = stage == "num_encoder_calls"
        values = times if is_count_metric else [t * unit_mult for t in times]

        metrics[stage_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            **{f"p{p}": float(np.percentile(values, p)) for p in selected_percentiles},
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

    if args.dataset_name == "hf" and not args.dataset_path:
        raise ValueError(
            "--dataset-path is required when using --dataset-name hf. "
            "For multimodal benchmarking, specify a dataset like "
            "'lmarena-ai/VisionArena-Chat'."
        )
    if args.dataset_name == "hf":
        supported_mm_datasets = (
            VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
            | MultiModalConversationDataset.SUPPORTED_DATASET_PATHS
        )
        if args.dataset_path not in supported_mm_datasets:
            raise ValueError(
                f"{args.dataset_path} is not a supported multimodal dataset. "
                f"Supported multimodal datasets are: {sorted(supported_mm_datasets)}"
            )


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

    num_warmups = getattr(args, "num_warmups", 0)
    if num_warmups > 0:
        print(f"Processing {num_warmups} warmup requests...")
        # Create a temporary args object for warmup requests
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.num_prompts = num_warmups
        warmup_args.seed += 1
        warmup_requests = get_requests(warmup_args, tokenizer)
        warmup_prompts = [req.prompt for req in warmup_requests]
        warmup_output_lens = [req.expected_output_len for req in warmup_requests]
        warmup_sampling_params = [
            SamplingParams(max_tokens=output_len) for output_len in warmup_output_lens
        ]
        llm.chat(
            warmup_prompts,
            warmup_sampling_params,
            use_tqdm=not getattr(args, "disable_tqdm", False),
        )

    # Clear stats from warmup requests
    collect_mm_processor_stats(llm.llm_engine)

    print(f"Processing {len(prompts)} requests...")
    start_time = time.perf_counter()

    outputs = llm.chat(
        prompts, sampling_params, use_tqdm=not getattr(args, "disable_tqdm", False)
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    mm_stats_by_stage = collect_mm_processor_stats(llm.llm_engine)

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
        # Calculate E2E latency as: TTFT + (last_token_ts - first_token_ts)
        if (
            getattr(metrics, "first_token_latency", None) is not None
            and getattr(metrics, "last_token_ts", None) is not None
            and getattr(metrics, "first_token_ts", None) is not None
        ):
            ttft = metrics.first_token_latency
            # Decode time is the duration between the first and last token generation
            decode_time = max(0.0, metrics.last_token_ts - metrics.first_token_ts)
            e2el_times.append((ttft + decode_time) * 1000)

    if not e2el_times and completed > 0:
        print(
            "\n⚠️  Warning: Detailed end-to-end latency metrics not available.\n"
            "   Falling back to average request latency "
            "(total_time / num_completed_requests).\n"
        )
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

    encoder_summary = {}
    if (
        "num_encoder_calls" in mm_stats_by_stage
        and mm_stats_by_stage["num_encoder_calls"]
    ):
        encoder_calls = mm_stats_by_stage["num_encoder_calls"]
        encoder_summary = {
            "total_encoder_calls": int(sum(encoder_calls)),
            "num_requests_with_encoder_calls": len(encoder_calls),
        }

    benchmark_result = {
        "completed": completed,
        "failed": failed,
        "mean_e2el_ms": mean_e2el_ms,
        "median_e2el_ms": median_e2el_ms,
        "std_e2el_ms": std_e2el_ms,
        "percentiles_e2el_ms": percentiles_e2el_ms,
        "mm_processor_stats": mm_processor_metrics,
        "encoder_summary": encoder_summary,
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
        choices=["random-mm", "hf"],
        help="Name of the dataset to benchmark on. Defaults to 'random-mm'.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=1,
        help="Number of warmup prompts to process.",
    )

    from vllm.benchmarks.datasets import (
        add_random_dataset_base_args,
        add_random_multimodal_dataset_args,
    )

    add_random_dataset_base_args(parser)
    add_random_multimodal_dataset_args(parser)

    # HuggingFace dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file or HuggingFace dataset name "
        "(e.g., 'yale-nlp/MMVU', 'lmarena-ai/VisionArena-Chat').",
    )
    parser.add_argument(
        "--hf-subset",
        type=str,
        default=None,
        help="Subset of the HuggingFace dataset (optional).",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="Split of the HuggingFace dataset (e.g., 'train', 'test', 'validation').",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. "
        "Overrides the default output lengths from the dataset.",
    )

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
        print("\nMM Processor Metrics:")
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

        if "encoder_summary" in result and result["encoder_summary"]:
            total_calls = result["encoder_summary"]["total_encoder_calls"]
            num_requests = result["encoder_summary"]["num_requests_with_encoder_calls"]
            print(
                f"\nSummary: {total_calls} total encoder calls "
                f"across {num_requests} requests."
            )

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
