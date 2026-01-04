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
import torch

from vllm.benchmarks.throughput import get_requests
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.processing import (
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


# ============================================================================
# Encoder Benchmarking Functions
# ============================================================================


def get_vision_encoder(llm_engine: Any) -> torch.nn.Module | None:
    """
    Extract vision encoder from vLLM engine.

    Args:
        llm_engine: The vLLM engine instance

    Returns:
        The vision encoder module, or None if not found
    """
    try:
        model = llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        return None

    # Handle different model architectures
    if hasattr(model, "vision_tower"):
        return model.vision_tower  # LLaVA-style
    elif hasattr(model, "vision_model"):
        return model.vision_model  # Other styles
    elif hasattr(model, "visual"):
        return model.visual  # Qwen-VL style
    else:
        return None


def is_supported_encoder(encoder: torch.nn.Module) -> bool:
    """Check if the encoder type is supported for benchmarking."""
    # Import here to avoid circular imports
    try:
        from vllm.model_executor.models.clip import CLIPVisionModel
        from vllm.model_executor.models.siglip import SiglipVisionModel

        return isinstance(encoder, (CLIPVisionModel, SiglipVisionModel))
    except ImportError:
        return False


def create_encoder_inputs(
    encoder: torch.nn.Module,
    batch_size: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create dummy input tensor for encoder forward pass.

    Args:
        encoder: The vision encoder module
        batch_size: Number of images in batch
        image_height: Height of images
        image_width: Width of images
        device: Device to create tensor on

    Returns:
        Random image tensor [batch, channels, height, width]
    """
    # Get encoder config for num_channels
    config = getattr(encoder, "config", None)
    if config is None:
        config = getattr(encoder, "vision_config", None)

    num_channels = getattr(config, "num_channels", 3)

    # Get dtype from encoder parameters
    dtype = torch.float16  # Default
    for param in encoder.parameters():
        dtype = param.dtype
        break

    return torch.randn(
        batch_size,
        num_channels,
        image_height,
        image_width,
        device=device,
        dtype=dtype,
    )


def measure_encoder_latency(
    encoder: torch.nn.Module,
    inputs: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
) -> list[float]:
    """
    Measure encoder forward pass latency.

    Args:
        encoder: The vision encoder module
        inputs: Input tensor for encoder
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        List of latency measurements in seconds
    """
    encoder.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(inputs)

    # Timed runs
    torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = encoder(inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append(end - start)

    return latencies


def benchmark_encoder_forward(
    llm_engine: Any,
    batch_sizes: list[int],
    image_sizes: list[tuple[int, int]],
    num_warmup: int,
    num_iterations: int,
    selected_percentiles: list[float],
) -> dict[str, Any] | None:
    """
    Benchmark encoder forward pass latency.

    Args:
        llm_engine: The vLLM engine instance
        batch_sizes: List of batch sizes to test
        image_sizes: List of (height, width) tuples to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        selected_percentiles: Percentiles to calculate

    Returns:
        Dict with encoder stats, or None if encoder not found
    """
    encoder = get_vision_encoder(llm_engine)
    if encoder is None:
        print("\n⚠️  Warning: Could not find vision encoder in model.")
        return None

    if not is_supported_encoder(encoder):
        encoder_type = type(encoder).__name__
        print(
            f"\n⚠️  Warning: Encoder type '{encoder_type}' is not fully supported. "
            "Benchmarking may not work correctly."
        )

    # Get device from encoder
    device = next(encoder.parameters()).device

    results = []
    for batch_size in batch_sizes:
        for height, width in image_sizes:
            print(
                f"  Benchmarking: batch_size={batch_size}, "
                f"image_size={height}x{width}..."
            )

            inputs = create_encoder_inputs(encoder, batch_size, height, width, device)

            latencies = measure_encoder_latency(
                encoder, inputs, num_warmup, num_iterations
            )

            # Convert to milliseconds
            latencies_ms = [t * 1000 for t in latencies]

            results.append(
                {
                    "batch_size": batch_size,
                    "image_size": f"{height}x{width}",
                    "mean": float(np.mean(latencies_ms)),
                    "median": float(np.median(latencies_ms)),
                    "std": float(np.std(latencies_ms)),
                    **{
                        f"p{p}": float(np.percentile(latencies_ms, p))
                        for p in selected_percentiles
                    },
                }
            )

            # Free memory
            del inputs
            torch.cuda.empty_cache()

    return {
        "encoder_type": type(encoder).__name__,
        "num_warmup": num_warmup,
        "num_iterations": num_iterations,
        "results": results,
    }


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

    # Add encoder benchmarking if requested
    if getattr(args, "benchmark_encoder", False):
        print("\nBenchmarking encoder forward pass...")
        batch_sizes = [int(x) for x in args.encoder_batch_sizes.split(",")]

        # Get image sizes from bucket config
        bucket_config = getattr(args, "random_mm_bucket_config", None)
        if bucket_config:
            image_sizes = [
                (h, w)
                for (h, w, frames) in bucket_config
                if frames == 1  # Only static images
            ]
        else:
            # Default image sizes if no bucket config
            image_sizes = [(224, 224), (336, 336)]

        encoder_results = benchmark_encoder_forward(
            llm.llm_engine,
            batch_sizes=batch_sizes,
            image_sizes=image_sizes,
            num_warmup=args.encoder_num_warmup,
            num_iterations=args.encoder_num_iterations,
            selected_percentiles=selected_percentiles,
        )
        benchmark_result["encoder_stats"] = encoder_results

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

    # Encoder benchmarking arguments
    parser.add_argument(
        "--benchmark-encoder",
        action="store_true",
        help="Also benchmark encoder forward pass latency.",
    )
    parser.add_argument(
        "--encoder-batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes for encoder benchmark.",
    )
    parser.add_argument(
        "--encoder-num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for encoder benchmark.",
    )
    parser.add_argument(
        "--encoder-num-iterations",
        type=int,
        default=100,
        help="Number of timed iterations for encoder benchmark.",
    )
    parser.add_argument(
        "--encoder-use-real-images",
        action="store_true",
        help="Use real images from dataset instead of random tensors.",
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

    # Display encoder benchmark results
    if "encoder_stats" in result and result["encoder_stats"] is not None:
        encoder_stats = result["encoder_stats"]
        print("\nEncoder Forward Pass Latency (ms):")
        print(f"  Encoder Type: {encoder_stats['encoder_type']}")
        print(
            f"  Warmup: {encoder_stats['num_warmup']}, "
            f"Iterations: {encoder_stats['num_iterations']}"
        )

        selected_percentiles = [
            float(p) for p in getattr(args, "metric_percentiles", "99").split(",")
        ]

        encoder_data = []
        for entry in encoder_stats["results"]:
            row = {
                "Batch": entry["batch_size"],
                "Image Size": entry["image_size"],
                "Mean": f"{entry['mean']:.2f}",
                "Median": f"{entry['median']:.2f}",
                "Std": f"{entry['std']:.2f}",
            }
            for p in selected_percentiles:
                row[f"P{p}"] = f"{entry.get(f'p{p}', 0.0):.2f}"
            encoder_data.append(row)

        encoder_df = pd.DataFrame(encoder_data)
        print(encoder_df.to_string(index=False))

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
