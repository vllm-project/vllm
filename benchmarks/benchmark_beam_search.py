# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark beam search latency and throughput in vLLM.

This benchmark follows the vLLM benchmark conventions and provides
both latency and throughput metrics for beam search operations.

Supports both random synthetic data and real datasets (ShareGPT, etc.).
"""

import argparse
import dataclasses
import json
import os
import random
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.tokenizers import get_tokenizer


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    """Save results in PyTorch benchmark format if enabled."""
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return

    pt_records = []
    metrics = {
        "avg_latency": [results["avg_latency"]],
        "requests_per_second": [results["requests_per_second"]],
        "output_tokens_per_second": [results["output_tokens_per_second"]],
    }

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM beam search benchmark",
                "extra_info": {"args": vars(args)},
            },
            "model": {"name": args.model},
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": {
                    k: results[k]
                    for k in ["percentiles", "total_output_tokens", "elapsed_time"]
                    if k in results
                },
            },
        }
        pt_records.append(record)

    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        with open(pt_file, "w") as f:
            json.dump(pt_records, f, indent=2)


def get_requests_from_dataset(args, tokenizer):
    """Load requests from a dataset file or generate random requests."""
    from vllm.benchmarks.datasets import (
        RandomDataset,
        ShareGPTDataset,
        SonnetDataset,
    )

    seed = getattr(args, "seed", 0) or 0

    if args.dataset_name == "random" or args.dataset_path is None:
        # Use random dataset
        dataset = RandomDataset(
            dataset_path=None,
            random_seed=seed,
        )
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            range_ratio=args.random_range_ratio,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    elif args.dataset_name == "sharegpt":
        dataset = ShareGPTDataset(
            dataset_path=args.dataset_path,
            random_seed=seed,
        )
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.output_len if args.output_len else None,
        )
    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(
            dataset_path=args.dataset_path,
            random_seed=seed,
        )
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            input_len=args.input_len if args.input_len else None,
            output_len=args.output_len if args.output_len else None,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    return requests


def get_prompts_from_requests(requests) -> tuple[list[PromptType], int]:
    """Convert SampleRequest objects to prompts for beam search."""
    prompts = []
    output_len = None

    for req in requests:
        if isinstance(req.prompt, dict) and "prompt_token_ids" in req.prompt:
            prompts.append({"prompt_token_ids": req.prompt["prompt_token_ids"]})
        elif isinstance(req.prompt, str):
            prompts.append({"prompt": req.prompt})
        else:
            prompts.append(req.prompt)

        # For beam search, all requests should have the same output length
        if output_len is None:
            output_len = req.expected_output_len
        else:
            # Use the maximum output length across all requests
            output_len = max(output_len, req.expected_output_len)

    return prompts, output_len


def generate_random_prompts(
    batch_size: int,
    input_len: int,
) -> list[PromptType]:
    """Generate random prompts with dummy token IDs."""
    dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, input_len))
    return [{"prompt_token_ids": batch.tolist()} for batch in dummy_prompt_token_ids]


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for beam search benchmark."""
    # Beam search specific arguments
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for beam search (default: 4).",
    )
    parser.add_argument(
        "--beam-widths",
        type=int,
        nargs="+",
        default=None,
        help="List of beam widths to sweep. Overrides --beam-width if set.",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "sharegpt", "sonnet"],
        help="Name of the dataset to use (default: random).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (required for sharegpt/sonnet).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts to use from dataset (default: 8).",
    )
    # Note: --seed is already defined in EngineArgs.add_cli_args()

    # Input/output configuration
    parser.add_argument(
        "--input-len",
        type=int,
        default=32,
        help="Input prompt length in tokens (default: 32).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length to generate (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of prompts per batch. If not set, uses --num-prompts.",
    )

    # Random dataset specific arguments
    parser.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens for random dataset (default: 0).",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length (default: 0.0).",
    )

    # Benchmark configuration
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the generation process of a single batch.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results in JSON format.",
    )

    # Add engine arguments
    parser = EngineArgs.add_cli_args(parser)
    # Disable prefix caching by default for consistent latency measurements
    parser.set_defaults(enable_prefix_caching=False)

    return parser


def run_beam_search_benchmark(
    llm,
    prompts: list[PromptType],
    beam_params: BeamSearchParams,
    num_iters_warmup: int,
    num_iters: int,
    do_profile: bool = False,
) -> dict[str, Any]:
    """Run beam search benchmark and return metrics."""

    def run_beam_search():
        return llm.beam_search(prompts, beam_params)

    def run_to_completion(profile: bool = False) -> float | None:
        if profile:
            llm.start_profile()
            run_beam_search()
            llm.stop_profile()
            return None
        else:
            start_time = time.perf_counter()
            run_beam_search()
            end_time = time.perf_counter()
            return end_time - start_time

    # Warmup
    print("Warming up...")
    for _ in tqdm(range(num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile=False)

    # Profiling mode
    if do_profile:
        print("Profiling...")
        run_to_completion(profile=True)
        return {"profiled": True}

    # Benchmark
    latencies = []
    print("Running benchmark...")
    for _ in tqdm(range(num_iters), desc="Benchmark iterations"):
        latency = run_to_completion(profile=False)
        latencies.append(latency)

    latencies = np.array(latencies)
    return {"latencies": latencies}


def compute_metrics(
    latencies: np.ndarray,
    num_prompts: int,
    output_len: int,
    beam_width: int,
) -> dict[str, Any]:
    """Compute latency and throughput metrics from raw latencies."""
    percentages = [10, 25, 50, 75, 90, 95, 99]
    percentiles = np.percentile(latencies, percentages)

    avg_latency = float(np.mean(latencies))
    total_output_tokens = num_prompts * output_len * beam_width

    return {
        "avg_latency": avg_latency,
        "std_latency": float(np.std(latencies)),
        "min_latency": float(np.min(latencies)),
        "max_latency": float(np.max(latencies)),
        "latencies": latencies.tolist(),
        "percentiles": dict(zip(percentages, [float(p) for p in percentiles])),
        # Throughput metrics
        "requests_per_second": num_prompts / avg_latency,
        "output_tokens_per_second": total_output_tokens / avg_latency,
        "latency_per_request_ms": (avg_latency / num_prompts) * 1000,
        "latency_per_token_ms": (avg_latency / total_output_tokens) * 1000,
        # Configuration
        "num_prompts": num_prompts,
        "output_len": output_len,
        "beam_width": beam_width,
        "total_output_tokens": total_output_tokens,
        "elapsed_time": float(np.sum(latencies)),
    }


def print_metrics(metrics: dict[str, Any], beam_width: int) -> None:
    """Print benchmark metrics in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"Results for beam_width={beam_width}")
    print(f"{'=' * 60}")

    print("\nLatency Metrics:")
    print(f"  Avg latency:    {metrics['avg_latency']:.4f} s")
    print(f"  Std latency:    {metrics['std_latency']:.4f} s")
    print(f"  Min latency:    {metrics['min_latency']:.4f} s")
    print(f"  Max latency:    {metrics['max_latency']:.4f} s")

    print("\nPercentile Latencies:")
    for pct, val in metrics["percentiles"].items():
        print(f"  P{pct}:           {val:.4f} s")

    print("\nThroughput Metrics:")
    print(f"  Requests/s:     {metrics['requests_per_second']:.2f}")
    print(f"  Output tok/s:   {metrics['output_tokens_per_second']:.2f}")
    print(f"  ms/request:     {metrics['latency_per_request_ms']:.2f}")
    print(f"  ms/token:       {metrics['latency_per_token_ms']:.4f}")


def print_summary_table(all_results: dict[int, dict[str, Any]]) -> None:
    """Print a summary comparison table for all beam widths."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    header = (
        f"{'Beam Width':<12} {'Avg Lat (s)':<14} {'P95 Lat (s)':<14} "
        f"{'Req/s':<10} {'Tok/s':<12}"
    )
    print(f"\n{header}")
    print("-" * 80)

    for beam_width in sorted(all_results.keys()):
        m = all_results[beam_width]
        p95 = m["percentiles"].get(95, m["percentiles"].get("95", 0))
        print(
            f"{beam_width:<12} {m['avg_latency']:<14.4f} {p95:<14.4f} "
            f"{m['requests_per_second']:<10.2f} {m['output_tokens_per_second']:<12.2f}"
        )

    # Scaling analysis if we have beam_width=1 as baseline
    if 1 in all_results and len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("SCALING ANALYSIS (relative to beam_width=1)")
        print(f"{'=' * 80}")

        baseline = all_results[1]
        print(f"\n{'Beam Width':<12} {'Latency Ratio':<16} {'Throughput Ratio':<18}")
        print("-" * 60)

        for beam_width in sorted(all_results.keys()):
            if beam_width == 1:
                continue
            m = all_results[beam_width]
            lat_ratio = m["avg_latency"] / baseline["avg_latency"]
            tput_ratio = (
                m["output_tokens_per_second"] / baseline["output_tokens_per_second"]
            )
            print(f"{beam_width:<12} {lat_ratio:<16.2f}x {tput_ratio:<18.2f}x")


def main(args: argparse.Namespace) -> None:
    """Main benchmark function."""
    # Lazy import to avoid importing LLM when not needed
    from vllm import LLM

    # Set random seed (use seed from EngineArgs if available)
    seed = getattr(args, "seed", 0) or 0
    random.seed(seed)
    np.random.seed(seed)

    engine_args = EngineArgs.from_cli_args(args)

    print(f"\n{'=' * 60}")
    print("vLLM Beam Search Benchmark")
    print(f"{'=' * 60}")
    print(f"Model:          {args.model}")
    print(f"Dataset:        {args.dataset_name}")
    if args.dataset_path:
        print(f"Dataset path:   {args.dataset_path}")
    print(f"Input length:   {args.input_len}")
    print(f"Output length:  {args.output_len}")
    print(f"Num prompts:    {args.num_prompts}")
    print(f"Warmup iters:   {args.num_iters_warmup}")
    print(f"Bench iters:    {args.num_iters}")

    # Determine beam widths to test
    beam_widths = args.beam_widths if args.beam_widths else [args.beam_width]
    print(f"Beam widths:    {beam_widths}")
    print(f"{'=' * 60}\n")

    # Initialize LLM
    print("Loading model...")
    llm = LLM(**dataclasses.asdict(engine_args))

    # Get prompts - either from dataset or generate random
    if args.dataset_name != "random" and args.dataset_path:
        print(f"Loading prompts from {args.dataset_name} dataset...")
        tokenizer = get_tokenizer(
            args.tokenizer
            if hasattr(args, "tokenizer") and args.tokenizer
            else args.model,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        requests = get_requests_from_dataset(args, tokenizer)
        prompts, dataset_output_len = get_prompts_from_requests(requests)
        num_prompts = len(prompts)
        # Use dataset output_len if not explicitly set
        output_len = args.output_len if args.output_len else dataset_output_len
        print(f"Loaded {num_prompts} prompts from dataset")
    else:
        # Generate random prompts
        num_prompts = args.batch_size if args.batch_size else args.num_prompts
        output_len = args.output_len
        print(f"Generating {num_prompts} random prompts...")
        prompts = generate_random_prompts(num_prompts, args.input_len)

    # Validate model configuration
    assert llm.llm_engine.model_config.max_model_len >= (args.input_len + output_len), (
        "Please ensure that max_model_len is greater than "
        "the sum of input_len and output_len."
    )

    all_results: dict[int, dict[str, Any]] = {}

    for beam_width in beam_widths:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking beam_width={beam_width}")
        print(f"{'=' * 60}")

        beam_params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=output_len,
            ignore_eos=True,
        )

        # Run benchmark
        result = run_beam_search_benchmark(
            llm=llm,
            prompts=prompts,
            beam_params=beam_params,
            num_iters_warmup=args.num_iters_warmup,
            num_iters=args.num_iters,
            do_profile=args.profile,
        )

        if args.profile:
            print("Profiling complete.")
            return

        # Compute and store metrics
        metrics = compute_metrics(
            latencies=result["latencies"],
            num_prompts=num_prompts,
            output_len=output_len,
            beam_width=beam_width,
        )
        all_results[beam_width] = metrics

        # Print metrics for this beam width
        print_metrics(metrics, beam_width)

    # Print summary table
    if len(all_results) > 1:
        print_summary_table(all_results)

    # Save results to JSON
    if args.output_json:
        output_data = {
            "config": {
                "model": args.model,
                "dataset_name": args.dataset_name,
                "dataset_path": args.dataset_path,
                "input_len": args.input_len,
                "output_len": output_len,
                "num_prompts": num_prompts,
                "beam_widths": beam_widths,
                "num_iters_warmup": args.num_iters_warmup,
                "num_iters": args.num_iters,
            },
            "results": {str(k): v for k, v in all_results.items()},
        }

        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

        # Save in PyTorch benchmark format if enabled
        if len(all_results) == 1:
            beam_width = list(all_results.keys())[0]
            save_to_pytorch_benchmark_format(args, all_results[beam_width])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark beam search latency and throughput in vLLM."
    )
    parser = add_cli_args(parser)
    args = parser.parse_args()
    main(args)
