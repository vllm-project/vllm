#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to measure inductor cache performance across multiple models.

This script benchmarks the vLLM inductor cache feature by running models twice:
1. Cold Start: First run compiles from scratch and saves artifacts to cache
2. Warm Start: Second run loads precompiled artifacts from cache

The script measures load time and inference performance for both runs and computes
the speedup achieved by using cached artifacts.

Usage:
    # Run all models with default settings
    python benchmark_inductor_compiled_artifacts.py

    # Run specific models by index
    python benchmark_inductor_compiled_artifacts.py --models 0 3 4

    # Run specific models by name (partial match)
    python benchmark_inductor_compiled_artifacts.py --models Qwen Llama

    # List available models
    python benchmark_inductor_compiled_artifacts.py --list-models

    # Customize cache directory and output
    python benchmark_inductor_compiled_artifacts.py \
        --cache-dir /path/to/cache --output results.json

    # Adjust generation parameters
    python benchmark_inductor_compiled_artifacts.py --max-tokens 256 --batch-size 4

Requirements:
    - PyTorch 2.10.0+ (for standalone_compile with serialization support)
    - vLLM with VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS feature enabled
    - Sufficient GPU memory (8x H100 recommended for largest models)

Environment Variables:
    VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS=1  (automatically set by script)
    VLLM_USE_AOT_COMPILE=1                  (automatically set by script)
    VLLM_USE_STANDALONE_COMPILE=1           (automatically set by script)
    VLLM_CACHE_ROOT                         (set via --cache-dir argument)

NOTE: FLASHINFER_WORKSPACE_BUFFER_SIZE needs to be set to 512MB for mistral...
      https://github.com/vllm-project/vllm/issues/25342 should fix. 
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""

    name: str
    tensor_parallel_size: int

    def __str__(self):
        return f"{self.name} (TP={self.tensor_parallel_size})"


# Model configurations with appropriate TP sizes
DEFAULT_MODELS = [
    ModelConfig("Qwen/Qwen3-32B", 1),
    # not working on b200 for some unrelated reason
    # ModelConfig("deepseek-ai/DeepSeek-R1-0528", 8),
    ModelConfig("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 1),
    ModelConfig("meta-llama/Llama-3.3-70B-Instruct", 2),
    ModelConfig("nvidia/Llama-3.3-70B-Instruct-FP8", 2),
    ModelConfig("mistralai/Mistral-Large-Instruct-2411", 2),
]


def run_vllm_inference(
    model: str,
    tensor_parallel_size: int,
    prompt: str,
    max_tokens: int,
    batch_size: int,
    compile_sizes: list[int | str] | None,
    compile_ranges_split_points: list[int] | None,
    enable_compile: bool,
    cache_dir: str,
    use_baseline: bool = False,
) -> dict[str, float]:
    """Run vLLM inference and measure timing.

    Args:
        model: Model name or path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        batch_size: Batch size
        compile_sizes: Which batch sizes to compile for
        compile_ranges_split_points: Split points for compile ranges
        enable_compile: Whether to enable torch compile
        cache_dir: Cache directory for compiled artifacts
        use_baseline: If True, use standalone compile WITHOUT inductor cache
            (baseline comparison)

    Returns:
        Dictionary with timing information
    """
    # Set environment variables FIRST, before any imports
    # IMPORTANT: Keep all environment variables IDENTICAL between cold and
    # warm start runs. Changes to env vars can affect the config hash and
    # prevent warm starts

    # Clear any existing VLLM_FORCE_AOT_LOAD that might prevent compilation
    os.environ.pop("VLLM_FORCE_AOT_LOAD", None)

    if enable_compile:
        os.environ["VLLM_CACHE_ROOT"] = cache_dir
        os.environ["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = str(512 * 1024 * 1024)
        if use_baseline:
            # Baseline: Use standalone compile WITHOUT inductor cache
            # This is the current/old approach for comparison
            os.environ.pop("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", None)
            os.environ["VLLM_USE_AOT_COMPILE"] = "1"
            os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
            print(
                "Environment: Using baseline "
                "(standalone compile WITHOUT inductor cache)"
            )
        else:
            # New approach: Use inductor cache
            # Always set all compilation flags the same way for both cold
            # and warm starts. Cold/warm start is determined by whether
            # artifacts exist, not by flags. Keeping flags identical ensures
            # the config hash remains the same
            os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"] = "1"
            os.environ["VLLM_USE_AOT_COMPILE"] = "1"
            os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
            print("Environment: Using inductor cache backend")
    else:
        # Eager mode: clear all compile-related flags
        os.environ.pop("VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS", None)
        os.environ.pop("VLLM_USE_AOT_COMPILE", None)
        os.environ.pop("VLLM_USE_STANDALONE_COMPILE", None)

    # Now import after environment variables are set
    import torch

    from vllm.entrypoints.llm import LLM
    from vllm.sampling_params import SamplingParams

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Prepare prompts
    prompts = [prompt] * batch_size

    # Measure model loading time
    load_start = time.perf_counter()

    if enable_compile:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
            compilation_config={
                "level": 3,  # CompilationMode.VLLM_COMPILE enables splitting
                "backend": "inductor",
                "compile_sizes": compile_sizes,
                "compile_ranges_split_points": compile_ranges_split_points,
            },
            # Increase from default 0.9 to use more GPU memory
            gpu_memory_utilization=0.95,
            # Limit context length to reduce memory usage
            max_model_len=2048,
        )
    else:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
        )
    load_time = time.perf_counter() - load_start

    # Warmup run (not timed)
    print("Running warmup...")
    _ = llm.generate(prompts[:1], sampling_params)

    # Clear GPU memory before timed run
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Timed inference run
    print("Running timed inference...")
    inference_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start

    # Calculate tokens generated
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / inference_time

    return {
        "load_time": load_time,
        "inference_time": inference_time,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def print_model_results(model_name: str, results: dict[str, float], run_type: str = ""):
    """Print results for a single model."""
    prefix = f"  [{run_type}] " if run_type else "  "
    print(f"\n{prefix}Load Time:        {format_time(results['load_time'])}")
    print(f"{prefix}Inference Time:   {format_time(results['inference_time'])}")
    print(f"{prefix}Total Tokens:     {results['total_tokens']:.0f}")
    print(f"{prefix}Tokens/Second:    {results['tokens_per_second']:.2f}")
    total_time = format_time(results["load_time"] + results["inference_time"])
    print(f"{prefix}Total Time:       {total_time}")


def print_summary(all_results: dict[str, dict[str, any]]):
    """Print summary table of all results."""
    print("\n" + "=" * 180)
    print("INDUCTOR CACHE BENCHMARK - Comparing Load Time Performance")
    print("=" * 180)
    print("\nComparison: Old Approach (Baseline AOT) vs New Approach (Inductor Cache)")
    print("=" * 180)

    # Table header - make it crystal clear what we're comparing
    header = (
        f"\n{'Model':<45} {'OLD: Baseline':<15} {'NEW: Inductor':<15} "
        f"{'BENEFIT':<12} {'Time Saved':<12} {'Miss Penalty':<15}"
    )
    print(header)
    subheader = (
        f"{'(Approach being compared)':<45} {'AOT Cache':<15} "
        f"{'Cache Hit':<15} {'(Speedup)':<12} {'(Seconds)':<12} "
        f"{'vs Baseline':<15}"
    )
    print(subheader)
    print("-" * 180)

    # Sort by model name for consistent output
    for model_name in sorted(all_results.keys()):
        results = all_results[model_name]

        # Check if we have all results
        if "cold_start" in results and "warm_start" in results:
            miss = results["cold_start"]
            hit = results["warm_start"]

            hit_load = format_time(hit["load_time"])

            # Check if baseline_hit exists (for apples-to-apples comparison)
            if "baseline_hit" in results:
                base_hit = results["baseline_hit"]
                base_load = format_time(base_hit["load_time"])

                # PRIMARY METRIC: Inductor cache hit vs baseline hit
                speedup = base_hit["load_time"] / hit["load_time"]
                speedup_str = f"{speedup:.2f}x"

                # Time saved in seconds
                time_saved = base_hit["load_time"] - hit["load_time"]
                if time_saved >= 0:
                    time_saved_str = f"-{format_time(time_saved)}"
                else:
                    time_saved_str = f"+{format_time(abs(time_saved))}"

                # Miss penalty: compare cache miss overhead vs baseline miss
                if "baseline_miss" in results:
                    base_miss = results["baseline_miss"]
                    miss_penalty_pct = (
                        (miss["load_time"] - base_miss["load_time"])
                        / base_miss["load_time"]
                    ) * 100
                    miss_penalty_str = f"{miss_penalty_pct:+.1f}%"
                else:
                    miss_penalty_str = "N/A"
            else:
                base_load = "N/A"
                speedup_str = "N/A"
                time_saved_str = "N/A"
                miss_penalty_str = "N/A"

            # Truncate model name if too long
            model_display = (
                model_name if len(model_name) <= 45 else model_name[:42] + "..."
            )

            row = (
                f"{model_display:<45} {base_load:<15} {hit_load:<15} "
                f"{speedup_str:<12} {time_saved_str:<12} {miss_penalty_str:<15}"
            )
            print(row)
        else:
            # Only have partial results
            model_display = (
                model_name if len(model_name) <= 45 else model_name[:42] + "..."
            )
            n_a_row = (
                f"{model_display:<45} {'N/A':<15} {'N/A':<15} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<15}"
            )
            print(n_a_row)

    print("=" * 180)
    print("\nKey Findings:")
    print(
        "  1. BENEFIT (Speedup): NEW approach (inductor cache) vs "
        "OLD approach (baseline AOT)"
    )
    print("     - Shows how much faster the new inductor cache is at loading")
    print("     - Example: 1.34x means inductor cache is 34% faster")
    print("\n  2. Time Saved: Absolute time reduction from using inductor cache")
    print("     - Direct time savings per model load")
    print(
        "\n  3. Miss Penalty: Overhead of first compile with inductor "
        "cache serialization"
    )
    print(
        "     - Negative % means even first compile is faster than "
        "baseline (excellent!)"
    )
    print(
        "     - Positive % means first compile is slower (overhead from serialization)"
    )
    print("=" * 180)


def clean_all_caches(cache_dir: str):
    """Clean all compilation caches before running benchmarks.

    Removes:
    1. torch_compile_cache: {cache_dir}/torch_compile_cache/ (Dynamo cache)
    2. torch_aot_compile cache: {cache_dir}/torch_aot_compile/ (AOT artifacts)
    3. vllm compile cache: /tmp/vllm_compile_cache_* (temp cache)
    """
    import glob

    print("\n" + "=" * 80)
    print("CLEANING ALL CACHES")
    print("=" * 80)

    # Clean torch_compile_cache (Dynamo cache - most important!)
    torch_compile_path = os.path.join(cache_dir, "torch_compile_cache")
    if os.path.exists(torch_compile_path):
        print(f"Removing: {torch_compile_path}")
        shutil.rmtree(torch_compile_path)
    else:
        print(f"Not found: {torch_compile_path}")

    # Clean torch_aot_compile cache
    torch_aot_path = os.path.join(cache_dir, "torch_aot_compile")
    if os.path.exists(torch_aot_path):
        print(f"Removing: {torch_aot_path}")
        shutil.rmtree(torch_aot_path)
    else:
        print(f"Not found: {torch_aot_path}")

    # Clean vllm_compile_cache_* in /tmp
    vllm_cache_pattern = "/tmp/vllm_compile_cache_*"
    matching_dirs = glob.glob(vllm_cache_pattern)
    if matching_dirs:
        for cache_path in matching_dirs:
            print(f"Removing: {cache_path}")
            shutil.rmtree(cache_path)
    else:
        print(f"No matching directories found for pattern: {vllm_cache_pattern}")

    print("=" * 80)
    print("CACHE CLEANING COMPLETE")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inductor precompile across multiple models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help=(
            "Specific models to benchmark (by name or index). "
            "If not specified, runs all models."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a story about a robot learning to paint.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=f"/data/users/{os.getenv('USER', 'unknown')}/vllm_cache",
        help="Cache directory for compiled artifacts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help=(
            "Include baseline run with standalone compile "
            "(without inductor cache) for comparison"
        ),
    )
    parser.add_argument(
        "--compile-sizes",
        type=int,
        nargs="+",
        default=None,
        help=("Batch sizes to compile ahead of time."),
    )
    parser.add_argument(
        "--compile-ranges-split-points",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Split points for compile ranges. Creates ranges like "
            "[1, split1), [split1, split2), ..., [splitN, max]. "
            "Example: --compile-ranges-split-points 256 512 1024"
        ),
    )
    args = parser.parse_args()
    # List models if requested
    if args.list_models:
        print("Available models:")
        for i, model_config in enumerate(DEFAULT_MODELS):
            print(f"  {i}: {model_config}")
        return

    # Determine which models to run
    models_to_run = []
    if args.models:
        for model_spec in args.models:
            # Check if it's an index
            try:
                idx = int(model_spec)
                if 0 <= idx < len(DEFAULT_MODELS):
                    models_to_run.append(DEFAULT_MODELS[idx])
                else:
                    max_idx = len(DEFAULT_MODELS) - 1
                    print(f"ERROR: Model index {idx} out of range (0-{max_idx})")
                    return
            except ValueError:
                # It's a model name - find matching config
                found = False
                for config in DEFAULT_MODELS:
                    if model_spec in config.name:
                        models_to_run.append(config)
                        found = True
                        break
                if not found:
                    # Create custom config with TP=1
                    models_to_run.append(ModelConfig(model_spec, 1))
    else:
        models_to_run = DEFAULT_MODELS

    # Setup cache directory
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    all_results = {}

    # Run benchmark for each model
    for model_config in models_to_run:
        print("\n" + "=" * 100)
        print(f"Benchmarking: {model_config}")
        print("=" * 100)

        model_results = {}

        try:
            # Clean ALL caches before cold start run
            clean_all_caches(cache_dir)

            # Run 1: Cold Start (compile from scratch)
            print("\n--- RUN 1: COLD START (compiling from scratch) ---")

            cold_start_results = run_vllm_inference(
                model=model_config.name,
                tensor_parallel_size=model_config.tensor_parallel_size,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                compile_sizes=args.compile_sizes,
                compile_ranges_split_points=args.compile_ranges_split_points,
                enable_compile=True,
                cache_dir=cache_dir,
            )
            model_results["cold_start"] = cold_start_results
            print_model_results(model_config.name, cold_start_results, "Cold Start")

            # Clean up GPU memory between runs
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("\nCleaned up GPU memory before cache hit run")

            # Run 2: Warm Start (load from cache)
            print("\n--- RUN 2: WARM START (loading from cache) ---")
            warm_start_results = run_vllm_inference(
                model=model_config.name,
                tensor_parallel_size=model_config.tensor_parallel_size,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                compile_sizes=args.compile_sizes,
                compile_ranges_split_points=args.compile_ranges_split_points,
                enable_compile=True,
                cache_dir=cache_dir,
            )
            model_results["warm_start"] = warm_start_results
            print_model_results(model_config.name, warm_start_results, "Warm Start")

            # Runs 3 & 4 (optional): Baseline with standalone AOT compile
            # (no inductor cache)
            if args.include_baseline:
                # Clean ALL caches to ensure baseline starts fresh
                clean_all_caches(cache_dir)

                # Clean up GPU memory
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("Cleaned up GPU memory before baseline runs")

                # Baseline Run 1: Cold Start
                print("\n--- RUN 3: BASELINE COLD START (first AOT compile) ---")
                note = (
                    "Note: This uses AOT compile but WITHOUT the inductor "
                    "cache backend feature"
                )
                print(note)
                baseline_miss_results = run_vllm_inference(
                    model=model_config.name,
                    tensor_parallel_size=model_config.tensor_parallel_size,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                    compile_sizes=args.compile_sizes,
                    compile_ranges_split_points=args.compile_ranges_split_points,
                    enable_compile=True,
                    cache_dir=cache_dir,
                    use_baseline=True,
                )
                model_results["baseline_miss"] = baseline_miss_results
                print_model_results(
                    model_config.name, baseline_miss_results, "Baseline Miss"
                )

                # Clean up GPU memory between baseline runs
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("\nCleaned up GPU memory before baseline warm start run")

                # Baseline Run 2: Warm Start
                print("\n--- RUN 4: BASELINE WARM START (loading from AOT cache) ---")
                baseline_hit_results = run_vllm_inference(
                    model=model_config.name,
                    tensor_parallel_size=model_config.tensor_parallel_size,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                    compile_sizes=args.compile_sizes,
                    compile_ranges_split_points=args.compile_ranges_split_points,
                    enable_compile=True,
                    cache_dir=cache_dir,
                    use_baseline=True,
                )
                model_results["baseline_hit"] = baseline_hit_results
                print_model_results(
                    model_config.name, baseline_hit_results, "Baseline Hit"
                )

            # Calculate and display speedup
            load_speedup = (
                cold_start_results["load_time"] / warm_start_results["load_time"]
            )
            total_speedup = (
                cold_start_results["load_time"] + cold_start_results["inference_time"]
            ) / (warm_start_results["load_time"] + warm_start_results["inference_time"])
            print("\n  Speedup (Warm Start vs Cold Start):")
            print(f"    Load Time:  {load_speedup:.2f}x faster with warm start")
            print(f"    Total Time: {total_speedup:.2f}x faster with warm start")

            if args.include_baseline and "baseline_hit" in model_results:
                baseline_hit_to_hit_speedup = (
                    baseline_hit_results["load_time"] / warm_start_results["load_time"]
                )
                print("\n  Speedup (Inductor Warm Start vs Baseline Hit):")
                speedup_msg = (
                    f"    Load Time:  {baseline_hit_to_hit_speedup:.2f}x - "
                    "inductor cache vs baseline AOT cache"
                )
                print(speedup_msg)

            all_results[model_config.name] = model_results

        except Exception as e:
            print(f"ERROR: Failed to benchmark {model_config.name}: {e}")
            import traceback

            traceback.print_exc()
            continue
        finally:
            # Always clean up GPU memory after each model
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("Cleaned up GPU memory")

    # Print summary
    if all_results:
        print_summary(all_results)

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        print("\nNo successful benchmarks to report.")


if __name__ == "__main__":
    main()
