#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to measure inductor cache performance across multiple models.

This script benchmarks the vLLM inductor cache feature by running models twice:
1. Cache MISS: First run compiles from scratch and saves artifacts to cache
2. Cache HIT: Second run loads precompiled artifacts from cache

The script measures load time and inference performance for both runs and computes
the speedup achieved by using cached artifacts.

Usage:
    # Run all models with default settings
    python benchmark_multimodel_precompile.py

    # Run specific models by index
    python benchmark_multimodel_precompile.py --models 0 3 4

    # Run specific models by name (partial match)
    python benchmark_multimodel_precompile.py --models Qwen Llama

    # List available models
    python benchmark_multimodel_precompile.py --list-models

    # Customize cache directory and output
    python benchmark_multimodel_precompile.py --cache-dir /path/to/cache --output results.json

    # Adjust generation parameters
    python benchmark_multimodel_precompile.py --max-tokens 256 --batch-size 4

Requirements:
    - PyTorch 2.10.0+ (for standalone_compile with serialization support)
    - vLLM with VLLM_USE_BACKEND_WITH_INDUCTOR_CACHE feature enabled
    - Sufficient GPU memory (8x H100 recommended for largest models)

Environment Variables:
    VLLM_USE_BACKEND_WITH_INDUCTOR_CACHE=1  (automatically set by script)
    VLLM_USE_AOT_COMPILE=1                  (automatically set by script)
    VLLM_USE_STANDALONE_COMPILE=1           (automatically set by script)
    VLLM_CACHE_ROOT                         (set via --cache-dir argument)
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
    ModelConfig("deepseek-ai/DeepSeek-R1-0528", 8),  # Very large model, needs 8 GPUs
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
    enable_compile: bool,
    cache_dir: str,
    use_cached: bool = False,
) -> dict[str, float]:
    """Run vLLM inference and measure timing.

    Args:
        model: Model name or path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        batch_size: Batch size
        enable_compile: Whether to enable torch compile
        cache_dir: Cache directory for compiled artifacts
        use_cached: Whether to use cached artifacts (only applies if enable_compile=True)

    Returns:
        Dictionary with timing information
    """
    # Set environment variables FIRST, before any imports
    # IMPORTANT: Keep all environment variables IDENTICAL between cache miss and cache hit runs
    # Changes to env vars can affect the config hash and cause cache misses

    # Clear any existing VLLM_FORCE_AOT_LOAD that might prevent compilation
    os.environ.pop("VLLM_FORCE_AOT_LOAD", None)

    if enable_compile:
        # Always set all compilation flags the same way for both cache miss and cache hit
        # The cache hit/miss is determined by whether artifacts exist, not by flags
        # Keeping flags identical ensures the config hash remains the same
        os.environ["VLLM_CACHE_ROOT"] = cache_dir
        os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_CACHE"] = "1"
        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
    else:
        # Eager mode: clear all compile-related flags
        os.environ.pop("VLLM_USE_BACKEND_WITH_INDUCTOR_CACHE", None)
        os.environ.pop("VLLM_USE_AOT_COMPILE", None)
        os.environ.pop("VLLM_USE_STANDALONE_COMPILE", None)

    # Now import after environment variables are set
    import torch

    from vllm import LLM, SamplingParams

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
        # Enable VLLM_COMPILE mode (level 3) which enables piecewise compilation
        # with splitting at attention ops for multiple submodules
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
            compilation_config={
                "level": 3,  # CompilationMode.VLLM_COMPILE enables splitting
                "use_inductor": True,
            },
            gpu_memory_utilization=0.95,  # Increase from default 0.9 to use more GPU memory
            max_model_len=2048,  # Limit context length to reduce memory usage
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
    print(
        f"{prefix}Total Time:       {format_time(results['load_time'] + results['inference_time'])}"
    )


def print_summary(all_results: dict[str, dict[str, any]]):
    """Print summary table of all results."""
    print("\n" + "=" * 140)
    print("BENCHMARK SUMMARY - Cache Hit vs Cache Miss Comparison")
    print("=" * 140)

    # Table header
    print(
        f"\n{'Model':<45} {'Miss Load':<12} {'Hit Load':<12} {'Speedup':<10} {'Miss Total':<12} {'Hit Total':<12} {'Speedup':<10}"
    )
    print("-" * 140)

    # Sort by model name for consistent output
    for model_name in sorted(all_results.keys()):
        results = all_results[model_name]

        # Check if we have both cache miss and cache hit results
        if "cache_miss" in results and "cache_hit" in results:
            miss = results["cache_miss"]
            hit = results["cache_hit"]

            miss_load = format_time(miss["load_time"])
            hit_load = format_time(hit["load_time"])
            load_speedup = f"{miss['load_time'] / hit['load_time']:.2f}x"

            miss_total = format_time(miss["load_time"] + miss["inference_time"])
            hit_total = format_time(hit["load_time"] + hit["inference_time"])
            total_speedup = f"{(miss['load_time'] + miss['inference_time']) / (hit['load_time'] + hit['inference_time']):.2f}x"

            # Truncate model name if too long
            model_display = (
                model_name if len(model_name) <= 45 else model_name[:42] + "..."
            )

            print(
                f"{model_display:<45} {miss_load:<12} {hit_load:<12} {load_speedup:<10} {miss_total:<12} {hit_total:<12} {total_speedup:<10}"
            )
        else:
            # Only have one result type
            result_type = "cache_miss" if "cache_miss" in results else "cache_hit"
            result = results[result_type]
            load_time = format_time(result["load_time"])
            total_time = format_time(result["load_time"] + result["inference_time"])
            model_display = (
                model_name if len(model_name) <= 45 else model_name[:42] + "..."
            )
            print(
                f"{model_display:<45} {load_time:<12} {'N/A':<12} {'N/A':<10} {total_time:<12} {'N/A':<12} {'N/A':<10}"
            )

    print("=" * 140)


def clean_all_caches(cache_dir: str):
    """Clean all compilation caches before running benchmarks.

    Removes:
    1. torch_aot_compile cache: {cache_dir}/torch_aot_compile/
    2. vllm compile cache: /tmp/vllm_compile_cache_*
    """
    import glob

    print("\n" + "=" * 80)
    print("CLEANING ALL CACHES")
    print("=" * 80)

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
        help="Specific models to benchmark (by name or index). If not specified, runs all models.",
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
                    print(
                        f"ERROR: Model index {idx} out of range (0-{len(DEFAULT_MODELS) - 1})"
                    )
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
            # Determine cache directory for this model
            model_cache_path = os.path.join(
                cache_dir, "torch_compile", model_config.name.replace("/", "_")
            )

            # Clean ALL caches before cache miss run
            clean_all_caches(cache_dir)

            # Run 1: Cache MISS (compile from scratch)
            print("\n--- RUN 1: CACHE MISS (compiling from scratch) ---")

            cache_miss_results = run_vllm_inference(
                model=model_config.name,
                tensor_parallel_size=model_config.tensor_parallel_size,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                enable_compile=True,
                cache_dir=cache_dir,
                use_cached=False,
            )
            model_results["cache_miss"] = cache_miss_results
            print_model_results(model_config.name, cache_miss_results, "Cache Miss")

            # Clean up GPU memory between runs
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("\nCleaned up GPU memory before cache hit run")

            # Run 2: Cache HIT (load from cache)
            print("\n--- RUN 2: CACHE HIT (loading from cache) ---")
            cache_hit_results = run_vllm_inference(
                model=model_config.name,
                tensor_parallel_size=model_config.tensor_parallel_size,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                enable_compile=True,
                cache_dir=cache_dir,
                use_cached=True,
            )
            model_results["cache_hit"] = cache_hit_results
            print_model_results(model_config.name, cache_hit_results, "Cache Hit")

            # Calculate and display speedup
            load_speedup = (
                cache_miss_results["load_time"] / cache_hit_results["load_time"]
            )
            total_speedup = (
                cache_miss_results["load_time"] + cache_miss_results["inference_time"]
            ) / (cache_hit_results["load_time"] + cache_hit_results["inference_time"])
            print("\n  Speedup:")
            print(f"    Load Time:  {load_speedup:.2f}x faster with cache")
            print(f"    Total Time: {total_speedup:.2f}x faster with cache")

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
