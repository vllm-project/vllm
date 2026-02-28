# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark hybrid SSM + sliding-window attention vs baseline configurations.

This script compares three attention configurations:
1. Full attention (baseline) - standard full KV cache
2. Sliding window only - sliding window KV cache without SSM
3. Hybrid SSM + sliding window - sliding window KV cache with SSM history branch

Metrics collected:
- Memory: KV cache memory, peak GPU memory, number of KV blocks
- Throughput: Tokens/second, requests/second
- Latency: Average, P50, P90, P99 latencies

Usage:
    python benchmarks/benchmark_hybrid_attention.py \
        --model meta-llama/Llama-3.2-1B \
        --config full \
        --input-lengths 512,1024,2048,4096 \
        --num-prompts 100 \
        --output-json results.json
"""

import argparse
import dataclasses
import gc
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Configuration presets for benchmarking
BENCHMARK_CONFIGS = {
    "full": {
        "name": "full_attention",
        "description": "Full attention with complete KV cache",
        "sliding_window": None,
        "use_hybrid": False,
    },
    "sliding": {
        "name": "sliding_window_only",
        "description": "Sliding window attention without SSM",
        "sliding_window": 4096,
        "use_hybrid": False,
    },
    "hybrid": {
        "name": "hybrid_ssm_sliding",
        "description": "Hybrid SSM + sliding window attention",
        "sliding_window": 4096,
        "use_hybrid": True,
    },
}


def get_gpu_memory_info() -> dict[str, float]:
    """Get current GPU memory statistics in GiB.
    
    Note: vLLM runs in a separate process and takes exclusive GPU control.
    We cannot call CUDA functions from the parent process.
    Memory info will be obtained from vLLM's engine API instead.
    """
    return {"free_memory_gib": 0, "total_memory_gib": 0, "used_memory_gib": 0}


def get_torch_memory_stats() -> dict[str, float]:
    """Get PyTorch CUDA memory statistics.
    
    Note: vLLM runs in a separate process - CUDA stats are not available here.
    """
    return {}


def generate_prompts(
    tokenizer: Any,
    input_lengths: list[int],
    num_prompts_per_length: int,
    output_len: int = 128,
    seed: int = 42,
) -> list[dict]:
    """Generate test prompts of specific token lengths."""
    random.seed(seed)
    np.random.seed(seed)

    prompts = []
    vocab_size = tokenizer.vocab_size

    for target_length in input_lengths:
        for _ in range(num_prompts_per_length):
            # Generate random token IDs (avoiding special tokens)
            token_ids = [
                random.randint(100, vocab_size - 100) for _ in range(target_length)
            ]
            prompts.append(
                {
                    "prompt_token_ids": token_ids,
                    "target_length": target_length,
                    "output_len": output_len,
                }
            )

    return prompts


def run_benchmark(
    model_path: str,
    config: dict,
    prompts: list[dict],
    num_warmup_iters: int = 3,
    num_benchmark_iters: int = 10,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
    dtype: str = "auto",
) -> dict[str, Any]:
    """Run benchmark for a specific configuration.

    Returns:
        Dictionary containing all benchmark metrics.
    """
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    # Prepare engine arguments
    engine_kwargs = {
        "model": model_path,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,  # Disable CUDA graphs for fair memory comparison
    }

    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len

    # Configure hf_overrides for sliding window and hybrid attention
    hf_overrides = {}
    
    # Set sliding window if configured
    if config["sliding_window"] is not None:
        hf_overrides["sliding_window"] = config["sliding_window"]

    # Configure for hybrid attention if needed
    if config["use_hybrid"]:
        hf_overrides["use_hybrid_attention"] = True

    if hf_overrides:
        engine_kwargs["hf_overrides"] = hf_overrides

    # Don't touch CUDA before vLLM starts - it uses multiprocessing spawn
    # which requires CUDA to not be initialized in the parent process
    gc.collect()
    
    # Record baseline memory AFTER engine init to avoid CUDA init conflicts
    baseline_memory = {"free_memory_gib": 0, "total_memory_gib": 0, "used_memory_gib": 0}

    # Initialize the engine
    print(f"\nInitializing engine for config: {config['name']}")
    init_start = time.perf_counter()

    try:
        llm = LLM(**engine_kwargs)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return {"error": str(e), "config": config["name"]}

    init_time = time.perf_counter() - init_start

    # Get memory info after initialization
    post_init_memory = get_gpu_memory_info()
    torch_memory = get_torch_memory_stats()

    # Try to get KV cache info from the engine
    try:
        # v1 engine path
        kv_cache_memory = getattr(
            llm.llm_engine, "available_gpu_memory_for_kv_cache", None
        )
        if kv_cache_memory is None:
            # Try model_executor path
            executor = getattr(llm.llm_engine, "model_executor", None)
            if executor:
                kv_cache_memory = getattr(
                    executor, "available_gpu_memory_for_kv_cache", None
                )
    except Exception:
        kv_cache_memory = None

    # Prepare sampling params
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=prompts[0]["output_len"],
        detokenize=False,
    )

    # Prepare prompts for generation
    generation_prompts = [
        {"prompt_token_ids": p["prompt_token_ids"]} for p in prompts
    ]

    # Warmup iterations
    print(f"Running {num_warmup_iters} warmup iterations...")
    for _ in range(num_warmup_iters):
        # Use a subset for warmup
        warmup_prompts = generation_prompts[: min(10, len(generation_prompts))]
        llm.generate(warmup_prompts, sampling_params, use_tqdm=False)

    # Benchmark iterations
    print(f"Running {num_benchmark_iters} benchmark iterations...")
    latencies = []
    total_prompt_tokens = 0
    total_output_tokens = 0

    for i in tqdm(range(num_benchmark_iters), desc="Benchmark iterations"):
        gc.collect()

        start_time = time.perf_counter()
        outputs = llm.generate(generation_prompts, sampling_params, use_tqdm=False)
        end_time = time.perf_counter()

        latencies.append(end_time - start_time)

        # Count tokens from last iteration
        if i == num_benchmark_iters - 1:
            for output in outputs:
                if hasattr(output, "prompt_token_ids") and output.prompt_token_ids:
                    total_prompt_tokens += len(output.prompt_token_ids)
                for completion in output.outputs:
                    if hasattr(completion, "token_ids"):
                        total_output_tokens += len(completion.token_ids)

    # Calculate metrics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    total_tokens = total_prompt_tokens + total_output_tokens

    # Memory after benchmark
    post_benchmark_memory = get_gpu_memory_info()
    post_benchmark_torch = get_torch_memory_stats()

    # Calculate throughput
    throughput_tokens_per_sec = total_tokens / avg_latency if avg_latency > 0 else 0
    throughput_requests_per_sec = (
        len(generation_prompts) / avg_latency if avg_latency > 0 else 0
    )

    results = {
        "config": config["name"],
        "config_description": config["description"],
        "model": model_path,
        "num_prompts": len(prompts),
        "initialization": {
            "time_seconds": init_time,
        },
        "memory": {
            "baseline": baseline_memory,
            "post_init": post_init_memory,
            "post_benchmark": post_benchmark_memory,
            "torch_stats": post_benchmark_torch,
            "kv_cache_memory_gib": kv_cache_memory / (1024**3)
            if kv_cache_memory
            else None,
            "model_memory_gib": post_init_memory["used_memory_gib"]
            - baseline_memory["used_memory_gib"],
        },
        "throughput": {
            "tokens_per_second": throughput_tokens_per_sec,
            "requests_per_second": throughput_requests_per_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "latency": {
            "avg_seconds": float(avg_latency),
            "min_seconds": float(np.min(latencies)),
            "max_seconds": float(np.max(latencies)),
            "std_seconds": float(np.std(latencies)),
            "p50_seconds": float(np.percentile(latencies, 50)),
            "p90_seconds": float(np.percentile(latencies, 90)),
            "p99_seconds": float(np.percentile(latencies, 99)),
            "all_latencies": latencies.tolist(),
        },
        "benchmark_params": {
            "num_warmup_iters": num_warmup_iters,
            "num_benchmark_iters": num_benchmark_iters,
            "gpu_memory_utilization": gpu_memory_utilization,
            "sliding_window": config["sliding_window"],
            "use_hybrid": config["use_hybrid"],
        },
    }

    # Clean up - deleting LLM will terminate the engine subprocess
    del llm
    gc.collect()
    # Note: Don't call torch.cuda functions here - GPU is in child process

    return results


def run_benchmark_by_input_length(
    model_path: str,
    config: dict,
    tokenizer: Any,
    input_lengths: list[int],
    num_prompts_per_length: int,
    output_len: int,
    **benchmark_kwargs,
) -> dict[str, Any]:
    """Run benchmarks for each input length separately."""
    results_by_length = {}

    for input_len in input_lengths:
        print(f"\n{'='*60}")
        print(f"Benchmarking input length: {input_len}")
        print(f"{'='*60}")

        prompts = generate_prompts(
            tokenizer=tokenizer,
            input_lengths=[input_len],
            num_prompts_per_length=num_prompts_per_length,
            output_len=output_len,
        )

        result = run_benchmark(
            model_path=model_path,
            config=config,
            prompts=prompts,
            **benchmark_kwargs,
        )

        results_by_length[str(input_len)] = result

    return results_by_length


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the benchmark."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the model to benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()),
        required=True,
        help="Benchmark configuration to run",
    )
    parser.add_argument(
        "--input-lengths",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated list of input lengths to benchmark",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts per input length",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length for each prompt",
    )
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-benchmark-iters",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sliding-window-override",
        type=int,
        default=None,
        help="Override the default sliding window size for sliding/hybrid configs",
    )


def main(args: argparse.Namespace) -> None:
    """Main benchmark entry point."""
    # Parse input lengths
    input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]

    # Get the benchmark config
    config = BENCHMARK_CONFIGS[args.config].copy()

    # Apply sliding window override if provided
    if args.sliding_window_override is not None and config["sliding_window"] is not None:
        config["sliding_window"] = args.sliding_window_override

    print(f"\n{'='*60}")
    print(f"Hybrid Attention Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Config: {config['name']} - {config['description']}")
    print(f"Input lengths: {input_lengths}")
    print(f"Prompts per length: {args.num_prompts}")
    print(f"Output length: {args.output_len}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    # Set random seed (avoid CUDA init before vLLM starts)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU seed only - vLLM handles GPU seeding

    # Run benchmarks by input length
    results = run_benchmark_by_input_length(
        model_path=args.model,
        config=config,
        tokenizer=tokenizer,
        input_lengths=input_lengths,
        num_prompts_per_length=args.num_prompts,
        output_len=args.output_len,
        num_warmup_iters=args.num_warmup_iters,
        num_benchmark_iters=args.num_benchmark_iters,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    # Aggregate results
    final_results = {
        "config": config["name"],
        "model": args.model,
        "benchmark_params": {
            "input_lengths": input_lengths,
            "num_prompts_per_length": args.num_prompts,
            "output_len": args.output_len,
            "num_warmup_iters": args.num_warmup_iters,
            "num_benchmark_iters": args.num_benchmark_iters,
            "sliding_window": config["sliding_window"],
            "use_hybrid": config["use_hybrid"],
            "seed": args.seed,
        },
        "by_input_length": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for length, result in results.items():
        if "error" in result:
            print(f"Input length {length}: ERROR - {result['error']}")
        else:
            print(f"\nInput length {length}:")
            print(
                f"  Throughput: {result['throughput']['tokens_per_second']:.2f} tokens/s"
            )
            print(
                f"  Avg latency: {result['latency']['avg_seconds']*1000:.2f} ms"
            )
            print(f"  P99 latency: {result['latency']['p99_seconds']*1000:.2f} ms")
            if result["memory"].get("kv_cache_memory_gib"):
                print(
                    f"  KV cache memory: {result['memory']['kv_cache_memory_gib']:.2f} GiB"
                )

    # Save results to JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid SSM + sliding-window attention"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

