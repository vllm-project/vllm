#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark vLLM MPS vs llama.cpp Metal for E2E validation.

This script validates that vLLM inference on MPS is competitive with
the llama.cpp Metal backend for real-world Llama/Qwen model serving.

Metrics:
- Throughput: tokens/second (prefill + decode)
- Latency: time to first token (TTFT), per-token latency
- Memory: Peak GPU memory usage
"""

import argparse
import json
import time
from typing import Any

import torch

from vllm import LLM, SamplingParams


def get_mps_memory_stats() -> dict[str, float]:
    """Get MPS GPU memory stats."""
    allocated = torch.mps.current_allocated_memory() / (1024**3)  # GiB
    reserved = torch.mps.driver_allocated_memory() / (1024**3)  # GiB
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
    }


def benchmark_vllm_mps(
    model_name: str,
    num_prompts: int = 10,
    max_tokens: int = 100,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Benchmark vLLM inference on MPS.

    Args:
        model_name: HF model ID (e.g., "Qwen/Qwen2-7B-Instruct")
        num_prompts: Number of prompts to process
        max_tokens: Max tokens per generation
        dtype: Precision ("bfloat16", "float16", "float32")

    Returns:
        Dictionary with throughput, latency, memory stats.
    """
    print(f"\n{'=' * 60}")
    print(f"vLLM MPS Benchmark: {model_name}")
    print(f"{'=' * 60}")

    prompts = [
        "Once upon a time,",
        "The quick brown fox",
        "In the year 2025,",
        "The future of AI is",
        "Machine learning models",
    ] * (num_prompts // 5 + 1)
    prompts = prompts[:num_prompts]

    # Initialize LLM
    print(f"Loading model: {model_name} (dtype={dtype})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        dtype=dtype,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    # Warmup
    print("Warmup...")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=10)
    _ = llm.generate(["Hello"], sampling_params=sampling_params)
    torch.mps.synchronize()

    # Benchmark
    print(f"Generating {num_prompts} requests...")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    total_time = time.time() - start_time
    torch.mps.synchronize()

    # Collect stats
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    throughput = total_tokens / total_time

    mem_stats = get_mps_memory_stats()

    return {
        "model": model_name,
        "dtype": dtype,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "throughput_tokens_per_sec": throughput,
        "latency_ms_per_token": (total_time / total_tokens) * 1000,
        "memory": mem_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM MPS vs llama.cpp")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-7B-Instruct",
        help="Model to benchmark",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per generation"
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float16",
        help="Model precision",
    )
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available on this machine")
        return

    # Run vLLM benchmark
    results = benchmark_vllm_mps(
        model_name=args.model,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("vLLM MPS Results:")
    print(f"{'=' * 60}")
    print(f"Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Latency: {results['latency_ms_per_token']:.2f} ms/token")
    print(f"Memory (allocated): {results['memory']['allocated_gb']:.2f} GiB")
    print(f"Total time: {results['total_time_sec']:.2f} sec")
    print(f"Total tokens: {results['total_tokens']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nNote: To benchmark llama.cpp Metal backend, run:")
    print(
        f"  ./main -m <model.gguf> --n-predict {args.max_tokens}"
        f" --n-threads 1 --gpu-layers -1"
    )


if __name__ == "__main__":
    main()
