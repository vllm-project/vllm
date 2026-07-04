#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark script for int4 embedding quantization performance."""

import argparse
import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def benchmark_memory(model_name: str, quantization: str | None = None):
    """Benchmark memory usage with and without int4 quantization."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking memory: {model_name}")
    print(f"Quantization: {quantization or 'None (baseline)'}")
    print(f"{'=' * 60}\n")

    torch.accelerator.reset_peak_memory_stats()
    torch.accelerator.empty_cache()

    start_time = time.time()
    llm = LLM(
        model=model_name,
        dtype="float16",
        quantization=quantization,
        gpu_memory_utilization=0.9,
    )
    load_time = time.time() - start_time

    # Get memory usage
    peak_memory_gb = torch.accelerator.max_memory_allocated() / 1024**3

    print(f"Model load time: {load_time:.2f}s")
    print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

    # Get embedding layer size if available
    try:
        model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        embed_layer = model.model.decoder.embed_tokens
        embed_weight = embed_layer.weight

        if hasattr(embed_layer, "weight_scale"):
            # Int4 quantized - weight is packed uint8
            # Each byte stores two int4 values, so actual storage is half
            embed_size_mb = embed_weight.numel() * embed_weight.element_size() / 1024**2
            scale_size_mb = (
                embed_layer.weight_scale.numel()
                * embed_layer.weight_scale.element_size()
                / 1024**2
            )
            total_embed_mb = embed_size_mb + scale_size_mb
            print(f"\nEmbedding layer size: {total_embed_mb:.2f} MB")
            print(f"  - Weight (int4 packed): {embed_size_mb:.2f} MB")
            print(f"  - Scale: {scale_size_mb:.2f} MB")

            # Calculate actual hidden size (before packing)
            vocab_size = embed_weight.shape[0]
            hidden_size = embed_weight.shape[1] * 2  # Packed: 2 values per byte
            print(f"Vocabulary size: {vocab_size:,}")
            print(f"Hidden size: {hidden_size}")
        else:
            # Baseline
            embed_size_mb = embed_weight.numel() * embed_weight.element_size() / 1024**2
            print(f"\nEmbedding layer size: {embed_size_mb:.2f} MB")

            vocab_size = embed_weight.shape[0]
            hidden_size = embed_weight.shape[1]
            print(f"Vocabulary size: {vocab_size:,}")
            print(f"Hidden size: {hidden_size}")
    except Exception as e:
        print(f"Could not get embedding info: {e}")

    del llm
    torch.accelerator.empty_cache()

    return {
        "load_time": load_time,
        "peak_memory_gb": peak_memory_gb,
    }


def benchmark_throughput(
    model_name: str,
    quantization: str | None = None,
    num_prompts: int = 100,
    input_len: int = 512,
    output_len: int = 128,
):
    """Benchmark throughput with and without int4 quantization."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking throughput: {model_name}")
    print(f"Quantization: {quantization or 'None (baseline)'}")
    print(f"Prompts: {num_prompts}, Input: {input_len}, Output: {output_len}")
    print(f"{'=' * 60}\n")

    # Generate test prompts
    prompts = ["This is a test prompt " * (input_len // 5) for _ in range(num_prompts)]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=output_len,
    )

    llm = LLM(
        model=model_name,
        dtype="float16",
        quantization=quantization,
        gpu_memory_utilization=0.9,
    )

    # Warmup
    print("Warming up...")
    llm.generate(prompts[:10], sampling_params)

    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - start_time

    # Calculate metrics
    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    tokens_per_second = total_output_tokens / elapsed_time
    requests_per_second = num_prompts / elapsed_time

    print("\nResults:")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"Requests/sec: {requests_per_second:.2f}")

    del llm
    torch.accelerator.empty_cache()

    return {
        "elapsed_time": elapsed_time,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "requests_per_second": requests_per_second,
    }


def benchmark_latency(
    model_name: str,
    quantization: str | None = None,
    num_runs: int = 50,
    input_len: int = 512,
    output_len: int = 128,
):
    """Benchmark latency with and without int4 quantization."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking latency: {model_name}")
    print(f"Quantization: {quantization or 'None (baseline)'}")
    print(f"Runs: {num_runs}, Input: {input_len}, Output: {output_len}")
    print(f"{'=' * 60}\n")

    prompt = "This is a test prompt " * (input_len // 5)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=output_len,
    )

    llm = LLM(
        model=model_name,
        dtype="float16",
        quantization=quantization,
        gpu_memory_utilization=0.9,
    )

    # Warmup
    print("Warming up...")
    for _ in range(10):
        llm.generate([prompt], sampling_params)

    # Benchmark
    print("Running benchmark...")
    latencies = []
    for i in range(num_runs):
        start_time = time.time()
        llm.generate([prompt], sampling_params)
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}")

    # Calculate statistics
    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p99_latency = latencies[int(len(latencies) * 0.99)]

    print("\nResults:")
    print(f"Average latency: {avg_latency * 1000:.2f}ms")
    print(f"P50 latency: {p50_latency * 1000:.2f}ms")
    print(f"P95 latency: {p95_latency * 1000:.2f}ms")
    print(f"P99 latency: {p99_latency * 1000:.2f}ms")

    del llm
    torch.accelerator.empty_cache()

    return {
        "avg_latency_ms": avg_latency * 1000,
        "p50_latency_ms": p50_latency * 1000,
        "p95_latency_ms": p95_latency * 1000,
        "p99_latency_ms": p99_latency * 1000,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark int4 embedding quantization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="int4_benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput benchmark",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency benchmark",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Int4 Embedding Quantization Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    device_count = torch.accelerator.device_count()
    print(f"GPUs: {device_count}")
    for i in range(device_count):
        device_name = current_platform.get_device_name(i)
        print(f"  GPU {i}: {device_name}")

    results = {"model": args.model, "gpu_count": device_count}

    # Baseline memory
    baseline_memory = benchmark_memory(args.model, quantization=None)
    results["baseline_memory"] = baseline_memory

    # Int4 memory
    int4_memory = benchmark_memory(
        args.model, quantization="int4_per_channel_weight_only"
    )
    results["int4_memory"] = int4_memory

    # Calculate memory savings
    memory_savings = baseline_memory["peak_memory_gb"] - int4_memory["peak_memory_gb"]
    memory_savings_pct = (memory_savings / baseline_memory["peak_memory_gb"]) * 100
    results["memory_savings_gb"] = memory_savings
    results["memory_savings_pct"] = memory_savings_pct

    print(f"\n{'=' * 60}")
    print("Memory Savings Summary")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline_memory['peak_memory_gb']:.2f} GB")
    print(f"Int4: {int4_memory['peak_memory_gb']:.2f} GB")
    print(f"Savings: {memory_savings:.2f} GB ({memory_savings_pct:.1f}%)")

    # Throughput benchmark
    if not args.skip_throughput:
        print(f"\n{'=' * 60}")
        print("Throughput Benchmark")
        print(f"{'=' * 60}")

        baseline_throughput = benchmark_throughput(args.model, quantization=None)
        results["baseline_throughput"] = baseline_throughput

        int4_throughput = benchmark_throughput(
            args.model, quantization="int4_per_channel_weight_only"
        )
        results["int4_throughput"] = int4_throughput

        baseline_tps = baseline_throughput["tokens_per_second"]
        int4_tps = int4_throughput["tokens_per_second"]
        throughput_diff = ((int4_tps - baseline_tps) / baseline_tps) * 100
        results["throughput_diff_pct"] = throughput_diff

        print(f"\n{'=' * 60}")
        print("Throughput Summary")
        print(f"{'=' * 60}")
        print(f"Baseline: {baseline_tps:.2f} tokens/sec")
        print(f"Int4: {int4_tps:.2f} tokens/sec")
        print(f"Change: {throughput_diff:+.1f}%")

    # Latency benchmark
    if not args.skip_latency:
        print(f"\n{'=' * 60}")
        print("Latency Benchmark")
        print(f"{'=' * 60}")

        baseline_latency = benchmark_latency(args.model, quantization=None)
        results["baseline_latency"] = baseline_latency

        int4_latency = benchmark_latency(
            args.model, quantization="int4_per_channel_weight_only"
        )
        results["int4_latency"] = int4_latency

        baseline_ms = baseline_latency["avg_latency_ms"]
        int4_ms = int4_latency["avg_latency_ms"]
        latency_diff = ((int4_ms - baseline_ms) / baseline_ms) * 100
        results["latency_diff_pct"] = latency_diff

        print(f"\n{'=' * 60}")
        print("Latency Summary")
        print(f"{'=' * 60}")
        print(f"Baseline: {baseline_ms:.2f}ms")
        print(f"Int4: {int4_ms:.2f}ms")
        print(f"Change: {latency_diff:+.1f}%")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
