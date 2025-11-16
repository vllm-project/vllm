# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: DeepSeek V3 TorchTitan vs vLLM Built-in

Compares performance of:
1. Custom DeepSeek V3 (TorchTitan + vLLM MLA)
2. vLLM's built-in DeepSeek implementation

Run from vLLM root:
    # Custom model only (default)
    python examples/custom_models/benchmark_deepseek_v3.py \\
        --num-requests 10 \\
        --max-batch-size 4

    # Built-in only
    python examples/custom_models/benchmark_deepseek_v3.py \\
        --use-builtin \\
        --num-requests 10

    # Both (for comparison)
    python examples/custom_models/benchmark_deepseek_v3.py \\
        --run-both \\
        --num-requests 10
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

# Add vLLM root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
vllm_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, vllm_root)

# Import vLLM first
from vllm import LLM, SamplingParams


def import_custom_model():
    """Import custom model to register it with vLLM."""
    deepseek_path = os.path.join(script_dir, "deepseek_v3_torchtitan.py")
    spec = importlib.util.spec_from_file_location(
        "deepseek_v3_torchtitan", deepseek_path
    )
    deepseek_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deepseek_module)
    print("✓ Custom model registered")
    return deepseek_module


def generate_requests(num_requests: int = 100, prompt_len: int = 128) -> list[str]:
    """Generate dummy requests for benchmarking."""
    # Simple prompt template
    base_prompt = "Write a detailed explanation about "
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "renewable energy",
        "space exploration",
        "genetic engineering",
        "blockchain technology",
        "neural networks",
        "machine learning",
        "robotics",
    ]

    requests = []
    for i in range(num_requests):
        topic = topics[i % len(topics)]
        # Pad to desired prompt length
        prompt = f"{base_prompt}{topic}. " + "Context: " * (prompt_len // 10)
        requests.append(prompt)

    return requests


def benchmark_model(
    model_name: str,
    requests: list[str],
    tp_size: int = 8,
    max_batch_size: int = 32,
    max_tokens: int = 128,
    max_model_len: int = 8192,
) -> dict:
    """Benchmark a model with given requests."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {model_name}")
    print(f"{'=' * 70}")
    print(f"Config: TP={tp_size}, Max Batch Size={max_batch_size}")
    print(f"Requests: {len(requests)}, Output Tokens: {max_tokens}")
    print(f"Max Model Length: {max_model_len}")

    # Create LLM instance
    print("\nInitializing LLM...")
    start_init = time.time()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_num_batched_tokens=max_batch_size * max_tokens,
        max_num_seqs=max_batch_size,
        max_model_len=max_model_len,  # Limit KV cache allocation
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graph for fair comparison
    )

    init_time = time.time() - start_init
    print(f"✓ Initialization time: {init_time:.2f}s")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=max_tokens,
        ignore_eos=True,  # Generate exactly max_tokens
    )

    # Warmup
    print("\nWarming up...")
    warmup_requests = requests[: min(10, len(requests))]
    llm.generate(warmup_requests, sampling_params)
    print("✓ Warmup complete")

    # Actual benchmark - track per-request timing
    print(f"\nRunning benchmark ({len(requests)} requests)...")
    start_time = time.time()

    # Track individual request timings
    request_start_times = []
    request_end_times = []

    # Generate one request at a time to get accurate per-request latency
    # (For batched generation, use total time / num_requests as approximation)
    outputs = []
    if len(requests) <= 20:  # Only do per-request timing for small benchmarks
        print("  (Measuring per-request latency...)")
        for req in requests:
            req_start = time.time()
            output = llm.generate([req], sampling_params)
            req_end = time.time()
            outputs.extend(output)
            request_start_times.append(req_start)
            request_end_times.append(req_end)
    else:
        # For large benchmarks, use batched generation (faster but no per-request latency)
        print("  (Using batched generation for speed...)")
        outputs = llm.generate(requests, sampling_params)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    num_requests = len(outputs)
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / total_time
    requests_per_sec = num_requests / total_time
    avg_latency = total_time / num_requests

    # Per-request latency distribution
    if request_end_times:
        per_request_latencies = [
            end - start for start, end in zip(request_start_times, request_end_times)
        ]
    else:
        # No per-request timing available (batched generation)
        per_request_latencies = []

    results = {
        "model": model_name,
        "num_requests": num_requests,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "requests_per_sec": requests_per_sec,
        "avg_latency": avg_latency,
        "p50_latency": (
            np.percentile(per_request_latencies, 50) if per_request_latencies else 0
        ),
        "p90_latency": (
            np.percentile(per_request_latencies, 90) if per_request_latencies else 0
        ),
        "p99_latency": (
            np.percentile(per_request_latencies, 99) if per_request_latencies else 0
        ),
        "init_time": init_time,
    }

    # Print results
    print(f"\n{'=' * 70}")
    print("Results:")
    print(f"{'=' * 70}")
    print(f"Total Time:        {results['total_time']:.2f}s")
    print(f"Total Tokens:      {results['total_tokens']:,}")
    print(f"Throughput:        {results['throughput']:.2f} tokens/s")
    print(f"Requests/sec:      {results['requests_per_sec']:.2f}")
    print(f"Avg Latency:       {results['avg_latency'] * 1000:.2f}ms")
    if per_request_latencies:
        print(f"P50 Latency:       {results['p50_latency'] * 1000:.2f}ms")
        print(f"P90 Latency:       {results['p90_latency'] * 1000:.2f}ms")
        print(f"P99 Latency:       {results['p99_latency'] * 1000:.2f}ms")
    else:
        print(
            f"P50/P90/P99:       N/A (use --num-requests ≤20 for per-request latency)"
        )
    print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek V3 custom vs built-in"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3-Base",
        help="Model name/path",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of requests"
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=32, help="Maximum batch size"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max output tokens per request"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=128, help="Prompt length (tokens)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (limits KV cache allocation)",
    )
    parser.add_argument(
        "--use-builtin",
        action="store_true",
        help="Use vLLM's built-in DeepSeek implementation (don't import custom)",
    )
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Run both custom and built-in for comparison",
    )
    parser.add_argument(
        "--skip-custom",
        action="store_true",
        help="Skip custom model benchmark",
    )
    parser.add_argument(
        "--skip-builtin",
        action="store_true",
        help="Skip built-in model benchmark (only run custom)",
    )

    args = parser.parse_args()

    # Determine which benchmarks to run
    run_custom = not args.use_builtin and not args.skip_custom
    run_builtin = args.use_builtin or args.run_both
    if args.run_both:
        run_custom = True  # Run both

    print(f"\n{'#' * 70}")
    print("DeepSeek V3 Benchmark: TorchTitan vs Built-in")
    print(f"{'#' * 70}")
    if run_custom and run_builtin:
        print("Mode: Comparing Custom vs Built-in")
    elif run_custom:
        print("Mode: Custom model only")
    elif run_builtin:
        print("Mode: Built-in model only")

    # Import custom model if needed
    if run_custom:
        print("\nImporting custom model...")
        import_custom_model()

    # Generate requests
    print(f"\nGenerating {args.num_requests} requests...")
    requests = generate_requests(args.num_requests, args.prompt_len)
    print(f"✓ Generated {len(requests)} requests")

    results = {}

    # Benchmark custom model (TorchTitan)
    if run_custom:
        try:
            print("\n" + "=" * 70)
            print("CUSTOM MODEL (TorchTitan + vLLM MLA)")
            print("=" * 70)
            results["custom"] = benchmark_model(
                model_name=args.model,
                requests=requests,
                tp_size=args.tp,
                max_batch_size=args.max_batch_size,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
            )
        except Exception as e:
            print(f"\n❌ Custom model benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Benchmark built-in vLLM model
    if run_builtin:
        try:
            print("\n" + "=" * 70)
            print("BUILT-IN vLLM MODEL")
            print("=" * 70)
            results["builtin"] = benchmark_model(
                model_name=args.model,
                requests=requests,
                tp_size=args.tp,
                max_batch_size=args.max_batch_size,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
            )
        except Exception as e:
            print(f"\n❌ Built-in model benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    if len(results) > 1:
        print(f"\n\n{'#' * 70}")
        print("Comparison Summary")
        print(f"{'#' * 70}")

        custom = results.get("custom", {})
        builtin = results.get("builtin", {})

        if custom and builtin:
            print(f"\n{'Metric':<25} {'Custom':<20} {'Built-in':<20} {'Speedup':<15}")
            print("-" * 80)

            metrics = [
                ("Throughput (tok/s)", "throughput", True),
                ("Requests/sec", "requests_per_sec", True),
                ("Avg Latency (ms)", "avg_latency", False),
                ("P50 Latency (ms)", "p50_latency", False),
                ("P90 Latency (ms)", "p90_latency", False),
                ("P99 Latency (ms)", "p99_latency", False),
            ]

            for name, key, higher_better in metrics:
                custom_val = custom.get(key, 0)
                builtin_val = builtin.get(key, 0)

                # Convert latency to ms
                if "latency" in key.lower():
                    custom_val *= 1000
                    builtin_val *= 1000

                if builtin_val > 0:
                    speedup = custom_val / builtin_val
                    speedup_str = f"{speedup:.2f}x"
                    if not higher_better:
                        speedup_str = f"{1 / speedup:.2f}x"
                else:
                    speedup_str = "N/A"

                print(
                    f"{name:<25} {custom_val:<20.2f} {builtin_val:<20.2f} {speedup_str:<15}"
                )

        print(f"\n{'#' * 70}\n")

    elif "custom" in results:
        print("\n✓ Custom model benchmark completed successfully!")
        print(
            f"   Throughput: {results['custom']['throughput']:.2f} tokens/s @ TP={args.tp}"
        )


if __name__ == "__main__":
    main()
