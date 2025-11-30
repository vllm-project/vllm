# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""testing core performance, strategies, penalties, FlashInfer, and Torch Compile."""

import argparse
import os
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.utils import FlexibleArgumentParser


def create_synthetic_logits(batch_size: int,
                            vocab_size: int,
                            device: str = "cuda") -> torch.Tensor:
    """Create synthetic logits for testing."""
    logits = torch.randn(batch_size,
                         vocab_size,
                         device=device,
                         dtype=torch.float32)

    # Make some tokens more likely (simulate realistic distribution)
    for i in range(batch_size):
        high_prob_indices = torch.randint(0,
                                          vocab_size, (vocab_size // 100, ),
                                          device=device)
        boost_values = (
            torch.randn(len(high_prob_indices), device=device) * 2 + 3)
        logits[i, high_prob_indices] += boost_values

    return logits


def sample_tokens(logits: torch.Tensor, params: dict) -> torch.Tensor:
    """Sample tokens based on strategy parameters."""
    temp = params.get("temp", 1.0)

    if temp == 0.0:
        # Greedy sampling
        return torch.argmax(logits, dim=-1)

    # Apply temperature
    logits = logits / temp

    if "k" in params:
        # Top-k sampling
        k = params["k"]
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, 1)
        return top_k_indices.gather(-1, sampled_indices).squeeze(-1)

    elif "p" in params:
        # Top-p sampling
        p = params["p"]
        sorted_logits, sorted_indices = torch.sort(logits,
                                                   descending=True,
                                                   dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Find the cutoff point
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    else:
        # Random sampling
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


def benchmark_core_performance(batch_sizes: list[int],
                               vocab_sizes: list[int],
                               num_iterations: int = 100,
                               device: str = "cuda") -> dict[str, Any]:
    """Benchmark core sampler performance - greedy sampling only."""

    results = {
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "latencies": {},
        "throughputs": {},
        "memory_usage": {}
    }

    print("Benchmarking core sampler performance (greedy sampling)...")

    for batch_size in batch_sizes:
        results["latencies"][batch_size] = {}
        results["throughputs"][batch_size] = {}
        results["memory_usage"][batch_size] = {}

        for vocab_size in vocab_sizes:
            print(f"Testing batch_size={batch_size}, vocab_size={vocab_size}")

            # Warmup
            for _ in range(10):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)
                with torch.no_grad():
                    _ = torch.argmax(logits, dim=-1)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None

            latencies = []
            memory_before = torch.cuda.memory_allocated(
            ) if device == "cuda" else 0

            for _ in tqdm(range(num_iterations),
                          desc=f"Batch={batch_size}, Vocab={vocab_size}"):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)

                start_time = time.perf_counter()

                with torch.no_grad():
                    _ = torch.argmax(logits, dim=-1)

                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                latencies.append(
                    (end_time - start_time) * 1000)  # Convert to ms

            memory_after = torch.cuda.memory_allocated(
            ) if device == "cuda" else 0
            memory_used = (memory_after -
                           memory_before) / 1024 / 1024  # Convert to MB

            # Calculate statistics
            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            throughput = batch_size / (avg_latency / 1000)  # tokens per second

            results["latencies"][batch_size][vocab_size] = {
                "avg_ms": avg_latency,
                "p99_ms": p99_latency,
                "all_latencies": latencies
            }

            results["throughputs"][batch_size][vocab_size] = throughput
            results["memory_usage"][batch_size][vocab_size] = memory_used

            print(
                f"  Avg latency: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                f"Throughput: {throughput:.2f} tokens/s, Memory: {memory_used:.2f}MB"
            )

    return results


def benchmark_sampling_strategies(batch_sizes: list[int],
                                  vocab_sizes: list[int],
                                  num_iterations: int = 50,
                                  device: str = "cuda") -> dict[str, Any]:
    """Benchmark different sampling strategies."""

    results = {
        "strategies": {},
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes
    }

    print("Benchmarking sampling strategies...")

    # Define sampling strategies
    strategies = {
        "greedy": {
            "temp": 0.0
        },
        "random_temp_1.0": {
            "temp": 1.0
        },
        "top_k_50": {
            "temp": 1.0,
            "k": 50
        },
        "top_p_0.9": {
            "temp": 1.0,
            "p": 0.9
        }
    }

    for strategy_name, params in strategies.items():
        print(f"Testing strategy: {strategy_name}")
        results["strategies"][strategy_name] = {}

        for batch_size in batch_sizes:
            results["strategies"][strategy_name][batch_size] = {}

            for vocab_size in vocab_sizes:
                print(f"  batch_size={batch_size}, vocab_size={vocab_size}")

                # Warmup
                for _ in range(5):
                    logits = create_synthetic_logits(batch_size, vocab_size,
                                                     device)
                    with torch.no_grad():
                        _ = sample_tokens(logits, params)

                # Benchmark
                torch.cuda.synchronize() if device == "cuda" else None

                latencies = []
                for _ in tqdm(range(num_iterations), desc=f"{strategy_name}"):
                    logits = create_synthetic_logits(batch_size, vocab_size,
                                                     device)

                    start_time = time.perf_counter()

                    with torch.no_grad():
                        _ = sample_tokens(logits, params)

                    torch.cuda.synchronize() if device == "cuda" else None
                    end_time = time.perf_counter()

                    latencies.append((end_time - start_time) * 1000)

                avg_latency = np.mean(latencies)
                p99_latency = np.percentile(latencies, 99)
                throughput = batch_size / (avg_latency / 1000)

                results["strategies"][strategy_name][batch_size][
                    vocab_size] = {
                        "avg_latency_ms": avg_latency,
                        "p99_latency_ms": p99_latency,
                        "throughput_tokens_per_sec": throughput,
                        "params": params
                    }

                print(
                    f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                    f"Throughput: {throughput:.2f} tokens/s")

    return results


def apply_penalties_and_sample(logits: torch.Tensor, prev_tokens: torch.Tensor,
                               penalty_params: dict) -> torch.Tensor:
    """Apply penalties and sample tokens."""
    modified_logits = logits.clone()

    # Apply repetition penalty
    if "repetition_penalty" in penalty_params:
        penalty = penalty_params["repetition_penalty"]
        for i in range(logits.size(0)):
            for token_id in prev_tokens[i]:
                if token_id < logits.size(1):
                    modified_logits[i, token_id] /= penalty

    # Apply frequency penalty
    if "frequency_penalty" in penalty_params:
        penalty = penalty_params["frequency_penalty"]
        for i in range(logits.size(0)):
            token_counts = torch.bincount(prev_tokens[i],
                                          minlength=logits.size(1))
            modified_logits[i] -= token_counts * penalty

    # Apply presence penalty
    if "presence_penalty" in penalty_params:
        penalty = penalty_params["presence_penalty"]
        for i in range(logits.size(0)):
            present_tokens = torch.unique(prev_tokens[i])
            for token_id in present_tokens:
                if token_id < logits.size(1):
                    modified_logits[i, token_id] -= penalty

    # Sample using softmax
    probs = torch.softmax(modified_logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def benchmark_penalties(batch_sizes: list[int],
                        vocab_sizes: list[int],
                        num_iterations: int = 50,
                        device: str = "cuda") -> dict[str, Any]:
    """Benchmark penalty overhead."""

    results = {
        "penalties": {},
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes
    }

    print("Benchmarking penalty overhead...")

    # Define penalty configurations
    penalties = {
        "no_penalty": {},
        "repetition_penalty_1.1": {
            "repetition_penalty": 1.1
        },
        "frequency_penalty_0.1": {
            "frequency_penalty": 0.1
        },
        "presence_penalty_0.1": {
            "presence_penalty": 0.1
        }
    }

    for penalty_name, penalty_params in penalties.items():
        print(f"Testing penalty: {penalty_name}")
        results["penalties"][penalty_name] = {}

        for batch_size in batch_sizes:
            results["penalties"][penalty_name][batch_size] = {}

            for vocab_size in vocab_sizes:
                print(f"  batch_size={batch_size}, vocab_size={vocab_size}")

                # Create previous tokens for penalty calculation
                prev_tokens = torch.randint(0,
                                            vocab_size, (batch_size, 10),
                                            device=device)

                # Warmup
                for _ in range(5):
                    logits = create_synthetic_logits(batch_size, vocab_size,
                                                     device)
                    with torch.no_grad():
                        _ = apply_penalties_and_sample(logits, prev_tokens,
                                                       penalty_params)

                # Benchmark
                torch.cuda.synchronize() if device == "cuda" else None

                latencies = []
                for _ in tqdm(range(num_iterations), desc=f"{penalty_name}"):
                    logits = create_synthetic_logits(batch_size, vocab_size,
                                                     device)

                    start_time = time.perf_counter()

                    with torch.no_grad():
                        _ = apply_penalties_and_sample(logits, prev_tokens,
                                                       penalty_params)

                    torch.cuda.synchronize() if device == "cuda" else None
                    end_time = time.perf_counter()

                    latencies.append((end_time - start_time) * 1000)

                avg_latency = np.mean(latencies)
                p99_latency = np.percentile(latencies, 99)
                throughput = batch_size / (avg_latency / 1000)

                results["penalties"][penalty_name][batch_size][vocab_size] = {
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                    "throughput_tokens_per_sec": throughput,
                    "penalty_params": penalty_params
                }

                print(
                    f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                    f"Throughput: {throughput:.2f} tokens/s")

    return results


def benchmark_torch_compile(batch_sizes: list[int],
                            vocab_sizes: list[int],
                            num_iterations: int = 50,
                            device: str = "cuda") -> dict[str, Any]:
    """Benchmark Torch Compile optimization."""

    results = {
        "torch_compile": {},
        "eager_mode": {},
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes
    }

    print("Benchmarking Torch Compile vs Eager Mode...")

    # Test eager mode
    print("Testing Eager Mode...")
    results["eager_mode"] = benchmark_eager_mode(batch_sizes, vocab_sizes,
                                                 num_iterations, device)

    # Test torch.compile
    print("Testing Torch Compile...")
    results["torch_compile"] = benchmark_compiled_mode(batch_sizes,
                                                       vocab_sizes,
                                                       num_iterations, device)

    return results


def benchmark_eager_mode(batch_sizes: list[int], vocab_sizes: list[int],
                         num_iterations: int, device: str) -> dict[str, Any]:
    """Benchmark eager mode sampling."""
    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}

        for vocab_size in vocab_sizes:
            print(
                f"  Eager Mode: batch_size={batch_size}, vocab_size={vocab_size}"
            )

            # Warmup
            for _ in range(5):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)
                with torch.no_grad():
                    _ = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None

            latencies = []
            for _ in tqdm(range(num_iterations), desc="Eager Mode"):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)

                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)

            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            throughput = batch_size / (avg_latency / 1000)

            results[batch_size][vocab_size] = {
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "throughput_tokens_per_sec": throughput
            }

            print(f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                  f"Throughput: {throughput:.2f} tokens/s")

    return results


def benchmark_compiled_mode(batch_sizes: list[int], vocab_sizes: list[int],
                            num_iterations: int,
                            device: str) -> dict[str, Any]:
    """Benchmark compiled mode sampling."""
    results = {}

    # Compile the sampling function
    def compiled_sampling(logits):
        return torch.multinomial(torch.softmax(logits, dim=-1), 1)

    compiled_fn = torch.compile(compiled_sampling, mode="reduce-overhead")

    for batch_size in batch_sizes:
        results[batch_size] = {}

        for vocab_size in vocab_sizes:
            print(
                f"  Torch Compile: batch_size={batch_size}, vocab_size={vocab_size}"
            )

            # Warmup (compilation happens here)
            for _ in range(10):  # More warmup for compilation
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)
                with torch.no_grad():
                    _ = compiled_fn(logits)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None

            latencies = []
            for _ in tqdm(range(num_iterations), desc="Torch Compile"):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)

                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = compiled_fn(logits)
                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)

            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            throughput = batch_size / (avg_latency / 1000)

            results[batch_size][vocab_size] = {
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "throughput_tokens_per_sec": throughput
            }

            print(f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                  f"Throughput: {throughput:.2f} tokens/s")

    return results


def benchmark_flashinfer(batch_sizes: list[int],
                         vocab_sizes: list[int],
                         num_iterations: int = 50,
                         device: str = "cuda") -> dict[str, Any]:
    """Benchmark FlashInfer vs PyTorch native sampling."""

    results = {
        "flashinfer": {},
        "pytorch_native": {},
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes
    }

    print("Benchmarking FlashInfer vs PyTorch native sampling...")

    # Check if FlashInfer is available
    import importlib.util
    flashinfer_available = importlib.util.find_spec("flashinfer") is not None
    if flashinfer_available:
        print("FlashInfer is available!")
    else:
        print("FlashInfer not available, skipping FlashInfer benchmarks")

    # Test PyTorch native implementation
    print("Testing PyTorch native implementation...")
    results["pytorch_native"] = benchmark_pytorch_native(
        batch_sizes, vocab_sizes, num_iterations, device)

    # Test FlashInfer if available
    if flashinfer_available:
        print("Testing FlashInfer implementation...")
        results["flashinfer"] = benchmark_flashinfer_impl(
            batch_sizes, vocab_sizes, num_iterations, device)
    else:
        results["flashinfer"] = {}

    return results


def benchmark_pytorch_native(batch_sizes: list[int], vocab_sizes: list[int],
                             num_iterations: int,
                             device: str) -> dict[str, Any]:
    """Benchmark PyTorch native sampling."""
    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}

        for vocab_size in vocab_sizes:
            print(
                f"  PyTorch native: batch_size={batch_size}, vocab_size={vocab_size}"
            )

            # Warmup
            for _ in range(5):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)
                with torch.no_grad():
                    _ = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None

            latencies = []
            for _ in tqdm(range(num_iterations), desc="PyTorch native"):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)

                start_time = time.perf_counter()

                with torch.no_grad():
                    _ = torch.multinomial(torch.softmax(logits, dim=-1), 1)

                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)

            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            throughput = batch_size / (avg_latency / 1000)

            results[batch_size][vocab_size] = {
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "throughput_tokens_per_sec": throughput
            }

            print(f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                  f"Throughput: {throughput:.2f} tokens/s")

    return results


def benchmark_flashinfer_impl(batch_sizes: list[int], vocab_sizes: list[int],
                              num_iterations: int,
                              device: str) -> dict[str, Any]:
    """Benchmark FlashInfer implementation."""
    import flashinfer

    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}

        for vocab_size in vocab_sizes:
            print(
                f"  FlashInfer: batch_size={batch_size}, vocab_size={vocab_size}"
            )

            # Warmup
            for _ in range(5):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)
                with torch.no_grad():
                    # Use FlashInfer's sampling function
                    _ = flashinfer.sample(logits, temperature=1.0, top_p=0.9)

            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None

            latencies = []
            for _ in tqdm(range(num_iterations), desc="FlashInfer"):
                logits = create_synthetic_logits(batch_size, vocab_size,
                                                 device)

                start_time = time.perf_counter()

                with torch.no_grad():
                    _ = flashinfer.sample(logits, temperature=1.0, top_p=0.9)

                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)

            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            throughput = batch_size / (avg_latency / 1000)

            results[batch_size][vocab_size] = {
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "throughput_tokens_per_sec": throughput
            }

            print(f"    Avg: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                  f"Throughput: {throughput:.2f} tokens/s")

    return results


def print_results(all_results: dict[str, Any]):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("SAMPLER BENCHMARK RESULTS")
    print("=" * 80)

    # Print core performance results
    if "core_performance" in all_results:
        print("\n" + "-" * 40)
        print("CORE PERFORMANCE (Greedy Sampling)")
        print("-" * 40)
        results = all_results["core_performance"]

        for batch_size in results["batch_sizes"]:
            print(f"\nBatch Size: {batch_size}")
            print(f"{'Vocab Size':<12} {'Avg Latency (ms)':<18} "
                  f"{'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20} "
                  f"{'Memory (MB)':<12}")
            print("-" * 90)

            for vocab_size in results["vocab_sizes"]:
                latency_data = results["latencies"][batch_size][vocab_size]
                throughput = results["throughputs"][batch_size][vocab_size]
                memory = results["memory_usage"][batch_size][vocab_size]

                print(f"{vocab_size:<12} {latency_data['avg_ms']:<18.2f} "
                      f"{latency_data['p99_ms']:<18.2f} {throughput:<20.2f} "
                      f"{memory:<12.2f}")

    # Print sampling strategies results
    if "sampling_strategies" in all_results:
        print("\n" + "-" * 40)
        print("SAMPLING STRATEGIES COMPARISON")
        print("-" * 40)
        results = all_results["sampling_strategies"]

        for batch_size in results["batch_sizes"]:
            for vocab_size in results["vocab_sizes"]:
                print(f"\nBatch Size: {batch_size}, Vocab Size: {vocab_size}")
                print(f"{'Strategy':<20} {'Avg Latency (ms)':<18} "
                      f"{'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20}")
                print("-" * 80)

                for strategy_name, strategy_data in results[
                        "strategies"].items():
                    if (batch_size in strategy_data
                            and vocab_size in strategy_data[batch_size]):
                        data = strategy_data[batch_size][vocab_size]
                        print(
                            f"{strategy_name:<20} {data['avg_latency_ms']:<18.2f} "
                            f"{data['p99_latency_ms']:<18.2f} "
                            f"{data['throughput_tokens_per_sec']:<20.2f}")

    # Print penalties results
    if "penalties" in all_results:
        print("\n" + "-" * 40)
        print("PENALTY OVERHEAD ANALYSIS")
        print("-" * 40)
        results = all_results["penalties"]

        for batch_size in results["batch_sizes"]:
            for vocab_size in results["vocab_sizes"]:
                print(f"\nBatch Size: {batch_size}, Vocab Size: {vocab_size}")
                print(f"{'Penalty':<25} {'Avg Latency (ms)':<18} "
                      f"{'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20}")
                print("-" * 85)

                for penalty_name, penalty_data in results["penalties"].items():
                    if (batch_size in penalty_data
                            and vocab_size in penalty_data[batch_size]):
                        data = penalty_data[batch_size][vocab_size]
                        print(
                            f"{penalty_name:<25} {data['avg_latency_ms']:<18.2f} "
                            f"{data['p99_latency_ms']:<18.2f} "
                            f"{data['throughput_tokens_per_sec']:<20.2f}")

    # Print Torch Compile results
    if "torch_compile" in all_results:
        print("\n" + "-" * 40)
        print("TORCH COMPILE VS EAGER MODE")
        print("-" * 40)
        results = all_results["torch_compile"]

        if "eager_mode" in results and "torch_compile" in results:
            for batch_size in results["batch_sizes"]:
                for vocab_size in results["vocab_sizes"]:

                    if (batch_size not in results["eager_mode"] or vocab_size
                            not in results["eager_mode"][batch_size]
                            or batch_size not in results["torch_compile"]
                            or vocab_size
                            not in results["torch_compile"][batch_size]):
                        continue

                    print(
                        f"\nBatch Size: {batch_size}, Vocab Size: {vocab_size}"
                    )
                    print(
                        f"{'Mode':<20} {'Avg Latency (ms)':<18} "
                        f"{'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20}")
                    print("-" * 80)

                    eager_data = results["eager_mode"][batch_size][vocab_size]
                    compile_data = results["torch_compile"][batch_size][
                        vocab_size]

                    print(f"{'Eager Mode':<20} "
                          f"{eager_data['avg_latency_ms']:<18.2f} "
                          f"{eager_data['p99_latency_ms']:<18.2f} "
                          f"{eager_data['throughput_tokens_per_sec']:<20.2f}")
                    print(
                        f"{'Torch Compile':<20} "
                        f"{compile_data['avg_latency_ms']:<18.2f} "
                        f"{compile_data['p99_latency_ms']:<18.2f} "
                        f"{compile_data['throughput_tokens_per_sec']:<20.2f}")

                    speedup = (eager_data['avg_latency_ms'] /
                               compile_data['avg_latency_ms'])
                    print(f"Compilation Speedup: {speedup:.2f}x")

    # Print FlashInfer results
    if "flashinfer" in all_results and all_results["flashinfer"]:
        print("\n" + "-" * 40)
        print("FLASHINFER VS PYTORCH NATIVE")
        print("-" * 40)
        results = all_results["flashinfer"]

        if ("pytorch_native" in all_results and "flashinfer" in results
                and results["flashinfer"]):
            for batch_size in results["batch_sizes"]:
                for vocab_size in results["vocab_sizes"]:
                    print(
                        f"\nBatch Size: {batch_size}, Vocab Size: {vocab_size}"
                    )
                    print(
                        f"{'Implementation':<20} {'Avg Latency (ms)':<18} "
                        f"{'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20}")
                    print("-" * 80)

                    pytorch_data = results["pytorch_native"][batch_size][
                        vocab_size]
                    flashinfer_data = results["flashinfer"][batch_size][
                        vocab_size]

                    print(
                        f"{'PyTorch Native':<20} "
                        f"{pytorch_data['avg_latency_ms']:<18.2f} "
                        f"{pytorch_data['p99_latency_ms']:<18.2f} "
                        f"{pytorch_data['throughput_tokens_per_sec']:<20.2f}")
                    print(
                        f"{'FlashInfer':<20} "
                        f"{flashinfer_data['avg_latency_ms']:<18.2f} "
                        f"{flashinfer_data['p99_latency_ms']:<18.2f} "
                        f"{flashinfer_data['throughput_tokens_per_sec']:<20.2f}"
                    )

                    speedup = (pytorch_data['avg_latency_ms'] /
                               flashinfer_data['avg_latency_ms'])
                    print(f"Speedup: {speedup:.2f}x")


def main(args: argparse.Namespace):
    print("Starting sampler benchmark...")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Device: {args.device}")
    print(f"Benchmarks to run: {args.benchmarks}")

    all_results = {}

    # Run selected benchmarks
    if "core" in args.benchmarks:
        print("\n" + "=" * 50)
        print("RUNNING CORE PERFORMANCE BENCHMARK")
        print("=" * 50)
        all_results["core_performance"] = benchmark_core_performance(
            batch_sizes=args.batch_sizes,
            vocab_sizes=args.vocab_sizes,
            num_iterations=args.num_iterations,
            device=args.device)

    if "strategies" in args.benchmarks:
        print("\n" + "=" * 50)
        print("RUNNING SAMPLING STRATEGIES BENCHMARK")
        print("=" * 50)
        all_results["sampling_strategies"] = benchmark_sampling_strategies(
            batch_sizes=args.batch_sizes,
            vocab_sizes=args.vocab_sizes,
            num_iterations=args.num_iterations,
            device=args.device)

    if "penalties" in args.benchmarks:
        print("\n" + "=" * 50)
        print("RUNNING PENALTIES BENCHMARK")
        print("=" * 50)
        all_results["penalties"] = benchmark_penalties(
            batch_sizes=args.batch_sizes,
            vocab_sizes=args.vocab_sizes,
            num_iterations=args.num_iterations,
            device=args.device)

    if "torch_compile" in args.benchmarks:
        print("\n" + "=" * 50)
        print("RUNNING TORCH COMPILE BENCHMARK")
        print("=" * 50)
        all_results["torch_compile"] = benchmark_torch_compile(
            batch_sizes=args.batch_sizes,
            vocab_sizes=args.vocab_sizes,
            num_iterations=args.num_iterations,
            device=args.device)

    if "flashinfer" in args.benchmarks:
        print("\n" + "=" * 50)
        print("RUNNING FLASHINFER BENCHMARK")
        print("=" * 50)
        all_results["flashinfer"] = benchmark_flashinfer(
            batch_sizes=args.batch_sizes,
            vocab_sizes=args.vocab_sizes,
            num_iterations=args.num_iterations,
            device=args.device)

    # Print results
    print_results(all_results)

    # Save results
    if args.output_json:
        write_to_json(args.output_json, all_results)
        print(f"\nResults saved to {args.output_json}")

    # Save in PyTorch benchmark format if requested
    if os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        pt_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={"sampler_benchmark": [all_results]},
            extra_info={"benchmark_type": "sampler_benchmark"})
        if pt_records:
            pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
            write_to_json(pt_file, pt_records)
            print(f"PyTorch benchmark format saved to {pt_file}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Sampler benchmark")

    # Benchmark configuration
    parser.add_argument("--batch-sizes",
                        type=int,
                        nargs="+",
                        default=[1, 8, 16, 32, 64],
                        help="Batch sizes to test")

    parser.add_argument("--vocab-sizes",
                        type=int,
                        nargs="+",
                        default=[32000, 50000, 100000],
                        help="Vocabulary sizes to test")

    parser.add_argument("--num-iterations",
                        type=int,
                        default=100,
                        help="Number of iterations for each test")

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run benchmark on")

    parser.add_argument("--benchmarks",
                        type=str,
                        nargs="+",
                        default=[
                            "core", "strategies", "penalties", "torch_compile",
                            "flashinfer"
                        ],
                        choices=[
                            "core", "strategies", "penalties", "torch_compile",
                            "flashinfer"
                        ],
                        help="Which benchmarks to run")

    parser.add_argument("--output-json",
                        type=str,
                        default="sampler_benchmark_results.json",
                        help="Output JSON file for results")

    args = parser.parse_args()
    main(args)
