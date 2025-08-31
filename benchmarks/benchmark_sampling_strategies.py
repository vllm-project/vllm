# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark different sampling strategies performance comparison."""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
import numpy as np
from tqdm import tqdm

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.utils import FlexibleArgumentParser


def create_synthetic_logits(
    batch_size: int, vocab_size: int, device: str = "cuda"
) -> torch.Tensor:
    """Create synthetic logits for testing."""
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    
    # Make some tokens more likely (simulate realistic distribution)
    for i in range(batch_size):
        high_prob_indices = torch.randint(0, vocab_size, (vocab_size // 100,), device=device)
        boost_values = torch.randn(len(high_prob_indices), device=device) * 2 + 3
        logits[i, high_prob_indices] += boost_values
    
    return logits


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling - always pick the highest probability token."""
    return torch.argmax(logits, dim=-1)


def sample_random(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Random sampling with temperature."""
    if temperature == 0.0:
        return sample_greedy(logits)
    
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Top-k sampling."""
    if temperature == 0.0:
        return sample_greedy(logits)
    
    scaled_logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k=k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
    return torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)


def sample_top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    if temperature == 0.0:
        return sample_greedy(logits)
    
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens within top-p
    mask = cumsum_probs <= p
    # Ensure at least one token is selected
    mask[:, 0] = True
    
    filtered_probs = sorted_probs * mask.float()
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled_indices).squeeze(-1)


def benchmark_sampling_strategies(
    strategies: List[str],
    batch_sizes: List[int],
    vocab_sizes: List[int],
    num_iterations: int = 100,
    device: str = "cuda",
    temperature_values: List[float] = [1.0],
    top_k_values: List[int] = [50],
    top_p_values: List[float] = [0.9]
) -> Dict[str, Any]:
    """Benchmark different sampling strategies."""
    
    results = {
        "strategies": strategies,
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "parameters": {
            "temperature_values": temperature_values,
            "top_k_values": top_k_values,
            "top_p_values": top_p_values
        },
        "latencies": {},
        "throughputs": {},
        "memory_usage": {}
    }
    
    print("Benchmarking sampling strategies...")
    
    for strategy in strategies:
        results["latencies"][strategy] = {}
        results["throughputs"][strategy] = {}
        results["memory_usage"][strategy] = {}
        
        # Get strategy parameters
        if strategy == "greedy":
            params = [{"temp": 0.0}]
        elif strategy == "random":
            params = [{"temp": temp} for temp in temperature_values]
        elif strategy == "top_k":
            params = [{"temp": 1.0, "k": k} for k in top_k_values]
        elif strategy == "top_p":
            params = [{"temp": 1.0, "p": p} for p in top_p_values]
        else:
            continue
        
        for param_set in params:
            param_key = str(param_set)
            results["latencies"][strategy][param_key] = {}
            results["throughputs"][strategy][param_key] = {}
            results["memory_usage"][strategy][param_key] = {}
            
            for batch_size in batch_sizes:
                results["latencies"][strategy][param_key][batch_size] = {}
                results["throughputs"][strategy][param_key][batch_size] = {}
                results["memory_usage"][strategy][param_key][batch_size] = {}
                
                for vocab_size in vocab_sizes:
                    print(f"Testing {strategy} {param_set}, batch_size={batch_size}, vocab_size={vocab_size}")
                    
                    # Warmup
                    for _ in range(10):
                        logits = create_synthetic_logits(batch_size, vocab_size, device)
                        with torch.no_grad():
                            if strategy == "greedy":
                                _ = sample_greedy(logits)
                            elif strategy == "random":
                                _ = sample_random(logits, param_set["temp"])
                            elif strategy == "top_k":
                                _ = sample_top_k(logits, param_set["k"], param_set["temp"])
                            elif strategy == "top_p":
                                _ = sample_top_p(logits, param_set["p"], param_set["temp"])
                    
                    # Benchmark
                    torch.cuda.synchronize() if device == "cuda" else None
                    
                    latencies = []
                    memory_before = torch.cuda.memory_allocated() if device == "cuda" else 0
                    
                    for _ in tqdm(range(num_iterations), desc=f"{strategy} {param_set}"):
                        logits = create_synthetic_logits(batch_size, vocab_size, device)
                        
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            if strategy == "greedy":
                                sampled_tokens = sample_greedy(logits)
                            elif strategy == "random":
                                sampled_tokens = sample_random(logits, param_set["temp"])
                            elif strategy == "top_k":
                                sampled_tokens = sample_top_k(logits, param_set["k"], param_set["temp"])
                            elif strategy == "top_p":
                                sampled_tokens = sample_top_p(logits, param_set["p"], param_set["temp"])
                        
                        torch.cuda.synchronize() if device == "cuda" else None
                        end_time = time.perf_counter()
                        
                        latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    memory_after = torch.cuda.memory_allocated() if device == "cuda" else 0
                    memory_used = (memory_after - memory_before) / 1024 / 1024  # Convert to MB
                    
                    # Calculate statistics
                    avg_latency = np.mean(latencies)
                    p99_latency = np.percentile(latencies, 99)
                    throughput = batch_size / (avg_latency / 1000)
                    
                    results["latencies"][strategy][param_key][batch_size][vocab_size] = {
                        "avg_ms": avg_latency,
                        "p99_ms": p99_latency,
                        "all_latencies": latencies
                    }
                    
                    results["throughputs"][strategy][param_key][batch_size][vocab_size] = throughput
                    results["memory_usage"][strategy][param_key][batch_size][vocab_size] = memory_used
                    
                    print(f"  Avg latency: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                          f"Throughput: {throughput:.2f} tokens/s")
    
    return results


def print_strategy_comparison(results: Dict[str, Any]):
    """Print strategy comparison results."""
    print("\n" + "="*80)
    print("SAMPLING STRATEGIES PERFORMANCE COMPARISON")
    print("="*80)
    
    # Compare strategies at a fixed configuration
    batch_size = results["batch_sizes"][0] if results["batch_sizes"] else 1
    vocab_size = results["vocab_sizes"][0] if results["vocab_sizes"] else 32000
    
    print(f"\nComparison at batch_size={batch_size}, vocab_size={vocab_size}")
    print(f"{'Strategy':<15} {'Parameters':<20} {'Avg Latency (ms)':<18} {'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20}")
    print("-" * 100)
    
    for strategy in results["strategies"]:
        for param_key in results["latencies"][strategy]:
            if batch_size in results["latencies"][strategy][param_key] and \
               vocab_size in results["latencies"][strategy][param_key][batch_size]:
                
                latency_data = results["latencies"][strategy][param_key][batch_size][vocab_size]
                throughput = results["throughputs"][strategy][param_key][batch_size][vocab_size]
                
                print(f"{strategy:<15} {param_key:<20} {latency_data['avg_ms']:<18.2f} "
                      f"{latency_data['p99_ms']:<18.2f} {throughput:<20.2f}")


def analyze_parameter_impact(results: Dict[str, Any]):
    """Analyze the impact of different parameters."""
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    batch_size = results["batch_sizes"][0] if results["batch_sizes"] else 1
    vocab_size = results["vocab_sizes"][0] if results["vocab_sizes"] else 32000
    
    # Analyze temperature impact for random sampling
    if "random" in results["strategies"]:
        print(f"\nTemperature Impact (Random Sampling)")
        print("Temperature -> Latency (ms) -> Throughput (tok/s)")
        print("-" * 50)
        
        for param_key in results["latencies"]["random"]:
            if batch_size in results["latencies"]["random"][param_key] and \
               vocab_size in results["latencies"]["random"][param_key][batch_size]:
                
                latency = results["latencies"]["random"][param_key][batch_size][vocab_size]["avg_ms"]
                throughput = results["throughputs"]["random"][param_key][batch_size][vocab_size]
                
                print(f"{param_key:>11} -> {latency:>12.2f} -> {throughput:>18.2f}")


def main(args: argparse.Namespace):
    print("Starting sampling strategies benchmark...")
    print(f"Strategies: {args.strategies}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Device: {args.device}")
    
    # Run benchmark
    results = benchmark_sampling_strategies(
        strategies=args.strategies,
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        num_iterations=args.num_iterations,
        device=args.device,
        temperature_values=args.temperature_values,
        top_k_values=args.top_k_values,
        top_p_values=args.top_p_values
    )
    
    # Print results
    print_strategy_comparison(results)
    analyze_parameter_impact(results)
    
    # Save results
    if args.output_json:
        write_to_json(args.output_json, results)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark sampling strategies performance")
    
    parser.add_argument("--strategies", 
                       type=str, 
                       nargs="+", 
                       default=["greedy", "random", "top_k", "top_p"],
                       choices=["greedy", "random", "top_k", "top_p"],
                       help="Sampling strategies to test")
    
    parser.add_argument("--batch-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[1, 8, 16],
                       help="Batch sizes to test")
    
    parser.add_argument("--vocab-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[32000, 50000],
                       help="Vocabulary sizes to test")
    
    parser.add_argument("--num-iterations", 
                       type=int, 
                       default=50,
                       help="Number of iterations for each test")
    
    parser.add_argument("--device", 
                       type=str, 
                       default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run benchmark on")
    
    parser.add_argument("--temperature-values",
                       type=float,
                       nargs="+",
                       default=[0.5, 1.0, 1.5],
                       help="Temperature values to test")
    
    parser.add_argument("--top-k-values",
                       type=int,
                       nargs="+", 
                       default=[10, 50, 100],
                       help="Top-k values to test")
    
    parser.add_argument("--top-p-values",
                       type=float,
                       nargs="+",
                       default=[0.8, 0.9, 0.95],
                       help="Top-p values to test")
    
    parser.add_argument("--output-json", 
                       type=str, 
                       default="sampling_strategies_benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
