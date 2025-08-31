# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark core sampler performance - basic latency and throughput testing."""

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
    # Create realistic logits with some high-probability tokens
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    
    # Make some tokens more likely (simulate realistic distribution)
    for i in range(batch_size):
        # Randomly boost some tokens to create realistic peaks
        high_prob_indices = torch.randint(0, vocab_size, (vocab_size // 100,), device=device)
        boost_values = torch.randn(len(high_prob_indices), device=device) * 2 + 3
        logits[i, high_prob_indices] += boost_values
    
    return logits


def benchmark_core_performance(
    batch_sizes: List[int],
    vocab_sizes: List[int],
    num_iterations: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
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
                logits = create_synthetic_logits(batch_size, vocab_size, device)
                with torch.no_grad():
                    _ = torch.argmax(logits, dim=-1)
            
            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None
            
            latencies = []
            memory_before = torch.cuda.memory_allocated() if device == "cuda" else 0
            
            for _ in tqdm(range(num_iterations), desc=f"Batch={batch_size}, Vocab={vocab_size}"):
                logits = create_synthetic_logits(batch_size, vocab_size, device)
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    sampled_tokens = torch.argmax(logits, dim=-1)
                
                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            memory_after = torch.cuda.memory_allocated() if device == "cuda" else 0
            memory_used = (memory_after - memory_before) / 1024 / 1024  # Convert to MB
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            throughput = batch_size / (avg_latency / 1000)  # tokens per second
            
            results["latencies"][batch_size][vocab_size] = {
                "avg_ms": avg_latency,
                "p50_ms": p50_latency,
                "p95_ms": p95_latency,
                "p99_ms": p99_latency,
                "all_latencies": latencies
            }
            
            results["throughputs"][batch_size][vocab_size] = throughput
            results["memory_usage"][batch_size][vocab_size] = memory_used
            
            print(f"  Avg latency: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms, "
                  f"Throughput: {throughput:.2f} tokens/s, Memory: {memory_used:.2f}MB")
    
    return results


def print_results(results: Dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print("\n" + "="*80)
    print("CORE SAMPLER PERFORMANCE RESULTS (Greedy Sampling)")
    print("="*80)
    
    for batch_size in results["batch_sizes"]:
        print(f"\nBatch Size: {batch_size}")
        print(f"{'Vocab Size':<12} {'Avg Latency (ms)':<18} {'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20} {'Memory (MB)':<12}")
        print("-" * 90)
        
        for vocab_size in results["vocab_sizes"]:
            latency_data = results["latencies"][batch_size][vocab_size]
            throughput = results["throughputs"][batch_size][vocab_size]
            memory = results["memory_usage"][batch_size][vocab_size]
            
            print(f"{vocab_size:<12} {latency_data['avg_ms']:<18.2f} {latency_data['p99_ms']:<18.2f} {throughput:<20.2f} {memory:<12.2f}")


def analyze_scaling(results: Dict[str, Any]):
    """Analyze scaling efficiency."""
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    # Analyze batch scaling for each vocab size
    for vocab_size in results["vocab_sizes"]:
        print(f"\nVocab Size: {vocab_size}")
        print("Batch Size -> Throughput -> Scaling Factor")
        print("-" * 40)
        
        base_throughput = None
        for batch_size in results["batch_sizes"]:
            throughput = results["throughputs"][batch_size][vocab_size]
            
            if base_throughput is None:
                base_throughput = throughput
                scaling_factor = 1.0
            else:
                scaling_factor = throughput / base_throughput
            
            print(f"{batch_size:>10} -> {throughput:>10.0f} -> {scaling_factor:>6.2f}x")


def main(args: argparse.Namespace):
    print("Starting core sampler performance benchmark...")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Device: {args.device}")
    
    # Run benchmark
    results = benchmark_core_performance(
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        num_iterations=args.num_iterations,
        device=args.device
    )
    
    # Print results
    print_results(results)
    
    # Analyze scaling
    analyze_scaling(results)
    
    # Save results
    if args.output_json:
        write_to_json(args.output_json, results)
        print(f"\nResults saved to {args.output_json}")
    
    # Save in PyTorch benchmark format if requested
    if os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        pt_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={"core_sampler_latency": [results]},
            extra_info={"benchmark_type": "core_sampler_performance"}
        )
        if pt_records:
            pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
            write_to_json(pt_file, pt_records)
            print(f"PyTorch benchmark format saved to {pt_file}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark core sampler performance")
    
    # Benchmark configuration
    parser.add_argument("--batch-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[1, 4, 8, 16, 32],
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
    
    parser.add_argument("--output-json", 
                       type=str, 
                       default="sampler_core_benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
