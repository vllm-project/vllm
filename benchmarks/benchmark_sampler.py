# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark sampler performance across different configurations."""

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.sampler import get_sampler
from vllm.sampling_params import SamplingType
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
        high_prob_indices = torch.randint(0, vocab_size, (vocab_size // 100,))
        logits[i, high_prob_indices] += torch.randn(len(high_prob_indices)) * 2 + 3
    
    return logits


def benchmark_sampler_latency(
    batch_sizes: List[int],
    vocab_sizes: List[int], 
    sampling_types: List[str],
    num_iterations: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Benchmark sampler latency across different configurations."""
    
    results = {
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "sampling_types": sampling_types,
        "latencies": {},
        "throughputs": {},
        "memory_usage": {}
    }
    
    sampler = get_sampler()
    if hasattr(sampler, 'to'):
        sampler = sampler.to(device)
    
    print("Benchmarking sampler latency...")
    
    for sampling_type in sampling_types:
        results["latencies"][sampling_type] = {}
        results["throughputs"][sampling_type] = {}
        results["memory_usage"][sampling_type] = {}
        
        for batch_size in batch_sizes:
            results["latencies"][sampling_type][batch_size] = {}
            results["throughputs"][sampling_type][batch_size] = {}
            results["memory_usage"][sampling_type][batch_size] = {}
            
            for vocab_size in vocab_sizes:
                print(f"Testing {sampling_type}, batch_size={batch_size}, vocab_size={vocab_size}")
                
                # Create sampling params based on type
                if sampling_type == "greedy":
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
                elif sampling_type == "random":
                    sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
                elif sampling_type == "top_k":
                    sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1)
                elif sampling_type == "top_p":
                    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=1)
                else:
                    raise ValueError(f"Unknown sampling type: {sampling_type}")
                
                # Warmup
                for _ in range(10):
                    logits = create_synthetic_logits(batch_size, vocab_size, device)
                    with torch.no_grad():
                        # Simulate sampling process
                        if sampling_type == "greedy":
                            _ = torch.argmax(logits, dim=-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)
                            if sampling_type == "top_k":
                                top_k_probs, top_k_indices = torch.topk(probs, k=50, dim=-1)
                                _ = torch.multinomial(top_k_probs, num_samples=1)
                            elif sampling_type == "top_p":
                                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                                mask = cumsum_probs <= 0.9
                                _ = torch.multinomial(sorted_probs * mask.float(), num_samples=1)
                            else:  # random
                                _ = torch.multinomial(probs, num_samples=1)
                
                # Benchmark
                torch.cuda.synchronize() if device == "cuda" else None
                
                latencies = []
                memory_before = torch.cuda.memory_allocated() if device == "cuda" else 0
                
                for _ in tqdm(range(num_iterations), desc=f"Benchmarking {sampling_type}"):
                    logits = create_synthetic_logits(batch_size, vocab_size, device)
                    
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        if sampling_type == "greedy":
                            sampled_tokens = torch.argmax(logits, dim=-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)
                            if sampling_type == "top_k":
                                top_k_probs, top_k_indices = torch.topk(probs, k=50, dim=-1)
                                sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
                                sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)
                            elif sampling_type == "top_p":
                                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                                mask = cumsum_probs <= 0.9
                                # Ensure at least one token is selected
                                mask[:, 0] = True
                                filtered_probs = sorted_probs * mask.float()
                                sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
                                sampled_tokens = torch.gather(sorted_indices, -1, sampled_indices).squeeze(-1)
                            else:  # random
                                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
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
                
                results["latencies"][sampling_type][batch_size][vocab_size] = {
                    "avg_ms": avg_latency,
                    "p50_ms": p50_latency,
                    "p95_ms": p95_latency,
                    "p99_ms": p99_latency,
                    "all_latencies": latencies
                }
                
                results["throughputs"][sampling_type][batch_size][vocab_size] = throughput
                results["memory_usage"][sampling_type][batch_size][vocab_size] = memory_used
                
                print(f"  Avg latency: {avg_latency:.2f}ms, Throughput: {throughput:.2f} tokens/s, Memory: {memory_used:.2f}MB")
    
    return results


def print_results(results: Dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print("\n" + "="*80)
    print("SAMPLER BENCHMARK RESULTS")
    print("="*80)
    
    for sampling_type in results["sampling_types"]:
        print(f"\n{sampling_type.upper()} SAMPLING:")
        print("-" * 40)
        
        for batch_size in results["batch_sizes"]:
            print(f"\nBatch Size: {batch_size}")
            print(f"{'Vocab Size':<12} {'Avg Latency (ms)':<18} {'P99 Latency (ms)':<18} {'Throughput (tok/s)':<20} {'Memory (MB)':<12}")
            print("-" * 90)
            
            for vocab_size in results["vocab_sizes"]:
                latency_data = results["latencies"][sampling_type][batch_size][vocab_size]
                throughput = results["throughputs"][sampling_type][batch_size][vocab_size]
                memory = results["memory_usage"][sampling_type][batch_size][vocab_size]
                
                print(f"{vocab_size:<12} {latency_data['avg_ms']:<18.2f} {latency_data['p99_ms']:<18.2f} {throughput:<20.2f} {memory:<12.2f}")


def main(args: argparse.Namespace):
    print("Starting sampler benchmark...")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Sampling types: {args.sampling_types}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Device: {args.device}")
    
    # Run benchmark
    results = benchmark_sampler_latency(
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        sampling_types=args.sampling_types,
        num_iterations=args.num_iterations,
        device=args.device
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output_json:
        write_to_json(args.output_json, results)
        print(f"\nResults saved to {args.output_json}")
    
    # Save in PyTorch benchmark format if requested
    if os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        pt_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={"sampler_latency": [results]},
            extra_info={"benchmark_type": "sampler_performance"}
        )
        if pt_records:
            pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
            write_to_json(pt_file, pt_records)
            print(f"PyTorch benchmark format saved to {pt_file}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark sampler performance")
    
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
    
    parser.add_argument("--sampling-types", 
                       type=str, 
                       nargs="+", 
                       default=["greedy", "random", "top_k", "top_p"],
                       choices=["greedy", "random", "top_k", "top_p"],
                       help="Sampling types to test")
    
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
                       default="sampler_benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
