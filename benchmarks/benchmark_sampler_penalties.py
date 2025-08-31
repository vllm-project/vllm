# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark sampler penalties overhead testing."""

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
    
    # Make some tokens more likely
    for i in range(batch_size):
        high_prob_indices = torch.randint(0, vocab_size, (vocab_size // 100,), device=device)
        boost_values = torch.randn(len(high_prob_indices), device=device) * 2 + 3
        logits[i, high_prob_indices] += boost_values
    
    return logits


def create_synthetic_token_history(
    batch_size: int, vocab_size: int, sequence_length: int, device: str = "cuda"
) -> torch.Tensor:
    """Create synthetic token history for penalty testing."""
    # Generate realistic token sequences with some repetition
    token_history = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    
    # Add some repetitive patterns to test penalties
    for i in range(batch_size):
        # Add some repeated tokens
        repeat_token = torch.randint(0, vocab_size, (1,), device=device).item()
        repeat_positions = torch.randint(0, sequence_length, (sequence_length // 4,))
        token_history[i, repeat_positions] = repeat_token
    
    return token_history


def apply_repetition_penalty(
    logits: torch.Tensor, 
    token_history: torch.Tensor, 
    penalty: float = 1.1
) -> torch.Tensor:
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique tokens in history
        unique_tokens = torch.unique(token_history[i])
        
        # Apply penalty
        for token in unique_tokens:
            if token < vocab_size:
                if penalized_logits[i, token] > 0:
                    penalized_logits[i, token] /= penalty
                else:
                    penalized_logits[i, token] *= penalty
    
    return penalized_logits


def apply_frequency_penalty(
    logits: torch.Tensor, 
    token_history: torch.Tensor, 
    penalty: float = 0.1
) -> torch.Tensor:
    """Apply frequency penalty to logits."""
    if penalty == 0.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Count token frequencies
        unique_tokens, counts = torch.unique(token_history[i], return_counts=True)
        
        # Apply frequency penalty
        for token, count in zip(unique_tokens, counts):
            if token < vocab_size:
                penalized_logits[i, token] -= penalty * count.float()
    
    return penalized_logits


def apply_presence_penalty(
    logits: torch.Tensor, 
    token_history: torch.Tensor, 
    penalty: float = 0.1
) -> torch.Tensor:
    """Apply presence penalty to logits."""
    if penalty == 0.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique tokens (presence)
        unique_tokens = torch.unique(token_history[i])
        
        # Apply presence penalty
        for token in unique_tokens:
            if token < vocab_size:
                penalized_logits[i, token] -= penalty
    
    return penalized_logits


def benchmark_penalties(
    penalty_types: List[str],
    batch_sizes: List[int],
    vocab_sizes: List[int],
    sequence_lengths: List[int],
    num_iterations: int = 100,
    device: str = "cuda",
    repetition_penalties: List[float] = [1.0, 1.1, 1.2],
    frequency_penalties: List[float] = [0.0, 0.1, 0.2],
    presence_penalties: List[float] = [0.0, 0.1, 0.2]
) -> Dict[str, Any]:
    """Benchmark penalties overhead."""
    
    results = {
        "penalty_types": penalty_types,
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "sequence_lengths": sequence_lengths,
        "parameters": {
            "repetition_penalties": repetition_penalties,
            "frequency_penalties": frequency_penalties,
            "presence_penalties": presence_penalties
        },
        "latencies": {},
        "throughputs": {},
        "memory_usage": {},
        "overhead_analysis": {}
    }
    
    print("Benchmarking penalties overhead...")
    
    # First, benchmark baseline (no penalties)
    baseline_results = {}
    
    for batch_size in batch_sizes:
        baseline_results[batch_size] = {}
        for vocab_size in vocab_sizes:
            for seq_len in sequence_lengths:
                print(f"Baseline: batch_size={batch_size}, vocab_size={vocab_size}, seq_len={seq_len}")
                
                # Warmup
                for _ in range(10):
                    logits = create_synthetic_logits(batch_size, vocab_size, device)
                    with torch.no_grad():
                        _ = torch.argmax(logits, dim=-1)
                
                # Benchmark baseline
                latencies = []
                for _ in range(num_iterations):
                    logits = create_synthetic_logits(batch_size, vocab_size, device)
                    
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        sampled_tokens = torch.argmax(logits, dim=-1)
                    torch.cuda.synchronize() if device == "cuda" else None
                    end_time = time.perf_counter()
                    
                    latencies.append((end_time - start_time) * 1000)
                
                baseline_results[batch_size][(vocab_size, seq_len)] = np.mean(latencies)
    
    # Now benchmark each penalty type
    for penalty_type in penalty_types:
        results["latencies"][penalty_type] = {}
        results["throughputs"][penalty_type] = {}
        results["memory_usage"][penalty_type] = {}
        results["overhead_analysis"][penalty_type] = {}
        
        # Get penalty values to test
        if penalty_type == "repetition":
            penalty_values = repetition_penalties
        elif penalty_type == "frequency":
            penalty_values = frequency_penalties
        elif penalty_type == "presence":
            penalty_values = presence_penalties
        else:
            continue
        
        for penalty_value in penalty_values:
            penalty_key = str(penalty_value)
            results["latencies"][penalty_type][penalty_key] = {}
            results["throughputs"][penalty_type][penalty_key] = {}
            results["memory_usage"][penalty_type][penalty_key] = {}
            results["overhead_analysis"][penalty_type][penalty_key] = {}
            
            for batch_size in batch_sizes:
                results["latencies"][penalty_type][penalty_key][batch_size] = {}
                results["throughputs"][penalty_type][penalty_key][batch_size] = {}
                results["memory_usage"][penalty_type][penalty_key][batch_size] = {}
                results["overhead_analysis"][penalty_type][penalty_key][batch_size] = {}
                
                for vocab_size in vocab_sizes:
                    for seq_len in sequence_lengths:
                        config_key = f"{vocab_size}_{seq_len}"
                        
                        print(f"Testing {penalty_type} penalty={penalty_value}, "
                              f"batch_size={batch_size}, vocab_size={vocab_size}, seq_len={seq_len}")
                        
                        # Create token history
                        token_history = create_synthetic_token_history(batch_size, vocab_size, seq_len, device)
                        
                        # Warmup
                        for _ in range(10):
                            logits = create_synthetic_logits(batch_size, vocab_size, device)
                            with torch.no_grad():
                                if penalty_type == "repetition":
                                    penalized_logits = apply_repetition_penalty(logits, token_history, penalty_value)
                                elif penalty_type == "frequency":
                                    penalized_logits = apply_frequency_penalty(logits, token_history, penalty_value)
                                elif penalty_type == "presence":
                                    penalized_logits = apply_presence_penalty(logits, token_history, penalty_value)
                                _ = torch.argmax(penalized_logits, dim=-1)
                        
                        # Benchmark
                        torch.cuda.synchronize() if device == "cuda" else None
                        
                        latencies = []
                        memory_before = torch.cuda.memory_allocated() if device == "cuda" else 0
                        
                        for _ in tqdm(range(num_iterations), desc=f"{penalty_type} {penalty_value}"):
                            logits = create_synthetic_logits(batch_size, vocab_size, device)
                            
                            start_time = time.perf_counter()
                            
                            with torch.no_grad():
                                if penalty_type == "repetition":
                                    penalized_logits = apply_repetition_penalty(logits, token_history, penalty_value)
                                elif penalty_type == "frequency":
                                    penalized_logits = apply_frequency_penalty(logits, token_history, penalty_value)
                                elif penalty_type == "presence":
                                    penalized_logits = apply_presence_penalty(logits, token_history, penalty_value)
                                
                                sampled_tokens = torch.argmax(penalized_logits, dim=-1)
                            
                            torch.cuda.synchronize() if device == "cuda" else None
                            end_time = time.perf_counter()
                            
                            latencies.append((end_time - start_time) * 1000)
                        
                        memory_after = torch.cuda.memory_allocated() if device == "cuda" else 0
                        memory_used = (memory_after - memory_before) / 1024 / 1024
                        
                        # Calculate statistics
                        avg_latency = np.mean(latencies)
                        p99_latency = np.percentile(latencies, 99)
                        throughput = batch_size / (avg_latency / 1000)
                        
                        # Calculate overhead compared to baseline
                        baseline_latency = baseline_results[batch_size][(vocab_size, seq_len)]
                        overhead_ms = avg_latency - baseline_latency
                        overhead_percent = (overhead_ms / baseline_latency) * 100
                        
                        results["latencies"][penalty_type][penalty_key][batch_size][config_key] = {
                            "avg_ms": avg_latency,
                            "p99_ms": p99_latency,
                            "all_latencies": latencies
                        }
                        
                        results["throughputs"][penalty_type][penalty_key][batch_size][config_key] = throughput
                        results["memory_usage"][penalty_type][penalty_key][batch_size][config_key] = memory_used
                        results["overhead_analysis"][penalty_type][penalty_key][batch_size][config_key] = {
                            "baseline_ms": baseline_latency,
                            "overhead_ms": overhead_ms,
                            "overhead_percent": overhead_percent
                        }
                        
                        print(f"  Avg latency: {avg_latency:.2f}ms, Overhead: +{overhead_ms:.2f}ms ({overhead_percent:.1f}%)")
    
    return results


def print_penalty_analysis(results: Dict[str, Any]):
    """Print penalty overhead analysis."""
    print("\n" + "="*80)
    print("PENALTIES OVERHEAD ANALYSIS")
    print("="*80)
    
    batch_size = results["batch_sizes"][0] if results["batch_sizes"] else 1
    vocab_size = results["vocab_sizes"][0] if results["vocab_sizes"] else 32000
    seq_len = results["sequence_lengths"][0] if results["sequence_lengths"] else 100
    config_key = f"{vocab_size}_{seq_len}"
    
    print(f"\nOverhead Analysis at batch_size={batch_size}, vocab_size={vocab_size}, seq_len={seq_len}")
    print(f"{'Penalty Type':<15} {'Value':<10} {'Latency (ms)':<15} {'Overhead (ms)':<15} {'Overhead (%)':<12}")
    print("-" * 80)
    
    for penalty_type in results["penalty_types"]:
        for penalty_value in results["overhead_analysis"][penalty_type]:
            if batch_size in results["overhead_analysis"][penalty_type][penalty_value] and \
               config_key in results["overhead_analysis"][penalty_type][penalty_value][batch_size]:
                
                analysis = results["overhead_analysis"][penalty_type][penalty_value][batch_size][config_key]
                latency = results["latencies"][penalty_type][penalty_value][batch_size][config_key]["avg_ms"]
                
                print(f"{penalty_type:<15} {penalty_value:<10} {latency:<15.2f} "
                      f"{analysis['overhead_ms']:<15.2f} {analysis['overhead_percent']:<12.1f}")


def main(args: argparse.Namespace):
    print("Starting penalties overhead benchmark...")
    print(f"Penalty types: {args.penalty_types}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Sequence lengths: {args.sequence_lengths}")
    print(f"Iterations: {args.num_iterations}")
    
    # Run benchmark
    results = benchmark_penalties(
        penalty_types=args.penalty_types,
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        sequence_lengths=args.sequence_lengths,
        num_iterations=args.num_iterations,
        device=args.device,
        repetition_penalties=args.repetition_penalties,
        frequency_penalties=args.frequency_penalties,
        presence_penalties=args.presence_penalties
    )
    
    # Print results
    print_penalty_analysis(results)
    
    # Save results
    if args.output_json:
        write_to_json(args.output_json, results)
        print(f"\nResults saved to {args.output_json}")
    
    # Save in PyTorch benchmark format if requested
    if os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        pt_records = convert_to_pytorch_benchmark_format(
            args=args,
            metrics={"sampler_penalties": [results]},
            extra_info={"benchmark_type": "sampler_penalties_overhead"}
        )
        if pt_records:
            pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
            write_to_json(pt_file, pt_records)
            print(f"PyTorch benchmark format saved to {pt_file}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark sampler penalties overhead")
    
    parser.add_argument("--penalty-types", 
                       type=str, 
                       nargs="+", 
                       default=["repetition", "frequency", "presence"],
                       choices=["repetition", "frequency", "presence"],
                       help="Penalty types to test")
    
    parser.add_argument("--batch-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[1, 4, 8],
                       help="Batch sizes to test")
    
    parser.add_argument("--vocab-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[32000, 50000],
                       help="Vocabulary sizes to test")
    
    parser.add_argument("--sequence-lengths", 
                       type=int, 
                       nargs="+", 
                       default=[100, 500, 1000],
                       help="Sequence lengths to test")
    
    parser.add_argument("--num-iterations", 
                       type=int, 
                       default=50,
                       help="Number of iterations for each test")
    
    parser.add_argument("--device", 
                       type=str, 
                       default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run benchmark on")
    
    parser.add_argument("--repetition-penalties",
                       type=float,
                       nargs="+",
                       default=[1.0, 1.1, 1.2],
                       help="Repetition penalty values to test")
    
    parser.add_argument("--frequency-penalties",
                       type=float,
                       nargs="+",
                       default=[0.0, 0.1, 0.2],
                       help="Frequency penalty values to test")
    
    parser.add_argument("--presence-penalties",
                       type=float,
                       nargs="+",
                       default=[0.0, 0.1, 0.2],
                       help="Presence penalty values to test")
    
    parser.add_argument("--output-json", 
                       type=str, 
                       default="sampler_penalties_benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
