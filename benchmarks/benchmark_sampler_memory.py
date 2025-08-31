# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark sampler memory usage analysis."""

import argparse
import json
import os
import time
import gc
from typing import Any, Dict, List

import torch
import numpy as np
from tqdm import tqdm

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.utils import FlexibleArgumentParser


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get current memory statistics."""
    if device == "cuda" and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024
        }
    else:
        # For CPU, we can't easily get detailed memory stats
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0
        }


def reset_memory_stats(device: str = "cuda"):
    """Reset memory statistics."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


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


def benchmark_logits_memory(
    batch_sizes: List[int],
    vocab_sizes: List[int],
    device: str = "cuda"
) -> Dict[str, Any]:
    """Benchmark memory usage for storing logits."""
    
    results = {
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "logits_memory": {},
        "theoretical_memory": {}
    }
    
    print("Benchmarking logits memory usage...")
    
    for batch_size in batch_sizes:
        results["logits_memory"][batch_size] = {}
        results["theoretical_memory"][batch_size] = {}
        
        for vocab_size in vocab_sizes:
            print(f"Testing logits memory: batch_size={batch_size}, vocab_size={vocab_size}")
            
            # Reset memory stats
            reset_memory_stats(device)
            
            # Measure baseline memory
            baseline_stats = get_memory_stats(device)
            
            # Create logits and measure memory
            logits = create_synthetic_logits(batch_size, vocab_size, device)
            
            # Force memory allocation
            torch.cuda.synchronize() if device == "cuda" else None
            
            # Measure memory after logits creation
            after_stats = get_memory_stats(device)
            
            # Calculate memory usage
            memory_used = after_stats["allocated_mb"] - baseline_stats["allocated_mb"]
            peak_memory = after_stats["max_allocated_mb"] - baseline_stats["allocated_mb"]
            
            # Calculate theoretical memory (float32 = 4 bytes)
            theoretical_mb = (batch_size * vocab_size * 4) / 1024 / 1024
            
            results["logits_memory"][batch_size][vocab_size] = {
                "actual_mb": memory_used,
                "peak_mb": peak_memory,
                "baseline_mb": baseline_stats["allocated_mb"],
                "after_mb": after_stats["allocated_mb"]
            }
            
            results["theoretical_memory"][batch_size][vocab_size] = theoretical_mb
            
            print(f"  Actual: {memory_used:.2f}MB, Theoretical: {theoretical_mb:.2f}MB, "
                  f"Peak: {peak_memory:.2f}MB")
            
            # Clean up
            del logits
            reset_memory_stats(device)
    
    return results


def benchmark_sampling_memory(
    sampling_strategies: List[str],
    batch_sizes: List[int],
    vocab_sizes: List[int],
    device: str = "cuda"
) -> Dict[str, Any]:
    """Benchmark memory usage for different sampling strategies."""
    
    results = {
        "sampling_strategies": sampling_strategies,
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "memory_usage": {},
        "peak_memory": {},
        "memory_efficiency": {}
    }
    
    print("Benchmarking sampling strategies memory usage...")
    
    for strategy in sampling_strategies:
        results["memory_usage"][strategy] = {}
        results["peak_memory"][strategy] = {}
        results["memory_efficiency"][strategy] = {}
        
        for batch_size in batch_sizes:
            results["memory_usage"][strategy][batch_size] = {}
            results["peak_memory"][strategy][batch_size] = {}
            results["memory_efficiency"][strategy][batch_size] = {}
            
            for vocab_size in vocab_sizes:
                print(f"Testing {strategy} memory: batch_size={batch_size}, vocab_size={vocab_size}")
                
                # Reset memory stats
                reset_memory_stats(device)
                
                # Measure baseline memory
                baseline_stats = get_memory_stats(device)
                
                # Create logits
                logits = create_synthetic_logits(batch_size, vocab_size, device)
                
                # Perform sampling based on strategy
                with torch.no_grad():
                    if strategy == "greedy":
                        sampled_tokens = torch.argmax(logits, dim=-1)
                    elif strategy == "random":
                        probs = torch.softmax(logits, dim=-1)
                        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    elif strategy == "top_k":
                        k = min(50, vocab_size)
                        top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
                        top_k_probs = torch.softmax(top_k_logits, dim=-1)
                        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
                        sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)
                    elif strategy == "top_p":
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumsum_probs <= 0.9
                        mask[:, 0] = True  # Ensure at least one token
                        filtered_probs = sorted_probs * mask.float()
                        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
                        sampled_tokens = torch.gather(sorted_indices, -1, sampled_indices).squeeze(-1)
                
                # Force memory allocation
                torch.cuda.synchronize() if device == "cuda" else None
                
                # Measure memory after sampling
                after_stats = get_memory_stats(device)
                
                # Calculate memory usage
                memory_used = after_stats["allocated_mb"] - baseline_stats["allocated_mb"]
                peak_memory = after_stats["max_allocated_mb"] - baseline_stats["allocated_mb"]
                
                # Calculate memory efficiency (tokens per MB)
                efficiency = batch_size / memory_used if memory_used > 0 else float('inf')
                
                results["memory_usage"][strategy][batch_size][vocab_size] = memory_used
                results["peak_memory"][strategy][batch_size][vocab_size] = peak_memory
                results["memory_efficiency"][strategy][batch_size][vocab_size] = efficiency
                
                print(f"  Memory: {memory_used:.2f}MB, Peak: {peak_memory:.2f}MB, "
                      f"Efficiency: {efficiency:.2f} tokens/MB")
                
                # Clean up
                del logits, sampled_tokens
                reset_memory_stats(device)
    
    return results


def benchmark_logprobs_memory(
    batch_sizes: List[int],
    vocab_sizes: List[int],
    top_k_values: List[int],
    device: str = "cuda"
) -> Dict[str, Any]:
    """Benchmark memory usage for logprobs computation."""
    
    results = {
        "batch_sizes": batch_sizes,
        "vocab_sizes": vocab_sizes,
        "top_k_values": top_k_values,
        "logprobs_memory": {},
        "topk_logprobs_memory": {}
    }
    
    print("Benchmarking logprobs memory usage...")
    
    for batch_size in batch_sizes:
        results["logprobs_memory"][batch_size] = {}
        results["topk_logprobs_memory"][batch_size] = {}
        
        for vocab_size in vocab_sizes:
            results["logprobs_memory"][batch_size][vocab_size] = {}
            results["topk_logprobs_memory"][batch_size][vocab_size] = {}
            
            # Test full logprobs
            print(f"Testing full logprobs: batch_size={batch_size}, vocab_size={vocab_size}")
            
            reset_memory_stats(device)
            baseline_stats = get_memory_stats(device)
            
            logits = create_synthetic_logits(batch_size, vocab_size, device)
            logprobs = torch.log_softmax(logits, dim=-1)
            
            torch.cuda.synchronize() if device == "cuda" else None
            after_stats = get_memory_stats(device)
            
            full_logprobs_memory = after_stats["allocated_mb"] - baseline_stats["allocated_mb"]
            results["logprobs_memory"][batch_size][vocab_size]["full"] = full_logprobs_memory
            
            print(f"  Full logprobs: {full_logprobs_memory:.2f}MB")
            
            # Test top-k logprobs
            for k in top_k_values:
                if k >= vocab_size:
                    continue
                    
                print(f"Testing top-{k} logprobs: batch_size={batch_size}, vocab_size={vocab_size}")
                
                reset_memory_stats(device)
                baseline_stats = get_memory_stats(device)
                
                logits = create_synthetic_logits(batch_size, vocab_size, device)
                logprobs = torch.log_softmax(logits, dim=-1)
                top_k_logprobs, top_k_indices = torch.topk(logprobs, k=k, dim=-1)
                
                torch.cuda.synchronize() if device == "cuda" else None
                after_stats = get_memory_stats(device)
                
                topk_memory = after_stats["allocated_mb"] - baseline_stats["allocated_mb"]
                results["topk_logprobs_memory"][batch_size][vocab_size][k] = topk_memory
                
                print(f"  Top-{k} logprobs: {topk_memory:.2f}MB")
                
                # Clean up
                del logits, logprobs, top_k_logprobs, top_k_indices
                reset_memory_stats(device)
    
    return results


def print_memory_analysis(
    logits_results: Dict[str, Any],
    sampling_results: Dict[str, Any],
    logprobs_results: Dict[str, Any]
):
    """Print comprehensive memory analysis."""
    print("\n" + "="*80)
    print("SAMPLER MEMORY USAGE ANALYSIS")
    print("="*80)
    
    # Logits memory analysis
    print("\n1. LOGITS MEMORY USAGE")
    print("-" * 40)
    batch_size = logits_results["batch_sizes"][0]
    print(f"Batch Size: {batch_size}")
    print(f"{'Vocab Size':<12} {'Actual (MB)':<12} {'Theoretical (MB)':<16} {'Efficiency (%)':<14}")
    print("-" * 60)
    
    for vocab_size in logits_results["vocab_sizes"]:
        actual = logits_results["logits_memory"][batch_size][vocab_size]["actual_mb"]
        theoretical = logits_results["theoretical_memory"][batch_size][vocab_size]
        efficiency = (theoretical / actual * 100) if actual > 0 else 0
        
        print(f"{vocab_size:<12} {actual:<12.2f} {theoretical:<16.2f} {efficiency:<14.1f}")
    
    # Sampling strategies memory comparison
    print("\n2. SAMPLING STRATEGIES MEMORY COMPARISON")
    print("-" * 50)
    batch_size = sampling_results["batch_sizes"][0]
    vocab_size = sampling_results["vocab_sizes"][0]
    print(f"Configuration: batch_size={batch_size}, vocab_size={vocab_size}")
    print(f"{'Strategy':<12} {'Memory (MB)':<12} {'Peak (MB)':<12} {'Efficiency (tok/MB)':<20}")
    print("-" * 60)
    
    for strategy in sampling_results["sampling_strategies"]:
        memory = sampling_results["memory_usage"][strategy][batch_size][vocab_size]
        peak = sampling_results["peak_memory"][strategy][batch_size][vocab_size]
        efficiency = sampling_results["memory_efficiency"][strategy][batch_size][vocab_size]
        
        print(f"{strategy:<12} {memory:<12.2f} {peak:<12.2f} {efficiency:<20.2f}")
    
    # Logprobs memory analysis
    print("\n3. LOGPROBS MEMORY USAGE")
    print("-" * 30)
    batch_size = logprobs_results["batch_sizes"][0]
    vocab_size = logprobs_results["vocab_sizes"][0]
    print(f"Configuration: batch_size={batch_size}, vocab_size={vocab_size}")
    print(f"{'Type':<15} {'Memory (MB)':<12}")
    print("-" * 30)
    
    full_memory = logprobs_results["logprobs_memory"][batch_size][vocab_size]["full"]
    print(f"{'Full logprobs':<15} {full_memory:<12.2f}")
    
    for k in logprobs_results["top_k_values"]:
        if k in logprobs_results["topk_logprobs_memory"][batch_size][vocab_size]:
            topk_memory = logprobs_results["topk_logprobs_memory"][batch_size][vocab_size][k]
            print(f"{f'Top-{k} logprobs':<15} {topk_memory:<12.2f}")


def main(args: argparse.Namespace):
    print("Starting sampler memory usage benchmark...")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Device: {args.device}")
    
    # Benchmark logits memory
    logits_results = benchmark_logits_memory(
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        device=args.device
    )
    
    # Benchmark sampling strategies memory
    sampling_results = benchmark_sampling_memory(
        sampling_strategies=args.sampling_strategies,
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        device=args.device
    )
    
    # Benchmark logprobs memory
    logprobs_results = benchmark_logprobs_memory(
        batch_sizes=args.batch_sizes,
        vocab_sizes=args.vocab_sizes,
        top_k_values=args.top_k_values,
        device=args.device
    )
    
    # Combine results
    combined_results = {
        "logits_memory": logits_results,
        "sampling_memory": sampling_results,
        "logprobs_memory": logprobs_results
    }
    
    # Print analysis
    print_memory_analysis(logits_results, sampling_results, logprobs_results)
    
    # Save results
    if args.output_json:
        write_to_json(args.output_json, combined_results)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark sampler memory usage")
    
    parser.add_argument("--batch-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    
    parser.add_argument("--vocab-sizes", 
                       type=int, 
                       nargs="+", 
                       default=[32000, 50000, 100000],
                       help="Vocabulary sizes to test")
    
    parser.add_argument("--sampling-strategies", 
                       type=str, 
                       nargs="+", 
                       default=["greedy", "random", "top_k", "top_p"],
                       choices=["greedy", "random", "top_k", "top_p"],
                       help="Sampling strategies to test")
    
    parser.add_argument("--top-k-values",
                       type=int,
                       nargs="+", 
                       default=[10, 50, 100],
                       help="Top-k values for logprobs testing")
    
    parser.add_argument("--device", 
                       type=str, 
                       default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run benchmark on")
    
    parser.add_argument("--output-json", 
                       type=str, 
                       default="sampler_memory_benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
