#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Alternative approach for measuring prefill and decode phases separately.

This version uses a different strategy to ensure we properly measure:
- Prefill: Time to process input and generate first token  
- Decode: Time to generate remaining tokens

Usage: torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode_v2.py --tensor-parallel-size 1 --batch-size 4 --input-length 512 --output-length 128
Output CSV includes: total decode time, average decode time per step, and decode throughput
"""

import argparse
import time
import csv
import os
from typing import List, Dict, Any
import torch.distributed as dist

from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="vLLM Prefill/Decode Benchmark v2")
    
    # Model and parallelism configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of tensor parallel processes (default: 1)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of data parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model length (default: 8192)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Benchmark configuration
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for benchmark (default: 1)")
    parser.add_argument("--input-length", type=int, default=512,
                        help="Input sequence length in tokens (default: 512)")
    parser.add_argument("--output-length", type=int, default=128,
                        help="Output sequence length in tokens (default: 128)")
    parser.add_argument("--data-path", type=str, default="HTTP/benchmark_results",
                        help="Directory to save results (default: current directory)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter (default: 1.0)")
    
    return parser.parse_args()


def generate_prompts(batch_size: int, input_length: int) -> List[str]:
    """Generate prompts with approximate token length."""
    base_text = "The quick brown fox jumps over the lazy dog. " * (input_length // 10)
    
    prompts = []
    for i in range(batch_size):
        prompt = f"Request {i+1}: Please continue this story. {base_text}"
        # Rough approximation: 4 chars per token
        target_chars = input_length * 4
        if len(prompt) > target_chars:
            prompt = prompt[:target_chars]
        prompts.append(prompt)
    
    return prompts


def measure_baseline_generation(llm: LLM, prompts: List[str], output_length: int, temperature: float, top_p: float) -> Dict[str, Any]:
    """
    Baseline measurement: Generate the entire sequence in one call to get total generation time.
    This serves as a reference to validate our separate prefill/decode measurements.
    """
    
    baseline_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=output_length
    )
    
    baseline_start = time.perf_counter()
    baseline_outputs = llm.generate(prompts, baseline_params)
    baseline_end = time.perf_counter()
    
    baseline_total_time = baseline_end - baseline_start
    
    # Count total tokens generated
    total_tokens = sum(len(output.outputs[0].text.split()) for output in baseline_outputs)
    
    if dist.get_rank() == 0:
        print(f"  Baseline total generation: {baseline_total_time:.4f}s")
        print(f"  Baseline total tokens: {total_tokens}")
    
    return {
        "baseline_total_time": baseline_total_time,
        "baseline_total_tokens": total_tokens
    }


def measure_prefill_decode_separately(llm: LLM, prompts: List[str], output_length: int, temperature: float, top_p: float) -> Dict[str, Any]:
    """
    Measure prefill and decode phases separately using two separate calls.
    
    Strategy:
    1. First call: Generate just 1 token (measures prefill + 1 token)
    2. Second call: Generate remaining tokens with prompt + first token (measures decode)
    
    Returns timing data including total decode time and average per-step decode time.
    """
    
    # Step 1: Measure prefill + first token
    prefill_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1  # Just one token to end prefill phase
    )
    
    prefill_start = time.perf_counter()
    prefill_outputs = llm.generate(prompts, prefill_params)
    prefill_end = time.perf_counter()
    
    prefill_time = prefill_end - prefill_start
    
    if dist.get_rank() == 0:
        print(f"  Prefill + 1 token: {prefill_time:.4f}s")
    
    # Step 2: Measure decode (remaining tokens)
    if output_length > 1:
        decode_steps = output_length - 1  # Number of remaining tokens to generate
        
        # Create new prompts with the first generated token
        # decode_prompts = []
        for i, output in enumerate(prefill_outputs):
            first_token = output.outputs[0].text
            # extended_prompt = prompts[i] + first_token
            # decode_prompts.append(extended_prompt)
        
        decode_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=output_length  
        )
        
        # decode_start = time.perf_counter()
        # decode_outputs = llm.generate(prompts, decode_params)
        # decode_end = time.perf_counter()
        
        total_start = time.perf_counter()
        total_outputs = llm.generate(prompts, decode_params)
        total_end = time.perf_counter()

        decode_time = total_end - total_start - prefill_time
        
        # Calculate average decode time per step
        avg_decode_time_per_step = decode_time / decode_steps if decode_steps > 0 else 0.0
        
        # Count total tokens generated
        # total_tokens = sum(len(output.outputs[0].text.split()) for output in prefill_outputs)
        total_tokens = sum(len(output.outputs[0].text.split()) for output in total_outputs)
        
        if dist.get_rank() == 0:
            print(f"  Decode ({decode_steps} tokens): {decode_time:.4f}s")
            print(f"  Avg decode time per step: {avg_decode_time_per_step*1000:.2f}ms")
    else:
        decode_time = 0.0
        decode_steps = 0
        avg_decode_time_per_step = 0.0
        total_tokens = sum(len(output.outputs[0].text.split()) for output in prefill_outputs)
    
    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "avg_decode_time_per_step": avg_decode_time_per_step,
        "decode_steps": decode_steps if output_length > 1 else 0,
        "total_tokens": total_tokens,
        "total_time": prefill_time + decode_time
    }


def calculate_metrics(timing_data: Dict[str, Any], baseline_data: Dict[str, Any], batch_size: int, input_length: int) -> Dict[str, float]:
    """Calculate prefill and decode metrics including per-step averages."""
    
    metrics = {}
    
    # Prefill metrics
    prefill_time = timing_data["prefill_time"]
    metrics["prefill_latency_ms"] = prefill_time * 1000
    
    # Prefill throughput: input tokens / prefill time
    total_input_tokens = batch_size * input_length
    metrics["prefill_throughput_tokens_per_sec"] = total_input_tokens / prefill_time if prefill_time > 0 else 0
    
    # Decode metrics
    decode_time = timing_data["decode_time"]
    metrics["decode_time_ms"] = decode_time * 1000
    
    # Average decode time per step
    avg_decode_time_per_step = timing_data["avg_decode_time_per_step"]
    metrics["avg_decode_time_per_step_ms"] = avg_decode_time_per_step * 1000
    
    # Decode throughput: batch_size / avg_decode_time_per_step (tokens per second)
    if avg_decode_time_per_step > 0:
        metrics["decode_throughput_tokens_per_sec"] = batch_size / avg_decode_time_per_step
    else:
        metrics["decode_throughput_tokens_per_sec"] = 0
    
    # Baseline metrics
    baseline_total_time = baseline_data["baseline_total_time"]
    metrics["baseline_total_time_ms"] = baseline_total_time * 1000
    
    return metrics


def write_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Write results to CSV with updated columns including baseline total generation time."""
    fieldnames = [
        "model_name",
        "tensor_parallel_size", 
        "pipeline_parallel_size",
        "data_parallel_size",
        "token_parallel_size",
        "batch_size",
        "sequence_length",
        "prefill_latency_ms",
        "prefill_throughput_tokens_per_sec",
        "decode_time_ms",
        "avg_decode_time_per_step_ms",
        "decode_throughput_tokens_per_sec",
        "baseline_total_time_ms"
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    
    # Ensure data path exists
    os.makedirs(args.data_path, exist_ok=True)
    
    # Generate output filename
    model_name_clean = args.model.replace('/', '_')
    output_file = os.path.join(args.data_path, f"{model_name_clean}_benchmark_v2.csv")
    
    # Initialize LLM
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    if dist.get_rank() == 0:
        print("=" * 80)
        print("vLLM Prefill/Decode Benchmark v2")
        print("=" * 80)
        print(f"Model: {args.model}")
        print(f"Parallelism: TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, DP={args.data_parallel_size}")
        if args.enable_token_parallel:
            print(f"Token Parallel: {args.token_parallel_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Input length: {args.input_length} tokens")
        print(f"Output length: {args.output_length} tokens")
        print(f"Output file: {output_file}")
        print("-" * 80)
    
    # Generate prompts
    prompts = generate_prompts(args.batch_size, args.input_length)
    
    # Warmup
    if dist.get_rank() == 0:
        print("Running warmup...")
    
    warmup_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=10)
    llm.generate(prompts, warmup_params)
    
    # Run benchmark
    if dist.get_rank() == 0:
        print("\nRunning benchmark...")
    
    # First run baseline measurement
    if dist.get_rank() == 0:
        print("Running baseline measurement...")
    baseline_data = measure_baseline_generation(
        llm, prompts, args.output_length, args.temperature, args.top_p
    )
    
    # Then run separated prefill/decode measurement
    if dist.get_rank() == 0:
        print("Running separated prefill/decode measurement...")
    timing_data = measure_prefill_decode_separately(
        llm, prompts, args.output_length, args.temperature, args.top_p
    )
    
    # Calculate metrics
    metrics = calculate_metrics(timing_data, baseline_data, args.batch_size, args.input_length)
    
    if dist.get_rank() == 0:
        # Create result record
        result = {
            "model_name": args.model.replace('/', '_'),
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "token_parallel_size": args.token_parallel_size if args.enable_token_parallel else 1,
            "batch_size": args.batch_size,
            "sequence_length": args.input_length,
            "prefill_latency_ms": f"{metrics['prefill_latency_ms']:.2f}",
            "prefill_throughput_tokens_per_sec": f"{metrics['prefill_throughput_tokens_per_sec']:.1f}",
            "decode_time_ms": f"{metrics['decode_time_ms']:.2f}",
            "avg_decode_time_per_step_ms": f"{metrics['avg_decode_time_per_step_ms']:.2f}",
            "decode_throughput_tokens_per_sec": f"{metrics['decode_throughput_tokens_per_sec']:.1f}",
            "baseline_total_time_ms": f"{metrics['baseline_total_time_ms']:.2f}"
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Prefill latency: {metrics['prefill_latency_ms']:.2f} ms")
        print(f"Prefill throughput: {metrics['prefill_throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"Decode time (total): {metrics['decode_time_ms']:.2f} ms")
        print(f"Decode time (avg per step): {metrics['avg_decode_time_per_step_ms']:.2f} ms")
        print(f"Decode throughput: {metrics['decode_throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"Baseline total time: {metrics['baseline_total_time_ms']:.2f} ms")
        print(f"Total tokens: {timing_data['total_tokens']}")
        print(f"Decode steps: {timing_data['decode_steps']}")
        print(f"Separated total time: {timing_data['total_time']:.4f}s")
        
        # Calculate and display timing comparison
        separated_total_ms = timing_data['total_time'] * 1000
        timing_difference = separated_total_ms - metrics['baseline_total_time_ms']
        timing_ratio = separated_total_ms / metrics['baseline_total_time_ms'] if metrics['baseline_total_time_ms'] > 0 else 0
        
        print(f"Timing validation:")
        print(f"  Baseline: {metrics['baseline_total_time_ms']:.2f} ms")
        print(f"  Separated (prefill + decode): {separated_total_ms:.2f} ms")
        print(f"  Difference: {timing_difference:.2f} ms ({timing_ratio:.2f}x)")
        
        # Write to CSV
        write_results_to_csv([result], output_file)
        print(f"\nResults saved to: {output_file}")
        print("=" * 60)


if __name__ == "__main__":
    main()
