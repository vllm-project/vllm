#!/usr/bin/env python3
# Benchmarking script for VLLM models with multiple prompts

import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VLLM model with multiple prompts")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--prompts", type=str, nargs="+", help="List of prompts to process")
    parser.add_argument("--prompts-file", type=str, help="Path to file containing prompts (one per line)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--output-len", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--num-iters", type=int, default=5, help="Number of iterations to run")
    parser.add_argument("--num-iters-warmup", type=int, default=2, help="Number of iterations for warmup")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager execution")
    parser.add_argument("--output-json", type=str, help="Path to save the results in JSON format")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load prompts from file if specified
    prompts = args.prompts if args.prompts else []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    if not prompts:
        raise ValueError("No prompts provided. Use --prompts or --prompts-file")
    
    print(f"Model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output length: {args.output_len}")
    print(f"Iterations: {args.num_iters}")
    print(f"Warmup iterations: {args.num_iters_warmup}")

    # Initialize model
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.output_len,
    )
    
    # Process prompts in batches
    def process_prompts():
        results = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            outputs = llm.generate(batch, sampling_params)
            results.extend(outputs)
        return results
    
    # Warmup runs
    print("\nWarming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        process_prompts()
    
    # Benchmark runs
    print("\nBenchmarking...")
    latencies = []
    for i in tqdm(range(args.num_iters), desc="Benchmark iterations"):
        start_time = time.perf_counter()
        results = process_prompts()
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
    
    # Calculate statistics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    
    # Print results
    print(f"\nResults for processing {len(prompts)} prompts:")
    print(f"Average latency: {avg_latency:.4f} seconds")
    print(f"Latency per prompt: {avg_latency / len(prompts):.4f} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile:.4f} seconds")
    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "model": args.model,
            "num_prompts": len(prompts),
            "batch_size": args.batch_size,
            "output_len": args.output_len,
            "avg_latency": float(avg_latency),
            "avg_latency_per_prompt": float(avg_latency / len(prompts)),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
