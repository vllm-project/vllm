#!/usr/bin/env python3
# Benchmarking script for VLLM models with multiple prompts

import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VLLM model with multiple prompts")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model name or path")
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (one per line)",
    )
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="Batch size for processing")
    parser.add_argument("--num-iters",
                        type=int,
                        default=5,
                        help="Number of iterations to run")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=2,
        help="Number of iterations for warmup",
    )
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="Enforce eager execution")
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save the results in JSON format",
    )
    parser.add_argument("--max-model-len",
                        type=int,
                        help="Maximum model length")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        help="Tensor parallel size")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        required=False,
                        help="KV cache dtype")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompts from file if specified
    prompts = []
    prompt_configs = []  # List of (output_length, prompt) tuples

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse format: output_length, prompt
                if "," in line:
                    parts = line.split(",", 1)  # Split only on first comma
                    try:
                        output_length = int(parts[0].strip())
                        prompt = parts[1].strip() if parts[1].strip() else ""
                        prompt_configs.append((output_length, prompt))
                        prompts.append(
                            prompt)  # Keep for backward compatibility
                    except ValueError:
                        print(
                            f"Warning: Invalid output length in line: {line}")
                        # Treat as regular prompt if parsing fails
                        prompts.append(line)
                        prompt_configs.append((args.output_len, line))
                else:
                    # No comma, treat as regular prompt with default output length
                    prompts.append(line)
                    prompt_configs.append((args.output_len, line))

    if not prompts:
        raise ValueError(
            "No prompts provided. Use --prompts or --prompts-file")

    print(f"Model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iters}")
    print(f"Warmup iterations: {args.num_iters_warmup}")

    effective_kv_cache_dtype = args.kv_cache_dtype if args.kv_cache_dtype else "auto"
    print(f"Effective kv_cache_dtype: {effective_kv_cache_dtype}")
    # Initialize model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        quantization="modelopt",
        kv_cache_dtype=effective_kv_cache_dtype,
        block_size=32,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
    )

    # Process prompts as a batch with individual sampling parameters
    def process_prompts():
        # Create list of prompts and corresponding sampling parameters
        batch_prompts = []
        batch_sampling_params = []

        for output_length, prompt in prompt_configs:
            batch_prompts.append(prompt)
            batch_sampling_params.append(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=output_length,
                ))

        batch_prompts = batch_prompts * 1
        batch_sampling_params = batch_sampling_params * 1
        # Process all prompts in one batch with individual sampling parameters
        outputs = llm.generate(batch_prompts, batch_sampling_params)
        return outputs

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
    print("\nPrompt-Output Pairs:")

    # Display results in original order
    for i, (output_length, prompt) in enumerate(prompt_configs):
        if i < len(results):
            print(f"\nPair {i + 1}:")
            print(f"Output Length:{output_length}")
            print(f"Prompt:{prompt}")
            print(f"Output:{results[i].outputs[0].text}")

    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile:.4f} seconds")

    # Output JSON results if specified
    if args.output_json:
        results_data = {  # Renamed to avoid conflict with 'results' from process_prompts
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
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
