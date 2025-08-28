# Example usage:
# Basic benchmark: torchrun --nproc-per-node=1 HTTP/benchmark_inference.py --tensor-parallel-size 1 --pipeline-parallel-size 1
# Custom configurations: torchrun --nproc-per-node=1 HTTP/benchmark_inference.py --tensor-parallel-size 1 --pipeline-parallel-size 1 --batch-sizes 1 4 8 --input-lengths 512 1024 --output-lengths 256 512
# With token parallelism: torchrun --nproc-per-node=1 HTTP/benchmark_inference.py --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-token-parallel --token-parallel-size 1


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""
Configurable benchmark for distributed vLLM inference with torchrun.

This script provides comprehensive benchmarking capabilities for measuring:
- Prefill and decode latency
- Request throughput (requests/second)
- Token throughput (tokens/second)

Across different configurations:
- Batch sizes
- Input sequence lengths
- Output sequence lengths
- Different parallelism strategies (tensor, pipeline, data, token)

Results are automatically saved to CSV for analysis.

see https://github.com/vllm-project/vllm/issues/11400 for
the motivation and use case for distributed inference.
run the script with `torchrun --nproc-per-node=N` where N matches the 
total parallel size (tensor_parallel_size * pipeline_parallel_size * data_parallel_size).
"""

import argparse
import time
import csv
import json
from typing import List, Dict, Any
import torch.distributed as dist

from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Distributed vLLM inference benchmark with torchrun")
    
    # Model and parallelism configuration
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of tensor parallel processes (default: 4)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of data parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model length (default: 32768)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    
    # Benchmark configuration
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="List of batch sizes to test (default: [1, 2, 4, 8])")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 512, 1024, 2048],
                        help="List of input sequence lengths to test (default: [128, 512, 1024, 2048])")
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[128, 256, 512],
                        help="List of output sequence lengths to test (default: [128, 256, 512])")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of iterations per configuration (default: 3)")
    parser.add_argument("--warmup-iterations", type=int, default=1,
                        help="Number of warmup iterations before measurement (default: 1)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.csv",
                        help="Output CSV file for results (default: benchmark_results.csv)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter (default: 1.0)")
    
    return parser.parse_args()


def generate_prompts(batch_size: int, input_length: int) -> List[str]:
    """Generate prompts with specified input length."""
    # Base prompt that will be repeated/truncated to reach target length
    base_text = ("This is a sample text for benchmarking purposes. " * 100)[:input_length]
    
    prompts = []
    for i in range(batch_size):
        # Add variety while maintaining target length
        prompt = f"Prompt {i+1}: {base_text}"
        # Truncate or pad to exact length
        if len(prompt) > input_length:
            prompt = prompt[:input_length]
        elif len(prompt) < input_length:
            prompt = prompt + " " * (input_length - len(prompt))
        prompts.append(prompt)
    
    return prompts


def calculate_metrics(total_time: float, num_prompts: int, total_tokens_generated: int) -> Dict[str, float]:
    """Calculate latency and throughput metrics."""
    return {
        "latency_per_request": total_time / num_prompts,
        "throughput_requests_per_sec": num_prompts / total_time,
        "throughput_tokens_per_sec": total_tokens_generated / total_time,
        "total_time": total_time,
        "total_tokens": total_tokens_generated
    }


def write_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Write benchmark results to CSV file."""
    if not results:
        return
    
    # Get all possible fieldnames from results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(list(fieldnames))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_benchmark_iteration(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> Dict[str, Any]:
    """Run a single benchmark iteration and return timing metrics."""
    start_time = time.perf_counter()
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Count total tokens generated
    total_tokens_generated = sum(len(output.outputs[0].text.split()) for output in outputs)
    
    # Calculate metrics
    metrics = calculate_metrics(total_time, len(prompts), total_tokens_generated)
    
    return {
        "total_time": total_time,
        "total_tokens_generated": total_tokens_generated,
        "metrics": metrics
    }


def main():
    args = parse_args()

    # Prepare LLM kwargs - only include token parallel args if enabled
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    
    # Only add token parallel configs if token parallelism is enabled
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    llm = LLM(**llm_kwargs)

    if dist.get_rank() == 0:
        if args.enable_token_parallel:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}, token_parallel_size={args.token_parallel_size}, enable_token_parallel={args.enable_token_parallel}")
        else:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}")
        
        print(f"\nStarting benchmark with configurations:")
        print(f"Batch sizes: {args.batch_sizes}")
        print(f"Input lengths: {args.input_lengths}")
        print(f"Output lengths: {args.output_lengths}")
        print(f"Iterations per config: {args.num_iterations}")
        print(f"Warmup iterations: {args.warmup_iterations}")
        print("-" * 80)

    # Store all benchmark results
    all_results = []
    
    # Benchmark loop over all configurations
    for input_length in args.input_lengths:
        for output_length in args.output_lengths:
            for batch_size in args.batch_sizes:
                if dist.get_rank() == 0:
                    print(f"\nTesting: batch_size={batch_size}, input_length={input_length}, output_length={output_length}")
                
                # Create sampling parameters for this configuration
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=output_length
                )
                
                # Generate prompts for this configuration
                prompts = generate_prompts(batch_size, input_length)
                
                # Warmup iterations
                for warmup_iter in range(args.warmup_iterations):
                    if dist.get_rank() == 0:
                        print(f"  Warmup iteration {warmup_iter + 1}/{args.warmup_iterations}")
                    llm.generate(prompts, sampling_params)
                
                # Benchmark iterations
                iteration_results = []
                for iter_num in range(args.num_iterations):
                    if dist.get_rank() == 0:
                        print(f"  Benchmark iteration {iter_num + 1}/{args.num_iterations}")
                    
                    result = run_benchmark_iteration(llm, prompts, sampling_params)
                    iteration_results.append(result)
                
                # Calculate average metrics across iterations (only on rank 0)
                if dist.get_rank() == 0:
                    avg_total_time = sum(r["total_time"] for r in iteration_results) / len(iteration_results)
                    avg_tokens_generated = sum(r["total_tokens_generated"] for r in iteration_results) / len(iteration_results)
                    
                    avg_metrics = calculate_metrics(avg_total_time, batch_size, int(avg_tokens_generated))
                    
                    # Create result record
                    result_record = {
                        "model": args.model,
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "pipeline_parallel_size": args.pipeline_parallel_size,
                        "data_parallel_size": args.data_parallel_size,
                        "token_parallel_size": args.token_parallel_size if args.enable_token_parallel else 1,
                        "enable_token_parallel": args.enable_token_parallel,
                        "batch_size": batch_size,
                        "input_length": input_length,
                        "output_length": output_length,
                        "num_iterations": args.num_iterations,
                        "avg_latency_per_request": avg_metrics["latency_per_request"],
                        "avg_throughput_requests_per_sec": avg_metrics["throughput_requests_per_sec"],
                        "avg_throughput_tokens_per_sec": avg_metrics["throughput_tokens_per_sec"],
                        "avg_total_time": avg_metrics["total_time"],
                        "avg_total_tokens": avg_metrics["total_tokens"],
                        "temperature": args.temperature,
                        "top_p": args.top_p
                    }
                    
                    all_results.append(result_record)
                    
                    # Print results for this configuration
                    print(f"  Results:")
                    print(f"    Avg latency per request: {avg_metrics['latency_per_request']:.4f} seconds")
                    print(f"    Avg throughput: {avg_metrics['throughput_requests_per_sec']:.2f} requests/sec")
                    print(f"    Avg throughput: {avg_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
                    print(f"    Avg total time: {avg_metrics['total_time']:.4f} seconds")
                    print(f"    Avg tokens generated: {avg_metrics['total_tokens']:.1f}")

    # Write results to CSV (only on rank 0)
    if dist.get_rank() == 0:
        write_results_to_csv(all_results, args.output_file)
        print(f"\nBenchmark completed! Results saved to {args.output_file}")
        print("-" * 80)


if __name__ == "__main__":
    main()

"""
Benchmark Results:

The benchmark outputs a CSV file with the following columns:
- model: Model name used
- tensor_parallel_size, pipeline_parallel_size, data_parallel_size, token_parallel_size: Parallelism configuration
- batch_size: Number of requests processed together
- input_length: Length of input sequences in characters
- output_length: Maximum output tokens generated
- avg_latency_per_request: Average time per request (seconds)
- avg_throughput_requests_per_sec: Average requests processed per second
- avg_throughput_tokens_per_sec: Average tokens generated per second
- avg_total_time: Average total time for the batch
- avg_total_tokens: Average total tokens generated

Example benchmark command:
torchrun --nproc-per-node=1 HTTP/benchmark_inference.py \
    --tensor-parallel-size 1 \
    --batch-sizes 1 2 4 8 \
    --input-lengths 128 512 1024 \
    --output-lengths 128 256 512 \
    --num-iterations 5 \
    --warmup-iterations 2 \
    --output-file my_benchmark_results.csv

Further distributed tips:

1. to communicate control messages across all ranks, use the cpu group,
a PyTorch ProcessGroup with GLOO backend.

```python
from vllm.distributed.parallel_state import get_world_group
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)
if torch_rank == 0:
    # do something for rank 0, e.g. saving the results to disk.
```

2. to communicate data across all ranks, use the model's device group,
a PyTorch ProcessGroup with NCCL backend.
```python
from vllm.distributed.parallel_state import get_world_group
device_group = get_world_group().device_group
```

3. to access the model directly in every rank, use the following code:
```python
llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
```
"""


