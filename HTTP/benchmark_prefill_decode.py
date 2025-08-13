# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Dedicated benchmark for measuring prefill and decode phases separately in vLLM.

This script provides precise timing measurements for:
- Prefill phase: Time to process input tokens and generate first token
- Decode phase: Time to generate subsequent tokens
- Per-token decode latency

Usage examples:
# Basic prefill/decode benchmark:
torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode.py --tensor-parallel-size 1 --batch-size 4 --input-length 512 --output-length 128

# With token parallelism:
torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode.py --tensor-parallel-size 1 --enable-token-parallel --token-parallel-size 1 --batch-size 2 --input-length 1024 --output-length 256
"""

import argparse
import time
import csv
import json
from typing import List, Dict, Any, Optional
import torch.distributed as dist
import uuid

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine


def parse_args():
    """Parse command line arguments for prefill/decode benchmark."""
    parser = argparse.ArgumentParser(description="vLLM Prefill/Decode Benchmark")
    
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
    parser.add_argument("--num-iterations", type=int, default=5,
                        help="Number of benchmark iterations (default: 5)")
    parser.add_argument("--warmup-iterations", type=int, default=2,
                        help="Number of warmup iterations (default: 2)")
    parser.add_argument("--output-file", type=str, default="prefill_decode_results.csv",
                        help="Output CSV file (default: prefill_decode_results.csv)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter (default: 1.0)")
    
    return parser.parse_args()


def generate_prompts_with_exact_length(batch_size: int, input_length: int) -> List[str]:
    """Generate prompts with exact token length using a simple tokenizer approximation."""
    # Create a base prompt that will be approximately the right length
    # Using rough estimate of ~4 characters per token
    base_text = "The quick brown fox jumps over the lazy dog. " * (input_length // 10)
    
    prompts = []
    for i in range(batch_size):
        # Add some variety while maintaining similar length
        prompt = f"Request {i+1}: Please continue this story. {base_text}"
        # Truncate to approximate token count (rough heuristic: 4 chars per token)
        target_chars = input_length * 4
        if len(prompt) > target_chars:
            prompt = prompt[:target_chars]
        prompts.append(prompt)
    
    return prompts


class PrefillDecodeTimer:
    """Custom timer that tracks prefill and decode phases separately."""
    
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.prefill_times = []
        self.decode_times = []
        self.per_token_decode_times = []
        self.total_tokens_generated = 0
        
    def benchmark_with_step_timing(self, prompts: List[str], sampling_params: SamplingParams) -> Dict[str, Any]:
        """Run benchmark using step-by-step execution to track prefill vs decode."""
        
        # Generate unique request IDs
        request_ids = [str(uuid.uuid4()) for _ in prompts]
        
        # Add all requests to the engine
        for i, prompt in enumerate(prompts):
            self.engine.add_request(
                request_id=request_ids[i],
                prompt=prompt,
                params=sampling_params
            )
        
        # Track timing state for each request
        request_metrics = {}
        for req_id in request_ids:
            request_metrics[req_id] = {
                'start_time': time.perf_counter(),
                'first_token_time': None,
                'decode_step_times': [],
                'total_tokens': 0,
                'finished': False
            }
        
        total_start_time = time.perf_counter()
        
        # Step through execution
        step_count = 0
        while self.engine.has_unfinished_requests():
            step_start = time.perf_counter()
            step_outputs = self.engine.step()
            step_end = time.perf_counter()
            
            step_time = step_end - step_start
            step_count += 1
            
            if dist.get_rank() == 0:
                print(f"    Step {step_count}: {step_time:.4f}s, {len(step_outputs)} outputs")
            
            for output in step_outputs:
                request_id = output.request_id
                metrics = request_metrics[request_id]
                
                # Count tokens in current output
                if hasattr(output.outputs[0], 'token_ids'):
                    current_token_count = len(output.outputs[0].token_ids)
                else:
                    # Fallback to text-based counting
                    generated_text = output.outputs[0].text
                    current_token_count = len(generated_text.split()) if generated_text.strip() else 0
                
                # Check if this is the first token generation (end of prefill)
                if metrics['first_token_time'] is None and current_token_count > 0:
                    metrics['first_token_time'] = step_end
                    prefill_time = step_end - metrics['start_time']
                    
                    if dist.get_rank() == 0:
                        print(f"      Request {request_id[:8]}... prefill complete: {prefill_time:.4f}s")
                
                # If we're past prefill, this is a decode step
                elif metrics['first_token_time'] is not None and not output.finished:
                    metrics['decode_step_times'].append(step_time)
                    
                    if dist.get_rank() == 0:
                        print(f"      Request {request_id[:8]}... decode step: {step_time:.4f}s (tokens: {current_token_count})")
                
                # If request finished, record final state
                if output.finished and not metrics['finished']:
                    metrics['finished'] = True
                    metrics['total_tokens'] = current_token_count
                    self.total_tokens_generated += current_token_count
                    
                    # Calculate total decode time
                    if metrics['first_token_time'] is not None:
                        total_decode_time = step_end - metrics['first_token_time']
                        
                        if dist.get_rank() == 0:
                            generated_text = output.outputs[0].text
                            print(f"      Request {request_id[:8]}... FINISHED - total decode: {total_decode_time:.4f}s, "
                                  f"tokens: {current_token_count} (text: '{generated_text[:50]}...')")
        
        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time
        
        # Extract timing data
        prefill_times = []
        decode_times = []
        per_token_decode_times = []
        
        for req_id, metrics in request_metrics.items():
            if metrics['first_token_time'] is not None:
                # Prefill time
                prefill_time = metrics['first_token_time'] - metrics['start_time']
                prefill_times.append(prefill_time)
                
                # Total decode time (if finished)
                if metrics['finished']:
                    decode_time = total_end_time - metrics['first_token_time']
                    decode_times.append(decode_time)
                
                # Per-step decode times
                per_token_decode_times.extend(metrics['decode_step_times'])
        
        return {
            "total_time": total_time,
            "prefill_times": prefill_times,
            "decode_times": decode_times,
            "per_token_decode_times": per_token_decode_times,
            "total_tokens_generated": self.total_tokens_generated,
            "step_count": step_count
        }


def calculate_detailed_metrics(timing_data: Dict[str, Any], batch_size: int) -> Dict[str, float]:
    """Calculate comprehensive prefill and decode metrics."""
    metrics = {}
    
    # Basic totals
    metrics["total_time"] = timing_data["total_time"]
    metrics["total_tokens_generated"] = timing_data["total_tokens_generated"]
    metrics["step_count"] = timing_data["step_count"]
    
    # Prefill metrics
    prefill_times = timing_data["prefill_times"]
    if prefill_times:
        metrics["avg_prefill_time"] = sum(prefill_times) / len(prefill_times)
        metrics["max_prefill_time"] = max(prefill_times)
        metrics["min_prefill_time"] = min(prefill_times)
        metrics["total_prefill_time"] = sum(prefill_times)
        metrics["prefill_throughput_requests_per_sec"] = len(prefill_times) / sum(prefill_times) if sum(prefill_times) > 0 else 0
    
    # Decode metrics
    decode_times = timing_data["decode_times"]
    if decode_times:
        metrics["avg_decode_time"] = sum(decode_times) / len(decode_times)
        metrics["max_decode_time"] = max(decode_times)
        metrics["min_decode_time"] = min(decode_times)
        metrics["total_decode_time"] = sum(decode_times)
        metrics["decode_throughput_tokens_per_sec"] = timing_data["total_tokens_generated"] / sum(decode_times) if sum(decode_times) > 0 else 0
    
    # Per-token decode metrics
    per_token_times = timing_data["per_token_decode_times"]
    if per_token_times:
        metrics["avg_per_token_decode_time"] = sum(per_token_times) / len(per_token_times)
        metrics["max_per_token_decode_time"] = max(per_token_times)
        metrics["min_per_token_decode_time"] = min(per_token_times)
        metrics["per_token_decode_throughput"] = len(per_token_times) / sum(per_token_times) if sum(per_token_times) > 0 else 0
    
    # Overall throughput
    metrics["overall_throughput_requests_per_sec"] = batch_size / timing_data["total_time"] if timing_data["total_time"] > 0 else 0
    metrics["overall_throughput_tokens_per_sec"] = timing_data["total_tokens_generated"] / timing_data["total_time"] if timing_data["total_time"] > 0 else 0
    
    # Efficiency ratios
    if prefill_times and decode_times and sum(decode_times) > 0:
        metrics["prefill_to_decode_ratio"] = sum(prefill_times) / sum(decode_times)
        metrics["prefill_percentage"] = (sum(prefill_times) / timing_data["total_time"]) * 100
        metrics["decode_percentage"] = (sum(decode_times) / timing_data["total_time"]) * 100
    elif prefill_times:
        metrics["prefill_percentage"] = (sum(prefill_times) / timing_data["total_time"]) * 100 if timing_data["total_time"] > 0 else 0
        metrics["decode_percentage"] = 0
        metrics["prefill_to_decode_ratio"] = float('inf')  # All prefill, no decode
    
    return metrics


def write_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Write benchmark results to CSV file."""
    if not results:
        return
    
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(list(fieldnames))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    
    # Prepare LLM configuration
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    
    # Add token parallel configs if enabled
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    # Initialize LLM
    llm = LLM(**llm_kwargs)
    
    if dist.get_rank() == 0:
        print("=" * 80)
        print("vLLM Prefill/Decode Benchmark")
        print("=" * 80)
        print(f"Model: {args.model}")
        print(f"Parallelism: TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, DP={args.data_parallel_size}")
        if args.enable_token_parallel:
            print(f"Token Parallel: {args.token_parallel_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Input length: {args.input_length} tokens")
        print(f"Output length: {args.output_length} tokens")
        print(f"Iterations: {args.num_iterations} (warmup: {args.warmup_iterations})")
        print("-" * 80)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.output_length
    )
    
    # Generate prompts
    prompts = generate_prompts_with_exact_length(args.batch_size, args.input_length)
    
    if dist.get_rank() == 0:
        print(f"Generated {len(prompts)} prompts")
        print(f"Sample prompt length: {len(prompts[0])} characters")
        print("-" * 80)
    
    # Warmup iterations
    if dist.get_rank() == 0:
        print("Running warmup iterations...")
    
    for warmup_iter in range(args.warmup_iterations):
        if dist.get_rank() == 0:
            print(f"  Warmup {warmup_iter + 1}/{args.warmup_iterations}")
        llm.generate(prompts, sampling_params)
    
    # Benchmark iterations
    if dist.get_rank() == 0:
        print("\nRunning benchmark iterations...")
    
    all_results = []
    timer = PrefillDecodeTimer(llm.llm_engine)
    
    for iter_num in range(args.num_iterations):
        if dist.get_rank() == 0:
            print(f"\n  Iteration {iter_num + 1}/{args.num_iterations}")
        
        # Reset timer state
        timer.prefill_times = []
        timer.decode_times = []
        timer.per_token_decode_times = []
        timer.total_tokens_generated = 0
        
        # Run benchmark with detailed timing
        timing_data = timer.benchmark_with_step_timing(prompts, sampling_params)
        
        # Calculate metrics
        metrics = calculate_detailed_metrics(timing_data, args.batch_size)
        
        if dist.get_rank() == 0:
            # Create result record
            result_record = {
                "iteration": iter_num + 1,
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "data_parallel_size": args.data_parallel_size,
                "token_parallel_size": args.token_parallel_size if args.enable_token_parallel else 1,
                "enable_token_parallel": args.enable_token_parallel,
                "batch_size": args.batch_size,
                "input_length": args.input_length,
                "output_length": args.output_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                **metrics
            }
            
            all_results.append(result_record)
            
            # Print iteration results
            print(f"    Total time: {metrics['total_time']:.4f}s")
            if 'avg_prefill_time' in metrics:
                print(f"    Avg prefill time: {metrics['avg_prefill_time']:.4f}s")
            if 'avg_decode_time' in metrics:
                print(f"    Avg decode time: {metrics['avg_decode_time']:.4f}s")
            if 'avg_per_token_decode_time' in metrics:
                print(f"    Avg per-token decode: {metrics['avg_per_token_decode_time']:.4f}s")
            print(f"    Tokens generated: {metrics['total_tokens_generated']}")
    
    # Calculate and display summary statistics
    if dist.get_rank() == 0 and all_results:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Calculate averages across iterations
        avg_total_time = sum(r["total_time"] for r in all_results) / len(all_results)
        avg_prefill_time = sum(r.get("avg_prefill_time", 0) for r in all_results) / len(all_results)
        avg_decode_time = sum(r.get("avg_decode_time", 0) for r in all_results) / len(all_results)
        avg_per_token_time = sum(r.get("avg_per_token_decode_time", 0) for r in all_results) / len(all_results)
        avg_total_tokens = sum(r["total_tokens_generated"] for r in all_results) / len(all_results)
        
        print(f"Average Results (across {len(all_results)} iterations):")
        print(f"  Total time: {avg_total_time:.4f}s")
        if avg_total_time > 0:
            print(f"  Prefill time: {avg_prefill_time:.4f}s ({avg_prefill_time/avg_total_time*100:.1f}%)")
            print(f"  Decode time: {avg_decode_time:.4f}s ({avg_decode_time/avg_total_time*100:.1f}%)")
        else:
            print(f"  Prefill time: {avg_prefill_time:.4f}s")
            print(f"  Decode time: {avg_decode_time:.4f}s")
        print(f"  Per-token decode: {avg_per_token_time:.4f}s")
        print(f"  Tokens generated: {avg_total_tokens:.1f}")
        print(f"  Overall throughput: {args.batch_size/avg_total_time:.2f} requests/sec")
        print(f"  Token throughput: {avg_total_tokens/avg_total_time:.2f} tokens/sec")
        
        # Write results to CSV
        # write_results_to_csv(all_results, args.output_file)
        # print(f"\nDetailed results saved to: {args.output_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()