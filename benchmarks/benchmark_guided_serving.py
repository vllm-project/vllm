"""Benchmark guided decoding server throughput.

On the server side, run:
    vllm serve <your_model> --guided-decoding-backend xgrammar

On the client side, run:
    python benchmark_guided_serving.py \
        --model <your_model> \
        --dataset <dataset_type> \
        --request-rate <request_rate> \
        --num-prompts <num_prompts>
"""
import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                RequestFuncOutput, get_tokenizer)

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking."""
    prompt: str
    prompt_len: int
    expected_output_len: int
    schema: dict
    structure_type: str = 'json'
    completion: str = None


async def get_request(
    input_requests: List[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """Generate requests at specified rate with optional burstiness."""
    input_requests = iter(input_requests)
    assert burstiness > 0, f"Expected positive burstiness, got {burstiness}"
    
    theta = 1.0 / (request_rate * burstiness) if request_rate != float("inf") else 0

    for request in input_requests:
        yield request
        
        if request_rate == float("inf"):
            continue
            
        # Sample interval from gamma distribution
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


def sample_requests(
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace
) -> List[SampleRequest]:
    """Sample requests based on the specified dataset type."""
    if args.dataset == 'xgrammar_bench':
        import datasets
        dataset = datasets.load_dataset("NousResearch/json-mode-eval", split="train")
        print(f"Loaded xgrammar dataset with {len(dataset)} entries")
        
        requests = []
        len_dataset = len(dataset)
        for data_point_idx in range(args.num_prompts):
            idx = data_point_idx
            while idx >= len_dataset:
                idx -= len_dataset
                
            schema = dataset["schema"][idx]
            prompt = tokenizer.apply_chat_template(dataset["prompt"][idx], tokenize=False)
            input_len = len(tokenizer(prompt).input_ids)
            completion = dataset["completion"][idx]
            
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=args.output_len,
                    schema=schema,
                    structure_type='json',
                    completion=completion
                )
            )
        return requests
        
    elif args.dataset == 'json':
        if args.json_schema_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            args.json_schema_path = os.path.join(
                dir_path, "structured_schemas", "structured_schema_1.json")
        with open(args.json_schema_path) as f:
            schema = json.load(f)
        prompt = f"Generate an example of a user profile given the following schema: {json.dumps(schema)}"
        input_len = len(tokenizer(prompt).input_ids)
        return [
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=schema,
                structure_type='json'
            ) for _ in range(args.num_prompts)
        ]
    
    elif args.dataset == "regex":
        regex = r"\w+@\w+\.com\n"
        prompt = "Generate an email address for Alan Turing, who works in Enigma. End in .com and new line."
        input_len = len(tokenizer(prompt).input_ids)
        return [
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=regex,
                structure_type='regex'
            ) for _ in range(args.num_prompts)
        ]
    
    elif args.dataset == "choice":
        choices = ["Positive", "Negative"]
        prompt = "Classify this sentiment: vLLM is wonderful!"
        input_len = len(tokenizer(prompt).input_ids)
        return [
            SampleRequest(
                prompt=prompt,
                prompt_len=input_len,
                expected_output_len=args.output_len,
                schema=choices,
                structure_type='choice'
            ) for _ in range(args.num_prompts)
        ]
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")


def calculate_metrics(
    requests: List[SampleRequest],
    results: List[RequestFuncOutput],
    duration: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Calculate benchmark metrics from results."""
    completed = sum(1 for r in results if r.success)
    total_input_tokens = sum(req.prompt_len for req in requests)
    total_output_tokens = sum(
        len(tokenizer(r.generated_text).input_ids) 
        for r in results if r.success
    )
    
    # Collect latency metrics
    ttfts = [r.ttft * 1000 for r in results if r.success]  # Convert to ms
    latencies = [r.latency * 1000 for r in results if r.success]  # Convert to ms
    itls = []
    for r in results:
        if r.success and r.itl:
            itls.extend([l * 1000 for l in r.itl])  # Convert to ms

    metrics = {
        "completed_requests": completed,
        "total_requests": len(requests),
        "completion_rate": completed / len(requests) * 100,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": completed / duration,
        "token_throughput": (total_input_tokens + total_output_tokens) / duration,
        "output_throughput": total_output_tokens / duration,
        "e2e_latency_stats": {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p90": np.percentile(latencies, 90),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        },
        "ttft_stats": {
            "mean": np.mean(ttfts),
            "median": np.median(ttfts),
            "p90": np.percentile(ttfts, 90),
            "p95": np.percentile(ttfts, 95),
            "p99": np.percentile(ttfts, 99),
        },
        "itl_stats": {
            "mean": np.mean(itls) if itls else 0,
            "median": np.median(itls) if itls else 0,
            "p90": np.percentile(itls, 90) if itls else 0,
            "p95": np.percentile(itls, 95) if itls else 0,
            "p99": np.percentile(itls, 99) if itls else 0,
        }
    }
    
    return metrics


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the benchmark and return results."""
    tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(tokenizer, args)
    
    # Construct API URL
    if args.base_url:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    # Get the appropriate request function
    if args.backend not in ASYNC_REQUEST_FUNCS:
        raise ValueError(f"Unknown backend: {args.backend}")
    request_func = ASYNC_REQUEST_FUNCS[args.backend]

    # Run warmup if enabled
    if args.warmup:
        print("Running warmup requests...")
        warmup_request = requests[0]
        warmup_input = RequestFuncInput(
            model=args.model,
            prompt=warmup_request.prompt,
            api_url=api_url,
            prompt_len=warmup_request.prompt_len,
            output_len=warmup_request.expected_output_len,
            ignore_eos=args.ignore_eos,
        )
        await request_func(warmup_input)
    
    print(f"Starting benchmark with {len(requests)} requests...")
    print(f"Request rate: {args.request_rate} requests/second")
    print(f"Guided decoding ratio: {args.guided_decoding_ratio}")
    
    # Set up progress bar
    pbar = None if args.disable_tqdm else tqdm(total=len(requests))
    
    # Create semaphore for concurrency control if specified
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    
    async def process_request(request: SampleRequest) -> RequestFuncOutput:
        # Prepare guided decoding parameters based on type
        extra_body = {}
        # Randomly decide whether to use guided decoding based on ratio
        if random.random() < args.guided_decoding_ratio:
            if request.structure_type == 'json':
                extra_body["guided_json"] = request.schema
            elif request.structure_type == 'regex':
                extra_body["guided_regex"] = request.schema
            elif request.structure_type == 'choice':
                extra_body["guided_choice"] = request.schema
            
            if args.guided_decoding_backend:
                extra_body["guided_decoding_backend"] = args.guided_decoding_backend

        request_input = RequestFuncInput(
            model=args.model,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            ignore_eos=args.ignore_eos,
            extra_body=extra_body,
        )
        
        if semaphore:
            async with semaphore:
                return await request_func(request_input, pbar)
        return await request_func(request_input, pbar)

    # Run the benchmark
    start_time = time.perf_counter()
    tasks = []
    async for request in get_request(requests, args.request_rate, args.burstiness):
        tasks.append(asyncio.create_task(process_request(request)))
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    if pbar:
        pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(requests, results, duration, tokenizer)
    
    # Add raw results for detailed analysis
    metrics["raw_results"] = [
        {
            "success": r.success,
            "latency": r.latency,
            "ttft": r.ttft,
            "generated_text": r.generated_text,
            "error": r.error
        }
        for r in results
    ]
    
    return metrics


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark guided decoding server throughput.")
    
    # Server connection arguments
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    
    # Model and tokenizer arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="xgrammar_bench",
        choices=["json", "regex", "choice", "xgrammar_bench"],
        help="Type of dataset to use for benchmarking"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. Use inf for maximum rate"
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request generation (default: 1.0 for Poisson process)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests"
    )
    parser.add_argument(
        "--json-schema-path",
        type=str,
        default=None,
        help="Path to JSON schema file for json dataset"
    )
    parser.add_argument(
        "--guided-decoding-ratio",
        type=float,
        default=1.0,
        help="Ratio of Guided Decoding requests"
    )
    parser.add_argument(
        "--guided-decoding-backend",
        type=str,
        choices=["xgrammar", "outlines"],
        default="xgrammar",
        help="Backend to use for guided decoding"
    )
    parser.add_argument(
        "--ignore-eos",
        type=bool,
        default=True,
        help="Set ignore_eos flag when sending requests"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run warmup requests before benchmark"
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to JSON file"
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Path to save benchmark results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run benchmark
    metrics = asyncio.run(run_benchmark(args))

    # Print results
    print("\n" + "="*50)
    print("Benchmark Results:")
    print("="*50)
    print(f"Completed requests: {metrics['completed_requests']}/{metrics['total_requests']} "
          f"({metrics['completion_rate']:.2f}%)")
    print(f"Request throughput: {metrics['request_throughput']:.2f} requests/s")
    print(f"Total Token throughput: {metrics['token_throughput']:.2f} tokens/s")
    print(f"Output Token throughput: {metrics['token_throughput']:.2f} tokens/s")
    print("\nE2E Latency Statistics (ms):")
    print(f"  Mean: {metrics['e2e_latency_stats']['mean']:.2f}")
    print(f"  Median: {metrics['e2e_latency_stats']['median']:.2f}")
    print(f"  P90: {metrics['e2e_latency_stats']['p90']:.2f}")
    print(f"  P95: {metrics['e2e_latency_stats']['p95']:.2f}")
    print(f"  P99: {metrics['e2e_latency_stats']['p99']:.2f}")
    print("\nTime to First Token (ms):")
    print(f"  Mean: {metrics['ttft_stats']['mean']:.2f}")
    print(f"  Median: {metrics['ttft_stats']['median']:.2f}")
    print(f"  P90: {metrics['ttft_stats']['p90']:.2f}")
    print(f"  P95: {metrics['ttft_stats']['p95']:.2f}")
    print(f"  P99: {metrics['ttft_stats']['p99']:.2f}")
    print("\nInter-token Latency (ms):")
    print(f"  Mean: {metrics['itl_stats']['mean']:.2f}")
    print(f"  Median: {metrics['itl_stats']['median']:.2f}")
    print(f"  P90: {metrics['itl_stats']['p90']:.2f}")
    print(f"  P95: {metrics['itl_stats']['p95']:.2f}")
    print(f"  P99: {metrics['itl_stats']['p99']:.2f}")
    print("="*50)

    # Save results if requested
    if args.save_result:
        result_file = args.result_file
        if result_file is None:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = args.model.split("/")[-1]
            result_file = (f"benchmark_guided_{model_name}_{args.dataset}_"
                         f"{args.request_rate}qps_{timestamp}.json")
        
        # Add benchmark configuration to results
        results = {
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "num_prompts": args.num_prompts,
                "request_rate": args.request_rate,
                "burstiness": args.burstiness,
                "max_concurrency": args.max_concurrency,
                "guided_decoding_ratio": args.guided_decoding_ratio,
                "output_len": args.output_len,
            },
            "metrics": metrics
        }
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()