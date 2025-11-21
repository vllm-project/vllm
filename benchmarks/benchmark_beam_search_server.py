#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple benchmark script to test beam search performance via OpenAI API.

This script sends beam search requests to a running vLLM server and handles
profiling via the /start_profile and /stop_profile endpoints.

Usage:
    # Start server with profiling enabled:
    # VLLM_TORCH_PROFILER_DIR=./vllm_profile vllm serve Qwen/Qwen3-8B --max-logprobs 100
    
    # Run this script:
    python benchmarks/benchmark_beam_search_server.py \    
        --base-url http://localhost:8000 \
        --beam-width 30 \
        --max-tokens 8 \
        --input-len 128 \
        --num-requests 10 --profile
"""

import argparse
import random
import time

import requests
from openai import OpenAI
from tqdm import tqdm


def start_profile(base_url: str) -> bool:
    """Start profiling on the server."""
    try:
        response = requests.post(f"{base_url}/start_profile", timeout=30)
        response.raise_for_status()
        print("✓ Profiling started")
        return True
    except Exception as e:
        print(f"✗ Failed to start profiling: {e}")
        return False


def stop_profile(base_url: str) -> bool:
    """Stop profiling on the server."""
    try:
        response = requests.post(f"{base_url}/stop_profile", timeout=30)
        response.raise_for_status()
        print("✓ Profiling stopped")
        return True
    except Exception as e:
        print(f"✗ Failed to stop profiling: {e}")
        return False


def send_beam_search_request(
    client: OpenAI,
    model: str,
    prompt: str,
    beam_width: int = 30,
    max_tokens: int = 8,
    temperature: float = 0.0,
) -> dict | None:
    """Send a beam search completion request."""
    start_time = time.perf_counter()
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        n=beam_width,  # Beam width
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"use_beam_search": True},  # This triggers beam search
    )
    end_time = time.perf_counter()

    latency = end_time - start_time

    # Convert completion object to dict-like structure for compatibility
    num_choices = len(completion.choices) if completion.choices else 0

    return {
        "latency": latency,
        "result": completion,
        "num_choices": num_choices,
    }


def generate_prompt(input_len: int, seed: int | None = None) -> str:
    """Generate a random prompt of approximately the specified length.

    Each call generates a unique prompt using random numbers to ensure
    disjoint prompts across requests.
    """
    if seed is not None:
        random.seed(seed)

    # Approximate numbers per token (numbers are typically 1 token each)
    num_numbers = int(input_len * 0.9)  # Slightly less to account for spaces

    # Generate random numbers
    numbers = [str(random.randint(0, 999999)) for _ in range(num_numbers)]

    prompt = " ".join(numbers)

    # Reset seed if it was set to avoid affecting other random calls
    if seed is not None:
        random.seed()

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark beam search performance via OpenAI API"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=30,
        help="Beam width (n parameter) (default: 30)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Maximum output tokens (default: 8)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Approximate input prompt length in tokens (default: 128)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to send (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for beam search (default: 1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (if not provided, generates dummy prompt)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Add profiling (call start/stop_profile endpoints)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (if not provided, will query server for available models)",
    )

    args = parser.parse_args()

    print("Beam Search Server Benchmark")
    print(f"Server URL: {args.base_url}")
    print(f"Beam width (n): {args.beam_width}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Input length: ~{args.input_len} tokens")
    print(f"Number of requests: {args.num_requests}")
    print(f"Temperature: {args.temperature}")

    # Initialize OpenAI client
    openai_api_base = f"{args.base_url}/v1"
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require a real API key
        base_url=openai_api_base,
    )

    # Get model name
    if args.model:
        model_name = args.model
        print(f"Model: {model_name}")
    else:
        print("\nQuerying server for available models...")
        try:
            models = client.models.list()
            if models.data:
                model_name = models.data[0].id
                print(f"Model: {model_name}")
            else:
                print("✗ No models available. Please specify --model")
                return
        except Exception as e:
            print(f"✗ Failed to query models: {e}")
            print("Please specify --model explicitly")
            return

    # Send test request with n=3
    test_prompt = (
        args.prompt if args.prompt else generate_prompt(args.input_len, seed=None)
    )
    print("\nSending test request (n=3)...", end=" ", flush=True)
    test_result = send_beam_search_request(
        client=client,
        model=model_name,
        prompt=test_prompt,
        beam_width=3,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if test_result:
        print(f"✓ Test request successful (latency: {test_result['latency']:.4f}s)")
    else:
        print("✗ Test request failed")
        print("Aborting benchmark due to test request failure.")
        return

    # Start profiling
    if args.profile:
        print("\nStarting profiler...")
        if not start_profile(args.base_url):
            print("Warning: Could not start profiling. Continuing anyway...")
        time.sleep(1)  # Brief pause to ensure profiler is ready

    print(f"\nSending {args.num_requests} requests...")
    latencies = []
    successful_requests = 0
    for _ in tqdm(range(args.num_requests)):
        # Generate a unique random prompt for each request
        prompt = args.prompt or generate_prompt(args.input_len, seed=None)
        result = send_beam_search_request(
            client,
            model_name,
            prompt,
            args.beam_width,
            args.max_tokens,
            args.temperature,
        )

        if result:
            latency = result["latency"]
            latencies.append(latency)
            successful_requests += 1
        else:
            print("✗ Failed request")

    # Stop profiling
    if args.profile:
        print("\nStopping profiler...")
        stop_profile(args.base_url)

    # Print statistics
    if latencies:
        import statistics

        print("\nResults")
        print(f"Successful requests: {successful_requests}/{args.num_requests}")
        print(f"Average latency: {statistics.mean(latencies) * 1000:.2f}ms")
        print(f"Median latency: {statistics.median(latencies) * 1000:.2f}ms")
        if len(latencies) > 1:
            print(f"Std deviation: {statistics.stdev(latencies) * 1000:.2f}ms")
        print(f"Min latency: {min(latencies) * 1000:.2f}ms")
        print(f"Max latency: {max(latencies) * 1000:.2f}ms")
        print("=" * 80)
    else:
        print("\n✗ No successful requests. Check server logs for errors.")


if __name__ == "__main__":
    main()
