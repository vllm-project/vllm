#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstration script for prefix-aware routing in vLLM.

This script shows how to use prefix-aware routing to improve cache hit rates
when serving requests with repeated prefixes.

Prerequisites:
    Start a vLLM server with prefix-aware routing enabled:

    vllm serve meta-llama/Llama-3.2-3B-Instruct \
        --data-parallel-size 2 \
        --enable-prefix-aware-routing \
        --prefix-routing-length 16

Then run this script:
    python prefix_routing_demo.py

The script will send requests with repeated prefixes and demonstrate
how prefix-aware routing ensures requests with the same prefix go to
the same engine for better cache utilization.
"""

import asyncio
import time
from typing import List

from openai import AsyncOpenAI

# Configuration
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Prefixes that will be repeated in requests
CACHED_PREFIXES = [
    "Once upon a time in a magical kingdom",
    "In the year 2050, scientists discovered",
    "The ancient manuscript revealed that",
]

# Unique prefixes for demonstrating load balancing
UNIQUE_PREFIXES = [
    f"This is unique request number {i}" for i in range(10)
]


async def send_request(client: AsyncOpenAI, prompt: str,
                       request_id: int) -> float:
    """Send a completion request and return the latency."""
    start_time = time.time()
    try:
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=20,
            temperature=0.7,
        )
        latency = time.time() - start_time
        print(f"  Request {request_id:3d}: {latency:.3f}s - "
              f"{response.choices[0].text[:50]}...")
        return latency
    except Exception as e:
        print(f"  Request {request_id:3d} failed: {e}")
        return -1


async def demo_cached_prefixes(client: AsyncOpenAI):
    """Demonstrate routing requests with same prefix to same engine."""
    print("\n" + "=" * 80)
    print("DEMO 1: Requests with Same Prefix (Should Hit Cache)")
    print("=" * 80)

    for prefix_idx, base_prefix in enumerate(CACHED_PREFIXES):
        print(f"\nSending 5 requests with prefix {prefix_idx + 1}: "
              f"\"{base_prefix[:40]}...\"")

        tasks = []
        for i in range(5):
            prompt = f"{base_prefix}, there was a story about {i}"
            tasks.append(
                send_request(client, prompt, prefix_idx * 5 + i + 1))

        latencies = await asyncio.gather(*tasks)
        valid_latencies = [l for l in latencies if l > 0]

        if valid_latencies:
            avg_latency = sum(valid_latencies) / len(valid_latencies)
            print(f"  Average latency: {avg_latency:.3f}s")


async def demo_load_balancing(client: AsyncOpenAI):
    """Demonstrate load balancing across engines for new prefixes."""
    print("\n" + "=" * 80)
    print("DEMO 2: Load Balancing New Prefixes Across Engines")
    print("=" * 80)
    print("\nSending requests with different prefixes (should balance)")

    tasks = []
    for i, prefix in enumerate(UNIQUE_PREFIXES):
        prompt = f"{prefix} and something interesting happened"
        tasks.append(send_request(client, prompt, i + 1))

    latencies = await asyncio.gather(*tasks)
    valid_latencies = [l for l in latencies if l > 0]

    if valid_latencies:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        print(f"\n  Average latency for new prefixes: {avg_latency:.3f}s")


async def demo_mixed_workload(client: AsyncOpenAI):
    """Demonstrate mixed workload with cached and new prefixes."""
    print("\n" + "=" * 80)
    print("DEMO 3: Mixed Workload (Cached + New Prefixes)")
    print("=" * 80)

    tasks = []
    request_id = 1

    # Add cached prefix requests
    print("\nSending 3 requests per cached prefix...")
    for base_prefix in CACHED_PREFIXES:
        for i in range(3):
            prompt = f"{base_prefix}, chapter {i + 1} begins"
            tasks.append(send_request(client, prompt, request_id))
            request_id += 1

    # Interleave with some unique prefix requests
    print("Interleaving unique prefix requests...")
    for i in range(5):
        prompt = f"Brand new prefix {i} for testing"
        tasks.append(send_request(client, prompt, request_id))
        request_id += 1

    latencies = await asyncio.gather(*tasks)
    valid_latencies = [l for l in latencies if l > 0]

    if valid_latencies:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        print(f"\n  Average latency for mixed workload: {avg_latency:.3f}s")


async def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("Prefix-Aware Routing Demo")
    print("=" * 80)
    print(f"\nConnecting to vLLM server at {API_BASE}")
    print(f"Model: {MODEL_NAME}")

    client = AsyncOpenAI(api_key="EMPTY", base_url=API_BASE)

    try:
        # Run demonstrations
        await demo_cached_prefixes(client)
        await asyncio.sleep(1)

        await demo_load_balancing(client)
        await asyncio.sleep(1)

        await demo_mixed_workload(client)

        print("\n" + "=" * 80)
        print("Demo completed!")
        print("=" * 80)
        print("\nKey observations:")
        print("1. Requests with the same prefix are routed to the same engine")
        print("2. This improves prefix cache hit rates and reduces latency")
        print("3. New prefixes are load-balanced across available engines")
        print("4. Mixed workloads benefit from both caching and load balancing")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the vLLM server is running with:")
        print("  vllm serve <model> --data-parallel-size 2 "
              "--enable-prefix-aware-routing")


if __name__ == "__main__":
    asyncio.run(main())
