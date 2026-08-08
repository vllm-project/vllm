#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark to measure GPU idle caused by late KV prefetch start.

This reproduces the scenario from https://github.com/vllm-project/vllm/issues/41784:
  - Several requests sit in the waiting queue while the running queue fills
    the token budget
  - Without the fix, the KV connector only learns about waiting requests
    when the scheduler polls the queue - which can be many iterations later
  - This shows up as "GPU idle (KV connector stall)" in the logs

Usage (requires LMCache with a disk backend configured):

    # Run the server first (terminal 1):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --kv-transfer-config \
            '{"kv_connector":"LMCacheConnector","kv_role":"kv_both",\
"kv_connector_extra_config":{"lmcache_config_file":"lmcache.yaml"}}' \
        --max-num-seqs 8 \
        --disable-log-requests

    # Then run this script (terminal 2):
    python benchmarks/kv_prefetch_stall_bench.py

What to look at in the server logs:
    - "GPU idle (KV connector stall): NNN ms" lines
    - Sum them up for before vs after to see the improvement
"""

import argparse
import asyncio
import time

import aiohttp

# A prompt long enough to have meaningful KV cache but not so long it
# dominates the measurement. 512 tokens of context is a reasonable stand-in
# for a typical RAG or chat history scenario.
PROMPT = (
    "The following is a detailed technical discussion about transformer models "
    "and their memory requirements during inference. " * 20
    + "\n\nQuestion: What is the main bottleneck in serving large language models?"
)


async def send_request(session, url, prompt, request_id):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        data = await resp.json()
    ttft = time.perf_counter() - t0
    return request_id, ttft, data


async def run_burst(url, num_requests, concurrency):
    """
    Fire num_requests in bursts of `concurrency` at a time.

    The burst pattern maximises the chance that the running queue is full
    when the new requests arrive, which is exactly when the prefetch delay
    hurts most.
    """
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm up - these will miss the KV cache and build it
        print("Warming up cache (first pass)...")
        warmup_tasks = [
            send_request(session, url, PROMPT, f"warmup-{i}")
            for i in range(num_requests)
        ]
        await asyncio.gather(*warmup_tasks)
        print("Warmup done. Cache should be populated.")

        # Now do the real measurement - these should be cache hits
        # but the prefetch delay determines whether the GPU stalls
        print(f"\nMeasurement run: {num_requests} concurrent requests "
              f"(all cache hits)...")
        t_start = time.perf_counter()
        tasks = [
            send_request(session, url, PROMPT, f"req-{i}")
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_start

    return results, wall_time


def print_results(results, wall_time):
    ttfts = [r[1] for r in results]
    ttfts.sort()
    n = len(ttfts)

    print("\n--- Results ---")
    print(f"  Requests:         {n}")
    print(f"  Wall time:        {wall_time:.3f}s")
    print(f"  Median TTFT:      {ttfts[n // 2] * 1000:.1f}ms")
    print(f"  p90 TTFT:         {ttfts[int(n * 0.9)] * 1000:.1f}ms")
    print(f"  p99 TTFT:         {ttfts[int(n * 0.99)] * 1000:.1f}ms")
    print(f"  Max TTFT:         {ttfts[-1] * 1000:.1f}ms")
    print(
        "\nAlso check server logs for 'GPU idle (KV connector stall)' lines."
        "\nSum of those values is the cumulative wasted GPU time."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Measure GPU idle from late KV prefetch (issue #41784)"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server base URL"
    )
    parser.add_argument(
        "--num-requests", type=int, default=16, help="Total requests to send"
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Max concurrent requests"
    )
    args = parser.parse_args()

    results, wall_time = asyncio.run(
        run_burst(args.url, args.num_requests, args.concurrency)
    )
    print_results(results, wall_time)


if __name__ == "__main__":
    main()
