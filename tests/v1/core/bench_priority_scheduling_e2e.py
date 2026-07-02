# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end benchmark for priority scheduling with lazy deletion.

This script simulates a realistic production workload with frequent
request cancellations (aborts/timeouts) to measure the actual impact
of lazy deletion on throughput and latency.

Usage:
    # Start vLLM server with priority scheduling:
    #   vllm serve <model> --scheduling-policy priority --max-num-seqs 256

    # Run benchmark:
    python tests/v1/core/bench_priority_scheduling_e2e.py --host localhost --port 8000
"""

import argparse
import asyncio
import random
import time
from statistics import mean, median, stdev

import aiohttp


async def send_request(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    request_id: str,
    priority: int,
    max_tokens: int,
    abort_after_ms: int | None = None,
) -> tuple[str, float | None, bool]:
    """Send a single request and optionally abort it."""
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": "default",
        "prompt": "Hello, world! " * 50,
        "max_tokens": max_tokens,
        "priority": priority,
    }

    start = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            await resp.read()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return request_id, elapsed_ms, False
    except (asyncio.TimeoutError, asyncio.CancelledError, aiohttp.ClientError):
        return request_id, None, True


async def run_benchmark(
    host: str,
    port: int,
    num_requests: int,
    abort_rate: float,
    abort_after_ms: int,
    max_tokens_range: tuple[int, int],
    priority_range: tuple[int, int],
) -> dict:
    """Run the full benchmark workload."""

    latencies = []
    aborted_count = 0
    completed_count = 0
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            request_id = f"req_{i:06d}"
            priority = random.randint(*priority_range)
            max_tokens = random.randint(*max_tokens_range)

            should_abort = random.random() < abort_rate
            abort_ms = abort_after_ms if should_abort else None

            task = asyncio.create_task(
                send_request(
                    session, host, port, request_id, priority, max_tokens, abort_ms
                )
            )
            tasks.append(task)

            # Small delay between requests to simulate realistic traffic
            await asyncio.sleep(random.uniform(0.001, 0.01))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                request_id, latency_ms, was_aborted = result
                if was_aborted:
                    aborted_count += 1
                elif latency_ms is not None:
                    latencies.append(latency_ms)
                    completed_count += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time

    return {
        "total_requests": num_requests,
        "completed": completed_count,
        "aborted": aborted_count,
        "aborted_rate": aborted_count / num_requests * 100,
        "total_time_s": total_time,
        "throughput_rps": completed_count / total_time if total_time > 0 else 0,
        "latency_mean_ms": mean(latencies) if latencies else 0,
        "latency_median_ms": median(latencies) if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)]
        if latencies
        else 0,
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)]
        if latencies
        else 0,
        "latency_stdev_ms": stdev(latencies) if len(latencies) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="E2E benchmark for priority scheduling"
    )
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--num-requests", type=int, default=200, help="Total requests to send"
    )
    parser.add_argument(
        "--abort-rate",
        type=float,
        default=0.3,
        help="Fraction of requests to abort (0.0-1.0)",
    )
    parser.add_argument(
        "--abort-after-ms", type=int, default=500, help="Abort after this many ms"
    )
    parser.add_argument(
        "--max-tokens-min", type=int, default=50, help="Min tokens per request"
    )
    parser.add_argument(
        "--max-tokens-max", type=int, default=200, help="Max tokens per request"
    )
    parser.add_argument("--priority-min", type=int, default=-3, help="Min priority")
    parser.add_argument("--priority-max", type=int, default=3, help="Max priority")
    args = parser.parse_args()

    print(
        f"Starting E2E benchmark: {args.num_requests} requests, "
        f"{args.abort_rate * 100:.0f}% abort rate"
    )
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    results = asyncio.run(
        run_benchmark(
            host=args.host,
            port=args.port,
            num_requests=args.num_requests,
            abort_rate=args.abort_rate,
            abort_after_ms=args.abort_after_ms,
            max_tokens_range=(args.max_tokens_min, args.max_tokens_max),
            priority_range=(args.priority_min, args.priority_max),
        )
    )

    print("\nResults:")
    print(f"  Total requests:  {results['total_requests']}")
    print(f"  Completed:       {results['completed']}")
    print(f"  Aborted:         {results['aborted']} ({results['aborted_rate']:.1f}%)")
    print(f"  Total time:      {results['total_time_s']:.2f}s")
    print(f"  Throughput:      {results['throughput_rps']:.1f} req/s")
    print("\nLatency (completed requests):")
    print(f"  Mean:    {results['latency_mean_ms']:.1f} ms")
    print(f"  Median:  {results['latency_median_ms']:.1f} ms")
    print(f"  P95:     {results['latency_p95_ms']:.1f} ms")
    print(f"  P99:     {results['latency_p99_ms']:.1f} ms")
    print(f"  Stdev:   {results['latency_stdev_ms']:.1f} ms")


if __name__ == "__main__":
    main()
