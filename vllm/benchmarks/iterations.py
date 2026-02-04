# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Batch iteration benchmark for precise prefill/decode phase measurement.

On the server side, run:
    vllm serve <your_model> --profiler-config.profiler torch \
        --profiler-config.torch_profiler_dir /path/to/traces

On the client side, run:
    # Prefill benchmark: measure prefill of 8K new tokens (no existing context)
    vllm bench iterations \
        --endpoints 127.0.0.1:8000 \
        --input-len 8192 \
        --batch-size 1 \
        --mode prefill \
        --profile \
        --model <your_model>

    # Prefill benchmark: measure prefill of 2K new tokens against 4K existing context
    vllm bench iterations \
        --endpoints 127.0.0.1:8000 \
        --context-len 4096 \
        --input-len 2048 \
        --batch-size 1 \
        --mode prefill \
        --profile \
        --model <your_model>

    # Decode benchmark: warmup with 8K context, measure 128 decode iterations
    vllm bench iterations \
        --endpoints 127.0.0.1:8000 \
        --context-len 8192 \
        --batch-size 64 \
        --mode decode \
        --iterations 128 \
        --profile \
        --model <your_model>

This benchmark uses sleep(level=0) to pause scheduling, queues requests,
then resumes scheduling to measure precise batch execution times.

Modes:
    prefill: Measures prefill latency for (context_len + input_len) total tokens.
             context_len=0 is valid (clean prefill of new input only).
             Generates 1 output token to complete the prefill phase.

    decode:  Warms up KV cache with context_len tokens, then measures decode
             throughput for --iterations output tokens.
             context_len > 0 is REQUIRED (cannot decode without context).

NOTE: For accurate prefill benchmarks, do NOT use --enable-chunked-prefill on the
server. Chunked prefill breaks long prefills into multiple steps, which interferes
with measuring true prefill performance.
"""

import argparse
import asyncio
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass

import aiohttp

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the iterations benchmark."""

    endpoints: list[str]
    context_lens: list[int]
    input_lens: list[int]
    batch_sizes: list[int]
    mode: str  # "prefill" or "decode"
    iterations: int
    profile: bool
    model: str


@dataclass
class IterationResult:
    """Result of a single benchmark iteration."""

    endpoint: str
    mode: str
    context_len: int
    input_len: int
    batch_size: int
    iteration: int
    total_latency_ms: float
    latency_per_iter_ms: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int


class EndpointRotator:
    """Round-robin endpoint selection (matches disagg_benchmarks pattern)."""

    def __init__(self, endpoints: list[str]):
        self.endpoints = [self._normalize(e) for e in endpoints]
        self.cycle = itertools.cycle(self.endpoints)

    def _normalize(self, endpoint: str) -> str:
        """Ensure endpoint has http:// prefix."""
        if not endpoint.startswith(("http://", "https://")):
            return f"http://{endpoint}"
        return endpoint.rstrip("/")

    def next(self) -> str:
        return next(self.cycle)

    def all(self) -> list[str]:
        return self.endpoints


async def call_debug_endpoint(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    path: str,
    params: dict | None = None,
) -> bool:
    """Call debug endpoint on ALL endpoints (for sleep/wake_up/profile).

    Returns True if all calls succeeded, False if any failed.
    """
    tasks = [
        session.post(f"{endpoint}{path}", params=params) for endpoint in rotator.all()
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_success = True
    for endpoint, resp in zip(rotator.all(), responses):
        if isinstance(resp, Exception):
            logger.warning("Failed to call %s%s: %s", endpoint, path, resp)
            all_success = False
        else:
            body = await resp.read()
            if resp.status >= 400:
                logger.warning(
                    "HTTP %d from %s%s: %s",
                    resp.status,
                    endpoint,
                    path,
                    body.decode()[:200],
                )
                all_success = False
    return all_success


def build_prompt(context_len: int, input_len: int, mode: str) -> str:
    """Build prompt with specified context and input lengths.

    For prefill mode: total prompt = context_len + input_len tokens.
        Measures prefill of new input_len tokens against existing context.
    For decode mode: prompt = context_len tokens (KV cache warmup).
        Then generates iterations tokens to measure decode throughput.
    """
    # Use a simple repeating pattern to approximate token count
    # "hello " is roughly 1-2 tokens depending on tokenizer
    if mode == "prefill":
        # Total prompt = context (existing history) + input (new tokens to prefill)
        num_words = (context_len + input_len) // 2
    else:
        # Decode: only context for KV cache warmup
        num_words = context_len // 2

    return "hello " * max(1, num_words)


def count_tokens(response_data: dict) -> tuple[int, int]:
    """Extract token counts from completion response."""
    usage = response_data.get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def run_single_iteration(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    rotator: EndpointRotator,
    context_len: int,
    input_len: int,
    batch_size: int,
) -> tuple[float, int, int]:
    """Run one iteration: sleep → queue requests → wake → measure."""

    # 1. Pause scheduling on ALL endpoints
    await call_debug_endpoint(session, rotator, "/debug/sleep", {"level": "0"})

    # 2. Build requests and start sending them (they queue while server sleeps)
    # We use asyncio.ensure_future to actually start the requests immediately,
    # not just create coroutines. The requests will be sent to the server
    # and queue there while scheduling is paused.
    tasks = []
    for _ in range(batch_size):
        endpoint = rotator.next()
        prompt = build_prompt(context_len, input_len, config.mode)
        max_tokens = 1 if config.mode == "prefill" else config.iterations

        # ensure_future schedules the coroutine immediately
        task = asyncio.ensure_future(
            session.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": config.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        )
        tasks.append(task)

    # Small delay to ensure requests are queued on server
    await asyncio.sleep(0.1)

    # 3. Resume scheduling on ALL endpoints and time the batch
    start = time.perf_counter()
    await call_debug_endpoint(session, rotator, "/debug/wake_up")
    responses = await asyncio.gather(*tasks)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 4. Count tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for resp in responses:
        try:
            data = await resp.json()
            prompt_tokens, completion_tokens = count_tokens(data)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        except Exception as e:
            logger.warning("Failed to parse response: %s", e)

    return elapsed_ms, total_prompt_tokens, total_completion_tokens


async def fetch_traces(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    prefix: str,
    output_dir: str,
) -> list[str]:
    """Download trace files from all endpoints."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []

    for i, endpoint in enumerate(rotator.all()):
        try:
            resp = await session.get(f"{endpoint}/debug/traces")
            data = await resp.json()

            for trace_file in data.get("traces", []):
                if prefix in trace_file:
                    trace_resp = await session.get(
                        f"{endpoint}/debug/traces/{trace_file}"
                    )
                    local_path = os.path.join(output_dir, f"endpoint{i}_{trace_file}")
                    with open(local_path, "wb") as f:
                        f.write(await trace_resp.read())
                    logger.info("Downloaded: %s", local_path)
                    downloaded.append(local_path)
        except Exception as e:
            logger.warning("Failed to fetch traces from %s: %s", endpoint, e)

    return downloaded


async def run_benchmark(config: BenchmarkConfig) -> list[IterationResult]:
    """Main benchmark loop with parameter sweeping."""

    rotator = EndpointRotator(config.endpoints)
    results: list[IterationResult] = []

    connector = aiohttp.TCPConnector(
        limit=0,  # No limit
        ttl_dns_cache=300,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        # Start profiling if requested
        prefix = None
        profiling_started = False
        if config.profile:
            prefix = f"{config.mode}_{int(time.time())}"
            profiling_started = await call_debug_endpoint(
                session, rotator, "/debug/profile/start", {"prefix": prefix}
            )
            if profiling_started:
                logger.info("Started profiling with prefix: %s", prefix)
            else:
                logger.warning(
                    "Profiling failed to start. Ensure server has "
                    "--profiler-config enabled. Continuing without profiling."
                )

        # Sweep all parameter combinations
        param_combos = list(
            itertools.product(
                config.context_lens,
                config.input_lens,
                config.batch_sizes,
            )
        )

        logger.info(
            "Running %d parameter combinations",
            len(param_combos),
        )

        # For prefill: 1 output token. For decode: config.iterations tokens.
        num_output_tokens = 1 if config.mode == "prefill" else config.iterations

        for ctx_len, in_len, batch_size in param_combos:
            logger.info(
                "Running: mode=%s, ctx=%d, input=%d, batch=%d, output_tokens=%d",
                config.mode,
                ctx_len,
                in_len,
                batch_size,
                num_output_tokens,
            )

            (
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
            ) = await run_single_iteration(
                session,
                config,
                rotator,
                ctx_len,
                in_len,
                batch_size,
            )

            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = (
                total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            )
            latency_per_iter = (
                elapsed_ms / num_output_tokens if num_output_tokens > 0 else elapsed_ms
            )

            results.append(
                IterationResult(
                    endpoint=",".join(rotator.all()),
                    mode=config.mode,
                    context_len=ctx_len,
                    input_len=in_len,
                    batch_size=batch_size,
                    iteration=num_output_tokens,
                    total_latency_ms=elapsed_ms,
                    latency_per_iter_ms=latency_per_iter,
                    tokens_per_second=tokens_per_second,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

            logger.info(
                "  Result: %.2fms total, %.2fms/iter, %.2f tok/s",
                elapsed_ms,
                latency_per_iter,
                tokens_per_second,
            )

        # Stop profiling and fetch traces (only if profiling actually started)
        if profiling_started and prefix:
            await call_debug_endpoint(session, rotator, "/debug/profile/stop")
            logger.info("Stopped profiling")
            await fetch_traces(session, rotator, prefix, "traces")

    return results


def print_results_summary(results: list[IterationResult]) -> None:
    """Print a summary of benchmark results."""
    if not results:
        logger.warning("No results to summarize")
        return

    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 110)
    print(
        f"{'Mode':>8} {'Context':>8} {'Input':>8} {'Batch':>6} {'Iters':>6} "
        f"{'Total (ms)':>12} {'Per-Iter (ms)':>14} {'Tokens/s':>12}"
    )
    print("-" * 110)

    for r in results:
        print(
            f"{r.mode:>8} {r.context_len:>8} {r.input_len:>8} {r.batch_size:>6} "
            f"{r.iteration:>6} {r.total_latency_ms:>12.2f} "
            f"{r.latency_per_iter_ms:>14.2f} {r.tokens_per_second:>12.2f}"
        )

    print("=" * 110 + "\n")


def write_results_json(results: list[IterationResult], output_path: str) -> None:
    """Write results to JSON file."""
    data = {
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results written to: %s", output_path)


def parse_comma_list(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in value.split(",")]


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the iterations benchmark."""
    parser.add_argument(
        "--endpoints",
        type=str,
        required=True,
        help="Comma-separated list of endpoints (e.g., host1:8000,host2:8000)",
    )
    parser.add_argument(
        "--context-len",
        type=str,
        default="0",
        help="Prompt tokens to prefill KV cache before decode (decode mode only, "
        "comma-separated for sweep, e.g., 512,1024,2048)",
    )
    parser.add_argument(
        "--input-len",
        type=str,
        default="128",
        help="Prompt tokens to measure prefill latency (prefill mode only, "
        "comma-separated for sweep, e.g., 128,256,512)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="Batch sizes, comma-separated for sweep (e.g., 1,4,8)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefill", "decode"],
        required=True,
        help="Benchmark mode: prefill or decode",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=128,
        help="Number of decode tokens to generate (decode mode only, default: 128)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable GPU profiling and download traces",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for requests",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser(
        description="Batch iteration benchmark for prefill/decode phases"
    )
    add_cli_args(parser)

    if args is None:
        args = parser.parse_args()

    # For decode mode, input_len is not used (only context matters)
    input_lens = [1] if args.mode == "decode" else parse_comma_list(args.input_len)

    config = BenchmarkConfig(
        endpoints=args.endpoints.split(","),
        context_lens=parse_comma_list(args.context_len),
        input_lens=input_lens,
        batch_sizes=parse_comma_list(args.batch_size),
        mode=args.mode,
        iterations=args.iterations,
        profile=args.profile,
        model=args.model,
    )

    # Validate config based on mode
    if config.mode == "decode" and any(ctx <= 0 for ctx in config.context_lens):
        parser.error(
            "Decode mode requires --context-len > 0 for KV cache warmup. "
            "Cannot decode without existing context."
        )
    if config.mode == "prefill" and any(inp <= 0 for inp in config.input_lens):
        parser.error(
            "Prefill mode requires --input-len > 0. "
            "Need tokens to measure prefill latency."
        )

    logger.info("Starting benchmark with config: %s", config)
    results = asyncio.run(run_benchmark(config))
    print_results_summary(results)

    if args.output:
        write_results_json(results, args.output)


if __name__ == "__main__":
    main()
