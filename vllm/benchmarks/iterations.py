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

Prefix Cache Warmup:
    Before each benchmark run, the client sends warmup requests with context_len
    tokens to populate the prefix cache. The benchmark requests share this prefix,
    so the server can skip prefilling the context portion (prefix cache hit).

Modes:
    prefill: First warms up prefix cache with context_len tokens.
             Then measures prefill of input_len NEW tokens against existing context.
             Total prompt = context_len + input_len tokens.
             context_len=0 is valid (clean prefill of new input only).

    decode:  First warms up prefix cache with context_len tokens.
             Then measures decode throughput for --iterations output tokens.
             The benchmark prompt matches the warmup (full prefix cache hit),
             so we measure ONLY decode latency, not prefill.
             context_len > 0 is REQUIRED (cannot decode without context).

Batch Size Semantics:
    --batch-size specifies the batch size PER DP domain, matching the standalone
    benchmark (fbcode) semantics. The client automatically queries the server's
    DP configuration and multiplies to get the global batch size.

    Example: With DP=8 and --batch-size 64, the client sends 64*8=512 total
    requests distributed round-robin across all DP ranks.

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
    trace_dir: str = "traces"


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


def build_prompts(
    context_len: int, input_len: int, mode: str
) -> tuple[str | None, str]:
    """Build context and benchmark prompts for a parameter combination.

    Returns (context_prompt, benchmark_prompt) where:
    - context_prompt: Used for prefix cache warmup (None if context_len <= 0)
    - benchmark_prompt: Used for the actual benchmark run

    For prefill mode:
        context_prompt = context_len tokens ("hello " repeated)
        benchmark_prompt = context_len + input_len tokens
        The first context_len tokens match context_prompt (prefix cache hit).
        We measure prefill of the remaining input_len new tokens.

    For decode mode:
        context_prompt = context_len tokens
        benchmark_prompt = same as context_prompt (full prefix cache hit)
        We measure only decode iterations (no prefill work).
    """
    # Build context portion ("hello " is roughly 1-2 tokens per word)
    context_words = context_len // 2
    context_part = "hello " * max(1, context_words) if context_len > 0 else ""

    # Context prompt for prefix cache warmup
    context_prompt = context_part if context_len > 0 else None

    # Build benchmark prompt
    if mode == "prefill":
        # Add new input tokens after context (these will be prefilled)
        input_words = input_len // 2
        input_part = "world " * max(1, input_words)
        benchmark_prompt = context_part + input_part
    else:
        # Decode: same as context (full prefix cache hit, no prefill)
        benchmark_prompt = context_part

    return context_prompt, benchmark_prompt


def count_tokens(response_data: dict) -> tuple[int, int]:
    """Extract token counts from completion response."""
    usage = response_data.get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


@dataclass
class ServerConfig:
    """Server parallelism configuration."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    world_size: int = 1
    # Real values from platform sharding config (e.g., TPU).
    # Used for display only, not for batch size calculation.
    real_data_parallel_size: int | None = None
    real_tensor_parallel_size: int | None = None


async def fetch_server_config(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
) -> ServerConfig:
    """Fetch server parallelism config from first endpoint."""
    endpoint = rotator.all()[0]
    try:
        resp = await session.get(f"{endpoint}/debug/config")
        if resp.status == 200:
            data = await resp.json()
            return ServerConfig(
                data_parallel_size=data.get("data_parallel_size", 1),
                tensor_parallel_size=data.get("tensor_parallel_size", 1),
                pipeline_parallel_size=data.get("pipeline_parallel_size", 1),
                world_size=data.get("world_size", 1),
                real_data_parallel_size=data.get("real_data_parallel_size"),
                real_tensor_parallel_size=data.get("real_tensor_parallel_size"),
            )
    except Exception as e:
        logger.warning("Failed to fetch server config: %s", e)
    return ServerConfig()


async def run_compilation_warmup(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    model: str,
) -> None:
    """Send a warmup request to trigger runtime compilation."""
    endpoint = rotator.all()[0]
    logger.info("Sending warmup request to trigger compilation...")
    try:
        resp = await session.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model,
                "prompt": "Hello",
                "max_tokens": 1,
                "stream": False,
            },
        )
        if resp.status == 200:
            await resp.json()
            logger.info("Compilation warmup complete")
        else:
            logger.warning("Warmup request failed: HTTP %d", resp.status)
    except Exception as e:
        logger.warning("Warmup request failed: %s", e)


async def run_prefix_cache_warmup(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    model: str,
    context_prompt: str | None,
    batch_size: int,
) -> None:
    """Populate prefix cache with context tokens before benchmarking.

    Sends batch_size requests with context_prompt to populate the
    prefix cache. The benchmark requests will share this prefix.
    """
    if context_prompt is None:
        return

    logger.info("Populating prefix cache...")

    # Send warmup requests to all endpoints (round-robin)
    tasks = []
    for _ in range(batch_size):
        endpoint = rotator.next()
        task = asyncio.ensure_future(
            session.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": context_prompt,
                    "max_tokens": 1,
                    "stream": False,
                },
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(
        1 for r in responses if not isinstance(r, Exception) and r.status == 200
    )
    # Consume response bodies
    for resp in responses:
        if not isinstance(resp, Exception):
            await resp.read()

    logger.info(
        "Prefix cache warmup: %d/%d requests succeeded", success_count, batch_size
    )


async def wait_for_batch_ready(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    target_running: int,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.1,
) -> bool:
    """Poll /debug/batch_info until all requests are running (in decode).

    On TPU, new requests must each go through a mandatory prefill step
    even with prefix cache hits.  With max_num_batched_tokens limiting
    how many can prefill per step, the batch ramps up incrementally
    (e.g. 8 -> 16 -> 32 -> 64).  This function waits for the ramp-up
    to finish so that profiling captures only steady-state decode.

    Returns True if the target was reached, False on timeout.
    """
    endpoint = rotator.all()[0]
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        try:
            resp = await session.get(f"{endpoint}/debug/batch_info")
            if resp.status == 200:
                info = await resp.json()
                num_running = info.get("num_running", 0)
                num_waiting = info.get("num_waiting", -1)
                if num_waiting == 0 and num_running >= target_running:
                    logger.info(
                        "Batch ready: %d running, %d waiting",
                        num_running, num_waiting,
                    )
                    return True
            elif resp.status == 404:
                # Server doesn't have /debug/batch_info — skip polling
                logger.info(
                    "Server does not support /debug/batch_info, "
                    "skipping batch readiness wait"
                )
                return True
        except Exception:
            pass
        await asyncio.sleep(poll_interval_s)

    logger.warning(
        "Timeout waiting for batch to reach %d running requests",
        target_running,
    )
    return False


async def run_single_iteration(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    rotator: EndpointRotator,
    benchmark_prompt: str,
    batch_size: int,
    num_tokens_to_generate: int = 0,
    trace_prefix: str | None = None,
) -> tuple[float, int, int, bool]:
    """Run one iteration: sleep -> queue requests -> wake -> measure.

    For decode mode with profiling, waits for the batch to fully ramp up
    before starting the profiler so the trace captures only steady-state
    decode iterations.

    All requests use non-streaming mode for robustness. Streaming was
    previously used for TPU warmup token exclusion but caused
    ServerDisconnectedError on large batch+context combinations.

    Returns:
        (elapsed_ms, prompt_tokens, completion_tokens, trace_started)
    """

    # 1. Pause scheduling on ALL endpoints
    await call_debug_endpoint(session, rotator, "/debug/sleep", {"level": "0"})

    # 2. For decode mode: enable prefill-only scheduling so that
    #    all requests finish prefill before any decode begins.
    #    Without this, the scheduler mixes prefill and decode, causing
    #    early requests to finish decoding before late ones even start
    #    prefilling (the batch never reaches full size simultaneously).
    if config.mode == "decode":
        await call_debug_endpoint(
            session, rotator, "/debug/prefill_only", {"enabled": "true"})

    # 3. Build requests and start sending them (they queue while server sleeps)
    max_tokens = 1 if config.mode == "prefill" else num_tokens_to_generate

    tasks = []
    for _ in range(batch_size):
        endpoint = rotator.next()
        request_body = {
            "model": config.model,
            "prompt": benchmark_prompt,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Force all requests to generate exactly the same number of
        # tokens so they finish in the same step.  Without min_tokens,
        # EOS or other stop conditions can cause some requests to finish
        # early, shrinking the batch during the profiled window.
        if config.mode == "decode":
            request_body["min_tokens"] = max_tokens
        task = asyncio.ensure_future(
            session.post(
                f"{endpoint}/v1/completions",
                json=request_body,
            )
        )
        tasks.append(task)

    # Wait for all requests to be queued on server before waking.
    # A fixed delay is unreliable — instead poll batch_info to confirm
    # the engine has received all requests.
    if config.mode == "decode":
        endpoint = rotator.all()[0]
        for _ in range(100):  # up to 10s
            try:
                resp = await session.get(f"{endpoint}/debug/batch_info")
                if resp.status == 200:
                    info = await resp.json()
                    total = info.get("num_running", 0) + info.get("num_waiting", 0)
                    if total >= batch_size:
                        logger.info(
                            "All %d requests queued (running=%d, waiting=%d)",
                            total, info["num_running"], info["num_waiting"],
                        )
                        break
            except Exception:
                pass
            await asyncio.sleep(0.1)
    else:
        await asyncio.sleep(0.1)

    # 4. Resume scheduling — with prefill_only, only prefills run
    await call_debug_endpoint(session, rotator, "/debug/wake_up")

    # 5. For decode mode: wait for all prefills to complete, then
    #    disable prefill_only, start profiling, and begin timing.
    trace_started = False
    if config.mode == "decode":
        await wait_for_batch_ready(session, rotator, batch_size)

        # All requests are now in decode — disable prefill_only
        # so decode scheduling resumes
        await call_debug_endpoint(
            session, rotator, "/debug/prefill_only", {"enabled": "false"})

        # Start profiling (trace captures only steady-state decode)
        if trace_prefix is not None:
            params = {"prefix": trace_prefix, "delay": 0}
            for attempt in range(3):
                trace_started = await call_debug_endpoint(
                    session, rotator, "/debug/profile/start", params,
                )
                if trace_started:
                    break
                logger.warning(
                    "Failed to start profiling for %s (attempt %d/3)",
                    trace_prefix, attempt + 1,
                )
                await asyncio.sleep(2.0)

    # 6. Start timing from here (after ramp-up, at steady-state)
    start = time.perf_counter()

    responses = await asyncio.gather(*tasks)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 7. Count tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for resp in responses:
        try:
            data = await resp.json()
            pt, ct = count_tokens(data)
            total_prompt_tokens += pt
            total_completion_tokens += ct
        except Exception as e:
            logger.warning("Failed to parse response: %s", e)

    return elapsed_ms, total_prompt_tokens, total_completion_tokens, trace_started


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
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(await trace_resp.read())
                    logger.info("Downloaded: %s", local_path)
                    downloaded.append(local_path)
        except Exception as e:
            logger.warning("Failed to fetch traces from %s: %s", endpoint, e)

    return downloaded


async def run_benchmark(
    config: BenchmarkConfig,
) -> tuple[list[IterationResult], ServerConfig]:
    """Main benchmark loop with parameter sweeping."""

    rotator = EndpointRotator(config.endpoints)
    results: list[IterationResult] = []

    connector = aiohttp.TCPConnector(
        limit=0,  # No limit
        ttl_dns_cache=300,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        # Fetch server config once
        server_config = await fetch_server_config(session, rotator)
        dp_size = server_config.real_data_parallel_size or server_config.data_parallel_size
        display_tp = (
            server_config.real_tensor_parallel_size
            or server_config.tensor_parallel_size
        )
        logger.info(
            "Server config: DP=%d, TP=%d, PP=%d",
            dp_size,
            display_tp,
            server_config.pipeline_parallel_size,
        )

        # Warmup: trigger runtime compilation before benchmarking
        await run_compilation_warmup(session, rotator, config.model)

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

        # On TPU, the first engine step for a new request always runs the
        # prefill XLA program (even with prefix cache). Generate extra tokens
        # so prefill overhead is amortized into the per-iteration latency.
        # The trace will include the prefill step, but with enough decode
        # steps it becomes a small fraction of the total.
        is_tpu = server_config.real_data_parallel_size is not None
        tpu_extra_tokens = 3 if (is_tpu and config.mode == "decode") else 0
        num_tokens_to_generate = num_output_tokens + tpu_extra_tokens

        # Track all trace prefixes for fetching at the end
        trace_prefixes: list[str] = []

        for ctx_len, in_len, batch_size_per_dp in param_combos:
            # Scale batch size by DP to match standalone benchmark semantics
            # User specifies per-DP batch size, we compute global batch size
            global_batch_size = batch_size_per_dp * dp_size

            # Build all prompts for this parameter combination
            context_prompt, benchmark_prompt = build_prompts(
                ctx_len, in_len, config.mode
            )

            logger.info(
                "Running: mode=%s, ctx=%d, input=%d, batch=%d/dp (global=%d), "
                "output_tokens=%d (+ %d overhead)",
                config.mode,
                ctx_len,
                in_len,
                batch_size_per_dp,
                global_batch_size,
                num_output_tokens,
                tpu_extra_tokens,
            )

            # Prefix cache warmup: populate KV cache before benchmarking
            await run_prefix_cache_warmup(
                session, rotator, config.model, context_prompt, global_batch_size
            )

            # Wait for engine to fully drain warmup requests before
            # starting the benchmark.  Without this, a stale warmup
            # request can linger in the scheduler's running queue and
            # be counted toward the batch size target, then finish
            # after 1 token — shrinking the batch during decode.
            if config.mode == "decode":
                endpoint = rotator.all()[0]
                for _ in range(100):
                    try:
                        resp = await session.get(f"{endpoint}/debug/batch_info")
                        if resp.status == 200:
                            info = await resp.json()
                            if (info.get("num_running", 1) == 0
                                    and info.get("num_waiting", 1) == 0):
                                break
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)

            # Build trace prefix for this param combo
            trace_prefix = None
            trace_started = False
            if config.profile:
                trace_prefix = (
                    f"{config.mode}_ctx{ctx_len}_in{in_len}_bs{batch_size_per_dp}"
                )

            # For prefill mode, start profiling BEFORE the iteration
            # (the entire prefill is what we want to capture).
            if config.profile and config.mode == "prefill" and trace_prefix:
                params = {"prefix": trace_prefix, "delay": 0}
                for attempt in range(3):
                    trace_started = await call_debug_endpoint(
                        session, rotator, "/debug/profile/start",
                        params,
                    )
                    if trace_started:
                        trace_prefixes.append(trace_prefix)
                        break
                    logger.warning(
                        "Failed to start profiling for %s (attempt %d/3)",
                        trace_prefix, attempt + 1,
                    )
                    await asyncio.sleep(2.0)

            (
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                decode_trace_started,
            ) = await run_single_iteration(
                session,
                config,
                rotator,
                benchmark_prompt,
                global_batch_size,
                num_tokens_to_generate,
                trace_prefix=trace_prefix if config.mode == "decode" else None,
            )

            # For decode mode, trace_started comes from run_single_iteration
            # (profiling starts after batch ramp-up inside that function).
            if config.mode == "decode" and decode_trace_started:
                trace_started = True
                trace_prefixes.append(trace_prefix)

            # Stop profiling only if start succeeded
            # Read server-reported decode-only elapsed_ms from response
            server_elapsed_ms = None
            if trace_started:
                for endpoint in rotator.all():
                    try:
                        resp = await session.post(
                            f"{endpoint}/debug/profile/stop"
                        )
                        if resp.status == 200:
                            data = await resp.json()
                            if "elapsed_ms" in data and server_elapsed_ms is None:
                                server_elapsed_ms = data["elapsed_ms"]
                    except Exception as e:
                        logger.warning("Failed to stop profiling on %s: %s",
                                       endpoint, e)

            # Use server-reported execute_model() time for decode mode only.
            # For decode, this gives pure kernel time excluding scheduler gaps.
            # For prefill, use client-side timing because prefill may span
            # multiple engine steps and the server only accumulates steps with
            # scheduled tokens, missing the full prefill latency.
            if server_elapsed_ms is not None and config.mode == "decode":
                decode_elapsed_ms = server_elapsed_ms
                logger.info("  Server decode-only elapsed: %.2fms "
                            "(client total: %.2fms)",
                            decode_elapsed_ms, elapsed_ms)
            else:
                decode_elapsed_ms = elapsed_ms

            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = (
                total_tokens / (decode_elapsed_ms / 1000)
                if decode_elapsed_ms > 0 else 0
            )
            # Compute per-iteration latency
            if server_elapsed_ms is not None and config.mode == "decode":
                # Decode: divide by actual decode steps (exclude prefill)
                num_decode_steps = num_tokens_to_generate - 1
                latency_per_iter = (
                    decode_elapsed_ms / num_decode_steps
                    if num_decode_steps > 0
                    else decode_elapsed_ms
                )
            else:
                # Prefill or fallback: divide by output tokens
                latency_per_iter = (
                    decode_elapsed_ms / num_output_tokens
                    if num_output_tokens > 0
                    else decode_elapsed_ms
                )

            # Record per-DP batch size for comparison with standalone benchmark
            results.append(
                IterationResult(
                    endpoint=",".join(rotator.all()),
                    mode=config.mode,
                    context_len=ctx_len,
                    input_len=in_len,
                    batch_size=batch_size_per_dp,
                    iteration=num_output_tokens,
                    total_latency_ms=decode_elapsed_ms,
                    latency_per_iter_ms=latency_per_iter,
                    tokens_per_second=tokens_per_second,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

            logger.info(
                "  Result: %.2fms total, %.2fms/iter, %.2f tok/s",
                decode_elapsed_ms,
                latency_per_iter,
                tokens_per_second,
            )

        # Fetch traces for all param combos
        if config.profile and trace_prefixes:
            logger.info("Fetching traces for %d runs...", len(trace_prefixes))

            # Retry fetching traces (async write by torch profiler)
            max_retries = 3
            all_downloaded: list[str] = []
            for attempt in range(max_retries):
                await asyncio.sleep(2.0)
                for prefix in trace_prefixes:
                    downloaded = await fetch_traces(session, rotator, prefix, config.trace_dir)
                    all_downloaded.extend(downloaded)
                if all_downloaded:
                    logger.info(
                        "Downloaded %d trace files to %s", len(all_downloaded),
                        config.trace_dir,
                    )
                    break
                logger.info(
                    "No traces yet, retrying (%d/%d)...", attempt + 1, max_retries
                )

            if not all_downloaded:
                logger.warning(
                    "No trace files found after %d attempts. "
                    "Check server profiler directory.",
                    max_retries,
                )

    return results, server_config


def print_results_summary(
    results: list[IterationResult],
    server_config: ServerConfig | None = None,
) -> None:
    """Print a summary of benchmark results."""
    if not results:
        logger.warning("No results to summarize")
        return

    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 110)

    if server_config:
        dp = server_config.real_data_parallel_size or server_config.data_parallel_size
        tp = (
            server_config.real_tensor_parallel_size
            or server_config.tensor_parallel_size
        )
        print(
            f"Server: DP={dp}, "
            f"TP={tp}, "
            f"PP={server_config.pipeline_parallel_size}, "
            f"World={server_config.world_size}"
        )
        print("-" * 110)

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
        help="Batch size per DP domain, comma-separated for sweep (e.g., 1,4,8). "
        "Automatically multiplied by server DP size to get global batch size.",
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
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="traces",
        help="Directory to save downloaded trace files (default: ./traces)",
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
        trace_dir=args.trace_dir,
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
    results, server_config = asyncio.run(run_benchmark(config))
    print_results_summary(results, server_config)

    if args.output:
        write_results_json(results, args.output)


if __name__ == "__main__":
    main()
