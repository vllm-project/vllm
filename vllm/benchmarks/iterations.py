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

Re-analyze traces (no server needed):
    vllm bench iterations \
        --reanalyze-traces ./run_dir/prefill_results.json \
        --trace-dir ./run_dir
"""

import argparse
import asyncio
import gzip
import itertools
import json
import os
import re
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
    server_total_latency_ms: float
    server_latency_per_iter_ms: float
    prompt_tokens: int
    completion_tokens: int
    client_latency_per_iter_ms: float = 0.0
    client_total_latency_ms: float = 0.0
    trace_breakdown: dict[str, float] | None = None
    trace_iter_duration_ms: float = 0.0
    trace_kernel_sum_ms: float = 0.0
    trace_request_pairs: list[tuple[int, int]] | None = None


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
    # Build context portion.
    # Use "a " which tokenizes to ~1 token per repetition on most models.
    # The exact token count may vary by model but will be close to the
    # requested length.  For precise control, a tokenizer-based approach
    # would be needed, but that requires loading the model tokenizer.
    context_part = "a " * max(1, context_len) if context_len > 0 else ""

    # Context prompt for prefix cache warmup
    context_prompt = context_part if context_len > 0 else None

    # Build benchmark prompt
    if mode == "prefill":
        # Add new input tokens after context (these will be prefilled)
        input_part = "b " * max(1, input_len)
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
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None


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
    config: "BenchmarkConfig | None" = None,
    dp_size: int = 1,
) -> None:
    """Send warmup requests to trigger runtime compilation.

    With SKIP_JAX_PRECOMPILE=1, each unique token bucket must be compiled
    before profiling.  This function sends 1 request per unique total token
    count (across all parameter combinations) to trigger compilation for
    every bucket that will be used during benchmarking.
    """
    if config is None:
        # Simple warmup with minimal tokens
        logger.info("Sending 1 warmup request to trigger compilation...")
        try:
            endpoint = rotator.next()
            resp = await session.post(
                f"{endpoint}/v1/completions",
                json={"model": model, "prompt": "Hello",
                      "max_tokens": 1, "stream": False},
            )
            await resp.read()
            logger.info("Compilation warmup complete")
        except Exception as e:
            logger.warning("Warmup request failed: %s", e)
        return

    # Compute unique total token counts for all parameter combinations.
    # Each combo may land in a different compiled bucket.
    input_lens = config.input_lens
    if config.mode == "decode":
        # In decode mode, input_len is not used (context_len matters)
        input_lens = [1]

    seen_totals: set[int] = set()
    # List of (total_tokens, per_req_tokens, num_requests) for each combo
    warmup_shapes: list[tuple[int, int, int]] = []
    for ctx_len in config.context_lens:
        for in_len in input_lens:
            for bs in config.batch_sizes:
                per_req = ctx_len + in_len + 1  # +1 for BOS token
                num_reqs = bs * dp_size
                total_tokens = per_req * num_reqs
                if total_tokens > 0 and total_tokens not in seen_totals:
                    seen_totals.add(total_tokens)
                    warmup_shapes.append((total_tokens, per_req, num_reqs))

    # For decode mode, also warmup the decode-step bucket
    # (1 token per request per step)
    if config.mode == "decode":
        for bs in config.batch_sizes:
            num_reqs = bs * dp_size
            # Decode step: each request generates 1 token
            # Send num_reqs requests each with 1 token prompt
            if num_reqs not in seen_totals:
                seen_totals.add(num_reqs)
                warmup_shapes.append((num_reqs, 1, num_reqs))

    # Sort ascending so smaller compilations finish first
    warmup_shapes.sort()

    logger.info(
        "Compilation warmup: %d unique token counts to compile: %s",
        len(warmup_shapes),
        [t for t, _, _ in warmup_shapes],
    )

    for total_tokens, per_req, num_reqs in warmup_shapes:
        prompt = "a " * max(1, per_req)
        logger.info("  Compiling for %d tokens (%d reqs × %d tok)...",
                     total_tokens, num_reqs, per_req)
        try:
            tasks = []
            for _ in range(num_reqs):
                endpoint = rotator.next()
                tasks.append(asyncio.ensure_future(
                    session.post(
                        f"{endpoint}/v1/completions",
                        json={"model": model, "prompt": prompt,
                              "max_tokens": 1, "stream": False},
                    )
                ))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success = sum(1 for r in responses
                          if not isinstance(r, Exception) and r.status == 200)
            for resp in responses:
                if not isinstance(resp, Exception):
                    await resp.read()
            logger.info("  Compiled %d tokens OK (%d/%d)",
                         total_tokens, success, num_reqs)
        except Exception as e:
            logger.warning("  Failed %d tokens: %s", total_tokens, e)


async def run_prefix_cache_warmup(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    model: str,
    context_prompt: str | None,
    dp_size: int,
) -> None:
    """Populate prefix cache with context tokens before benchmarking.

    Sends one request per DP rank to populate the prefix cache.
    The prefix cache is content-addressable, so a single request per rank
    is sufficient — all subsequent requests with the same prefix get a
    cache hit regardless of batch size.
    """
    if context_prompt is None:
        return

    num_warmup = max(1, dp_size)
    logger.info("Populating prefix cache (%d requests for %d DP ranks)...",
                num_warmup, dp_size)

    # Send one warmup request per DP rank (round-robin)
    tasks = []
    for _ in range(num_warmup):
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
        "Prefix cache warmup: %d/%d requests succeeded", success_count, num_warmup
    )


async def wait_for_batch_ready(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    target_running: int,
    timeout_s: float = 1800.0,
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
    use_sleep_wake: bool = True,
) -> tuple[float, int, int, bool]:
    """Run one iteration: queue requests and measure decode.

    Two modes depending on use_sleep_wake:

    DP=1 (use_sleep_wake=True):
        sleep -> prefill_only -> queue -> wake -> wait for all prefilled
        -> disable prefill_only -> profile -> time decode -> stop profile
        Gives exact batch control via scheduler pause.

    DP>1 (use_sleep_wake=False):
        queue requests normally -> wait for batch to fill -> profile
        -> time remaining decode -> stop profile
        Cannot pause scheduler because DP engines must execute in lockstep.
        Uses extra max_tokens so requests survive the ramp-up.

    Returns:
        (elapsed_ms, prompt_tokens, completion_tokens, trace_started)
    """

    max_tokens = 1 if config.mode == "prefill" else num_tokens_to_generate
    request_body_base = {
        "model": config.model,
        "prompt": benchmark_prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if config.mode == "decode":
        request_body_base["min_tokens"] = max_tokens
        request_body_base["ignore_eos"] = True

    if use_sleep_wake:
        # === DP=1 path: full scheduler control ===

        # 1. Pause scheduling
        await call_debug_endpoint(
            session, rotator, "/debug/sleep", {"level": "0"})

        # 2. Enable prefill-only scheduling
        if config.mode == "decode":
            await call_debug_endpoint(
                session, rotator, "/debug/prefill_only", {"enabled": "true"})

        # 3. Queue all requests (they accumulate while scheduler is paused)
        tasks = []
        for _ in range(batch_size):
            endpoint = rotator.next()
            task = asyncio.ensure_future(
                session.post(
                    f"{endpoint}/v1/completions", json=request_body_base))
            tasks.append(task)

        # Wait for all requests to arrive at the engine
        if config.mode == "decode":
            endpoint = rotator.all()[0]
            for _ in range(100):
                try:
                    resp = await session.get(f"{endpoint}/debug/batch_info")
                    if resp.status == 200:
                        info = await resp.json()
                        total = (info.get("num_running", 0)
                                 + info.get("num_waiting", 0))
                        if total >= batch_size:
                            logger.info(
                                "All %d requests queued (running=%d, "
                                "waiting=%d)",
                                total, info["num_running"],
                                info["num_waiting"])
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.1)

        # 4. Resume scheduling (prefill_only: only prefills run)
        await call_debug_endpoint(session, rotator, "/debug/wake_up")

        # 5. Wait for all prefills, then switch to decode + profile
        trace_started = False
        if config.mode == "decode":
            await wait_for_batch_ready(session, rotator, batch_size)
            await call_debug_endpoint(
                session, rotator, "/debug/prefill_only",
                {"enabled": "false"})

            if trace_prefix is not None:
                params = {"prefix": trace_prefix, "delay": 0}
                for attempt in range(3):
                    trace_started = await call_debug_endpoint(
                        session, rotator, "/debug/profile/start", params)
                    if trace_started:
                        break
                    logger.warning(
                        "Failed to start profiling for %s (attempt %d/3)",
                        trace_prefix, attempt + 1)
                    await asyncio.sleep(2.0)

        # 6. Time decode (steady-state)
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        elapsed_ms = (time.perf_counter() - start) * 1000

    else:
        # === DP>1 path: no scheduler control ===

        # 1. Send all requests normally (no sleep/wake)
        tasks = []
        for _ in range(batch_size):
            endpoint = rotator.next()
            task = asyncio.ensure_future(
                session.post(
                    f"{endpoint}/v1/completions", json=request_body_base))
            tasks.append(task)

        # 2. Wait for batch to reach full size (ramp-up completes)
        trace_started = False
        if config.mode == "decode":
            await wait_for_batch_ready(session, rotator, batch_size)
            logger.info("Batch ramped up, starting profiling")

            # 3. Start profiling (captures steady-state decode)
            if trace_prefix is not None:
                params = {"prefix": trace_prefix, "delay": 0}
                for attempt in range(3):
                    trace_started = await call_debug_endpoint(
                        session, rotator, "/debug/profile/start", params)
                    if trace_started:
                        break
                    logger.warning(
                        "Failed to start profiling for %s (attempt %d/3)",
                        trace_prefix, attempt + 1)
                    await asyncio.sleep(2.0)

        # 4. Time remaining decode
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        elapsed_ms = (time.perf_counter() - start) * 1000

    # Count tokens
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


def trim_prefill_trace(trace_path: str) -> None:
    """Remove the decode step from a prefill trace file.

    On TPU, prefill traces contain 2 main steps (prefill + 1 decode)
    because jax.profiler.stop_trace() can't be called between them.
    This function trims the trace to keep only the first (prefill) step.

    Identifies main steps via jit_step_fun events on tid=2 with a JIT hash
    in the name (format: jit_step_fun(HASH)). Keeps all events up to the
    start of the second main step.
    """
    if not trace_path.endswith(('.json', '.json.gz')):
        return

    try:
        if trace_path.endswith('.gz'):
            with gzip.open(trace_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(trace_path, 'r') as f:
                data = json.load(f)
    except Exception:
        return

    events = data.get('traceEvents', [])
    if not events:
        return

    # Find main execution steps: tid=2 jit_step_fun with hash, dur > 50ms
    main_events = sorted(
        [e for e in events
         if 'jit_step_fun(' in str(e.get('name', ''))
         and e.get('tid') == 2
         and e.get('dur', 0) > 50000],  # > 50ms in microseconds
        key=lambda e: e['ts']
    )

    if not main_events:
        logger.info("No main step events found, no trimming needed")
        return

    # Group into steps (events within 10ms are same step, different devices)
    steps: list[list[dict]] = [[main_events[0]]]
    for e in main_events[1:]:
        if e['ts'] - steps[-1][0]['ts'] < 10000:  # 10ms in microseconds
            steps[-1].append(e)
        else:
            steps.append([e])

    if len(steps) < 2:
        logger.info("Trace has %d step(s), no trimming needed", len(steps))
        return

    # Keep events that start before the second step
    cutoff_ts = steps[1][0]['ts']
    trimmed = [e for e in events if e.get('ts', 0) < cutoff_ts]

    logger.info("Trimmed prefill trace: %d → %d events (removed decode step "
                "at %.1fms)", len(events), len(trimmed), cutoff_ts / 1000)

    data['traceEvents'] = trimmed
    if trace_path.endswith('.gz'):
        with gzip.open(trace_path, 'wt') as f:
            json.dump(data, f)
    else:
        with open(trace_path, 'w') as f:
            json.dump(data, f)


async def wait_for_all_first_tokens(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    model: str,
    prompt: str,
    batch_size: int,
    max_tokens: int,
) -> tuple[list[aiohttp.ClientResponse], float]:
    """Send streaming requests and wait until all produce their first token.

    Returns (responses, prefill_elapsed_ms) where responses are the open
    streaming connections that still need to be read for remaining tokens.

    The first token from each request signals that its prefill is complete.
    Once all requests have produced a first token, the entire batch is in
    steady-state decode — safe to start profiling.
    """
    request_body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "min_tokens": max_tokens,
        "ignore_eos": True,
    }

    # Open all streaming connections
    responses: list[aiohttp.ClientResponse] = []
    for _ in range(batch_size):
        endpoint = rotator.next()
        resp = await session.post(
            f"{endpoint}/v1/completions", json=request_body)
        responses.append(resp)

    # Track first-token arrival for each request
    first_token_seen = [False] * batch_size
    first_token_count = 0
    start_time = time.perf_counter()

    async def read_until_first_token(idx: int, resp: aiohttp.ClientResponse):
        nonlocal first_token_count
        async for chunk in resp.content:
            line = chunk.decode('utf-8', errors='ignore').strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                if not first_token_seen[idx]:
                    first_token_seen[idx] = True
                    first_token_count += 1
                    return  # Stop reading, leave stream open

    # Read first token from all streams concurrently
    await asyncio.gather(
        *[read_until_first_token(i, resp) for i, resp in enumerate(responses)]
    )

    prefill_elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("All %d requests produced first token in %.1fms "
                "(prefill ramp-up complete)", batch_size, prefill_elapsed_ms)

    return responses, prefill_elapsed_ms


async def drain_streaming_responses(
    responses: list[aiohttp.ClientResponse],
) -> tuple[int, int]:
    """Read remaining tokens from streaming responses and count totals."""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    async def drain_one(resp: aiohttp.ClientResponse):
        nonlocal total_prompt_tokens, total_completion_tokens
        async for chunk in resp.content:
            line = chunk.decode('utf-8', errors='ignore').strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                try:
                    data = json.loads(line[6:])
                    usage = data.get('usage', {})
                    if usage:
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get(
                            'completion_tokens', 0)
                except Exception:
                    pass

    await asyncio.gather(*[drain_one(resp) for resp in responses])
    return total_prompt_tokens, total_completion_tokens


async def list_server_traces(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
) -> set[str]:
    """Get the set of all trace files currently on the server."""
    existing: set[str] = set()
    for endpoint in rotator.all():
        try:
            resp = await session.get(f"{endpoint}/debug/traces")
            data = await resp.json()
            existing.update(data.get("traces", []))
        except Exception:
            pass
    return existing


async def fetch_traces(
    session: aiohttp.ClientSession,
    rotator: EndpointRotator,
    prefix: str,
    output_dir: str,
    exclude: set[str] | None = None,
) -> list[str]:
    """Download trace files from all endpoints."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    _exclude = exclude or set()

    for i, endpoint in enumerate(rotator.all()):
        try:
            resp = await session.get(f"{endpoint}/debug/traces")
            data = await resp.json()

            for trace_file in data.get("traces", []):
                if trace_file in _exclude:
                    continue
                if re.match(r"endpoint\d+_", trace_file):
                    continue
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

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=1800, sock_read=900),
    ) as session:
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

        # Warmup: trigger runtime compilation for all benchmark shapes.
        # With SKIP_JAX_PRECOMPILE=1, each unique token bucket needs to be
        # compiled before profiling.  Sends 1 request per unique total token
        # count to avoid compilation during profiled iterations.
        await run_compilation_warmup(
            session, rotator, config.model,
            config=config, dp_size=dp_size,
        )

        # Drain engine pipeline after warmup to prevent race with first
        # benchmark iteration. Without this, warmup's last engine step
        # can overlap with benchmark requests.
        await call_debug_endpoint(
            session, rotator, "/debug/sleep", {"level": "0"})
        await asyncio.sleep(0.5)
        await call_debug_endpoint(session, rotator, "/debug/wake_up")
        await call_debug_endpoint(
            session, rotator, "/debug/step_stats/reset")
        logger.info("Engine drained after warmup")

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

        # For decode with profiling, extra tokens are needed for:
        # - DP=1: steady-state polling (3 steps × 0.5s) consumes ~300+ tokens
        # - DP>1: ramp-up survival (requests prefill over multiple steps)
        # Use sleep/wake to batch all requests into one step:
        # - Prefill: always use sleep/wake (works with all DP sizes)
        # - Decode DP=1: use sleep/wake for scheduler control
        # - Decode DP>1: can't use sleep/wake, use streaming approach instead
        use_sleep_wake = (config.mode == "prefill") or (dp_size <= 1)
        if config.mode == "decode":
            if use_sleep_wake:
                # DP=1: need tokens for steady-state polling + profiling window
                # 3 polls × 0.5s = 1.5s at ~5ms/step = ~300 steps
                steady_state_tokens = 500
                tpu_extra_tokens = max(tpu_extra_tokens, steady_state_tokens)
            else:
                # DP>1: need tokens for ramp-up + steady-state polling
                # + profiled decode window
                max_ctx = max(config.context_lens)
                max_bs = max(config.batch_sizes) * dp_size
                max_batched = 10240  # conservative default
                ramp_up_steps = int(max_bs * max_ctx / max_batched) + 1
                # Tokens consumed: ramp-up + 3 steady-state polls + profiled
                ramp_up_tokens = ramp_up_steps + 50
                tpu_extra_tokens = max(tpu_extra_tokens, ramp_up_tokens)
                logger.info(
                    "DP>1: adding %d extra tokens for ramp-up survival "
                    "(total output=%d)",
                    ramp_up_tokens,
                    config.iterations + tpu_extra_tokens,
                )

        num_tokens_to_generate = num_output_tokens + tpu_extra_tokens

        # Track all trace prefixes for fetching at the end
        trace_prefixes: list[str] = []

        # Snapshot existing traces so we only download new ones
        pre_existing_traces: set[str] = set()
        if config.profile:
            pre_existing_traces = await list_server_traces(session, rotator)

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

            # Reset server-side step stats before each run
            await call_debug_endpoint(
                session, rotator, "/debug/step_stats/reset")

            # Prefix cache warmup: populate KV cache before benchmarking
            await run_prefix_cache_warmup(
                session, rotator, config.model, context_prompt, dp_size
            )

            # Wait for engine to drain warmup requests
            if config.mode == "decode":
                await asyncio.sleep(1.0)

            # Build trace prefix for this param combo
            trace_prefix = None
            trace_started = False
            if config.profile:
                trace_prefix = (
                    f"{config.mode}_ctx{ctx_len}_in{in_len}_bs{batch_size_per_dp}"
                )

            if config.mode == "decode" and config.profile and trace_prefix:
                # === Decode with profiling: streaming-based approach ===
                # Send streaming requests, wait for ALL first tokens
                # (= all prefills done), then start profiling for pure decode.
                responses, _ = await wait_for_all_first_tokens(
                    session, rotator, config.model,
                    benchmark_prompt, global_batch_size,
                    num_tokens_to_generate,
                )

                # wait_for_all_first_tokens confirmed all prefills are done.
                # Now poll step_stats for decode_steps >= 3 to ensure
                # steady-state decode before starting the profiler.
                # Skip wait_for_batch_ready — it's redundant after
                # wait_for_all_first_tokens and can timeout on fast decode.
                steady_deadline = time.perf_counter() + 30.0
                while time.perf_counter() < steady_deadline:
                    try:
                        resp = await session.get(
                            f"{rotator.all()[0]}/debug/step_stats")
                        if resp.status == 200:
                            stats = await resp.json()
                            if stats.get("decode_steps", 0) >= 3:
                                logger.info(
                                    "Steady-state reached: %d decode steps",
                                    stats["decode_steps"])
                                break
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    logger.warning(
                        "Timed out waiting for steady-state decode "
                        "(30s). Proceeding with profiling anyway.")

                # Reset stats again so profiled measurement is clean
                await call_debug_endpoint(
                    session, rotator, "/debug/step_stats/reset")

                # All prefills done → start profiling (pure decode)
                params = {"prefix": trace_prefix}
                for attempt in range(3):
                    trace_started = await call_debug_endpoint(
                        session, rotator, "/debug/profile/start", params)
                    if trace_started:
                        trace_prefixes.append(trace_prefix)
                        break
                    logger.warning(
                        "Failed to start profiling for %s (attempt %d/3)",
                        trace_prefix, attempt + 1)
                    await asyncio.sleep(2.0)

                # Time the remaining decode (traced)
                start = time.perf_counter()
                prompt_tokens, completion_tokens = (
                    await drain_streaming_responses(responses))
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Stop profiling
                if trace_started:
                    for endpoint_url in rotator.all():
                        try:
                            await session.post(
                                f"{endpoint_url}/debug/profile/stop")
                        except Exception as e:
                            logger.warning(
                                "Failed to stop profiling: %s", e)

            elif config.mode == "prefill" and config.profile and trace_prefix:
                # === Prefill with profiling ===
                # Start profiling, run iteration normally, stop.
                # Trace has prefill + 1 decode step; we trim the decode
                # step from the trace file after download.
                params = {"prefix": trace_prefix}
                for attempt in range(3):
                    trace_started = await call_debug_endpoint(
                        session, rotator, "/debug/profile/start", params)
                    if trace_started:
                        trace_prefixes.append(trace_prefix)
                        break
                    logger.warning(
                        "Failed to start profiling for %s (attempt %d/3)",
                        trace_prefix, attempt + 1)
                    await asyncio.sleep(2.0)

                (
                    elapsed_ms, prompt_tokens,
                    completion_tokens, _,
                ) = await run_single_iteration(
                    session, config, rotator, benchmark_prompt,
                    global_batch_size, num_tokens_to_generate,
                    trace_prefix=None, use_sleep_wake=use_sleep_wake,
                )

                # Stop profiling
                if trace_started:
                    for endpoint_url in rotator.all():
                        try:
                            await session.post(
                                f"{endpoint_url}/debug/profile/stop")
                        except Exception as e:
                            logger.warning(
                                "Failed to stop profiling: %s", e)

            else:
                # === Non-profiled run ===
                (
                    elapsed_ms, prompt_tokens,
                    completion_tokens, _,
                ) = await run_single_iteration(
                    session, config, rotator, benchmark_prompt,
                    global_batch_size, num_tokens_to_generate,
                    trace_prefix=None, use_sleep_wake=use_sleep_wake,
                )

            decode_elapsed_ms = elapsed_ms

            # Client-side per-iteration latency
            if config.mode == "decode":
                num_decode_steps = num_tokens_to_generate - 1
                client_latency_per_iter = (
                    decode_elapsed_ms / num_decode_steps
                    if num_decode_steps > 0
                    else decode_elapsed_ms
                )
            else:
                client_latency_per_iter = (
                    decode_elapsed_ms / num_output_tokens
                    if num_output_tokens > 0
                    else decode_elapsed_ms
                )

            # Server-side per-iteration latency (steady-state only)
            server_latency_per_iter = None
            step_stats = None
            try:
                resp = await session.get(
                    f"{rotator.all()[0]}/debug/step_stats")
                if resp.status == 200:
                    step_stats = await resp.json()
                    if config.mode == "decode" and step_stats.get(
                            "decode_steps", 0) > 0:
                        server_latency_per_iter = step_stats["decode_avg_ms"]
                        logger.info(
                            "  Server-side decode: %.1fms avg over %d steps "
                            "(%d toks/step)",
                            server_latency_per_iter,
                            step_stats["decode_steps"],
                            step_stats.get("decode_toks_per_step", 0))
                    elif config.mode == "prefill" and step_stats.get(
                            "prefill_steps", 0) > 0:
                        server_latency_per_iter = step_stats["prefill_avg_ms"]
                        logger.info(
                            "  Server-side prefill: %.1fms avg over %d steps",
                            server_latency_per_iter,
                            step_stats["prefill_steps"])
            except Exception as e:
                logger.warning("Failed to fetch step stats: %s", e)

            # Use server-side if available, client-side as fallback
            latency_per_iter = (server_latency_per_iter
                                if server_latency_per_iter is not None
                                else client_latency_per_iter)

            # Server-side total
            if server_latency_per_iter is not None and step_stats:
                n_steps = step_stats.get("decode_steps", 0) or step_stats.get(
                    "prefill_steps", 0) or 1
                server_total_ms = server_latency_per_iter * n_steps
            else:
                server_total_ms = decode_elapsed_ms

            # Record per-DP batch size for comparison with standalone benchmark
            results.append(
                IterationResult(
                    endpoint=",".join(rotator.all()),
                    mode=config.mode,
                    context_len=ctx_len,
                    input_len=in_len,
                    batch_size=batch_size_per_dp,
                    iteration=num_output_tokens,
                    server_total_latency_ms=server_total_ms,
                    server_latency_per_iter_ms=latency_per_iter,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    client_latency_per_iter_ms=client_latency_per_iter,
                    client_total_latency_ms=decode_elapsed_ms,
                )
            )

            logger.info(
                "  Result: %.2fms total (server), %.2fms/iter (server), "
                "%.2fms total (client), %.2fms/iter (client)",
                server_total_ms,
                latency_per_iter,
                decode_elapsed_ms,
                client_latency_per_iter,
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
                    downloaded = await fetch_traces(session, rotator, prefix, config.trace_dir, exclude=pre_existing_traces)
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
            elif config.mode == "prefill":
                # Trim decode step from prefill traces
                for trace_path in all_downloaded:
                    trim_prefill_trace(trace_path)

            # Run trace analysis on downloaded traces
            if all_downloaded:
                _analyze_and_attach_trace_breakdown(
                    all_downloaded, trace_prefixes, results, param_combos,
                    config, dp_size)

    return results, server_config


def _analyze_and_attach_trace_breakdown(
    downloaded_traces: list[str],
    trace_prefixes: list[str],
    results: list[IterationResult],
    param_combos: list[tuple],
    config: BenchmarkConfig,
    dp_size: int,
) -> None:
    """Run TPU trace analyzer on downloaded traces and attach breakdowns to results."""
    try:
        from vllm.benchmarks.tpu_trace_analyzer import analyze_tpu_trace_breakdown
    except ImportError:
        logger.warning("tpu_trace_analyzer not available, skipping trace analysis")
        return

    # Map trace prefix to result index
    prefix_to_result_idx: dict[str, int] = {}
    for i, (ctx_len, in_len, batch_size_per_dp) in enumerate(param_combos):
        prefix = f"{config.mode}_ctx{ctx_len}_in{in_len}_bs{batch_size_per_dp}"
        prefix_to_result_idx[prefix] = i

    # Analyze each trace and attach to corresponding result
    for trace_path in downloaded_traces:
        logger.info("  Analyzing trace: %s", trace_path)
        # Find which prefix this trace belongs to
        # Match longest prefix first to avoid bs1 matching bs128
        matched_prefix = None
        for prefix in sorted(trace_prefixes, key=len, reverse=True):
            if f"{prefix}/" in trace_path or trace_path.startswith(prefix + "/"):
                matched_prefix = prefix
                break
        if matched_prefix is None or matched_prefix not in prefix_to_result_idx:
            logger.info("    No matching prefix for trace (skipping)")
            continue

        result_idx = prefix_to_result_idx[matched_prefix]
        if result_idx >= len(results):
            continue

        # Skip if already analyzed (multiple endpoints may have traces)
        if results[result_idx].trace_breakdown is not None:
            continue

        try:
            breakdown, avg_iter_ms, req_pairs = analyze_tpu_trace_breakdown(
                trace_path)
            results[result_idx].trace_breakdown = breakdown
            results[result_idx].trace_iter_duration_ms = avg_iter_ms
            results[result_idx].trace_kernel_sum_ms = sum(breakdown.values())
            results[result_idx].trace_request_pairs = req_pairs

            # Log summary
            pairs_str = ", ".join(f"({r},{t})" for r, t in req_pairs) if req_pairs else "N/A"
            parts = [f"{cat}: {ms:.1f}ms" for cat, ms in
                     sorted(breakdown.items(), key=lambda x: -x[1])]
            logger.info("  Trace breakdown (%s): %s (iter: %.1fms, reqs/toks: %s)",
                        matched_prefix, ", ".join(parts), avg_iter_ms, pairs_str)
        except Exception as e:
            logger.warning("Failed to analyze trace %s: %s", trace_path, e)


def print_results_summary(
    results: list[IterationResult],
    server_config: ServerConfig | None = None,
) -> None:
    """Print a summary of benchmark results."""
    if not results:
        logger.warning("No results to summarize")
        return

    print("\n" + "=" * 130)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 130)

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
        f"{'Srvr Total':>12} {'Srvr/Iter':>12} "
        f"{'Clnt Total':>12} {'Clnt/Iter':>12}"
    )
    print("-" * 130)

    for r in results:
        print(
            f"{r.mode:>8} {r.context_len:>8} {r.input_len:>8} {r.batch_size:>6} "
            f"{r.iteration:>6} {r.server_total_latency_ms:>12.2f} {r.server_latency_per_iter_ms:>12.2f} "
            f"{r.client_total_latency_ms:>12.2f} {r.client_latency_per_iter_ms:>12.2f}"
        )

    print("=" * 130)

    # Trace kernel breakdown (if available)
    has_breakdown = any(r.trace_breakdown for r in results)
    if has_breakdown:
        # Collect all categories across results
        all_categories: set[str] = set()
        for r in results:
            if r.trace_breakdown:
                all_categories.update(r.trace_breakdown.keys())

        # Sort categories by total time descending
        cat_totals: dict[str, float] = {}
        for cat in all_categories:
            cat_totals[cat] = sum(
                r.trace_breakdown.get(cat, 0)
                for r in results if r.trace_breakdown)
        sorted_cats = sorted(cat_totals.keys(), key=lambda c: -cat_totals[c])

        print("\nTRACE KERNEL BREAKDOWN (per-device per-iteration, ms)")
        print("-" * 130)

        # Header
        cat_header = "".join(f"{c:>10}" for c in sorted_cats)
        print(f"{'Mode':>8} {'Context':>8} {'Input':>8} {'Batch':>6} "
              f"{'TraceIter':>10} {'KernelSum':>10}{cat_header}  {'(reqs,toks)'}")
        print("-" * 130)

        for r in results:
            if r.trace_breakdown:
                cat_vals = "".join(
                    f"{r.trace_breakdown.get(c, 0):>10.2f}" for c in sorted_cats)
                pairs_str = " ".join(
                    f"({rq},{tk})" for rq, tk in r.trace_request_pairs
                ) if r.trace_request_pairs else "N/A"
                print(f"{r.mode:>8} {r.context_len:>8} {r.input_len:>8} "
                      f"{r.batch_size:>6} {r.trace_iter_duration_ms:>10.2f} "
                      f"{r.trace_kernel_sum_ms:>10.2f}{cat_vals}  {pairs_str}")
            else:
                print(f"{r.mode:>8} {r.context_len:>8} {r.input_len:>8} "
                      f"{r.batch_size:>6} {'N/A':>10} {'N/A':>10}")

        print("=" * 130)

    print()


def write_results_json(results: list[IterationResult], output_path: str) -> None:
    """Write results to JSON file."""
    data = {
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results written to: %s", output_path)


def _find_trace_for_result(
    trace_dir: str,
    mode: str,
    context_len: int,
    input_len: int,
    batch_size: int,
) -> str | None:
    """Find a trace file in trace_dir matching the given parameters.

    Supports two trace directory layouts:
      Client download layout (from fetch_traces):
        {trace_dir}/endpoint{i}_{mode}_ctx{ctx}_in{in}_bs{bs}/
          plugins/profile/DATE/trace.json.gz
      Server-side layout (direct from profiler):
        {trace_dir}/{mode}_ctx{ctx}_in{in}_bs{bs}/
          plugins/profile/DATE/trace.json.gz

    Args:
        trace_dir: Directory containing trace files.
        mode: "prefill" or "decode".
        context_len: Context length.
        input_len: Input length.
        batch_size: Per-DP batch size.

    Returns:
        Path to the trace file, or None if not found.
    """
    import glob as glob_mod

    prefix = f"{mode}_ctx{context_len}_in{input_len}_bs{batch_size}"

    # Try client download layout: endpoint*_{prefix}/plugins/profile/*/*.trace.json.gz
    pattern = os.path.join(
        trace_dir,
        f"endpoint*_{prefix}",
        "plugins", "profile", "*", "*.trace.json.gz",
    )
    matches = sorted(glob_mod.glob(pattern))
    if matches:
        return matches[0]

    # Try server-side layout: {prefix}/plugins/profile/*/*.trace.json.gz
    pattern = os.path.join(
        trace_dir,
        prefix,
        "plugins", "profile", "*", "*.trace.json.gz",
    )
    matches = sorted(glob_mod.glob(pattern))
    if matches:
        return matches[0]

    # Fallback: any subdirectory containing the prefix
    pattern = os.path.join(trace_dir, f"*{prefix}*", "**", "*.trace.json.gz")
    matches = sorted(glob_mod.glob(pattern, recursive=True))
    if matches:
        return matches[0]

    return None


def reanalyze_traces_from_json(
    results_json: str,
    trace_dir: str,
) -> list[IterationResult]:
    """Re-parse traces and update an existing results JSON file.

    Reads the results JSON, finds the matching trace file for each result
    entry, re-runs analyze_tpu_trace_breakdown(), and updates trace fields.

    Args:
        results_json: Path to the results JSON file (prefill or decode).
        trace_dir: Directory containing downloaded trace files.

    Returns:
        List of updated IterationResult objects.
    """
    try:
        from vllm.benchmarks.tpu_trace_analyzer import (
            analyze_tpu_trace_breakdown,
        )
    except ImportError:
        logger.error("tpu_trace_analyzer not available")
        return []

    with open(results_json) as f:
        data = json.load(f)

    results: list[IterationResult] = []
    updated = 0
    skipped = 0

    for entry in data.get("results", []):
        result = IterationResult(
            endpoint=entry.get("endpoint", ""),
            mode=entry.get("mode", ""),
            context_len=entry.get("context_len", 0),
            input_len=entry.get("input_len", 0),
            batch_size=entry.get("batch_size", 0),
            iteration=entry.get("iteration", 0),
            server_total_latency_ms=entry.get(
                "server_total_latency_ms",
                entry.get("total_latency_ms", 0.0)),
            server_latency_per_iter_ms=entry.get(
                "server_latency_per_iter_ms",
                entry.get("latency_per_iter_ms", 0.0)),
            prompt_tokens=entry.get("prompt_tokens", 0),
            completion_tokens=entry.get("completion_tokens", 0),
            client_latency_per_iter_ms=entry.get(
                "client_latency_per_iter_ms", 0.0),
            client_total_latency_ms=entry.get(
                "client_total_latency_ms", 0.0),
        )

        trace_file = _find_trace_for_result(
            trace_dir,
            result.mode,
            result.context_len,
            result.input_len,
            result.batch_size,
        )

        if not trace_file:
            logger.warning(
                "No trace for %s ctx=%d in=%d bs=%d",
                result.mode, result.context_len,
                result.input_len, result.batch_size,
            )
            skipped += 1
            results.append(result)
            continue

        try:
            breakdown, avg_iter_ms, req_pairs = (
                analyze_tpu_trace_breakdown(trace_file))
            result.trace_breakdown = breakdown
            result.trace_iter_duration_ms = avg_iter_ms
            result.trace_kernel_sum_ms = sum(breakdown.values())
            result.trace_request_pairs = req_pairs
            updated += 1

            parts = [f"{cat}: {ms:.1f}ms" for cat, ms in
                     sorted(breakdown.items(), key=lambda x: -x[1])]
            logger.info(
                "  Updated %s ctx=%d in=%d bs=%d: %s (iter=%.1fms)",
                result.mode, result.context_len,
                result.input_len, result.batch_size,
                ", ".join(parts), avg_iter_ms,
            )
        except Exception as e:
            logger.warning(
                "Failed to analyze trace for %s ctx=%d in=%d bs=%d: %s",
                result.mode, result.context_len,
                result.input_len, result.batch_size, e,
            )
            skipped += 1

        results.append(result)

    logger.info("Reanalysis complete: %d updated, %d skipped", updated, skipped)

    # Write back
    write_results_json(results, results_json)

    return results


def parse_comma_list(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in value.split(",")]


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the iterations benchmark."""
    parser.add_argument(
        "--endpoints",
        type=str,
        default="",
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
        default=None,
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
        default="",
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
    parser.add_argument(
        "--reanalyze-traces",
        type=str,
        metavar="RESULTS_JSON",
        help="Re-parse traces in --trace-dir and update the given results JSON "
        "with fresh trace breakdowns. Skips benchmarking entirely. "
        "Example: --reanalyze-traces prefill_results.json --trace-dir ./run_dir",
    )


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser(
        description="Batch iteration benchmark for prefill/decode phases"
    )
    add_cli_args(parser)

    if args is None:
        args = parser.parse_args()

    # Handle --reanalyze-traces mode (no server needed)
    if args.reanalyze_traces:
        results_json = args.reanalyze_traces
        if not os.path.exists(results_json):
            parser.error(f"Results file not found: {results_json}")
        trace_dir = args.trace_dir
        if not os.path.isdir(trace_dir):
            parser.error(f"Trace directory not found: {trace_dir}")

        logger.info("Re-analyzing traces in %s for %s", trace_dir, results_json)
        results = reanalyze_traces_from_json(results_json, trace_dir)
        print_results_summary(results)
        return

    # Validate required args for benchmark mode
    if not args.endpoints:
        parser.error("--endpoints is required for benchmarking")
    if not args.mode:
        parser.error("--mode is required for benchmarking")
    if not args.model:
        parser.error("--model is required for benchmarking")

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
