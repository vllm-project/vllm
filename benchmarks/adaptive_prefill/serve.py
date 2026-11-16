# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded mixed-load streaming benchmark against a real OpenAI endpoint.

Run unchanged against baseline and adaptive servers. No simulated numbers are
emitted here. Token gaps are reported only if every content chunk contains
exactly one token according to the returned logprobs, otherwise they are null.
"""

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))]


async def request(client, args, index, kind, delay):
    await asyncio.sleep(delay)
    rng = random.Random(args.seed + index)
    size = args.long_tokens if kind == "long" else args.short_tokens
    # Same token IDs on both runs; random prefixes avoid accidental prefix hits.
    prompt = [rng.randrange(10, args.vocab_size) for _ in range(size)]
    payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": args.long_output if kind == "long" else args.short_output,
        "ignore_eos": True,
        "stream": True,
        "logprobs": 0,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    arrivals = []
    all_single_token = True
    output_tokens = None
    error = None
    finish_reason = None
    done = False
    try:
        async with client.stream(
            "POST", args.url.rstrip("/") + "/v1/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                usage = event.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens")
                for choice in event.get("choices", []):
                    finish_reason = choice.get("finish_reason") or finish_reason
                    tokens = (choice.get("logprobs") or {}).get("tokens", [])
                    if tokens or choice.get("text"):
                        arrivals.append(time.perf_counter())
                        all_single_token &= len(tokens) == 1
        if not done or finish_reason is None:
            raise RuntimeError("Stream ended without completion")
        if not arrivals or output_tokens != payload["max_tokens"]:
            raise RuntimeError("Missing output/usage or unexpected output token count")
    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
        error = str(exc)
    gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "request": index,
        "kind": kind,
        "error": error,
        "ttft_ms": (arrivals[0] - start) * 1000 if arrivals else None,
        "duration_ms": (time.perf_counter() - start) * 1000,
        "stream_gap_ms": gaps,
        "token_gap_ms": gaps if all_single_token and arrivals else None,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
    }


async def run(args):
    limits = httpx.Limits(max_connections=args.long_requests + args.short_requests)
    async with httpx.AsyncClient(
        timeout=args.timeout, limits=limits, trust_env=False
    ) as client:
        health = await client.get(args.url.rstrip("/") + "/health")
        health.raise_for_status()
        for i in range(args.warmup_requests):
            warmup = await request(client, args, 1000000 + i, "short", 0)
            if warmup["error"]:
                raise RuntimeError("Warmup failed: " + warmup["error"])
        start = time.perf_counter()
        jobs = [
            request(client, args, i, "long", i * args.long_interval)
            for i in range(args.long_requests)
        ]
        jobs += [
            request(
                client,
                args,
                args.long_requests + i,
                "short",
                args.short_delay + i * args.short_interval,
            )
            for i in range(args.short_requests)
        ]
        # An overall timeout bounds slow-drip streams as well as socket inactivity.
        rows = await asyncio.wait_for(asyncio.gather(*jobs), args.timeout)
        elapsed = time.perf_counter() - start
    short = [r for r in rows if r["kind"] == "short" and not r["error"]]
    token_gaps_available = bool(short) and all(
        r["token_gap_ms"] is not None for r in short
    )
    token_gaps = [g for r in short for g in (r["token_gap_ms"] or [])]
    stream_gaps = [g for r in short for g in r["stream_gap_ms"]]
    counts_available = all(r["output_tokens"] is not None for r in rows)
    return {
        "evidence": "live_streaming_endpoint",
        "label": args.label,
        "config": vars(args),
        "elapsed_s": elapsed,
        "failures": sum(r["error"] is not None for r in rows),
        "decode_token_gap_p99_ms": percentile(token_gaps, 0.99)
        if token_gaps_available
        else None,
        "decode_stream_event_gap_p99_ms": percentile(stream_gaps, 0.99),
        "short_ttft_p95_ms": percentile(
            [r["ttft_ms"] for r in short if r["ttft_ms"] is not None], 0.95
        ),
        "long_ttft_p95_ms": percentile(
            [
                r["ttft_ms"]
                for r in rows
                if r["kind"] == "long" and r["ttft_ms"] is not None and not r["error"]
            ],
            0.95,
        ),
        "output_tokens_per_s": sum(r["output_tokens"] for r in rows) / elapsed
        if counts_available
        else None,
        "all_long_requests_completed": all(
            not r["error"] for r in rows if r["kind"] == "long"
        ),
        "requests": rows,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    for name, default in (
        ("long-tokens", 16384),
        ("short-tokens", 16),
        ("long-output", 8),
        ("short-output", 128),
        ("long-requests", 4),
        ("short-requests", 32),
        ("warmup-requests", 3),
        ("vocab-size", 10000),
        ("seed", 62574),
    ):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("long-interval", 0.05),
        ("short-interval", 0.2),
        ("short-delay", 3.0),
        ("timeout", 300.0),
    ):
        p.add_argument("--" + name, type=float, default=default)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.vocab_size <= 10 or args.long_requests + args.short_requests <= 0:
        raise SystemExit("Need vocab-size > 10 and at least one request")
    try:
        result = asyncio.run(run(args))
    except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(
            f"Benchmark blocked; no valid performance result: {exc}"
        ) from exc
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("requests", "config")},
            indent=2,
        )
    )
    raise SystemExit(1 if result["failures"] else 0)
