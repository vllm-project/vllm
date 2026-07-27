#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate closed-loop load and report throughput plus acceptance metrics."""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp

METRICS = {
    "output_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "requests": "vllm:request_success_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
CHAT_KWARGS = {
    "hy3": {"reasoning_effort": "no_think"},
    "qwen3-8b": {"enable_thinking": False},
}


def load_prompts(path: Path, limit: int, seed: int = 980406) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if prompt is None:
                turns = row.get("turns")
                prompt = turns[0] if isinstance(turns, list) and turns else None
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"missing prompt in {path}")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    if limit and len(prompts) > limit:
        random.Random(seed).shuffle(prompts)
        prompts = prompts[:limit]
    return prompts


def parse_metrics(text: str) -> dict[str, float]:
    values = dict.fromkeys(METRICS, 0.0)
    by_name = {metric: key for key, metric in METRICS.items()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        if key := by_name.get(name):
            values[key] += float(fields[1])
    return values


async def scrape(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        values = parse_metrics(await response.text())
    values["time"] = time.monotonic()
    return values


def request_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": CHAT_KWARGS[model],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    request_id: str,
) -> None:
    async with session.post(
        url,
        json=payload,
        headers={"x-request-id": request_id},
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        await response.read()


def window(
    before: dict[str, float],
    after: dict[str, float],
    errors: int,
) -> dict[str, Any]:
    elapsed = after["time"] - before["time"]
    delta = {key: after[key] - before[key] for key in METRICS}
    drafts = delta["drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    return {
        "duration_s": elapsed,
        "output_throughput": delta["output_tokens"] / elapsed,
        "request_throughput": delta["requests"] / elapsed,
        "total_token_throughput": (delta["prompt_tokens"] + delta["output_tokens"])
        / elapsed,
        "completed_requests": delta["requests"],
        "client_errors": errors,
        "generation_tokens": delta["output_tokens"],
        "prompt_tokens": delta["prompt_tokens"],
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "mean_acceptance_length": (None if drafts == 0 else 1 + accepted / drafts),
        "draft_acceptance_rate": (
            None if draft_tokens == 0 else accepted / draft_tokens
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.dataset, args.max_prompts, args.prompt_seed)
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 1)
    stop = asyncio.Event()
    cursor = 0
    error_count = 0
    error_samples: list[str] = []

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:

        async def worker(worker_id: int) -> None:
            nonlocal cursor, error_count
            while not stop.is_set():
                index = cursor
                cursor += 1
                prompt = prompts[index % len(prompts)]
                try:
                    await send(
                        session,
                        f"{args.base_url}/v1/chat/completions",
                        request_payload(args.model, prompt, args.max_tokens),
                        f"{args.request_prefix}-{worker_id}-{index}",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    error_count += 1
                    if len(error_samples) < 5:
                        error_samples.append(repr(error))
                    await asyncio.sleep(0.1)

        tasks = [
            asyncio.create_task(worker(index)) for index in range(args.concurrency)
        ]
        await asyncio.sleep(args.warmup_seconds)
        snapshot = await scrape(session, args.base_url)
        windows = []
        for repeat in range(args.repeats):
            errors_before = error_count
            await asyncio.sleep(args.measure_seconds)
            next_snapshot = await scrape(session, args.base_url)
            windows.append(
                {
                    "repeat": repeat,
                    **window(
                        snapshot,
                        next_snapshot,
                        error_count - errors_before,
                    ),
                }
            )
            snapshot = next_snapshot

        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    valid = all(
        item["generation_tokens"] > 0
        and item["completed_requests"] > 0
        and item["client_errors"] == 0
        and (not args.require_spec or item["drafts"] > 0)
        for item in windows
    )
    return {
        "status": "ok" if valid else "failed",
        "dataset": str(args.dataset),
        "model": args.model,
        "concurrency": args.concurrency,
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "max_prompts": args.max_prompts,
        "prompt_count": len(prompts),
        "prompt_seed": args.prompt_seed,
        "request": {
            **request_payload(args.model, "<prompt>", args.max_tokens),
            "messages": "<dataset prompt>",
        },
        "windows": windows,
        "error_samples": error_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=30)
    parser.add_argument("--measure-seconds", type=float, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--prompt-seed", type=int, default=980406)
    parser.add_argument("--request-prefix", default="throughput")
    parser.add_argument("--require-spec", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
