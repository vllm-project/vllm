#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRCR long-context evaluation for vLLM's OpenAI-compatible server.

Streams samples from `openai/mrcr` on HuggingFace, sends chat completions to
the server, and scores each response with a prefix-gated SequenceMatcher ratio
against the reference answer.
"""

import argparse
import asyncio
import json
import time
from difflib import SequenceMatcher

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

DATASET_REPO = "openai/mrcr"
NEEDLE_SHARDS = {
    2: "2needle/2needle_0.parquet",
    4: "4needle/4needle_0.parquet",
    8: "8needle/8needle_0.parquet",
}
# Reserve headroom for chat-template tokens on top of the messages.
PROMPT_SAFETY_BUFFER = 256
# Pre-filter heuristic before the authoritative /tokenize check.
CHARS_PER_TOKEN = 4
# Skip chain-of-thought on reasoning models; ignored by non-reasoning templates.
DEFAULT_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}


def discover_server_model(base_url: str) -> tuple[str, int | None]:
    """Return (model_id, max_model_len) from /v1/models."""
    resp = requests.get(f"{base_url}/v1/models", timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models advertised at {base_url}/v1/models")
    entry = data[0]
    return entry["id"], entry.get("max_model_len")


def count_chat_tokens(base_url: str, model: str, messages: list[dict]) -> int:
    """Return the chat-template-rendered token count via /tokenize."""
    resp = requests.post(
        f"{base_url}/tokenize",
        json={"model": model, "messages": messages, "add_generation_prompt": True},
        timeout=120,
    )
    resp.raise_for_status()
    return int(resp.json()["count"])


def _load_mrcr_samples(
    needles: list[int],
    max_prompt_tokens: int,
    num_samples: int,
    seed: int,
    base_url: str,
    model_name: str,
) -> list[dict]:
    """Stream MRCR samples balanced across needle buckets, token-verified."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "MRCR eval requires `datasets`. Install with: uv pip install datasets"
        ) from e

    max_chars = max_prompt_tokens * CHARS_PER_TOKEN
    per_bucket = num_samples // len(needles)
    leftover = num_samples - per_bucket * len(needles)

    samples: list[dict] = []
    for idx, n in enumerate(needles):
        if n not in NEEDLE_SHARDS:
            raise ValueError(f"Unsupported needle count {n}")
        target = per_bucket + (1 if idx < leftover else 0)
        if target == 0:
            continue

        ds = load_dataset(
            DATASET_REPO,
            data_files=NEEDLE_SHARDS[n],
            split="train",
            streaming=True,
        ).shuffle(seed=seed + n, buffer_size=16)

        taken = 0
        for row in ds:
            if int(row.get("n_chars", 0)) > max_chars:
                continue
            prompt = row["prompt"]
            messages = json.loads(prompt) if isinstance(prompt, str) else list(prompt)
            n_tokens = count_chat_tokens(base_url, model_name, messages)
            if n_tokens > max_prompt_tokens:
                continue
            samples.append(
                {
                    "messages": messages,
                    "answer": row["answer"],
                    "random_string_to_prepend": row["random_string_to_prepend"],
                    "n_needles": int(row["n_needles"]),
                    "n_tokens": n_tokens,
                }
            )
            taken += 1
            if taken >= target:
                break

        if taken < target:
            print(f"Warning: only {taken}/{target} samples for n_needles={n}")

    if not samples:
        raise RuntimeError("No MRCR samples fit; loosen max_prompt_tokens.")
    return samples


def score_mrcr(response: str, answer: str, random_prefix: str) -> float:
    """Prefix-gated SequenceMatcher ratio; 0 if the prefix is missing."""
    if not response.startswith(random_prefix):
        return 0.0
    stripped = response[len(random_prefix) :]
    return SequenceMatcher(a=answer, b=stripped, autojunk=False).ratio()


async def _call_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    seed: int | None,
    extra_body: dict,
) -> tuple[str, int]:
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra_body,
    }
    if seed is not None:
        data["seed"] = seed
    try:
        async with session.post(f"{url}/v1/chat/completions", json=data) as resp:
            resp.raise_for_status()
            result = await resp.json()
            text = result["choices"][0]["message"]["content"] or ""
            return text, result.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        print(f"chat request failed: {e}")
        return "", 0


def evaluate_mrcr(
    model_name: str | None = None,
    num_samples: int = 40,
    needles: list[int] | None = None,
    max_prompt_tokens: int | None = None,
    max_tokens: int = 2048,
    host: str = "http://127.0.0.1",
    port: int = 8000,
    temperature: float = 0.0,
    seed: int | None = 42,
    concurrency: int = 8,
    extra_body: dict | None = None,
) -> dict:
    """Run MRCR against a vLLM server; auto-discovers model and context."""
    needles = needles or [2, 4, 8]
    extra_body = DEFAULT_EXTRA_BODY if extra_body is None else extra_body
    base_url = f"{host}:{port}"

    discovered_model, server_max_len = discover_server_model(base_url)
    if model_name is None:
        model_name = discovered_model
    if max_prompt_tokens is None:
        if server_max_len is None:
            raise RuntimeError(
                "Server did not advertise max_model_len; pass --max-prompt-tokens."
            )
        max_prompt_tokens = max(512, server_max_len - max_tokens - PROMPT_SAFETY_BUFFER)
    print(
        f"Model: {model_name} | max_prompt_tokens={max_prompt_tokens} "
        f"(server max_model_len={server_max_len}, max_tokens={max_tokens})"
    )

    samples = _load_mrcr_samples(
        needles=needles,
        max_prompt_tokens=max_prompt_tokens,
        num_samples=num_samples,
        seed=seed or 0,
        base_url=base_url,
        model_name=model_name,
    )
    tok_counts = [s["n_tokens"] for s in samples]
    print(
        f"Loaded {len(samples)} samples (needles={needles}, "
        f"tokens={min(tok_counts)}-{max(tok_counts)})"
    )

    async def run():
        sem = asyncio.Semaphore(concurrency)
        responses = [""] * len(samples)
        out_tokens = [0] * len(samples)

        async def one(session, i):
            async with sem:
                text, toks = await _call_chat(
                    session=session,
                    url=base_url,
                    model=model_name,
                    messages=samples[i]["messages"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    extra_body=extra_body,
                )
                responses[i] = text
                out_tokens[i] = toks

        timeout = aiohttp.ClientTimeout(total=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await tqdm.gather(
                *[one(session, i) for i in range(len(samples))], desc="MRCR"
            )
        return responses, out_tokens

    tic = time.perf_counter()
    responses, out_tokens = asyncio.run(run())
    latency = time.perf_counter() - tic

    scores = np.array(
        [
            score_mrcr(r, s["answer"], s["random_string_to_prepend"])
            for r, s in zip(responses, samples)
        ]
    )
    prefix_hits = np.array(
        [
            r.startswith(s["random_string_to_prepend"])
            for r, s in zip(responses, samples)
        ]
    )
    per_needle = {
        f"match_ratio_n{n}": float(
            scores[np.array([s["n_needles"] == n for s in samples])].mean()
        )
        for n in needles
        if any(s["n_needles"] == n for s in samples)
    }

    total_out = int(sum(out_tokens))
    return {
        "model": model_name,
        "match_ratio": float(scores.mean()),
        "prefix_hit_rate": float(prefix_hits.mean()),
        "per_needle": per_needle,
        "num_samples": len(samples),
        "latency": latency,
        "total_output_tokens": total_out,
        "tokens_per_second": total_out / latency if latency > 0 else 0.0,
        "max_tokens": max_tokens,
        "needles": needles,
        "max_prompt_tokens": max_prompt_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MRCR evaluation for vLLM serve")
    p.add_argument("--model", default=None, help="Default: discovered from /v1/models")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument(
        "--needles", type=int, nargs="+", default=[2, 4, 8], choices=[2, 4, 8]
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Default: server max_model_len - max_tokens - buffer",
    )
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--host", default="http://127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--extra-body",
        default=None,
        help="JSON merged into each request. "
        "Pass '{}' to disable the default enable_thinking=false.",
    )
    p.add_argument("--save-results", default=None)
    args = p.parse_args()

    extra_body = json.loads(args.extra_body) if args.extra_body else None

    result = evaluate_mrcr(
        model_name=args.model,
        num_samples=args.num_samples,
        needles=args.needles,
        max_prompt_tokens=args.max_prompt_tokens,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        seed=args.seed,
        concurrency=args.concurrency,
        extra_body=extra_body,
    )

    print("\nResults:")
    print(f"  match_ratio:     {result['match_ratio']:.4f}")
    print(f"  prefix_hit_rate: {result['prefix_hit_rate']:.4f}")
    for k, v in result["per_needle"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  samples:         {result['num_samples']}")
    print(f"  latency:         {result['latency']:.1f}s")
    print(f"  output tok/s:    {result['tokens_per_second']:.1f}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
