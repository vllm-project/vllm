#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible correctness gate for speculative decoding.

This utility compares a baseline autoregressive server against a speculative
decoding server under deterministic requests. It is intentionally method
agnostic: the candidate can use MTP, DFlash, EAGLE, a draft model, n-gram, or
another speculative proposer.

Run this gate before latency/throughput sweeps. If a tokenizer is provided, the
strict gate compares generated token IDs; otherwise it falls back to exact text.
JSON prompts additionally validate that required keys are present.

Example:

    python benchmarks/spec_decode_correctness.py \\
      --ar-url http://127.0.0.1:8000 \\
      --spec-url http://127.0.0.1:8001 \\
      --ar-model meta-llama/Llama-3.1-8B-Instruct \\
      --spec-model meta-llama/Llama-3.1-8B-Instruct \\
      --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
      --concurrency-sweep 1,4,8,16 \\
      --output-json spec_decode_correctness.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "short_explain",
        "kind": "text",
        "prompt": (
            "In two concise sentences, explain why speculative decoding "
            "requires a correctness check."
        ),
    },
    {
        "id": "bandwidth_bound",
        "kind": "text",
        "prompt": (
            "In one concise paragraph, explain why single-token decode can be "
            "memory-bandwidth bound."
        ),
    },
    {
        "id": "code",
        "kind": "text",
        "prompt": (
            "Write a compact Python function that returns the first n "
            "Fibonacci numbers."
        ),
    },
    {
        "id": "json_decision",
        "kind": "json",
        "required_keys": ["decision", "risk", "next_step"],
        "prompt": (
            "Return compact JSON with keys decision, risk, and next_step. "
            "Topic: whether to enable a speculative decoding feature by "
            "default after a drift finding."
        ),
    },
    {
        "id": "json_tool",
        "kind": "json",
        "required_keys": ["tool", "allowed", "reason"],
        "prompt": (
            "Return JSON only with keys tool, allowed, and reason. Scenario: "
            "a user asks to run a destructive filesystem command without "
            "approval."
        ),
    },
    {
        "id": "routing_rules",
        "kind": "text",
        "prompt": (
            "Summarize three practical routing rules for choosing between a "
            "single-stream local runtime and a high-concurrency serving "
            "runtime."
        ),
    },
]


@dataclass(frozen=True)
class PromptInstance:
    prompt_id: str
    kind: str
    prompt: str
    required_keys: list[str]
    replica: int


@dataclass
class Completion:
    text: str
    latency_s: float
    raw: dict[str, Any]


def parse_concurrency_sweep(value: str) -> list[int]:
    concurrency = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item < 1:
            raise argparse.ArgumentTypeError(
                f"concurrency values must be positive, got {item}"
            )
        concurrency.append(item)
    if not concurrency:
        raise argparse.ArgumentTypeError("at least one concurrency value is required")
    return concurrency


def load_prompts(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_PROMPTS

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise SystemExit(f"{path}:{line_no}: missing 'prompt'")
            rows.append(obj)

    if not rows:
        raise SystemExit(f"{path}: no prompts found")
    return rows


def make_prompt_instances(
    prompts: list[dict[str, Any]], concurrency: int
) -> list[PromptInstance]:
    replicas = max(1, math.ceil(concurrency / len(prompts)))
    instances: list[PromptInstance] = []
    for replica in range(replicas):
        for index, item in enumerate(prompts):
            instances.append(
                PromptInstance(
                    prompt_id=str(item.get("id") or f"prompt_{index}"),
                    kind=str(item.get("kind") or "text"),
                    prompt=str(item["prompt"]),
                    required_keys=list(item.get("required_keys") or []),
                    replica=replica,
                )
            )
    return instances


def build_headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"{api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    endpoint_type: str,
    headers: dict[str, str],
    max_tokens: int,
    timeout_s: int,
    seed: int | None,
) -> Completion:
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed

    if endpoint_type == "chat":
        path = "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": prompt}]
    elif endpoint_type == "completions":
        path = "/v1/completions"
        body["prompt"] = prompt
    else:
        raise ValueError(f"unknown endpoint type: {endpoint_type}")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    latency_s = time.perf_counter() - start

    if endpoint_type == "chat":
        text = raw["choices"][0]["message"].get("content") or ""
    else:
        text = raw["choices"][0].get("text") or ""
    return Completion(text=text, latency_s=latency_s, raw=raw)


def run_batch(
    *,
    instances: list[PromptInstance],
    concurrency: int,
    base_url: str,
    model: str,
    endpoint_type: str,
    headers: dict[str, str],
    max_tokens: int,
    timeout_s: int,
    seed: int | None,
) -> list[Completion]:
    def run_one(instance: PromptInstance) -> Completion:
        return post_completion(
            base_url=base_url,
            model=model,
            prompt=instance.prompt,
            endpoint_type=endpoint_type,
            headers=headers,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            seed=seed,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(run_one, instances))


def edit_distance(a: str, b: str, cap: int = 4096) -> int:
    a = a[:cap]
    b = b[:cap]
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def first_diff(a: list[int] | str, b: list[int] | str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def json_keys_ok(parsed: Any, required_keys: list[str]) -> bool:
    return isinstance(parsed, dict) and all(key in parsed for key in required_keys)


def load_tokenizer(model: str | None) -> Any:
    if not model:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            f"--tokenizer was set but transformers import failed: {exc}"
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def encode(tokenizer: Any, text: str) -> list[int] | None:
    if tokenizer is None:
        return None
    return tokenizer.encode(text, add_special_tokens=False)


def compare_row(
    *,
    instance: PromptInstance,
    concurrency: int,
    ar: Completion,
    spec: Completion,
    tokenizer: Any,
) -> dict[str, Any]:
    ar_ids = encode(tokenizer, ar.text)
    spec_ids = encode(tokenizer, spec.text)
    token_exact = None
    if ar_ids is not None and spec_ids is not None:
        token_exact = ar_ids == spec_ids

    text_exact = ar.text == spec.text
    norm_dist = edit_distance(ar.text, spec.text) / max(
        1, max(len(ar.text), len(spec.text))
    )

    ar_json_ok = None
    spec_json_ok = None
    if instance.kind == "json":
        ar_json_ok = json_keys_ok(parse_json(ar.text), instance.required_keys)
        spec_json_ok = json_keys_ok(parse_json(spec.text), instance.required_keys)

    return {
        "concurrency": concurrency,
        "id": instance.prompt_id,
        "replica": instance.replica,
        "kind": instance.kind,
        "text_exact": text_exact,
        "token_exact": token_exact,
        "first_text_diff": first_diff(ar.text, spec.text),
        "first_token_diff": (
            first_diff(ar_ids, spec_ids)
            if ar_ids is not None and spec_ids is not None
            else None
        ),
        "edit_distance_norm": norm_dist,
        "length_ratio": len(spec.text) / max(1, len(ar.text)),
        "ar_latency_s": ar.latency_s,
        "spec_latency_s": spec.latency_s,
        "speed_ratio_latency": (
            ar.latency_s / spec.latency_s if spec.latency_s > 0 else math.inf
        ),
        "ar_json_ok": ar_json_ok,
        "spec_json_ok": spec_json_ok,
        "required_keys": instance.required_keys,
        "ar_text_prefix": ar.text[:240],
        "spec_text_prefix": spec.text[:240],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-url", required=True)
    parser.add_argument("--spec-url", required=True)
    parser.add_argument("--ar-model", required=True)
    parser.add_argument("--spec-model", required=True)
    parser.add_argument(
        "--endpoint-type",
        choices=("chat", "completions"),
        default="chat",
        help="OpenAI-compatible endpoint family to call.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Read a bearer token from this environment variable.",
    )
    parser.add_argument(
        "--tokenizer",
        help="HF/local tokenizer path for token-ID comparison.",
    )
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--concurrency-sweep",
        type=parse_concurrency_sweep,
        default=[1],
        help="Comma-separated max parallel request counts, e.g. 1,4,8,16.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("spec_decode_correctness.json"),
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_jsonl)
    tokenizer = load_tokenizer(args.tokenizer)
    headers = build_headers(args.api_key_env)

    rows: list[dict[str, Any]] = []
    for concurrency in args.concurrency_sweep:
        instances = make_prompt_instances(prompts, concurrency)
        ar_outputs = run_batch(
            instances=instances,
            concurrency=concurrency,
            base_url=args.ar_url,
            model=args.ar_model,
            endpoint_type=args.endpoint_type,
            headers=headers,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            seed=args.seed,
        )
        spec_outputs = run_batch(
            instances=instances,
            concurrency=concurrency,
            base_url=args.spec_url,
            model=args.spec_model,
            endpoint_type=args.endpoint_type,
            headers=headers,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            seed=args.seed,
        )
        for instance, ar, spec in zip(instances, ar_outputs, spec_outputs):
            rows.append(
                compare_row(
                    instance=instance,
                    concurrency=concurrency,
                    ar=ar,
                    spec=spec,
                    tokenizer=tokenizer,
                )
            )

    strict_pass = all(
        row["token_exact"] if row["token_exact"] is not None else row["text_exact"]
        for row in rows
    )
    schema_pass = all(
        row["kind"] != "json" or (row["ar_json_ok"] and row["spec_json_ok"])
        for row in rows
    )
    result = {
        "schema_version": "vllm-spec-decode-correctness-v1",
        "strict_pass": strict_pass,
        "schema_pass": schema_pass,
        "tokenizer": args.tokenizer,
        "endpoint_type": args.endpoint_type,
        "concurrency_sweep": args.concurrency_sweep,
        "prompts": len(prompts),
        "rows": rows,
        "decision": (
            "GO_STRICT_LOSSLESS"
            if strict_pass and schema_pass
            else "NO_GO_STRICT"
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "strict_pass": strict_pass,
                "schema_pass": schema_pass,
                "rows": len(rows),
                "decision": result["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if strict_pass and schema_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
