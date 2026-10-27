#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke, long-context, and memory checks for MiniMax-M3.1-Flash on AMD.

Precision actually served must be recorded as MXFP4 routed experts with FP8
KV/indexer. This is not native NVFP4 serving.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = "http://127.0.0.1:8000/v1"
SERVED_PRECISION = "mxfp4_experts_fp8_kv_indexer"


def _request(url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base: str, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    last_error = "server did not become ready"
    models_url = base.rstrip("/") + "/models"
    while time.time() < deadline:
        try:
            body = _request(models_url, timeout=10)
            model = body["data"][0]["id"]
            print(f"ready model={model}")
            return model
        except (urllib.error.URLError, KeyError, IndexError) as exc:
            last_error = str(exc)
            time.sleep(5)
    raise SystemExit(f"timeout waiting for {models_url}: {last_error}")


def smoke(base: str, model: str) -> dict:
    body = _request(
        base.rstrip("/") + "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say only: ok"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    text = body["choices"][0]["message"]["content"]
    print(f"smoke={text!r}")
    return {"text": text, "usage": body.get("usage", {})}


def long_context(base: str, model: str, tokens: int) -> dict:
    filler = "word " * tokens
    prompt = (
        f"The marker TOKEN_ALPHA appears once in this preamble.\n{filler}\n"
        "Reply with only TOKEN_ALPHA."
    )
    body = _request(
        base.rstrip("/") + "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=600,
    )
    text = body["choices"][0]["message"]["content"]
    print(f"long_context tokens~={tokens} text={text!r}")
    return {"text": text, "usage": body.get("usage", {})}


def metrics(metrics_url: str) -> str:
    req = urllib.request.Request(metrics_url)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        print(f"metrics unavailable: {exc}")
        return ""
    keep = [
        line
        for line in raw.splitlines()
        if any(
            key in line
            for key in (
                "vllm:kv_cache_usage",
                "vllm:gpu_cache",
                "vllm:num_requests",
                "process_resident_memory_bytes",
            )
        )
    ]
    print("\n".join(keep[:40]))
    return "\n".join(keep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", DEFAULT_BASE))
    parser.add_argument("--output", default="")
    parser.add_argument("--wait-s", type=float, default=1800)
    parser.add_argument("--long-context-tokens", type=int, default=8192)
    parser.add_argument("--skip-long-context", action="store_true")
    args = parser.parse_args()
    model = wait_ready(args.base_url, args.wait_s)
    result = {
        "served_precision": SERVED_PRECISION,
        "model": model,
        "base_url": args.base_url,
        "smoke": smoke(args.base_url, model),
        "metrics": metrics(args.base_url.rstrip("/").removesuffix("/v1") + "/metrics"),
    }
    if not args.skip_long_context:
        result["long_context"] = long_context(
            args.base_url, model, args.long_context_tokens
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
