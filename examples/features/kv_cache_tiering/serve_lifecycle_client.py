#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise FS tiering and session lifecycle through the OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def fs_stats(root: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not root.exists():
        return files, total_bytes
    for path in root.rglob("*.bin"):
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes


def post_chat(
    url: str,
    model: str,
    prompt: str,
    session_id: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "kv_transfer_params": {"session_id": session_id},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", default="/root/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--fs-root", type=Path, default=Path("/tmp/vllm_kv_tiering"))
    parser.add_argument("--session-id", default="tiering-lifecycle-demo")
    parser.add_argument("--prompt-repetitions", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ttl-wait", type=float, default=12.0)
    args = parser.parse_args()

    prompt = (
        "KV cache tiering keeps reusable attention state across memory tiers. "
        * args.prompt_repetitions
    )
    before = fs_stats(args.fs_root)
    print(f"FS before: files={before[0]} bytes={before[1]}")

    for attempt in (1, 2):
        started = time.perf_counter()
        response = post_chat(
            args.url,
            args.model,
            prompt,
            args.session_id,
            args.max_tokens,
            args.timeout,
        )
        elapsed = time.perf_counter() - started
        usage = response.get("usage", {})
        print(
            f"request={attempt} elapsed={elapsed:.3f}s "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')}"
        )

    after = fs_stats(args.fs_root)
    print(f"FS after requests: files={after[0]} bytes={after[1]}")
    if after[0] <= before[0]:
        raise RuntimeError("FS tier did not create additional KV block files")

    print(f"Waiting {args.ttl_wait:.1f}s for lifecycle expiration")
    time.sleep(args.ttl_wait)
    final = fs_stats(args.fs_root)
    print(f"FS after TTL: files={final[0]} bytes={final[1]}")
    print("PASS: requests completed and FS tier stored KV blocks")


if __name__ == "__main__":
    main()
