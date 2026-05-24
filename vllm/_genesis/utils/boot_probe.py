# SPDX-License-Identifier: Apache-2.0
"""Boot-time probe for spec-decode cross-rank issues (#41190-class).

Sends a single small chat-completion request after `vllm serve` is
responsive. Catches the following classes of failure BEFORE the first
real user request hits them:

- **#41190 cross-rank event ptr mismatch** — TP=2 spec-decode crashes
  on `num_accepted_tokens_event.synchronize()` with
  `cudaErrorIllegalAddress` on first request. Reported on 2× R6000 Ada
  (sm_89), Qwen3.6-35B-A3B-AWQ + qwen3_next_mtp.

- **Workspace lock unwarmed shape (PR #40941)** — `WorkspaceManager._locked`
  flag flips after warmup; if MTP draft proposer's first call shape
  was not in warmup set, AssertionError on workspace size grow.

- **Cudagraph capture invalidation** — text-patches change file mtime,
  Triton hash mismatch, first inference triggers cold compile. If
  >120s, likely autotune regression.

Usage (CLI):
    python3 -m vllm._genesis.utils.boot_probe \\
        --url http://localhost:8000/v1/chat/completions \\
        --model qwen3.6-27b \\
        --api-key genesis-local

    # Or via env vars:
    GENESIS_BOOT_PROBE_URL=http://localhost:8000/v1/chat/completions \\
    GENESIS_BOOT_PROBE_MODEL=qwen3.6-27b \\
    GENESIS_API_KEY=genesis-local \\
    python3 -m vllm._genesis.utils.boot_probe

Exit codes:
    0 — probe passed (server responsive, response parseable, no errors)
    1 — probe failed (HTTP error, timeout, parse error, or connect error)
    2 — invalid invocation (missing required args)

Recommended placement: in container start scripts AFTER `vllm serve`
is detected as ready (e.g. in a tail-poll loop checking
`/v1/models` for 200 OK), BEFORE the script returns control.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

log = logging.getLogger("genesis.boot_probe")


def probe(
    url: str,
    model: str,
    api_key: str,
    timeout: float = 120.0,
    max_tokens: int = 32,
) -> tuple[bool, str]:
    """Send a single dummy request, return (success, message).

    Catches:
    - Connection errors (server not up)
    - HTTP 500/4xx (server crashed)
    - Timeouts (cold compile or hung)
    - Parse errors (malformed response)

    Returns:
        (True, "probe passed in Xms") on success
        (False, "<diagnostic>") on any failure
    """
    try:
        import requests
    except ImportError:
        return False, "requests library not available"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return False, (
            f"TIMEOUT after {timeout}s — cold compile or hung kernel. "
            "Check container logs for stuck JIT compile."
        )
    except requests.exceptions.ConnectionError as e:
        return False, f"CONNECTION ERROR — server not responsive: {e}"
    except Exception as e:
        return False, f"REQUEST ERROR: {type(e).__name__}: {e}"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        snippet = r.text[:300] if r.text else "(empty)"
        # Heuristic detection of #41190 / workspace-lock signatures
        marker = ""
        if "cudaErrorIllegalAddress" in snippet:
            marker = " [#41190-class cross-rank event sync]"
        elif "AssertionError" in snippet and "workspace" in snippet.lower():
            marker = " [#40941 workspace-lock unwarmed shape]"
        elif "CUDA" in snippet and "out of memory" in snippet.lower():
            marker = " [OOM — check memory_pool patches]"
        return False, (
            f"HTTP {r.status_code}{marker} after {elapsed_ms:.0f}ms — "
            f"response: {snippet}"
        )

    # Parse response — Qwen3 reasoning models put output in `reasoning_content`
    # before visible `content`, both count as successful generation.
    try:
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return False, f"EMPTY choices array after {elapsed_ms:.0f}ms"
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        tool_calls = msg.get("tool_calls") or []
        # Also check finish_reason — `length` indicates we hit max_tokens
        # which counts as model-was-generating successfully
        finish_reason = choices[0].get("finish_reason")
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens", 0)

        if not content and not reasoning and not tool_calls:
            if completion_tokens == 0:
                return False, (
                    f"NO output (0 completion tokens) after {elapsed_ms:.0f}ms "
                    "— sampler/decoder may be wedged"
                )
            # Tokens were generated but all hidden in <think> — model is alive,
            # this is OK as boot probe (not quality test).
            return True, (
                f"probe passed in {elapsed_ms:.0f}ms (model={model}, "
                f"{completion_tokens} tokens generated, all reasoning, "
                f"finish_reason={finish_reason})"
            )
    except (ValueError, KeyError, AttributeError) as e:
        return False, f"PARSE ERROR after {elapsed_ms:.0f}ms: {e}"

    return True, (
        f"probe passed in {elapsed_ms:.0f}ms (model={model}, "
        f"{completion_tokens} tokens, finish_reason={finish_reason})"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns exit code.

    Reads from CLI args first, falls back to env vars:
      GENESIS_BOOT_PROBE_URL, GENESIS_BOOT_PROBE_MODEL, GENESIS_API_KEY,
      GENESIS_BOOT_PROBE_TIMEOUT (seconds, default 120)
    """
    ap = argparse.ArgumentParser(
        description="Genesis boot-time probe for spec-decode cross-rank issues"
    )
    ap.add_argument("--url", default=os.environ.get("GENESIS_BOOT_PROBE_URL"))
    ap.add_argument("--model", default=os.environ.get("GENESIS_BOOT_PROBE_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("GENESIS_API_KEY", "genesis-local"))
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("GENESIS_BOOT_PROBE_TIMEOUT", "120.0")),
    )
    ap.add_argument(
        "--max-tokens", type=int, default=32,
        help=(
            "Tokens to generate. Default 32 — enough headroom for Qwen3 "
            "reasoning models that consume tokens for <think> section "
            "before emitting visible content"
        ),
    )
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Suppress success log; only print on failure")
    args = ap.parse_args(argv)

    if not args.url or not args.model:
        sys.stderr.write(
            "Error: --url and --model required (or set "
            "GENESIS_BOOT_PROBE_URL + GENESIS_BOOT_PROBE_MODEL)\n"
        )
        return 2

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s:%(name)s] %(message)s",
        )

    ok, msg = probe(
        url=args.url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    if ok:
        if not args.quiet:
            log.info("[boot probe] %s", msg)
        return 0
    else:
        log.error("[boot probe] FAILED: %s", msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
