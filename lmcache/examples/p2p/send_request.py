#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send a long prompt to a vLLM + LMCache server and print its cache-hit stats.

Used to demonstrate P2P KV cache sharing: send the prompt to node A first
(populates A's cache), then send the *same* prompt to node B. Because B never
served the prompt and its only remote cache source is its P2P peer (A), a
non-zero ``num_lmcache_cached_tokens`` on B proves the KV was read from A over
P2P.

Usage:
    python send_request.py --port 8000                 # node A (cold)
    python send_request.py --port 8001                 # node B (P2P hit)
    python send_request.py --host 10.0.0.3 --port 8000 # a remote node
"""

# Standard
import argparse
import json
import urllib.request

# A prompt long enough to span several LMCache chunks (default 256 tokens each).
LONG_PROMPT = "Explain the history of computer science in great detail. " + (
    "The Turing machine is a fundamental concept in theoretical computer "
    "science that defines an abstract machine capable of manipulating "
    "symbols on a strip of tape according to a table of rules. " * 20
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    args = parser.parse_args()

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 1,
        # Opt in to per-request cache-hit accounting in the response.
        "kv_transfer_params": {"cached_token_stats": True},
    }
    request = urllib.request.Request(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())

    stats = (data.get("kv_transfer_params") or {}).get("cached_token_stats")
    if stats is None:
        print("No cached_token_stats in response (is LMCache enabled?):")
        print(json.dumps(data, indent=2))
        return

    print(json.dumps(stats, indent=2))
    print(f"\nnum_lmcache_cached_tokens = {stats['num_lmcache_cached_tokens']}")


if __name__ == "__main__":
    main()
