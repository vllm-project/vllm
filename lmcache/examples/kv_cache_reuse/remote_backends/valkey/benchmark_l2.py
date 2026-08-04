# SPDX-License-Identifier: Apache-2.0
"""
End-to-end L2 benchmark for LMCache with vLLM.

Solves the 64k L2 eviction problem from the 12-March benchmark session:
flood prompts that share a prefix with the test prompt produce identical
chunk hashes (rolling prefix hashes), so LRU never evicts the overlapping
chunks.  This script generates flood prompts with **completely disjoint
token content** to guarantee different chunk hashes and full L1 eviction.

Usage (run from the benchmark EC2 instance):

    # Generate prompts + flood files (one-time)
    python benchmark_l2.py generate \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --context-tokens 65536 \
        --num-floods 3 \
        --output-dir /home/ubuntu/bench_prompts

    # Run the full benchmark (cold → flood → L2)
    python benchmark_l2.py run \
        --prompt-dir /home/ubuntu/bench_prompts \
        --vllm-url http://localhost:8000 \
        --valkey-nodes <node1>,<node2>,<node3> \
        --valkey-port 6379

Requirements:
    - vLLM running with LMCacheConnectorV1
    - transformers (for tokenizer)
    - redis-py (for keyspace_hits checking)
"""

# Standard
from pathlib import Path
import argparse
import json
import random
import string
import sys
import time

# Third Party
import requests


def _random_text(num_chars: int, seed: int) -> str:
    """Generate random ASCII text that tokenizes to unique tokens.

    Uses distinct vocabulary per seed so that no two generated texts
    share a token-level prefix, which would produce identical rolling
    chunk hashes in LMCache.
    """
    rng = random.Random(seed)
    # Mix words of varying length to get dense, unique tokenization.
    # Each "word" is 3-10 random lowercase chars followed by a space.
    parts = []
    written = 0
    while written < num_chars:
        word_len = rng.randint(3, 10)
        word = "".join(rng.choices(string.ascii_lowercase, k=word_len))
        parts.append(word)
        written += word_len + 1  # +1 for space
    return " ".join(parts)[:num_chars]


def _make_prompt_payload(
    text: str,
    model: str,
    max_tokens: int = 1,
) -> dict:
    """Build an OpenAI-compatible /v1/completions payload."""
    return {
        "model": model,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def cmd_generate(args):
    """Generate test prompt and disjoint flood prompts."""
    # Third Party
    from transformers import AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    target_tokens = args.context_tokens

    # We over-generate text and then truncate to exactly target_tokens
    # after tokenization.  Factor of 6 chars/token is conservative.
    chars_estimate = target_tokens * 6

    # ── Test prompt ──
    print(f"Generating test prompt (~{target_tokens} tokens)...")
    test_text = _random_text(chars_estimate, seed=42)
    test_ids = tokenizer.encode(test_text, add_special_tokens=False)
    test_ids = test_ids[:target_tokens]
    test_text_truncated = tokenizer.decode(test_ids)
    actual_tokens = len(tokenizer.encode(test_text_truncated, add_special_tokens=False))
    print(f"  Test prompt: {actual_tokens} tokens")

    payload = _make_prompt_payload(test_text_truncated, args.model)
    test_path = out / "prompt.json"
    test_path.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"  Saved: {test_path}")

    # ── Flood prompts (completely disjoint content) ──
    for i in range(args.num_floods):
        # Use seeds far apart from test (42) and from each other
        seed = 1000 + i * 1000
        print(f"Generating flood prompt {i + 1}/{args.num_floods} (seed={seed})...")
        flood_text = _random_text(chars_estimate, seed=seed)
        flood_ids = tokenizer.encode(flood_text, add_special_tokens=False)
        flood_ids = flood_ids[:target_tokens]
        flood_text_truncated = tokenizer.decode(flood_ids)
        flood_tokens = len(
            tokenizer.encode(flood_text_truncated, add_special_tokens=False)
        )
        print(f"  Flood {i + 1}: {flood_tokens} tokens")

        # Verify zero prefix overlap at the token level
        test_first_chunk = test_ids[:256]
        flood_first_chunk = flood_ids[:256]
        if test_first_chunk == flood_first_chunk:
            print(
                "  WARNING: first chunk matches test prompt! Retrying with "
                "different seed would be needed."
            )
        else:
            overlap = 0
            for a, b in zip(test_ids, flood_ids, strict=False):
                if a != b:
                    break
                overlap += 1
            print(
                f"  Token prefix overlap with test: {overlap} "
                f"(< chunk_size=256 → OK, different chunk hashes)"
            )

        flood_payload = _make_prompt_payload(flood_text_truncated, args.model)
        flood_path = out / f"flood_{i + 1}.json"
        flood_path.write_text(json.dumps(flood_payload, ensure_ascii=False))
        print(f"  Saved: {flood_path}")

    print(f"\nAll prompts saved to {out}/")
    print(f"  prompt.json        — test prompt ({actual_tokens} tokens)")
    for i in range(args.num_floods):
        print(f"  flood_{i + 1}.json      — flood prompt (disjoint content)")


def _get_keyspace_hits(nodes: list[str], port: int) -> tuple[int, list[int]]:
    """Query keyspace_hits from all Valkey/Redis cluster nodes."""
    # Third Party
    import redis

    per_node = []
    for node in nodes:
        try:
            r = redis.Redis(host=node, port=port, socket_timeout=5)
            info = r.info("stats")
            hits = info.get("keyspace_hits", 0)
            per_node.append(hits)
            r.close()
        except Exception as e:
            print(f"  WARNING: could not reach {node}:{port} — {e}")
            per_node.append(-1)
    total = sum(h for h in per_node if h >= 0)
    return total, per_node


def _send_request(url: str, payload_path: Path, timeout: int = 600) -> float:
    """Send a completion request and return wall-clock time in ms."""
    payload = json.loads(payload_path.read_text())
    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return elapsed_ms


def cmd_run(args):
    """Run the full L2 benchmark: cold → flood → L2 retrieval."""
    prompt_dir = Path(args.prompt_dir)
    prompt_path = prompt_dir / "prompt.json"
    if not prompt_path.exists():
        print(f"ERROR: {prompt_path} not found. Run 'generate' first.")
        sys.exit(1)

    flood_paths = sorted(prompt_dir.glob("flood_*.json"))
    if not flood_paths:
        print(f"ERROR: no flood_*.json found in {prompt_dir}. Run 'generate' first.")
        sys.exit(1)

    nodes = [n.strip() for n in args.valkey_nodes.split(",") if n.strip()]
    url = args.vllm_url.rstrip("/")

    print(f"Prompt: {prompt_path}")
    print(f"Floods: {[p.name for p in flood_paths]}")
    print(f"Valkey nodes: {nodes}")
    print(f"vLLM: {url}")
    print()

    # ── Step 1: Cold request (compute + store to L1 + L2) ──
    print("=== Step 1: Cold request (store to L1 + L2) ===")
    cold_ms = _send_request(url, prompt_path)
    print(f"  Cold TTFT: {cold_ms:.0f}ms")
    time.sleep(3)

    # ── Step 2: Flood L1 with disjoint prompts ──
    print(f"\n=== Step 2: Flood L1 ({len(flood_paths)} disjoint prompts) ===")
    for fp in flood_paths:
        print(f"  Sending {fp.name}...", end="", flush=True)
        flood_ms = _send_request(url, fp)
        print(f" {flood_ms:.0f}ms")
        time.sleep(1)
    time.sleep(3)

    # ── Step 3: Record keyspace_hits before L2 ──
    print("\n=== Step 3: Check keyspace_hits (before L2) ===")
    before_total, before_per_node = _get_keyspace_hits(nodes, args.valkey_port)
    print(f"  TOTAL hits: {before_total}  per-node: {before_per_node}")

    # ── Step 4: L2 retrieval ──
    print("\n=== Step 4: L2 retrieval (same prompt, L1 should be evicted) ===")
    l2_ms = _send_request(url, prompt_path)
    print(f"  L2 Hit TTFT: {l2_ms:.0f}ms")
    time.sleep(1)

    # ── Step 5: Record keyspace_hits after L2 ──
    print("\n=== Step 5: Check keyspace_hits (after L2) ===")
    after_total, after_per_node = _get_keyspace_hits(nodes, args.valkey_port)
    print(f"  TOTAL hits: {after_total}  per-node: {after_per_node}")

    delta = after_total - before_total
    delta_per_node = [
        a - b
        for a, b in zip(after_per_node, before_per_node, strict=True)
        if a >= 0 and b >= 0
    ]
    print(f"\n  keyspace_hits Δ: +{delta} (per-node: {delta_per_node})")

    # ── Verdict ──
    print("\n" + "=" * 60)
    if delta > 0:
        print("✓ L2 RETRIEVAL CONFIRMED")
        print(f"  Cold TTFT:   {cold_ms:.0f}ms")
        print(f"  L2 Hit TTFT: {l2_ms:.0f}ms")
        print(f"  Speedup:     {cold_ms / l2_ms:.1f}x")
        print(f"  keyspace_hits Δ: +{delta}")
    else:
        print("✗ L2 RETRIEVAL NOT DETECTED (keyspace_hits unchanged)")
        print("  The L1 cache was likely not fully evicted.")
        print("  Possible causes:")
        print("    - max_local_cpu_size too large (floods fit alongside test data)")
        print("    - Not enough flood prompts to fill L1")
        print("    - Flood prompts share prefix with test prompt")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end L2 benchmark for LMCache + vLLM"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──
    gen = sub.add_parser("generate", help="Generate test + flood prompts")
    gen.add_argument("--model", required=True, help="HF model name for tokenizer")
    gen.add_argument("--context-tokens", type=int, default=65536)
    gen.add_argument(
        "--num-floods", type=int, default=3, help="Number of disjoint flood prompts"
    )
    gen.add_argument("--output-dir", required=True)

    # ── run ──
    run = sub.add_parser("run", help="Run the L2 benchmark")
    run.add_argument(
        "--prompt-dir",
        required=True,
        help="Directory with prompt.json and flood_*.json",
    )
    run.add_argument("--vllm-url", default="http://localhost:8000")
    run.add_argument(
        "--valkey-nodes", required=True, help="Comma-separated Valkey cluster node IPs"
    )
    run.add_argument("--valkey-port", type=int, default=6379)

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
