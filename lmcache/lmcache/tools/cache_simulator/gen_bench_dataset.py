# SPDX-License-Identifier: Apache-2.0
"""
Convert LMCache lookup-hash JSONL logs into a ``vllm bench serve`` custom
dataset (JSONL with ``"prompt"`` and ``"output_tokens"`` fields).

The conversion preserves the **prefix-sharing structure** of the original
requests: requests that shared a chunk hash in the lookup logs will share the
same token prefix in the generated prompts.  This allows the synthetic
benchmark to exercise LMCache prefix caching in the same pattern as the
original production workload.

Algorithm
---------
1. Build a *safe vocabulary*: token IDs from the tokenizer whose single-token
   round-trip is stable (``decode([id])`` re-encodes back to exactly ``[id]``).
   Tokens that start with a leading space are preferred to avoid BPE merges at
   chunk boundaries.
2. For each unique chunk hash, deterministically seed an RNG from
   ``SHA-256(hash)`` and sample ``chunk_size`` token IDs from the safe vocab.
   The same hash always produces the same token sequence.
3. Per request: concatenate the token sequences for each full chunk, add
   ``tail_len = seq_len mod chunk_size`` tail tokens (unique per request so
   they are never accidentally cached), decode the whole list to text, and
   write ``{"prompt": <text>, "output_tokens": <N>}``.

Usage (module mode)::

    python3 -m lmcache.tools.cache_simulator.gen_bench_dataset \\
        -i /path/to/lookup_hashes/ \\
        --tokenizer /models/DeepSeek-V3 \\
        --output-len 128 \\
        -o bench_dataset.jsonl

Usage (CLI)::

    lmcache tool cache-simulator gen-dataset \\
        -i /path/to/lookup_hashes/ \\
        --tokenizer /models/DeepSeek-V3 \\
        --output-len 128 \\
        -o bench_dataset.jsonl
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import random
import sys

# ---------------------------------------------------------------------------
# Safe-vocabulary helpers
# ---------------------------------------------------------------------------


def build_safe_vocab(tokenizer: Any) -> list[int]:
    """Return token IDs that round-trip stably through the tokenizer.

    A *safe* token is one where::

        tokenizer.encode(tokenizer.decode([id]), add_special_tokens=False) == [id]

    These tokens can be concatenated safely — decoding a list of safe tokens
    and re-tokenizing it will reproduce the same token IDs as long as each
    token starts with a leading space (preventing BPE merges at boundaries).

    Tokens that begin with a space (``Ġ``, ``▁``, or ``" "``) are placed
    first so that ``hash_to_tokens`` preferentially picks space-prefixed
    tokens, making BPE boundary merges much less likely.

    Args:
        tokenizer: A HuggingFace ``PreTrainedTokenizer`` or
            ``PreTrainedTokenizerFast`` instance.

    Returns:
        A non-empty list of safe token IDs.  Raises ``RuntimeError`` if
        fewer than 10 safe tokens are found (the tokenizer is probably
        unsupported).
    """
    vocab_size = tokenizer.vocab_size
    safe_with_space: list[int] = []
    safe_other: list[int] = []

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode(
                [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception:
            continue
        if not decoded or not decoded.isprintable():
            continue
        # Re-encode and check round-trip
        try:
            re_encoded = tokenizer.encode(decoded, add_special_tokens=False)
        except Exception:
            continue
        if re_encoded != [token_id]:
            continue
        # Prefer tokens that start with whitespace (prevents inter-chunk merges)
        if decoded[0] in (" ", "\u0120", "\u2581"):
            safe_with_space.append(token_id)
        else:
            safe_other.append(token_id)

    result = safe_with_space + safe_other
    if len(result) < 10:
        raise RuntimeError(
            f"Only {len(result)} safe tokens found in the vocabulary. "
            "This tokenizer may not be supported."
        )
    return result


def hash_to_tokens(chunk_hash: str, n_tokens: int, vocab_ids: list[int]) -> list[int]:
    """Deterministically map *chunk_hash* to *n_tokens* token IDs.

    The mapping is a pure function of ``chunk_hash`` and ``n_tokens`` — the
    same inputs always produce the same output regardless of call order.

    Args:
        chunk_hash: Hex string identifying the chunk (e.g. ``"0xabcd1234"``).
        n_tokens: Number of token IDs to generate.
        vocab_ids: Pool of safe token IDs to sample from.

    Returns:
        A list of ``n_tokens`` token IDs drawn from ``vocab_ids``.
    """
    seed_bytes = hashlib.sha256(chunk_hash.encode()).digest()
    seed = int.from_bytes(seed_bytes, "big") % (2**31)
    rng = random.Random(seed)
    return [rng.choice(vocab_ids) for _ in range(n_tokens)]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def gen_bench_dataset(
    events: list[dict[str, Any]],
    tokenizer: Any,
    output_len: int,
    output_path: Path,
) -> int:
    """Convert lookup-hash events into a vllm bench serve custom dataset.

    Each event becomes one JSONL line with fields ``"prompt"`` (text) and
    ``"output_tokens"`` (int).  Chunk hashes are deterministically mapped to
    token sequences so that prefix-sharing in the original workload is
    reproduced in the synthetic prompts.

    Args:
        events: Lookup-hash events as returned by
            :func:`~lmcache.tools.cache_simulator.simulator.load_lookup_events`.
        tokenizer: A HuggingFace tokenizer loaded for the target model.
        output_len: Number of output tokens for every request in the dataset.
        output_path: Where to write the output JSONL file.

    Returns:
        Number of records written.
    """
    print("Building safe vocabulary …", flush=True)
    vocab_ids = build_safe_vocab(tokenizer)
    space_chars = (" ", "\u0120", "\u2581")
    n_space = sum(1 for t in vocab_ids if tokenizer.decode([t])[0] in space_chars)
    print(
        f"  Safe vocab size: {len(vocab_ids)} tokens ({n_space} space-prefixed)",
        flush=True,
    )

    # Cache: chunk_hash -> token_id list (memoized across requests)
    hash_cache: dict[str, list[int]] = {}
    n_written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for event in events:
            chunk_hashes: list[str] = event.get("chunk_hashes", [])
            chunk_size: int = event["chunk_size"]
            seq_len: int = event["seq_len"]

            # --- Full-chunk tokens ----------------------------------------
            all_token_ids: list[int] = []
            for h in chunk_hashes:
                if h not in hash_cache:
                    hash_cache[h] = hash_to_tokens(h, chunk_size, vocab_ids)
                all_token_ids.extend(hash_cache[h])

            # --- Tail tokens ----------------------------------------------
            # seq_len mod chunk_size tail tokens; always a miss in LMCache.
            # Use a per-request unique seed so tails are never accidentally
            # shared across requests.
            tail_len = seq_len - len(chunk_hashes) * chunk_size
            if tail_len > 0:
                # Build a per-request identifier: prefer request_id, fall back
                # to timestamp+index to ensure uniqueness.
                req_id = event.get(
                    "request_id", f"{event.get('timestamp', n_written)}_{n_written}"
                )
                tail_hash = f"__tail__{req_id}"
                all_token_ids.extend(hash_to_tokens(tail_hash, tail_len, vocab_ids))

            # --- Decode and write -----------------------------------------
            prompt_text = tokenizer.decode(
                all_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            record = {"prompt": prompt_text, "output_tokens": output_len}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 1000 == 0:
                print(f"  {n_written} / {len(events)} records written …", flush=True)

    return n_written


# ---------------------------------------------------------------------------
# CLI helpers (shared by main() and lmcache tool cache-simulator gen-dataset)
# ---------------------------------------------------------------------------


def add_gen_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all gen-dataset CLI flags onto *parser*.

    Called by both :func:`main` (``python -m`` entry point) and the
    ``lmcache tool cache-simulator gen-dataset`` sub-command in
    ``lmcache/cli/commands/tool/cache_simulator.py``.

    **When adding or removing a flag**, edit only this function — the
    ``lmcache tool`` command picks up the change automatically.

    Args:
        parser: The :class:`argparse.ArgumentParser` (or sub-parser) to
            populate.
    """
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        type=Path,
        help=(
            "One or more JSONL files or directories containing "
            "``lookup_hashes_*.jsonl`` files."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        metavar="PATH",
        help=(
            "Path or HuggingFace model name for the tokenizer used to decode "
            "synthetic token IDs into text.  Should match the model that will "
            "be benchmarked."
        ),
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        metavar="N",
        help="Number of output tokens for every request (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="bench_dataset.jsonl",
        metavar="FILE",
        help="Output JSONL file path (default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Truncate to the first N events after sorting by timestamp.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help=(
            "Filter events by ``model_name`` (exact match).  Useful when logs "
            "contain traffic for multiple models."
        ),
    )


def run_gen_dataset(args: argparse.Namespace) -> None:
    """Execute the gen-dataset workflow from a parsed argument namespace.

    Loads events via
    :func:`~lmcache.tools.cache_simulator.simulator.load_lookup_events`,
    loads the tokenizer, calls :func:`gen_bench_dataset`, and prints a
    summary.

    Args:
        args: Parsed CLI namespace produced by a parser that was populated
            with :func:`add_gen_dataset_arguments`.
    """
    # Lazy imports — keeps CLI startup fast
    # Third Party
    from transformers import AutoTokenizer

    # First Party
    from lmcache.tools.cache_simulator.simulator import load_lookup_events

    # ---- Load events -------------------------------------------------------
    print(f"Loading events from: {[str(p) for p in args.input]}", flush=True)
    events = load_lookup_events(
        args.input,
        model=args.model,
        max_samples=args.max_samples,
    )
    if not events:
        print("No events found — nothing to write.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(events):,} events.", flush=True)

    # ---- Load tokenizer ----------------------------------------------------
    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # ---- Convert -----------------------------------------------------------
    output_path = Path(args.output)
    n = gen_bench_dataset(
        events=events,
        tokenizer=tokenizer,
        output_len=args.output_len,
        output_path=output_path,
    )
    print(f"\nWrote {n:,} records to {output_path}")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m``."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert LMCache lookup-hash JSONL logs into a "
            "vllm bench serve custom dataset."
        )
    )
    add_gen_dataset_arguments(parser)
    args = parser.parse_args()
    run_gen_dataset(args)


if __name__ == "__main__":
    main()
