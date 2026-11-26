# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense attention prefix-cache alignments on a tokenized JSONL trace."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_trace(path: Path) -> Iterator[tuple[list[int], list[int]]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                prompt = row["prompt_token_ids"]
                output = row.get("output_token_ids", [])
                if not prompt or not all(
                    isinstance(ids, list)
                    and all(type(token) is int and token >= 0 for token in ids)
                    for ids in (prompt, output)
                ):
                    raise ValueError("expected token ID arrays and a nonempty prompt")
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
            yield prompt, output


def replay(
    requests: Iterable[tuple[list[int], list[int]]], alignments: set[int]
) -> tuple[int, int, Counter[int], Counter[int]]:
    """Measure all policies in one pass over an initially empty, dense cache."""
    unit = math.gcd(*alignments)
    # Intern each (prefix, next unit) pair. Equal chunks after different
    # histories must never hit, even if Python's tuple hashes collide.
    edges: dict[tuple[int, tuple[int, ...]], int] = {}
    hits: Counter[int] = Counter()
    saved: Counter[int] = Counter()
    count = prompt_tokens = 0
    for prompt, output in requests:
        count += 1
        prompt_tokens += len(prompt)
        # The last sampled token has no KV until it is fed back to the model.
        computed = prompt + output[:-1]
        parent = matched = 0
        for end in range(unit, len(computed) + 1, unit):
            key = (parent, tuple(computed[end - unit : end]))
            child = edges.get(key)
            if child is None:
                child = edges[key] = len(edges) + 1
            elif end < len(prompt):
                # A fully cached prompt still needs its last token for logits.
                matched = end
            parent = child

        for alignment in alignments:
            reused = matched // alignment * alignment
            hits[alignment] += reused > 0
            saved[alignment] += reused
    return count, prompt_tokens, hits, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--block-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--hash-units", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if any(size <= 0 for size in args.block_sizes + args.hash_units):
        parser.error("block sizes and hash units must be positive")
    if any(size % unit for size in args.block_sizes for unit in args.hash_units):
        parser.error("every hash unit must divide every group block size")

    baseline = math.lcm(*args.block_sizes)
    units = sorted(set(args.hash_units))
    try:
        count, tokens, hits, saved = replay(read_trace(args.trace), {baseline, *units})
    except (ValueError, OSError) as e:
        parser.error(str(e))
    if not count:
        parser.error("empty trace")

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "policy",
            "alignment",
            "requests",
            "prompt_tokens",
            "request_hit_rate",
            "token_hit_rate",
            "prefill_tokens_saved",
            "extra_saved_vs_main",
        ]
    )
    for policy, alignment in [("main", baseline), *(("rfc", u) for u in units)]:
        writer.writerow(
            [
                policy,
                alignment,
                count,
                tokens,
                f"{hits[alignment] / count:.4%}",
                f"{saved[alignment] / tokens:.4%}",
                saved[alignment],
                saved[alignment] - saved[baseline],
            ]
        )


if __name__ == "__main__":
    main()
