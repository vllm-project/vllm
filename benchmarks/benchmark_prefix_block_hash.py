# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simple benchmark to compare prefix-cache block hashing algorithms.

Example:
    python benchmark_prefix_block_hash.py --num-blocks 20000 --block-size 32
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, hash_block_tokens, init_none_hash

SUPPORTED_ALGOS = ("sha256", "sha256_cbor", "xxhash", "xxhash_cbor")


def _generate_blocks(
    num_blocks: int, block_size: int, vocab_size: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(block_size)]
        for _ in range(num_blocks)
    ]


def _hash_all_blocks(
    hash_fn: Callable[[object], bytes],
    blocks: Iterable[Sequence[int]],
) -> float:
    parent_hash: BlockHash | None = None
    start = time.perf_counter()
    for block in blocks:
        parent_hash = hash_block_tokens(hash_fn, parent_hash, block, extra_keys=None)
    end = time.perf_counter()
    return end - start


def _benchmark(
    hash_algo: str,
    blocks: list[list[int]],
    trials: int,
) -> tuple[float, float, float] | None:
    try:
        hash_fn = get_hash_fn_by_name(hash_algo)
        init_none_hash(hash_fn)
        timings = [_hash_all_blocks(hash_fn, blocks) for _ in range(trials)]
    except ModuleNotFoundError as exc:
        print(f"Skipping {hash_algo}: {exc}", file=sys.stderr)
        return None

    avg = statistics.mean(timings)
    best = min(timings)
    # throughput: tokens / second
    tokens_hashed = len(blocks) * len(blocks[0])
    throughput = tokens_hashed / best
    return avg, best, throughput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-blocks", type=int, default=10000, help="Block count.")
    parser.add_argument("--block-size", type=int, default=32, help="Tokens per block.")
    parser.add_argument(
        "--vocab-size", type=int, default=32000, help="Token id range [0, vocab_size)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of timed trials per algorithm."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=SUPPORTED_ALGOS,
        choices=SUPPORTED_ALGOS,
        help="Hash algorithms to benchmark.",
    )
    args = parser.parse_args()

    blocks = _generate_blocks(
        args.num_blocks, args.block_size, args.vocab_size, args.seed
    )
    print(
        f"Benchmarking {len(args.algorithms)} algorithms on "
        f"{args.num_blocks} blocks (block size={args.block_size})."
    )

    for algo in args.algorithms:
        result = _benchmark(algo, blocks, args.trials)
        if result is None:
            continue

        avg, best, throughput = result
        print(
            f"{algo:14s} avg: {avg:.6f}s  best: {best:.6f}s  "
            f"throughput: {throughput / 1e6:.2f}M tokens/s"
        )


if __name__ == "__main__":
    main()
