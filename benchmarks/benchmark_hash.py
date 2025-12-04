# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Micro benchmark comparing built-in hash(), SHA-256, and xxHash.

This focuses on a single test payload shaped like the prefix-cache hash input:
    (32-byte bytes object, 32-int tuple)

Usage:
    python benchmarks/hash_micro_benchmark.py --iterations 20000
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections.abc import Callable, Iterable

from vllm.utils.hashing import sha256, xxhash


def _generate_test_data(seed: int) -> tuple[bytes, tuple[int, ...]]:
    """Generate a deterministic test payload."""
    random.seed(seed)
    bytes_data = bytes(random.getrandbits(8) for _ in range(32))
    int_tuple = tuple(random.randint(1, 1_000_000) for _ in range(32))
    return (bytes_data, int_tuple)


def _benchmark_func(func: Callable[[tuple], object], data: tuple, iterations: int):
    """Return (avg_seconds, std_seconds) for hashing `data` `iterations` times."""
    times: list[float] = []

    # Warm-up to avoid first-run noise.
    for _ in range(200):
        func(data)

    for _ in range(iterations):
        start = time.perf_counter()
        func(data)
        end = time.perf_counter()
        times.append(end - start)

    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return avg, std


def _run_benchmarks(
    benchmarks: Iterable[tuple[str, Callable[[tuple], object]]],
    data: tuple,
    iterations: int,
):
    """Yield (name, avg, std) for each benchmark, skipping unavailable ones."""
    for name, func in benchmarks:
        try:
            avg, std = _benchmark_func(func, data, iterations)
        except ModuleNotFoundError as exc:
            print(f"Skipping {name}: {exc}")
            continue
        yield name, avg, std


def builtin_hash(data: tuple) -> int:
    """Wrapper for Python's built-in hash()."""
    return hash(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Number of measured iterations per hash function.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for test payload."
    )
    args = parser.parse_args()

    data = _generate_test_data(args.seed)
    benchmarks = (
        ("SHA256 (pickle)", sha256),
        ("xxHash (pickle)", xxhash),
        ("built-in hash()", builtin_hash),
    )

    print("=" * 60)
    print("HASH FUNCTION MICRO BENCHMARK")
    print("=" * 60)
    print("Test data: (32-byte bytes object, 32-int tuple)")
    print(f"Iterations: {args.iterations:,}")
    print("=" * 60)

    results = list(_run_benchmarks(benchmarks, data, args.iterations))
    builtin_entry = next((r for r in results if r[0] == "built-in hash()"), None)

    print("\nResults:")
    for name, avg, std in results:
        print(f"  {name:16s}: {avg * 1e6:8.2f} ± {std * 1e6:6.2f} μs")

    if builtin_entry:
        _, builtin_avg, _ = builtin_entry
        print("\n" + "=" * 60)
        print("SUMMARY (relative to built-in hash())")
        print("=" * 60)
        for name, avg, _ in results:
            if name == "built-in hash()":
                continue
            speed_ratio = avg / builtin_avg
            print(f"• {name} is {speed_ratio:.1f}x slower than built-in hash()")
    else:
        print("\nBuilt-in hash() result missing; cannot compute speed ratios.")


if __name__ == "__main__":
    main()
