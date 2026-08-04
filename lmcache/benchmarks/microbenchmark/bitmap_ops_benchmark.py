#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the ranked fold (:func:`fold_unfold_ranked`).

Compares the pure-Python reference against the native C++ implementation across
request sizes, including a DeepSeek-scale hybrid case (1M tokens, 8 object
groups mixing full attention and sliding window). Run with::

    python benchmarks/microbenchmark/bitmap_ops_benchmark.py

The native op scans the packed ``Bitmap`` buffer directly -- no Python per-bit
loop and no ``Bitmap``<->tensor conversion -- so it stays sub-millisecond even
at multi-million-key scale where the Python scan takes hundreds of ms.
"""

# Standard
from collections.abc import Sequence
import time

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.bitmap_ops import fold_unfold_ranked, highest_set_bit
from lmcache.v1.distributed.bitmap_ops.fold import _fold_python, _unfold_python


def _python_pipeline(found, num_chunks, num_ranks, group_windows):
    """Pure-Python fold -> highest_set_bit -> unfold (no native ops)."""
    servable = _fold_python(found, num_chunks, num_ranks, group_windows)
    hit = highest_set_bit(servable) + 1  # -1 (no servable prefix) -> 0
    return hit, _unfold_python(hit, num_chunks, num_ranks, group_windows)


def _best_ms(fn, reps: int) -> float:
    """Best wall-clock time of ``fn`` over ``reps`` runs, in milliseconds."""
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def bench_case(
    label: str,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
    present_fraction: float,
    reps: int = 5,
) -> None:
    """Print Python vs native timings for one (size, fill) configuration."""
    num_keys = len(group_windows) * num_chunks * num_ranks
    if present_fraction >= 1.0:
        found = Bitmap(num_keys, num_keys)
    else:
        found = Bitmap(num_keys, int(num_keys * present_fraction))

    windows = list(group_windows)
    py_ms = _best_ms(
        lambda: _python_pipeline(found, num_chunks, num_ranks, windows),
        reps,
    )
    native_ms = _best_ms(
        lambda: fold_unfold_ranked(found, num_chunks, num_ranks, windows),
        reps,
    )
    speedup = py_ms / native_ms if native_ms else float("inf")
    print(
        f"{label:<34}keys={num_keys:>9}  python={py_ms:>9.2f}ms  "
        f"native={native_ms:>8.3f}ms  speedup={speedup:>7.1f}x"
    )


def main() -> None:
    """Run the benchmark grid."""
    # 8 groups, mix of full attention and sliding window (DeepSeek-like hybrid).
    dpsk_windows = (-1, -1, -1, -1, 4, 4, 8, 1)

    print("== DeepSeek 1M tokens @ chunk_size=256 (num_chunks=4096), all present ==")
    bench_case("dpsk, world_size=1", 4096, 1, dpsk_windows, 1.0)
    bench_case("dpsk, world_size=8", 4096, 8, dpsk_windows, 1.0)

    print("\n== same, 50% prefix present (realistic) ==")
    bench_case("dpsk, world_size=8", 4096, 8, dpsk_windows, 0.5)

    print("\n== small request ==")
    bench_case("4K tokens @256 (16 chunks)", 16, 8, dpsk_windows, 1.0)

    print("\n== stress: chunk_size=16 -> 62500 chunks (4M keys) ==")
    bench_case("stress, world_size=8", 62500, 8, dpsk_windows, 1.0)


if __name__ == "__main__":
    main()
