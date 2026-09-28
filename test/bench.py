#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark for ``memcpy_utils``: random-block throughput.

Measures the copy path used by ``memcpy_mt`` (and its underlying numba
kernel) on a realistic workload: many small, randomly placed block
copies per timed window. Five sweeps are provided:

  * ``threads``  — vary the numba thread-pool size
  * ``block``    — vary the block size (launch cost amortization)
  * ``subchunk`` — vary the sub-chunk size handed to the kernel
  * ``small``    — numpy slice assignment vs ``_copy_kernel`` on small blocks
  * ``memcpy``   — end-to-end ``memcpy_mt`` across block sizes

Notes on the measurement setup:

  * ``_copy_kernel`` copies one contiguous span per launch, so a
    random-block workload becomes one launch per block. The per-launch
    fixed cost (thread wake-up, prange split, final barrier) is visible
    in every row of the ``block`` sweep.
  * ``warmup_kernel`` compiles the JIT once with the same thread count
    the timed runs will use. Resetting the pool to 1 there would leak
    into sweeps that only call ``set_threads`` once outside their loop.
"""

import argparse
import time
from typing import Callable, List

import numpy as np

from vllm.utils import memcpy_utils as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_size(text: str) -> int:
    """Parse ``'4GiB'`` / ``'512MiB'`` / ``'1073741824'`` -> bytes."""
    text = text.strip()
    suffixes = {
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
    }
    low = text.lower()
    for suf, mul in suffixes.items():
        if low.endswith(suf):
            return int(float(low[: -len(suf)]) * mul)
    return int(text)


def format_size(num_bytes: float, decimal_places: int = 2) -> str:
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    base = 1024
    size = float(num_bytes)
    e = 0
    while size >= base and e < len(units) - 1:
        size /= base
        e += 1
    return f"{size:.{decimal_places}f} {units[e]}"


def bench(
    fn: Callable[[], None],
    bytes_per_run: int,
    n_runs: int,
    warmup: int = 1,
) -> float:
    """Return the mean bandwidth in GiB/s over ``n_runs`` timed runs."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        samples.append(bytes_per_run / dt / (1024 ** 3))
    return sum(samples) / len(samples)


def print_row(cols: List[str], widths: List[int]) -> None:
    print(" | ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# Kernel binding (from memcpy_utils, no local re-definition)
# ---------------------------------------------------------------------------

kernel = mod._copy_kernel

if mod._HAS_NUMBA:
    def set_threads(n: int) -> None:
        mod.set_num_threads(min(n, mod._MAX_NUMBA_THREADS))
else:
    def set_threads(n: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Buffers and index arrays
# ---------------------------------------------------------------------------

def make_buffers(size: int):
    src = np.arange(size, dtype=np.uint8)
    dst = np.empty_like(src)
    return src, dst


def make_indices(n_blocks: int, iters: int):
    """Draw a fixed random workload: ``iters`` source/destination blocks."""
    idx_s = np.random.randint(0, n_blocks, size=iters, dtype=np.intp)
    idx_d = np.random.randint(0, n_blocks, size=iters, dtype=np.intp)
    return idx_s, idx_d


def make_offsets(idx_s, idx_d, block):
    """Precompute byte offsets as Python lists for the hot loop."""
    return (idx_s * block).tolist(), (idx_d * block).tolist()


def warmup_kernel(src, dst, off_s, off_d, block, sub_chunk, n_threads) -> None:
    """
    Trigger the lazy JIT compile so it is not counted in any timing.

    ``n_threads`` must match the thread count the timed runs will use.
    Numba's compilation is thread-count agnostic, but resetting the pool
    here would leak into sweeps that only call ``set_threads`` once
    outside their loop.
    """
    if not mod._HAS_NUMBA:
        return
    set_threads(n_threads)
    kernel(src[off_s[0]:off_s[0] + block],
           dst[off_d[0]:off_d[0] + block],
           block, sub_chunk)


def run_kernel_window(src, dst, off_s, off_d, block, sub_chunk, iters) -> None:
    """One timed window: ``iters`` consecutive random-block copies."""
    k = kernel
    for i in range(iters):
        s = off_s[i]
        d = off_d[i]
        k(src[s:s + block], dst[d:d + block], block, sub_chunk)


def run_numpy_window(src, dst, off_s, off_d, block, iters) -> None:
    """Same window but with numpy slice assignment instead of the kernel."""
    for i in range(iters):
        s = off_s[i]
        d = off_d[i]
        dst[d:d + block] = src[s:s + block]


def run_memcpy_mt_window(src, dst, off_s, off_d, block, iters,
                         max_copy_threads, sub_chunk) -> None:
    """Same window but through the public ``memcpy_mt`` wrapper."""
    fn = mod.memcpy_mt
    for i in range(iters):
        s = off_s[i]
        d = off_d[i]
        fn(src[s:s + block], dst[d:d + block], block,
           sub_chunk_bytes=sub_chunk,
           max_copy_threads=max_copy_threads)


def fixed_threads(args) -> int:
    """
    Thread count used by sweeps that do not vary the thread pool size.

    Prefer the tuned default from ``memcpy_utils`` (``_MAX_COPY_THREADS``);
    if the user passed ``--threads``, honour that instead. The idea is to
    avoid silently picking the largest value in ``--threads-list``, which
    would push those sweeps into the barrier-bound regime.
    """
    if args.threads is not None:
        return args.threads
    return min(args.threads_list,
               key=lambda t: abs(t - mod._MAX_COPY_THREADS))


# ---------------------------------------------------------------------------
# Sweep: thread count
# ---------------------------------------------------------------------------

def sweep_threads(args) -> None:
    """Vary the numba thread-pool size at fixed block and sub-chunk."""
    block = args.block
    sc = args.sub_chunk
    iters = args.iters
    n_blocks = args.size // block
    if n_blocks < 1 or iters < 1:
        return

    src, dst = make_buffers(args.size)
    idx_s, idx_d = make_indices(n_blocks, iters)
    off_s, off_d = make_offsets(idx_s, idx_d, block)

    threads_list = ([args.threads] if args.threads is not None
                    else args.threads_list)
    warmup_kernel(src, dst, off_s, off_d, block, sc, threads_list[0])

    bytes_per_run = iters * block

    print("=== Thread sweep (_copy_kernel) ===")
    print(f"  size        : {format_size(args.size)} ({n_blocks} blocks)")
    print(f"  block       : {format_size(block)}")
    print(f"  iters       : {iters}")
    print(f"  bytes/window: {format_size(bytes_per_run)}")
    print(f"  sub_chunk   : {format_size(sc)}")
    print(f"  launches/win: {iters}")
    print()

    widths = [10, 22]
    print_row(["Threads", "Mean (GiB/s)"], widths)
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for T in threads_list:
        if T < 1:
            continue
        set_threads(T)

        def fn() -> None:
            run_kernel_window(src, dst, off_s, off_d, block, sc, iters)

        mean = bench(fn, bytes_per_run, args.runs, warmup=args.warmup)
        print_row([str(T), f"{mean:.2f}"], widths)
    print()


# ---------------------------------------------------------------------------
# Sweep: block size
# ---------------------------------------------------------------------------

def sweep_block(args) -> None:
    """Vary the block size at fixed thread count and sub-chunk.

    Larger blocks amortize the per-launch fixed cost; small blocks expose
    it. This is the sweep that shows how much of the bandwidth is eaten
    by kernel launch overhead rather than data movement.
    """
    T = fixed_threads(args)
    sc = args.sub_chunk
    iters = args.iters
    set_threads(T)

    src, dst = make_buffers(args.size)

    widths = [12, 10, 22, 22, 12]
    print("=== Block sweep (_copy_kernel) ===")
    print(f"  size        : {format_size(args.size)}")
    print(f"  threads     : {T}")
    print(f"  iters       : {iters}")
    print(f"  sub_chunk   : {format_size(sc)}")
    print()
    print_row(
        ["Block", "Blocks", "Bytes/window", "Mean (GiB/s)", "us/launch"],
        widths,
    )
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for block in args.blocks:
        if block <= 0 or block > args.size:
            continue
        n_blocks = args.size // block
        if n_blocks < 1:
            continue
        run_iters = min(iters, n_blocks)

        idx_s, idx_d = make_indices(n_blocks, run_iters)
        off_s, off_d = make_offsets(idx_s, idx_d, block)
        warmup_kernel(src, dst, off_s, off_d, block, sc, T)
        bytes_per_run = run_iters * block

        def fn(off_s=off_s, off_d=off_d, block=block,
               iters=run_iters) -> None:
            run_kernel_window(src, dst, off_s, off_d, block, sc, iters)

        mean = bench(fn, bytes_per_run, args.runs, warmup=args.warmup)
        us_per_launch = 1e6 * (bytes_per_run / (mean * (1024 ** 3))) / run_iters

        print_row(
            [
                format_size(block),
                str(n_blocks),
                format_size(bytes_per_run),
                f"{mean:.2f}",
                f"{us_per_launch:.1f}",
            ],
            widths,
        )
    print()


# ---------------------------------------------------------------------------
# Sweep: sub-chunk size
# ---------------------------------------------------------------------------

def sweep_subchunk(args) -> None:
    """Vary the sub-chunk size at fixed thread count and block size.

    For each launch the kernel splits ``block`` into ``ceil(block / sc)``
    sub-chunks and distributes them across ``prange``. Small sub-chunks
    give more parallel tasks per launch but more per-task overhead;
    large sub-chunks reduce task count and can under-utilize threads.
    """
    T = fixed_threads(args)
    block = args.block
    iters = args.iters
    n_blocks = args.size // block
    if n_blocks < 1 or iters < 1:
        return
    set_threads(T)

    src, dst = make_buffers(args.size)
    idx_s, idx_d = make_indices(n_blocks, iters)
    off_s, off_d = make_offsets(idx_s, idx_d, block)
    bytes_per_run = iters * block

    widths = [12, 10, 12, 22]
    print("=== Sub-chunk sweep (_copy_kernel) ===")
    print(f"  size        : {format_size(args.size)}")
    print(f"  block       : {format_size(block)}")
    print(f"  threads     : {T}")
    print(f"  iters       : {iters}")
    print(f"  bytes/window: {format_size(bytes_per_run)}")
    print()
    print_row(
        ["Sub-chunk", "Splits/blk", "Tasks/launch", "Mean (GiB/s)"],
        widths,
    )
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for sc in args.sub_chunks:
        if sc <= 0:
            continue
        eff_sc = min(sc, block)
        splits = (block + eff_sc - 1) // eff_sc
        tasks = iters * splits

        warmup_kernel(src, dst, off_s, off_d, block, eff_sc, T)

        def fn(eff_sc=eff_sc) -> None:
            run_kernel_window(src, dst, off_s, off_d, block, eff_sc, iters)

        mean = bench(fn, bytes_per_run, args.runs, warmup=args.warmup)
        print_row(
            [
                format_size(sc),
                str(splits),
                str(tasks),
                f"{mean:.2f}",
            ],
            widths,
        )
    print()


# ---------------------------------------------------------------------------
# Sweep: small blocks — numpy vs numba kernel
# ---------------------------------------------------------------------------

def sweep_small(args) -> None:
    """
    Compare numpy slice assignment against ``_copy_kernel`` on small blocks.

    ``memcpy_mt`` short-circuits to numpy when ``size <= sub_chunk_bytes``
    (8 KiB by default). This sweep checks whether that threshold should be
    raised: at 64 KiB–256 KiB, the numpy path should still be much faster
    than paying a numba launch per block.
    """
    T = fixed_threads(args)
    iters = args.iters
    set_threads(T)

    src, dst = make_buffers(args.size)

    widths = [12, 22, 22, 16]
    print("=== Small-block: numpy vs _copy_kernel ===")
    print(f"  size        : {format_size(args.size)}")
    print(f"  threads     : {T}")
    print(f"  iters       : {iters}")
    print()
    print_row(
        ["Block", "numpy (GiB/s)", "numba (GiB/s)", "numba/numpy"],
        widths,
    )
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for block in args.small_blocks:
        if block <= 0 or block > args.size:
            continue
        n_blocks = args.size // block
        if n_blocks < 1:
            continue
        run_iters = min(iters, n_blocks)

        idx_s, idx_d = make_indices(n_blocks, run_iters)
        off_s, off_d = make_offsets(idx_s, idx_d, block)
        bytes_per_run = run_iters * block

        eff_sc = min(args.sub_chunk, block)
        warmup_kernel(src, dst, off_s, off_d, block, eff_sc, T)

        def fn_np(off_s=off_s, off_d=off_d, block=block,
                  iters=run_iters) -> None:
            run_numpy_window(src, dst, off_s, off_d, block, iters)

        def fn_nb(off_s=off_s, off_d=off_d, block=block,
                  iters=run_iters, eff_sc=eff_sc) -> None:
            run_kernel_window(src, dst, off_s, off_d, block, eff_sc, iters)

        np_bw = bench(fn_np, bytes_per_run, args.runs, warmup=args.warmup)
        nb_bw = bench(fn_nb, bytes_per_run, args.runs, warmup=args.warmup)
        ratio = nb_bw / np_bw if np_bw > 0 else float("inf")

        print_row(
            [
                format_size(block),
                f"{np_bw:.2f}",
                f"{nb_bw:.2f}",
                f"{ratio:.2f}x",
            ],
            widths,
        )
    print()


# ---------------------------------------------------------------------------
# Sweep: end-to-end memcpy_mt
# ---------------------------------------------------------------------------

def sweep_memcpy(args) -> None:
    """
    End-to-end ``memcpy_mt`` across block sizes.

    Unlike the ``small`` sweep this goes through the public wrapper, so
    it includes ``_flatten_u8``, argument validation, and the two
    ``set_num_threads`` calls per invocation performed by ``_dispatch``.
    The gap to the raw-kernel numbers is the wrapper's per-call cost.
    """
    T = fixed_threads(args)
    sc = args.sub_chunk
    iters = args.iters

    src, dst = make_buffers(args.size)

    widths = [12, 22, 22]
    print("=== memcpy_mt end-to-end ===")
    print(f"  size        : {format_size(args.size)}")
    print(f"  threads     : {T}")
    print(f"  sub_chunk   : {format_size(sc)}")
    print(f"  iters       : {iters}")
    print()
    print_row(
        ["Block", "Mean (GiB/s)", "GB/s (decimal)"],
        widths,
    )
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for block in args.blocks:
        if block <= 0 or block > args.size:
            continue
        n_blocks = args.size // block
        if n_blocks < 1:
            continue
        run_iters = min(iters, n_blocks)

        idx_s, idx_d = make_indices(n_blocks, run_iters)
        off_s, off_d = make_offsets(idx_s, idx_d, block)
        bytes_per_run = run_iters * block

        # Warm up the numpy short-circuit and the numba kernel once.
        mod.memcpy_mt(src[:block], dst[:block], block,
                      sub_chunk_bytes=sc, max_copy_threads=T)

        def fn(off_s=off_s, off_d=off_d, block=block,
               iters=run_iters) -> None:
            run_memcpy_mt_window(src, dst, off_s, off_d, block, iters, T, sc)

        mean = bench(fn, bytes_per_run, args.runs, warmup=args.warmup)

        print_row(
            [
                format_size(block),
                f"{mean:.2f}",
                f"{mean * (1024 ** 3) / 1e9:.2f}",
            ],
            widths,
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="memcpy_utils benchmark: random-block throughput",
    )
    parser.add_argument(
        "--sweep",
        choices=["threads", "block", "subchunk", "small", "memcpy", "all"],
        default="all",
        help="which dimension to sweep (default: all)",
    )

    parser.add_argument("--size", type=parse_size, default=4 * (1 << 30),
                        help="total buffer size (default 4GiB)")
    parser.add_argument("--block", type=parse_size, default=1 << 20,
                        help="block size for threads/subchunk sweeps (default 1MiB)")
    parser.add_argument("--sub-chunk", type=parse_size, default=8 << 10,
                        help="sub-chunk size for threads/block sweeps (default 8KiB)")
    parser.add_argument("--iters", type=int, default=2048,
                        help="number of random blocks per timed window (default 2048)")

    parser.add_argument("--threads", type=int, default=None,
                        help="single-mode: one thread count to test; also used "
                             "as the fixed thread count for non-thread sweeps")
    parser.add_argument("--threads-list", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12, 16, 24, 32],
                        help="multi-mode: thread counts to sweep; the entry "
                             "closest to _MAX_COPY_THREADS is used as the "
                             "default for non-thread sweeps")

    parser.add_argument("--blocks", type=parse_size, nargs="+",
                        default=[4 << 10, 16 << 10, 64 << 10, 256 << 10,
                                 1 << 20, 4 << 20, 16 << 20],
                        help="block sizes for block/memcpy sweeps")

    parser.add_argument("--sub-chunks", type=parse_size, nargs="+",
                        default=[1 << 10, 2 << 10, 4 << 10, 8 << 10,
                                 16 << 10, 32 << 10, 64 << 10, 1 << 20],
                        help="sub-chunk sizes for the subchunk sweep")

    parser.add_argument("--small-blocks", type=parse_size, nargs="+",
                        default=[1 << 10, 4 << 10, 16 << 10, 64 << 10,
                                 256 << 10, 1 << 20],
                        help="block sizes for the numpy-vs-numba sweep")

    parser.add_argument("--runs", type=int, default=5,
                        help="timed runs per configuration (default 5)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warm-up runs before timing (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if not mod._HAS_NUMBA:
        print("[note] numba is not available; _copy_kernel is a no-op stub "
              "and all timings will reflect Python overhead only.")
        print()

    # Show the environment once so the numbers are interpretable.
    print("memcpy_utils environment")
    print(f"  numba available:   {mod._HAS_NUMBA}")
    print(f"  max_numba_threads: {mod._MAX_NUMBA_THREADS}")
    print(f"  auto_mt:           {mod.use_multithread()}")
    print(f"  copy_threads:      {mod.get_copy_threads()}")
    print(f"  default sub_chunk: {format_size(mod._SUBCHUNK_BYTES)}")
    print(f"  default max copy threads: {mod._MAX_COPY_THREADS}")
    print(f"  fixed threads for non-thread sweeps: {fixed_threads(args)}")
    print()

    if args.threads is None and args.threads_list:
        max_t = max(args.threads_list)
        if max_t > mod._MAX_NUMBA_THREADS:
            print(f"[warn] requested up to {max_t} threads but "
                  f"NUMBA_NUM_THREADS={mod._MAX_NUMBA_THREADS}; "
                  f"re-run with NUMBA_NUM_THREADS>={max_t} to go higher.")
            print()

    if args.sweep in ("threads", "all"):
        sweep_threads(args)
    if args.sweep in ("block", "all"):
        sweep_block(args)
    if args.sweep in ("subchunk", "all"):
        sweep_subchunk(args)
    if args.sweep in ("small", "all"):
        sweep_small(args)
    if args.sweep in ("memcpy", "all"):
        sweep_memcpy(args)


if __name__ == "__main__":
    main()