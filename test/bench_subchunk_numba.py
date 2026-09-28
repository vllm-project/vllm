#!/usr/bin/env python3
"""
Pure numpy + numba block copy benchmark (no Cython).

Test setup:
  1. 4 GiB total buffer, random data
  2. 1 MiB blocks
  3. Random block indices

Dependencies: numpy, numba
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# NUMBA_NUM_THREADS must be set BEFORE `import numba`.
# It caps the process-wide thread pool size; set_num_threads can only
# reduce within this limit. Set this to your socket's physical core count
# (or higher if you want to use HT threads).
_MAX_NUMBA_THREADS = int(os.environ.get("NUMBA_NUM_THREADS", "0")) or 64
os.environ["NUMBA_NUM_THREADS"] = str(_MAX_NUMBA_THREADS)

from numba import njit, prange, set_num_threads, get_num_threads


def format_size(num_bytes, decimal_places=2):
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB"]
    base = 1024
    size = float(num_bytes)
    e = 0
    while size >= base and e < len(units) - 1:
        size /= base
        e += 1
    return f"{size:.{decimal_places}f} {units[e]}"


def generate_random_indices(num_blocks, n_iters):
    src_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    dst_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    return src_idx, dst_idx


# ---------------------------------------------------------------------------
# numpy kernel
# ---------------------------------------------------------------------------
def np_block_mt(src, dst, block_bytes, src_idx, dst_idx, n_threads):
    n = len(src_idx)

    def worker(lo_hi):
        lo, hi = lo_hi
        for k in range(lo, hi):
            sbase = int(src_idx[k]) * block_bytes
            dbase = int(dst_idx[k]) * block_bytes
            dst[dbase:dbase + block_bytes] = src[sbase:sbase + block_bytes]

    chunk = (n + n_threads - 1) // n_threads
    ranges = [(i * chunk, min((i + 1) * chunk, n)) for i in range(n_threads)]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(worker, ranges))


# ---------------------------------------------------------------------------
# numba kernel (unified: one task per sub-chunk)
# ---------------------------------------------------------------------------
@njit(nogil=True, parallel=True, cache=True)
def numba_copy_kernel(src, dst, src_idx, dst_idx,
                      block_bytes, subchunk_bytes):
    splits = (block_bytes + subchunk_bytes - 1) // subchunk_bytes
    n = src_idx.shape[0]
    total = n * splits
    for t in prange(total):
        blk = t // splits
        sidx = t % splits
        off = sidx * subchunk_bytes
        if off < block_bytes:
            length = subchunk_bytes
            if off + length > block_bytes:
                length = block_bytes - off
            sbase = src_idx[blk] * block_bytes + off
            dbase = dst_idx[blk] * block_bytes + off
            dst[dbase:dbase + length] = src[sbase:sbase + length]


def numba_copy(src, dst, block_bytes, subchunk_bytes,
               src_idx, dst_idx, n_threads):
    set_num_threads(n_threads)
    numba_copy_kernel(src, dst, src_idx, dst_idx,
                      block_bytes, subchunk_bytes)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def bench(fn, bytes_per_run, n_runs, *args, **kwargs):
    fn(*args, **kwargs)
    best = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        bw = bytes_per_run / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4 * 1024 ** 3)
    parser.add_argument("--block", type=int, default=1 * 1024 ** 2)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads-list", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12, 16, 24, 32, 36])
    parser.add_argument("--subchunk-exp", type=int, nargs="+",
                        default=[10, 12, 14, 16, 17, 18, 19, 20])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    block_bytes = args.block
    n_blocks = args.size // block_bytes
    bytes_per_run = args.iters * block_bytes

    print(f"Buffer:   {format_size(args.size)}")
    print(f"Block:    {format_size(block_bytes)}  ({n_blocks} blocks)")
    print(f"Iters:    {args.iters} blocks per run  "
          f"({format_size(bytes_per_run)} copied per run)")
    print(f"Runs:     {args.runs} (best taken)")
    print(f"numba max threads: {_MAX_NUMBA_THREADS}")
    print()

    print(f"Allocating {format_size(args.size)} random source + destination...")
    src = np.random.randint(0, 256, size=args.size, dtype=np.uint8)
    dst = np.zeros(args.size, dtype=np.uint8)
    print()

    # JIT pre-warm both code paths (subchunk == block_bytes, and small subchunk)
    si0, di0 = generate_random_indices(n_blocks, 1)
    print("JIT compiling numba kernel...")
    set_num_threads(1)
    numba_copy_kernel(src[:block_bytes*2], dst[:block_bytes*2],
                      si0, di0, block_bytes, block_bytes)
    numba_copy_kernel(src[:block_bytes*2], dst[:block_bytes*2],
                      si0, di0, block_bytes, 64 * 1024)
    print("done.")
    print()

    # ------------------------------------------------------------------
    # Thread scaling (block-level: subchunk == block_bytes)
    # ------------------------------------------------------------------
    print("=== Block-level: numpy ThreadPool vs numba ===")
    header = ["Threads", "np_block_mt (GiB/s)", "numba_block (GiB/s)"]
    print(" | ".join(f"{h:>22}" for h in header))
    print("-" * (24 * len(header)))

    for T in args.threads_list:
        si, di = generate_random_indices(n_blocks, args.iters)
        bw_np = bench(np_block_mt, bytes_per_run, args.runs,
                      src, dst, block_bytes, si, di, T)
        bw_nb = bench(numba_copy, bytes_per_run, args.runs,
                      src, dst, block_bytes, block_bytes, si, di, T)
        print(f"{T:>22d} | {bw_np:>22.2f} | {bw_nb:>22.2f}")

    print()

    # ------------------------------------------------------------------
    # Sub-chunk scan at 8 threads
    # ------------------------------------------------------------------
    T = 8
    print(f"=== Sub-chunk kernels ({T} threads) ===")
    header = ["Sub-chunk", "np_block_mt (GiB/s)",
              "numba_block (GiB/s)", "numba_subchunk (GiB/s)",
              "tasks/block"]
    print(" | ".join(f"{h:>22}" for h in header))
    print("-" * (24 * len(header)))

    si, di = generate_random_indices(n_blocks, args.iters)
    bw_np = bench(np_block_mt, bytes_per_run, args.runs,
                  src, dst, block_bytes, si, di, T)
    bw_nb_blk = bench(numba_copy, bytes_per_run, args.runs,
                      src, dst, block_bytes, block_bytes, si, di, T)

    for exp in args.subchunk_exp:
        sc = 2 ** exp
        splits = (block_bytes + sc - 1) // sc
        si, di = generate_random_indices(n_blocks, args.iters)
        bw_nb_sub = bench(numba_copy, bytes_per_run, args.runs,
                          src, dst, block_bytes, sc, si, di, T)
        row = [
            f"{format_size(sc):>22}",
            f"{bw_np:>22.2f}",
            f"{bw_nb_blk:>22.2f}",
            f"{bw_nb_sub:>22.2f}",
            f"{splits:>22}",
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()