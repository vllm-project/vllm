#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark for paged_shm memcpy_utils: single-thread vs multi-thread.

Test setup (aligned with the earlier numba copy benchmark):

  - 4 GiB paged buffer
  - 1 MiB blocks
  - random block indices
  - 8 threads for the multi-threaded run

Dependencies: numpy, (optional) numba, (optional) vllm for PagedShmStorage.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from vllm.multimodal.paged_shm import memcpy_utils
from vllm.multimodal.paged_shm.memcpy_utils import (
    copy_blocks_to_contig,
    copy_contig_to_blocks,
    warmup,
)


def format_size(num_bytes: float, decimal_places: int = 2) -> str:
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


def bench(fn, bytes_per_run: int, n_runs: int, *args, **kwargs) -> float:
    """Best-of-N bandwidth in GiB/s."""
    fn(*args, **kwargs)  # warm
    best = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        bw = bytes_per_run / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4 * 1024 ** 3)
    parser.add_argument("--block", type=int, default=1 * 1024 ** 2)
    parser.add_argument("--iters", type=int, default=512,
                        help="blocks copied per run (default covers 512 MiB)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    block_bytes = args.block
    n_blocks = args.size // block_bytes
    bytes_per_run = args.iters * block_bytes

    print(f"Buffer:      {format_size(args.size)}")
    print(f"Block:       {format_size(block_bytes)}  ({n_blocks} blocks)")
    print(f"Iters:       {args.iters} blocks per run  "
          f"({format_size(bytes_per_run)} copied per run)")
    print(f"Runs:        {args.runs} (best taken)")
    print(f"Threads:     {args.threads} (MT run)")
    print()

    # ---- module state --------------------------------------------------
    cores, numa, mem = memcpy_utils.get_topology()
    print(f"Host:        cores={cores} numa={numa} "
          f"mem={format_size(mem)}")
    print(f"numba:       {'yes' if memcpy_utils._HAS_NUMBA else 'no'}")
    print(f"Heuristic:   use_multithread()={memcpy_utils.use_multithread()} "
          f"get_copy_threads()={memcpy_utils.get_copy_threads()}")
    print(f"Sub-chunk:   {format_size(memcpy_utils._SUBCHUNK_BYTES)}")
    print()

    # ---- buffers -------------------------------------------------------
    print(f"Allocating {format_size(args.size)} paged buffer + "
          f"{format_size(bytes_per_run)} staging...")
    flat = np.random.randint(
        0, 256, size=args.size, dtype=np.uint8
    )
    src = np.random.randint(
        0, 256, size=bytes_per_run, dtype=np.uint8
    )
    dst = np.empty(bytes_per_run, dtype=np.uint8)
    print()

    # ---- random block indices -----------------------------------------
    src_blocks = np.random.randint(
        0, n_blocks, size=args.iters, dtype=np.int64
    ).tolist()
    dst_blocks = np.random.randint(
        0, n_blocks, size=args.iters, dtype=np.int64
    ).tolist()

    # ---- JIT warm-up ---------------------------------------------------
    print("JIT compiling numba kernel...")
    warmup()
    print("done.")
    print()

    # ---- thread sweep --------------------------------------------------
    print(f"=== ST vs MT  ({format_size(bytes_per_run)} per run) ===")
    header = ["Dir", "Threads", "Bandwidth (GiB/s)", "Time (ms)", "Speedup"]
    print(" | ".join(f"{h:>18}" for h in header))
    print("-" * (21 * len(header)))

    results: dict[tuple[str, int], float] = {}

    for label, fn, args_tuple in [
        ("scatter", copy_contig_to_blocks, (src, flat, dst_blocks, block_bytes)),
        ("gather",  copy_blocks_to_contig, (flat, dst, src_blocks, block_bytes)),
    ]:
        for t in (1, args.threads):
            bw = bench(
                fn, bytes_per_run, args.runs, *args_tuple, n_threads=t
            )
            elapsed_ms = bytes_per_run / (bw * (1024 ** 3)) * 1e3
            results[(label, t)] = bw
            speedup = bw / results[(label, 1)] if t != 1 else 1.0
            print(
                f"{label:>18} | {t:>18d} | {bw:>18.2f} | "
                f"{elapsed_ms:>18.2f} | {speedup:>17.2f}x"
            )

    print()

    # ---- summary -------------------------------------------------------
    print("=== Summary ===")
    for label in ("scatter", "gather"):
        st = results[(label, 1)]
        mt = results[(label, args.threads)]
        print(
            f"{label:>8}: ST={st:6.2f} GiB/s  "
            f"MT({args.threads}T)={mt:6.2f} GiB/s  "
            f"speedup={mt / st:.2f}x"
        )


if __name__ == "__main__":
    main()