#!/usr/bin/env python3
"""
Benchmark CPU block copy bandwidth with random block indices (Cython + OpenMP).

Mirrors ``bench.py`` (the memcpy_utils benchmark) *exactly*: each Cython
call copies ONE contiguous block via a small OpenMP ``prange`` over the
sub-chunks of that block. A timed window is ``iters`` consecutive calls,
one per random block, matching ``_copy_kernel``'s per-launch semantics.

This is deliberately *not* the batch/``_indexed_copy_kernel`` style. The
goal is to reproduce, in Cython + OpenMP, the same per-launch overhead
profile that ``memcpy_utils._copy_kernel`` has — so the two benchmarks
can be compared row by row.

  * same default buffer size      (4 GiB)
  * same default block size       (1 MiB)
  * same default sub-chunk size   (8 KiB)
  * same default iters per window (2048 random blocks)
  * same three sweeps             (threads, block, subchunk)

Only the libc ``memcpy`` backend is used — no SIMD, no NT stores.

Dependencies:
    pip install cython numpy setuptools

Recommended run (NUMA binding on dual-socket machines):
    OMP_PROC_BIND=close OMP_PLACES=cores \
    numactl --cpunodebind=0 --membind=0 \
    python bench_cython_perblock.py --runs 20
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List

import numpy as np


# ---------------------------------------------------------------------------
# Cython source
# ---------------------------------------------------------------------------
CYTHON_SOURCE = r'''
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.string cimport memcpy
from cython.parallel cimport prange

cdef extern from "omp.h" nogil:
    void omp_set_num_threads(int num_threads)
    int  omp_get_max_threads()


def set_omp_threads(int n):
    """Set the OpenMP thread-pool size for subsequent parallel regions."""
    with nogil:
        omp_set_num_threads(n)


def get_omp_max_threads():
    cdef int n
    with nogil:
        n = omp_get_max_threads()
    return n


cpdef void cython_copy_block(
    unsigned char[:] src,
    unsigned char[:] dst,
    Py_ssize_t src_off,
    Py_ssize_t dst_off,
    Py_ssize_t block_bytes,
    Py_ssize_t subchunk_bytes,
):
    """Copy ONE contiguous block, splitting it into sub-chunk tasks.

    Matches ``memcpy_utils._copy_kernel(src_flat, dst_flat, size, chunk)``
    semantics: a single call handles a single contiguous span, and the
    parallel region is exactly ``ceil(block_bytes / subchunk_bytes)``
    iterations. The thread-pool size is whatever was last passed to
    ``set_omp_threads`` — no per-call ``omp_set_num_threads``.

    Callers that want to amortize the OMP region setup should issue many
    of these calls in a loop, exactly like the Python side does with
    ``memcpy_utils._copy_kernel``.
    """
    if block_bytes <= 0 or subchunk_bytes <= 0:
        return

    cdef unsigned char* s = &src[0]
    cdef unsigned char* d = &dst[0]

    cdef Py_ssize_t splits = (block_bytes + subchunk_bytes - 1) // subchunk_bytes
    if splits < 1:
        splits = 1

    cdef Py_ssize_t i, off, length
    with nogil:
        for i in prange(splits, schedule='static'):
            off = i * subchunk_bytes
            if off < block_bytes:
                length = subchunk_bytes
                if off + length > block_bytes:
                    length = block_bytes - off
                memcpy(d + dst_off + off, s + src_off + off, length)
'''


# ---------------------------------------------------------------------------
# Build module
# ---------------------------------------------------------------------------
def build_module(verbose: bool = True):
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    build_dir = Path(tempfile.mkdtemp(prefix="bench_cython_perblock_"))
    pyx_path = build_dir / "_block_copy_perblock.pyx"
    pyx_path.write_text(CYTHON_SOURCE)

    common = {
        "name": "_block_copy_perblock",
        "sources": [str(pyx_path)],
        "include_dirs": [np.get_include()],
        "extra_link_args": ["-fopenmp"],
    }
    flags_list = [
        ["-O3", "-fopenmp", "-march=native", "-funroll-loops"],
        ["-O3", "-fopenmp", "-mavx2"],
        ["-O3", "-fopenmp"],
    ]
    last_err = None
    for flags in flags_list:
        try:
            ext = Extension(extra_compile_args=flags, **common)
            dist = Distribution({
                "ext_modules": cythonize(
                    [ext],
                    compiler_directives={
                        "boundscheck": False,
                        "wraparound": False,
                        "cdivision": True,
                        "language_level": 3,
                    },
                    quiet=not verbose,
                )
            })
            cmd = dist.get_command_obj("build_ext")
            cmd.inplace = 1
            cmd.build_lib = str(build_dir)
            cmd.build_temp = str(build_dir / "build")
            cmd.ensure_finalized()
            if not verbose:
                cmd.verbose = 0
            cmd.run()
            if verbose:
                print(f"[build] flags={flags}")
            sys.path.insert(0, str(build_dir))
            return __import__("_block_copy_perblock")
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[build] flags={flags} failed: {e}")
            continue
    raise RuntimeError(f"compile failed: {last_err}")


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


def print_row(cols: List[str], widths: List[int]) -> None:
    print(" | ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))


def make_buffers(size: int):
    """Random-content source; zero destination. Matches bench.py layout."""
    src = np.random.randint(0, 256, size=size, dtype=np.uint8)
    dst = np.zeros(size, dtype=np.uint8)
    return src, dst


def make_indices(n_blocks: int, iters: int):
    """Random source/destination block indices, one array each."""
    idx_s = np.random.randint(0, n_blocks, size=iters, dtype=np.intp)
    idx_d = np.random.randint(0, n_blocks, size=iters, dtype=np.intp)
    return idx_s, idx_d


def make_offsets(idx_s, idx_d, block):
    """Precompute byte offsets as Python lists for the hot loop."""
    return (idx_s * block).tolist(), (idx_d * block).tolist()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_call(
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


def fixed_threads(args) -> int:
    """Thread count used by sweeps that do not vary the pool size.

    Mirrors ``bench.py``: prefer the value closest to 8 (the
    ``memcpy_utils._MAX_COPY_THREADS`` constant).
    """
    if args.threads is not None:
        return args.threads
    return min(args.threads_list, key=lambda t: abs(t - 8))


def warmup_kernel(module, src, dst, off_s, off_d, block, sub_chunk,
                  n_threads) -> None:
    """
    Compile the Cython module and prime the OMP thread pool.

    Sets the OMP pool size here (not per call) so that timed runs do not
    pay for ``omp_set_num_threads`` — matching ``memcpy_utils`` where the
    pool is sized once by ``_dispatch`` before the kernel loop.
    """
    module.set_omp_threads(n_threads)
    module.cython_copy_block(src, dst, off_s[0], off_d[0], block, sub_chunk)


def run_window(module, src, dst, off_s, off_d, block, sub_chunk, iters) -> None:
    """One timed window: ``iters`` consecutive per-block Cython calls."""
    fn = module.cython_copy_block
    for i in range(iters):
        fn(src, dst, off_s[i], off_d[i], block, sub_chunk)


# ---------------------------------------------------------------------------
# Sweep: thread count
# ---------------------------------------------------------------------------
def sweep_threads(module, args) -> None:
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
    warmup_kernel(module, src, dst, off_s, off_d, block, sc, threads_list[0])

    bytes_per_run = iters * block

    print("=== Thread sweep (cython per-block, memcpy) ===")
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
        module.set_omp_threads(T)

        def fn() -> None:
            run_window(module, src, dst, off_s, off_d, block, sc, iters)

        mean = bench_call(fn, bytes_per_run, args.runs, warmup=args.warmup)
        print_row([str(T), f"{mean:.2f}"], widths)
    print()


# ---------------------------------------------------------------------------
# Sweep: block size
# ---------------------------------------------------------------------------
def sweep_block(module, args) -> None:
    T = fixed_threads(args)
    sc = args.sub_chunk
    iters = args.iters
    module.set_omp_threads(T)

    src, dst = make_buffers(args.size)

    widths = [12, 10, 22, 22, 12]
    print("=== Block sweep (cython per-block, memcpy) ===")
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
        warmup_kernel(module, src, dst, off_s, off_d, block, sc, T)
        bytes_per_run = run_iters * block

        def fn(off_s=off_s, off_d=off_d, block=block,
               iters=run_iters) -> None:
            run_window(module, src, dst, off_s, off_d, block, sc, iters)

        mean = bench_call(fn, bytes_per_run, args.runs, warmup=args.warmup)
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
def sweep_subchunk(module, args) -> None:
    T = fixed_threads(args)
    block = args.block
    iters = args.iters
    n_blocks = args.size // block
    if n_blocks < 1 or iters < 1:
        return
    module.set_omp_threads(T)

    src, dst = make_buffers(args.size)
    idx_s, idx_d = make_indices(n_blocks, iters)
    off_s, off_d = make_offsets(idx_s, idx_d, block)
    bytes_per_run = iters * block

    widths = [12, 10, 12, 22]
    print("=== Sub-chunk sweep (cython per-block, memcpy) ===")
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
        tasks = splits

        warmup_kernel(module, src, dst, off_s, off_d, block, eff_sc, T)

        def fn(eff_sc=eff_sc) -> None:
            run_window(module, src, dst, off_s, off_d, block, eff_sc, iters)

        mean = bench_call(fn, bytes_per_run, args.runs, warmup=args.warmup)
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cython + OpenMP per-block copy benchmark (memcpy only)",
    )
    parser.add_argument("--sweep",
                        choices=["threads", "block", "subchunk", "all"],
                        default="all",
                        help="which dimension to sweep (default: all)")

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
                        help="multi-mode: thread counts to sweep")

    parser.add_argument("--blocks", type=parse_size, nargs="+",
                        default=[4 << 10, 16 << 10, 64 << 10, 256 << 10,
                                 1 << 20, 4 << 20, 16 << 20],
                        help="block sizes to sweep")

    parser.add_argument("--sub-chunks", type=parse_size, nargs="+",
                        default=[1 << 10, 2 << 10, 4 << 10, 8 << 10,
                                 16 << 10, 32 << 10, 64 << 10, 1 << 20],
                        help="sub-chunk sizes to sweep")

    parser.add_argument("--runs", type=int, default=5,
                        help="timed runs per configuration (default 5)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warm-up runs before timing (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-build", action="store_true",
                        help="suppress Cython build output")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("Cython + OpenMP per-block memcpy benchmark")
    module = build_module(verbose=not args.quiet_build)
    print(f"  omp_max_threads:   {module.get_omp_max_threads()}")
    print(f"  default sub_chunk: {format_size(8 << 10)}")
    print(f"  fixed threads for non-thread sweeps: {fixed_threads(args)}")
    print()

    if args.threads is None and args.threads_list:
        max_t = max(args.threads_list)
        omp_max = module.get_omp_max_threads()
        if max_t > omp_max:
            print(f"[warn] requested up to {max_t} threads but "
                  f"OMP max is {omp_max}; runs will be clamped by OpenMP.")
            print()

    if args.sweep in ("threads", "all"):
        sweep_threads(module, args)
    if args.sweep in ("block", "all"):
        sweep_block(module, args)
    if args.sweep in ("subchunk", "all"):
        sweep_subchunk(module, args)


if __name__ == "__main__":
    main()