# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmarks for the CPU KV-offload copy path.

Modes:

``--merge`` (default)
    Descriptor run-coalescing (``merge_contiguous_descriptors``) on the C++
    ``cuMemcpyBatchAsync`` path: sweeps page size x contiguous-run length and
    reports, for unmerged vs merged descriptor lists, the CPU submission cost
    (cuMemcpyBatchAsync charges ~0.5us of CPU per descriptor), the solo-call
    latency (submission stall exposed, as for a single prefix-hit load), and
    the pipelined back-to-back throughput. The Triton fast path is shown for
    reference where it applies.

``--crossover``
    DMA vs Triton fast path across page size x batch size, for re-validating
    ``THRESHOLD_BYTES`` / ``MIN_N`` / ``NUM_SMS`` when the kernel changes.

All timed calls run on a dedicated CUDA stream: cuMemcpyBatchAsync rejects
the legacy default stream and would otherwise silently take the per-copy
fallback loop.

Run on a CUDA host:

    .venv/bin/python benchmarks/kernels/benchmark_swap_blocks_batch.py
    .venv/bin/python benchmarks/kernels/benchmark_swap_blocks_batch.py --crossover
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.v1.kv_offload.cpu.gpu_worker import merge_contiguous_descriptors
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    MIN_N,
    NUM_SMS,
    THRESHOLD_BYTES,
    _swap_blocks_kernel,
)

KiB = 1024


def _make_batch(n: int, page_size: int, run_len: int = 1):
    """Build a descriptor list of ``n`` pages in contiguous runs of ``run_len``.

    Runs are contiguous on both host and device (as consecutive blocks within
    a layer are in production), while run order is shuffled with a one-page
    gap so no two runs are accidentally adjacent.
    """
    num_runs = (n + run_len - 1) // run_len
    pool_pages = num_runs * (run_len + 1)
    host = torch.empty(pool_pages * page_size, dtype=torch.uint8, pin_memory=True)
    host.random_()
    dev = torch.zeros(pool_pages * page_size, dtype=torch.uint8, device="cuda")

    rng = np.random.default_rng(seed=42)
    run_order = rng.permutation(num_runs)
    starts = run_order * (run_len + 1)
    offsets = (starts[:, None] + np.arange(run_len)[None, :]).ravel()[:n]
    offsets = offsets * page_size

    src = (host.data_ptr() + offsets).astype(np.int64)
    dst = (dev.data_ptr() + offsets).astype(np.int64)
    sizes = np.full(n, page_size, dtype=np.int64)
    page_idx = torch.from_numpy((offsets // page_size).astype(np.int64))
    return host, dev, src, dst, sizes, page_idx


def _verify_copied_pages(host, dev, page_idx, page_size):
    """Compare only the pages the descriptor list covers (runs have gaps)."""
    hv = host.view(-1, page_size)[page_idx]
    dv = dev.view(-1, page_size)[page_idx].cpu()
    assert torch.equal(dv, hv), "copy produced wrong bytes"


def _to_pinned(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).pin_memory()


def _dma_call(src_addrs, dst_addrs, sizes):
    ops.swap_blocks_batch(src_addrs, dst_addrs, sizes, is_src_access_order_any=True)


def _triton_call(src_addrs, dst_addrs, sizes, bytes_per_chunk: int):
    n = src_addrs.numel()
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )


def _submit_us(fn, stream, iters: int = 10) -> float:
    """Average CPU cost of enqueuing one call (no synchronization)."""
    fn()
    stream.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t0
    stream.synchronize()
    return dt / iters * 1e6


def _solo_ms(fn, stream, iters: int = 10) -> float:
    """Median submit+complete latency of an isolated call."""
    fn()
    stream.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        stream.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def _b2b_ms(fn, stream, iters: int = 10, warmup: int = 3) -> float:
    """Average per-call time of back-to-back (pipelined) calls."""
    for _ in range(warmup):
        fn()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / iters


def _gbps(total_bytes: int, ms: float) -> float:
    return total_bytes / (ms * 1e-3) / 1e9


def run_merge(n: int, page_kib_values: list[int], run_lengths: list[int]) -> None:
    stream = torch.cuda.Stream()
    header = (
        f"{'page':>6}{'run':>5}{'n_merged':>9}{'merge_us':>9}"
        f"{'submit_us u/m':>16}{'solo_ms u/m':>16}"
        f"{'b2b_GBps u/m':>16}{'triton_solo':>12}"
    )
    print(header)
    print("-" * len(header))
    for pk in page_kib_values:
        page = pk * KiB
        for run_len in run_lengths:
            host, dev, src, dst, sizes, page_idx = _make_batch(n, page, run_len)
            total = n * page

            t0 = time.perf_counter()
            m_src, m_dst, m_sizes = src.copy(), dst.copy(), sizes.copy()
            n_merged = merge_contiguous_descriptors(m_src, m_dst, m_sizes)
            merge_us = (time.perf_counter() - t0) * 1e6

            sa_u, da_u, sz_u = _to_pinned(src), _to_pinned(dst), _to_pinned(sizes)
            sa_m = _to_pinned(m_src[:n_merged])
            da_m = _to_pinned(m_dst[:n_merged])
            sz_m = _to_pinned(m_sizes[:n_merged])

            with torch.cuda.stream(stream):
                dev.zero_()
                _dma_call(sa_m, da_m, sz_m)
            stream.synchronize()
            _verify_copied_pages(host, dev, page_idx, page)

            unmerged = partial(_dma_call, sa_u, da_u, sz_u)
            merged = partial(_dma_call, sa_m, da_m, sz_m)
            with torch.cuda.stream(stream):
                sub_u = _submit_us(unmerged, stream)
                sub_m = _submit_us(merged, stream)
                solo_u = _solo_ms(unmerged, stream)
                solo_m = _solo_ms(merged, stream)
                b2b_u = _gbps(total, _b2b_ms(unmerged, stream))
                b2b_m = _gbps(total, _b2b_ms(merged, stream))
                tri = "-"
                if page < THRESHOLD_BYTES and n >= MIN_N:
                    chunk = min(triton.next_power_of_2(page), 8192)
                    tri_fn = partial(_triton_call, sa_u, da_u, sz_u, chunk)
                    tri = f"{_solo_ms(tri_fn, stream):.2f}"

            print(
                f"{pk:>5}K{run_len:>5}{n_merged:>9}{merge_us:>9.0f}"
                f"{sub_u:>8.0f}/{sub_m:<7.0f}{solo_u:>8.2f}/{solo_m:<7.2f}"
                f"{b2b_u:>8.1f}/{b2b_m:<7.1f}{tri:>12}"
            )
            del host, dev
    print(
        f"\nn={n} descriptors; run = contiguous pages per run (both sides); "
        "u/m = unmerged/merged.\nsubmit_us = CPU enqueue cost per call; "
        "solo_ms = isolated submit+complete latency;\nb2b_GBps = pipelined "
        "back-to-back throughput; triton_solo (ms) shown below the "
        f"{THRESHOLD_BYTES // KiB}KiB\nfast-path threshold for reference."
    )


def run_crossover(n_values: list[int], max_kib: int) -> None:
    stream = torch.cuda.Stream()
    page_sizes = list(range(4 * KiB, max_kib * KiB + 1, 4 * KiB))
    header = "page".rjust(8)
    for n in n_values:
        header += f" | DMA n={n}".rjust(13) + "Triton".rjust(10) + "win".rjust(5)
    print(header)
    print("-" * len(header))
    for ps in page_sizes:
        chunk = min(triton.next_power_of_2(ps), 8192)
        row = f"{ps // KiB:>6}KB"
        for n in n_values:
            host, dev, src, dst, sizes, page_idx = _make_batch(n, ps)
            sa, da, sz = _to_pinned(src), _to_pinned(dst), _to_pinned(sizes)
            with torch.cuda.stream(stream):
                dev.zero_()
                _triton_call(sa, da, sz, chunk)
            stream.synchronize()
            _verify_copied_pages(host, dev, page_idx, ps)
            with torch.cuda.stream(stream):
                dma = _solo_ms(partial(_dma_call, sa, da, sz), stream)
                tri = _solo_ms(partial(_triton_call, sa, da, sz, chunk), stream)
            win = "T" if tri < dma else "D"
            row += f" | {dma * 1e3:>9.0f}us{tri * 1e3:>8.0f}us{win:>5}"
            del host, dev
        print(row)
    print(f"\n(threshold={THRESHOLD_BYTES / KiB:g}KiB, min_n={MIN_N}, sms={NUM_SMS})")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merge",
        action="store_true",
        help="benchmark descriptor run-coalescing on the DMA path",
    )
    p.add_argument(
        "--crossover",
        action="store_true",
        help="sweep page size vs. batch size, mark DMA/Triton winner",
    )
    p.add_argument("--n", type=int, default=8192, help="descriptors for --merge")
    p.add_argument("--n-values", type=int, nargs="+", default=[16, 256, 4096])
    p.add_argument("--max-kib", type=int, default=64)
    p.add_argument("--run-lengths", type=int, nargs="+", default=[1, 4, 16, 128, 8192])
    p.add_argument("--page-kib", type=int, nargs="+", default=[8, 16, 32])
    args = p.parse_args()

    if not args.crossover and not args.merge:
        args.merge = True

    if args.merge:
        print("=== descriptor run-coalescing on cuMemcpyBatchAsync ===")
        run_merge(args.n, args.page_kib, args.run_lengths)
    if args.crossover:
        print("\n=== DMA (cuMemcpyBatchAsync) vs. Triton fast path ===")
        run_crossover(args.n_values, args.max_kib)


if __name__ == "__main__":
    main()
