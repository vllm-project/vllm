#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark for the EC region's batched block copies.

Measures `swap_blocks_batch` over a real `ECSharedRegion` for one
encoder-cache entry, across descriptor layouts and both directions, to find
out why the load (host->device) path runs far below the save (device->host)
path.

`EmbeddingCache.alloc` returns an entry's block ids sorted ascending, so the
two layouts that describe a real entry are:

  asc_consec    consecutive and ascending — a free list with contiguous space
  random_sorted ascending but scattered over the region — a free list
                fragmented by eviction churn

The other two are the same ids in orders `alloc` cannot produce, kept as the
contrast that shows what the sort buys:

  desc_consec   consecutive but DESCENDING
  random        scattered and unordered

`--coalesce` merges address-consecutive runs into one descriptor, which turns
`asc_consec` into a single large memcpy and leaves every other layout at
roughly one descriptor per block.

Ordering an entry's block ids is safe because save and load walk the same list:
slot j of the entry lives in region block ids[j] in both directions, so sorting
preserves the data layout while making both directions address-ascending.

`--host-descriptors` measures the other half of a transfer: the host time spent
building a step's descriptors before any copy is issued. That cost never
reaches the GPU, so the copy timings above cannot see it, and it scales with
entries per step rather than with bytes. The arms differ only in the dtype the
addresses are computed in -- see `descriptor_dtype_arms`.

`--coalesce-breakdown` drills into the largest term of that host time: turning
one entry's block ids into runs. It reports both the operations
`_coalesce_runs` performs and whole-function alternatives, including the floor
a per-entry cache would reach.

Run inside the pod, on a GPU the server is not using:

    CUDA_VISIBLE_DEVICES=1 /vllm-workspace/venv-vllm/bin/python \
        /vllm-workspace/bench/micro_swap_blocks.py --verify

    CUDA_VISIBLE_DEVICES=1 /vllm-workspace/venv-vllm/bin/python \
        /vllm-workspace/bench/micro_swap_blocks.py --host-descriptors
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import torch

from vllm._custom_ops import swap_blocks_batch
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import _coalesce_runs
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
)
from vllm.platforms import current_platform

# Qwen2.5-VL-7B: hidden 3584 x 2 bytes (bfloat16).
DEFAULT_BLOCK_SIZE = 3584 * 2
# 1288x728 image -> 46x26 merged patches.
DEFAULT_ENTRY_BLOCKS = 46 * 26
_ENGINE_ID = "microbench"


def build_ids(
    layout: str, num_blocks: int, n: int, seed: int, base: int = 0
) -> list[int]:
    """Block ids for one entry. `base` shifts the slice so successive reps use
    disjoint host pages and cannot inherit a previous rep's cache warmth."""
    if layout == "desc_consec":
        return [base + n - 1 - i for i in range(n)]
    if layout == "asc_consec":
        return [base + i for i in range(n)]
    if layout in ("random", "random_sorted"):
        g = torch.Generator().manual_seed(seed + base)
        ids = torch.randperm(num_blocks, generator=g)[:n].tolist()
        return sorted(ids) if layout == "random_sorted" else ids
    raise ValueError(layout)


def build_descriptors(
    ids: list[int],
    region_ptr: int,
    gpu_ptr: int,
    block_size: int,
    direction: str,
    coalesce: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host/device pointer pairs for one entry.

    Entry slot j <-> region block ids[j] <-> gpu offset j*block_size. The
    `coalesce` path calls the connector's own `_coalesce_runs`, so the numbers
    below describe the production descriptor layout rather than a reimplementation
    of it.
    """
    if coalesce:
        slots, first_blocks, num_blocks = _coalesce_runs(ids)
    else:
        slots = np.arange(len(ids), dtype=np.int64)
        first_blocks = np.asarray(ids, dtype=np.int64)
        num_blocks = np.ones(len(ids), dtype=np.int64)

    host = region_ptr + first_blocks * block_size
    gpu = gpu_ptr + slots * block_size
    src, dst = (host, gpu) if direction == "load" else (gpu, host)
    return (
        torch.from_numpy(src),
        torch.from_numpy(dst),
        torch.from_numpy(num_blocks * block_size),
    )


def time_batch(
    descriptors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    any_order: bool,
    warmup: int,
) -> float:
    """Median ms per batched copy.

    Each rep gets its own descriptor set (a distinct slice of the region), so
    no rep reads host pages a previous rep just touched — the cold state a real
    load runs in, since an entry is reloaded many steps after it was saved.
    """
    stream = torch.cuda.Stream()
    samples: list[float] = []
    with torch.cuda.stream(stream):
        for i, (src, dst, sizes) in enumerate(descriptors):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record(stream)
            swap_blocks_batch(src, dst, sizes, is_src_access_order_any=any_order)
            end.record(stream)
            end.synchronize()
            if i >= warmup:
                samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def verify(region: ECSharedRegion, ids: list[int], block_size: int, coalesce: bool):
    """Prove the descriptor math: load region blocks, compare against source."""
    n = len(ids)
    slot_ids = ids
    for j, b in enumerate(slot_ids):
        region.blocks[b].fill_((j % 127) + 1)
    gpu = torch.zeros(n * block_size, dtype=torch.int8, device="cuda")
    src, dst, sizes = build_descriptors(
        slot_ids, region.blocks.data_ptr(), gpu.data_ptr(), block_size, "load", coalesce
    )
    swap_blocks_batch(src, dst, sizes, is_src_access_order_any=True)
    torch.cuda.synchronize()
    got = gpu.view(n, block_size).cpu()
    bad = [j for j in range(n) if int(got[j][0]) != (j % 127) + 1]
    return len(bad) == 0, len(src)


def _coalesce_runs_u64(block_ids: list[int]):
    """`_coalesce_runs` producing offsets already in the address dtype.

    Runs are found by comparing against `ids[:-1] + 1` rather than by
    `np.diff`, whose unsigned difference wraps on any descending id list -- the
    `desc_consec` layout here, and anything an allocator that did not sort
    would hand out. The `+ 1` itself cannot wrap for a real block id.
    """
    ids = np.fromiter(block_ids, dtype=np.uint64, count=len(block_ids))
    breaks = np.flatnonzero(ids[1:] != ids[:-1] + np.uint64(1)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [ids.size]))
    return starts.astype(np.uint64), ids[starts], ends - starts


def descriptor_dtype_arms(block_size: int):
    """Save-side address arithmetic variants, all yielding the same addresses.

    Each arm coalesces one entry's block ids and writes that entry's
    descriptors into the pooled buffers at a running offset, exactly as
    `ECCPUWorker.save_caches` does, and returns how many it wrote.

      int64           Python-int base added to int64 offsets. Cheapest, and
                      raises OverflowError once a device pointer reaches 2**63,
                      which is why the other two exist.
      uint64          base and stride boxed as numpy scalars per call, offsets
                      cast up at the multiply.
      uint64_hoisted  the same, with the stride boxed once and the offsets
                      produced as uint64 by the coalescing step. The base is
                      still boxed per entry: each entry is its own tensor, so
                      there is nothing to hoist it out of.
      platform_dtype  uint64 only where a device pointer can reach 2**63,
                      leaving the cast a no-op everywhere else.
    """
    bs_u64 = np.uint64(block_size)

    def int64(bufs, first, ids, gpu_ptr, region_ptr) -> int:
        slots, first_blocks, num_blocks = _coalesce_runs(ids)
        bufs.add_copies(
            first,
            gpu_ptr + slots * block_size,
            region_ptr + first_blocks * block_size,
            num_blocks * block_size,
        )
        return slots.size

    def uint64(bufs, first, ids, gpu_ptr, region_ptr) -> int:
        slots, first_blocks, num_blocks = _coalesce_runs(ids)
        bufs.add_copies(
            first,
            np.uint64(gpu_ptr) + slots.astype(np.uint64) * np.uint64(block_size),
            region_ptr + first_blocks * block_size,
            num_blocks * block_size,
        )
        return slots.size

    def uint64_hoisted(bufs, first, ids, gpu_ptr, region_ptr) -> int:
        slots, first_blocks, num_blocks = _coalesce_runs_u64(ids)
        bufs.add_copies(
            first,
            np.uint64(gpu_ptr) + slots * bs_u64,
            np.uint64(region_ptr) + first_blocks * bs_u64,
            num_blocks * block_size,
        )
        return slots.size

    # The widest address dtype this platform's descriptors need. Only XPU USM
    # reaches 2**63, so on every other platform the cast below is a no-op and
    # the arithmetic stays exactly as cheap as the int64 arm.
    addr = np.uint64 if current_platform.is_xpu() else np.int64
    addr_bs = addr(block_size)

    def platform_dtype(bufs, first, ids, gpu_ptr, region_ptr) -> int:
        slots, first_blocks, num_blocks = _coalesce_runs(ids)
        bufs.add_copies(
            first,
            addr(gpu_ptr) + slots.astype(addr, copy=False) * addr_bs,
            region_ptr + first_blocks * block_size,
            num_blocks * block_size,
        )
        return slots.size

    return {
        "int64": int64,
        "uint64": uint64,
        "uint64_hoisted": uint64_hoisted,
        "platform_dtype": platform_dtype,
    }


def time_interleaved(fns: dict, reps: int, warmup: int) -> dict[str, tuple[int, float]]:
    """Min and median host ns per call, one entry per named zero-arg callable.

    The arms are interleaved within each rep, and their order rotates, because
    where the arms differ by less than the host drift over a run, timing one arm
    to completion and then the next attributes that drift to whichever arm went
    last. The minimum is the least scheduler-contaminated estimate and the one
    to compare on; the median shows how much noise sat on top of it.
    """
    names = list(fns)
    samples: dict[str, list[int]] = {name: [] for name in names}
    for r in range(warmup + reps):
        for name in names[r % len(names) :] + names[: r % len(names)]:
            fn = fns[name]
            t0 = time.perf_counter_ns()
            fn()
            elapsed = time.perf_counter_ns() - t0
            if r >= warmup:
                samples[name].append(elapsed)
    return {name: (min(s), statistics.median(s)) for name, s in samples.items()}


def time_descriptor_build(
    ids_per_entry: list[list[int]],
    region_ptr: int,
    gpu_ptrs: list[int],
    arms: dict,
    reps: int,
    warmup: int,
) -> dict[str, tuple[int, float, int]]:
    """Host ns to build one step's save descriptors, per arm.

    One iteration is one step: acquire a buffer sized for every entry the step
    saves, then fill it entry by entry. Buffer acquisition is inside the timed
    region because it is inside `save_caches` too, and after warmup it is a
    pool hit rather than an allocation. The release is a list append the worker
    makes on the completion path instead; it is timed here only to keep the
    pool from growing, and costs a fraction of one descriptor.
    """
    total = sum(len(ids) for ids in ids_per_entry)
    pools = {name: DescriptorBufferPool() for name in arms}
    ndescs: dict[str, int] = {}

    def step(name):
        def run():
            bufs = pools[name].acquire(total)
            first = 0
            for ids, gpu_ptr in zip(ids_per_entry, gpu_ptrs):
                first += arms[name](bufs, first, ids, gpu_ptr, region_ptr)
            pools[name].release(bufs)
            ndescs[name] = first

        return run

    timed = time_interleaved({n: step(n) for n in arms}, reps, warmup)
    return {name: (lo, med, ndescs[name]) for name, (lo, med) in timed.items()}


def coalesce_steps(block_ids: list[int]) -> dict:
    """The individual numpy operations `_coalesce_runs` performs, timed apart.

    Each step is fed inputs the previous one would have produced, so the times
    are additive and show which operation owns the total. The two `convert`
    steps and the two `breaks` steps are alternatives, not a sequence.
    """
    n = len(block_ids)
    ids = np.asarray(block_ids, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    starts = np.concatenate(([0], breaks))
    return {
        "convert fromiter": lambda: np.fromiter(block_ids, dtype=np.int64, count=n),
        "convert asarray": lambda: np.asarray(block_ids, dtype=np.int64),
        "breaks diff": lambda: np.flatnonzero(np.diff(ids) != 1) + 1,
        "breaks slice-cmp": lambda: np.flatnonzero(ids[1:] != ids[:-1] + 1) + 1,
        "assemble concat x2": lambda: (
            np.concatenate(([0], breaks)),
            np.concatenate((breaks, [ids.size])),
        ),
        "index ids[starts]": lambda: ids[starts],
    }


def coalesce_variants(block_ids: list[int]) -> dict:
    """Whole-function candidates, each returning `(slots, first_blocks, runs)`.

    current       `_coalesce_runs` as the connector calls it
    asarray       the same, converting with `np.asarray` instead of
                  `np.fromiter`
    slice_cmp     `asarray` plus finding breaks without `np.diff`'s temporary
    preconverted  arithmetic only, ids already an ndarray -- the ceiling on
                  what caching the array per entry could buy
    fastpath      an O(1) test for "one run covers everything" ahead of the
                  numpy path. `alloc` returns sorted distinct ids, so
                  `last - first == n - 1` proves consecutive without looking
                  at the middle. Helps only a contiguous free list.
    fastpath_pre  both: the O(1) test, falling back to preconverted ids
    memoized      the result handed back unchanged -- the floor, and the
                  ceiling on caching the runs themselves per entry
    """
    pre = np.asarray(block_ids, dtype=np.int64)
    cached = _coalesce_runs(block_ids)

    def assemble(ids, breaks):
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [ids.size]))
        return starts, ids[starts], ends - starts

    def current():
        return _coalesce_runs(block_ids)

    def asarray():
        ids = np.asarray(block_ids, dtype=np.int64)
        return assemble(ids, np.flatnonzero(np.diff(ids) != 1) + 1)

    def slice_cmp():
        ids = np.asarray(block_ids, dtype=np.int64)
        return assemble(ids, np.flatnonzero(ids[1:] != ids[:-1] + 1) + 1)

    def preconverted():
        return assemble(pre, np.flatnonzero(pre[1:] != pre[:-1] + 1) + 1)

    n = len(block_ids)

    def single_run():
        return (
            np.zeros(1, dtype=np.int64),
            np.array([block_ids[0]], dtype=np.int64),
            np.array([n], dtype=np.int64),
        )

    def fastpath():
        if block_ids[-1] - block_ids[0] == n - 1:
            return single_run()
        return current()

    def fastpath_pre():
        if block_ids[-1] - block_ids[0] == n - 1:
            return single_run()
        return preconverted()

    def memoized():
        return cached

    return {
        "current": current,
        "asarray": asarray,
        "slice_cmp": slice_cmp,
        "preconverted": preconverted,
        "fastpath": fastpath,
        "fastpath_pre": fastpath_pre,
        "memoized": memoized,
    }


def run_coalesce_breakdown(
    num_blocks: int, entry_blocks: int, reps: int, warmup: int, seed: int
) -> None:
    """Where one entry's coalescing time goes, and what removing it could buy.

    One entry, not a step: `--host-descriptors` already reports the per-step
    total, and this asks which operation inside it to attack. Only the layouts
    `EmbeddingCache.alloc` can produce are swept.
    """
    for layout in ("asc_consec", "random_sorted"):
        ids = build_ids(layout, num_blocks, entry_blocks, seed)
        runs = len(_coalesce_runs(ids)[0])
        print(f"--- {layout}: {entry_blocks} blocks -> {runs} runs ---")
        for title, fns in (
            ("whole-function candidates", coalesce_variants(ids)),
            ("steps of the current one", coalesce_steps(ids)),
        ):
            results = time_interleaved(fns, reps, warmup)
            baseline = next(iter(results.values()))[0]
            print(f"  {title}")
            for name, (lo, med) in results.items():
                share = f"{lo / baseline * 100:5.0f}%"
                print(
                    f"    {name:<20} min={lo / 1000:7.2f}us "
                    f"med={med / 1000:7.2f}us  {share}"
                )
        print()


def run_host_descriptors(
    region: ECSharedRegion,
    num_blocks: int,
    block_size: int,
    entry_blocks: int,
    entries: int,
    reps: int,
    warmup: int,
    seed: int,
) -> None:
    """Compare the address-arithmetic arms on host time per step.

    Addresses are all this measures, so no copy is issued and the destination
    pointers need not be real allocations -- but the region's is real, and the
    ids come from the same layouts the copy benchmark uses, so descriptor counts
    match what production sees.
    """
    region_ptr = region.blocks.data_ptr()
    gpu_ptrs = [0x7F0000000000 + i * entry_blocks * block_size for i in range(entries)]
    arms = descriptor_dtype_arms(block_size)
    stride = num_blocks // entries
    assert stride >= entry_blocks, "region too small for disjoint per-entry slices"

    header = (
        f"{'layout':<14} {'arm':<15} {'ndesc':>6} {'min us':>8} "
        f"{'med us':>8} {'min vs int64':>13}"
    )
    print(header)
    print("-" * len(header))
    for layout in ("desc_consec", "asc_consec", "random", "random_sorted"):
        ids_per_entry = [
            build_ids(layout, num_blocks, entry_blocks, seed, base=e * stride)
            for e in range(entries)
        ]
        results = time_descriptor_build(
            ids_per_entry, region_ptr, gpu_ptrs, arms, reps, warmup
        )
        baseline = results["int64"][0]
        for name, (lo, med, ndesc) in results.items():
            delta = (lo - baseline) / baseline * 100
            print(
                f"{layout:<14} {name:<15} {ndesc:>6} {lo / 1000:>8.1f} "
                f"{med / 1000:>8.1f} {delta:>12.1f}%"
            )
        print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--region-bytes", type=int, default=2 * 1024**3)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--entry-blocks", type=int, default=DEFAULT_ENTRY_BLOCKS)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--host-descriptors",
        action="store_true",
        help="measure host descriptor-build time instead of the copies",
    )
    p.add_argument(
        "--coalesce-breakdown",
        action="store_true",
        help="break one entry's _coalesce_runs down by operation",
    )
    p.add_argument(
        "--entries",
        type=int,
        default=8,
        help="entries saved per step, for --host-descriptors",
    )
    p.add_argument(
        "--host-reps",
        type=int,
        default=2000,
        help="steps timed per arm; host timings need many more reps than copies",
    )
    args = p.parse_args()

    num_blocks = args.region_bytes // args.block_size
    n = args.entry_blocks
    entry_bytes = n * args.block_size

    # Pure host arithmetic over block ids: needs neither the region nor a device.
    if args.coalesce_breakdown:
        print(f"{args.host_reps} calls per arm\n")
        run_coalesce_breakdown(num_blocks, n, args.host_reps, args.warmup, args.seed)
        return 0

    region = ECSharedRegion(
        engine_id=_ENGINE_ID,
        num_blocks=num_blocks,
        block_size_bytes=args.block_size,
    )
    try:
        print(
            f"region {args.region_bytes / 1024**3:.2f} GiB / {num_blocks} blocks "
            f"x {args.block_size} B | entry {n} blocks = "
            f"{entry_bytes / 1e6:.2f} MB\n"
        )

        # No copy is issued, so this arm needs neither pinning nor a device.
        if args.host_descriptors:
            print(f"{args.entries} entries/step, {args.host_reps} steps per arm\n")
            run_host_descriptors(
                region,
                num_blocks,
                args.block_size,
                n,
                args.entries,
                args.host_reps,
                args.warmup,
                args.seed,
            )
            return 0

        region.pin_memory()
        gpu = torch.zeros(entry_bytes, dtype=torch.int8, device="cuda")
        region_ptr, gpu_ptr = region.blocks.data_ptr(), gpu.data_ptr()

        if args.verify:
            for coalesce in (False, True):
                ids = build_ids(
                    "random_sorted" if coalesce else "random",
                    num_blocks,
                    n,
                    args.seed,
                )
                ok, ndesc = verify(region, ids, args.block_size, coalesce)
                print(
                    f"verify coalesce={coalesce!s:<5} descriptors={ndesc:<5} "
                    f"bytes correct: {'YES' if ok else 'NO'}"
                )
            print()

        # Each rep reads a different slice of the region, spread far apart, so
        # no rep benefits from the previous rep's host cache state.
        n_sets = args.warmup + args.reps
        stride = num_blocks // n_sets
        assert stride >= n, "region too small for disjoint per-rep slices"
        header = (
            f"{'layout':<14} {'coal':<5} {'dir':<5} {'ndesc':>6} {'ms':>8} {'GB/s':>8}"
        )
        print(header)
        print("-" * len(header))
        for layout in ("desc_consec", "asc_consec", "random", "random_sorted"):
            ids_per_rep = [
                build_ids(layout, num_blocks, n, args.seed, base=r * stride)
                for r in range(n_sets)
            ]
            for coalesce in (False, True):
                for direction in ("load", "save"):
                    # The connector's own choice per direction; the flag made no
                    # measurable difference in either direction.
                    any_order = direction == "load"
                    descriptors = [
                        build_descriptors(
                            ids,
                            region_ptr,
                            gpu_ptr,
                            args.block_size,
                            direction,
                            coalesce,
                        )
                        for ids in ids_per_rep
                    ]
                    ms = time_batch(descriptors, any_order, args.warmup)
                    gbps = entry_bytes / (ms / 1000) / 1e9
                    print(
                        f"{layout:<14} {str(coalesce):<5} {direction:<5} "
                        f"{len(descriptors[0][0]):>6} {ms:>8.3f} {gbps:>8.1f}"
                    )
    finally:
        region.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
