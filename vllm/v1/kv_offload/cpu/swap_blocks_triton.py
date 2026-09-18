# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel + tuned constants for the ``swap_blocks_batch`` fast path."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

# Constants tuned empirically on H100 (PCIe Gen5):
#   NUM_SMS         - smallest SM slice within 5% of peak bandwidth at the
#                     8-32 KB block sizes that matter in practice
#   THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA; above
#                     this the C++ cuMemcpyBatchAsync path takes the lead
#   MIN_N           - minimum batch size where Triton's per-launch cost is
#                     amortized; below this DMA wins
NUM_SMS = 12
THRESHOLD_BYTES = 28 * 1024
MIN_N = 16

# Init-time calibration of the DMA/Triton crossover (opt-in):
#   CALIBRATION_NS                - batch sizes timed on both paths
#   CALIBRATION_REPS              - timed runs per batch size (median is used)
#   CALIBRATION_MAX_SCRATCH_BYTES - largest GPU scratch destination we allocate;
#                                   bigger copy sizes keep the defaults
CALIBRATION_NS = (16, 32, 64, 128, 256)
CALIBRATION_REPS = 5
CALIBRATION_MAX_SCRATCH_BYTES = 16 * 1024 * 1024


def default_min_n(copy_size_bytes: int) -> int | None:
    """Smallest batch size from which Triton is used for copies of this size,
    or None if it never is."""
    return MIN_N if copy_size_bytes < THRESHOLD_BYTES else None


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
    min_n: int = MIN_N,
) -> None:
    """Triton implementation of ``swap_blocks_batch`` for small CPU->GPU batches."""
    n = src_addrs.numel()
    # Too few descriptors to amortize Triton's launch cost.
    if n < min_n:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )


def pick_min_n(
    dma_ms: Sequence[float],
    triton_ms: Sequence[float],
    ns: Sequence[int] = CALIBRATION_NS,
) -> int | None:
    """Smallest probed batch size from which Triton is at least as fast as DMA
    at every larger probed size, or None if it loses at the largest one."""
    min_n = None
    for n, dma, tri in reversed(list(zip(ns, dma_ms, triton_ms))):
        if tri > dma:
            break
        min_n = n
    return min_n


def measure_load_paths(
    copy_size: int,
    bytes_per_chunk: int,
    host: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[float]] | None:
    """Time DMA and Triton loads of ``copy_size`` bytes at each of
    CALIBRATION_NS. Returns (dma_ms, triton_ms), or None if it can't be done.

    ``host`` is the pinned CPU tensor real loads read from. The destination is
    a scratch GPU buffer, so no KV block is written."""
    n_max = CALIBRATION_NS[-1]
    scratch_bytes = n_max * copy_size
    # Source slots are taken every other one so neighbours are not contiguous.
    if (
        scratch_bytes > CALIBRATION_MAX_SCRATCH_BYTES
        or not host.is_contiguous()
        or host.numel() * host.element_size() < 2 * scratch_bytes
    ):
        return None
    try:
        scratch = torch.empty(scratch_bytes, dtype=torch.int8, device=device)
    except torch.OutOfMemoryError:
        return None

    idx = torch.arange(n_max, dtype=torch.int64)
    src = (host.data_ptr() + 2 * copy_size * idx).pin_memory()
    dst = (scratch.data_ptr() + copy_size * idx).pin_memory()
    sizes = torch.full((n_max,), copy_size, dtype=torch.int64).pin_memory()

    def run_ms(fn, n: int) -> float:
        times = []
        # The first run is a warm-up (and compiles the Triton kernel).
        for _ in range(1 + CALIBRATION_REPS):
            torch.accelerator.synchronize()
            t0 = time.perf_counter()
            fn(src[:n], dst[:n], sizes[:n])
            torch.accelerator.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        return sorted(times[1:])[CALIBRATION_REPS // 2]

    def dma(s, d, z):
        ops.swap_blocks_batch(s, d, z, is_src_access_order_any=True)

    def tri(s, d, z):
        # min_n=0 so the wrapper never falls back to DMA while we time it.
        swap_blocks_batch(s, d, z, bytes_per_chunk=bytes_per_chunk, min_n=0)

    dma_ms = [run_ms(dma, n) for n in CALIBRATION_NS]
    triton_ms = [run_ms(tri, n) for n in CALIBRATION_NS]
    return dma_ms, triton_ms
