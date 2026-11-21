# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Init-time autotuning of the DMA/Triton cutoff for CPU->GPU KV loads."""

import statistics
import time
from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    MIN_N,
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)

# Batch sizes timed on both paths; the cutoff is one of them.
_BATCH_SIZES = (16, 32, 64, 128, 256)


class LoadPathAutotuneError(RuntimeError):
    """Autotuning left copies in flight that could not be waited for."""


def autotune_load_min_n(
    copy_sizes: Sequence[int],
    bytes_per_chunk: int,
    host: torch.Tensor,
    device: torch.device,
) -> int | None:
    """Batch size from which a load handler uses Triton (None: DMA only).

    Times `cuMemcpyBatchAsync` against the Triton kernel on `device` for each
    of the handler's `copy_sizes`, reading from `host`, the pinned CPU tensor
    real loads read. A copy size that can't be measured keeps the tuned
    default.

    A transfer sends all of a handler's copy sizes in one call, so the
    per-size cutoffs are reduced to one: the largest, or DMA for everything
    if any size never wins on Triton.
    """
    min_n_by_size = {
        size: _autotune_copy_size(size, bytes_per_chunk, host, device)
        for size in copy_sizes
    }
    min_ns = list(min_n_by_size.values())
    reduced = None if None in min_ns else max(m for m in min_ns if m is not None)
    if len(set(min_ns)) > 1:
        logger.info(
            "KV offload load handler mixes copy sizes (%s); using %s for all",
            ", ".join(
                f"{size} B: {_describe(min_n)}" for size, min_n in min_n_by_size.items()
            ),
            _describe(reduced),
        )
    return reduced


def _describe(min_n: int | None) -> str:
    return "DMA only" if min_n is None else f"Triton from N={min_n}"


def _autotune_copy_size(
    copy_size: int,
    bytes_per_chunk: int,
    host: torch.Tensor,
    device: torch.device,
) -> int | None:
    default = MIN_N if copy_size < THRESHOLD_BYTES else None
    t0 = time.perf_counter()
    try:
        ratios = _measure_copy_size(
            copy_size, bytes_per_chunk, host, device, _BATCH_SIZES
        )
    except LoadPathAutotuneError:
        raise
    except Exception:
        logger.warning(
            "KV offload load path autotuning failed for %d-byte copies, "
            "keeping the default (%s)",
            copy_size,
            _describe(default),
            exc_info=True,
        )
        return default
    if ratios is None:
        logger.info(
            "KV offload load path not autotuned for %d-byte copies (the copy "
            "size, the CPU region or free GPU memory is out of range), "
            "keeping the default (%s)",
            copy_size,
            _describe(default),
        )
        return default
    min_n = _pick_min_n(_BATCH_SIZES, ratios, default)
    logger.info(
        "KV offload load path autotuned for %d-byte copies in %.1f ms: %s "
        "(default: %s); Triton/DMA time at N=%s: %s",
        copy_size,
        (time.perf_counter() - t0) * 1e3,
        _describe(min_n),
        _describe(default),
        "/".join(map(str, _BATCH_SIZES)),
        " ".join(f"{r:.2f}" for r in ratios),
    )
    return min_n


def _pick_min_n(
    batch_sizes: Sequence[int],
    ratios: Sequence[float],
    default: int | None,
) -> int | None:
    """Smallest batch size from which Triton is used at every larger one, or
    None if it is not used at the largest.

    `ratios` are Triton time / DMA time per batch size. Where the two paths
    are within the margin of each other, the default (Triton from `default`,
    never if None) decides, so noise around parity does not move the result.
    """
    # How much faster a path must be at a batch size to override the default.
    margin = 0.05
    min_n = None
    for n, ratio in reversed(list(zip(batch_sizes, ratios))):
        if ratio < 1 - margin:
            use_triton = True
        elif ratio > 1 + margin:
            use_triton = False
        else:
            use_triton = default is not None and n >= default
        if not use_triton:
            break
        min_n = n
    return min_n


def _measure_copy_size(
    copy_size: int,
    bytes_per_chunk: int,
    host: torch.Tensor,
    device: torch.device,
    batch_sizes: Sequence[int],
) -> list[float] | None:
    """Median Triton/DMA time ratio of `copy_size`-byte loads at each of
    `batch_sizes`, or None if this copy size can't be measured.

    `host` holds one CPU chunk per row. With the shared offload region it is
    a strided view, so the copies read row starts rather than assuming
    contiguous memory. The destination is a scratch GPU buffer, so no KV
    block is written.
    """
    # Passes over batch_sizes, so each one is sampled at separate moments.
    rounds = 3
    # DMA/Triton pairs timed per batch size and pass.
    reps = 5
    # Largest scratch destination; bigger copy sizes are not measured.
    max_scratch_bytes = 16 * 1024 * 1024

    n_max = max(batch_sizes)
    scratch_bytes = n_max * copy_size
    # Copies read every other row so neighbours are not contiguous.
    if (
        scratch_bytes > max_scratch_bytes
        or host.dim() != 2
        or host.stride(1) != 1
        or host.size(1) * host.element_size() < copy_size
        or host.size(0) < 2 * n_max
    ):
        return None
    try:
        scratch = torch.empty(scratch_bytes, dtype=torch.int8, device=device)
    except torch.OutOfMemoryError:
        return None

    idx = torch.arange(n_max, dtype=torch.int64)
    row_bytes = host.stride(0) * host.element_size()
    src = (host.data_ptr() + 2 * row_bytes * idx).pin_memory()
    dst = (scratch.data_ptr() + copy_size * idx).pin_memory()
    sizes = torch.full((n_max,), copy_size, dtype=torch.int64).pin_memory()

    def run_ms(fn, n: int) -> float:
        torch.accelerator.synchronize(device)
        t0 = time.perf_counter()
        fn(src[:n], dst[:n], sizes[:n])
        torch.accelerator.synchronize(device)
        return (time.perf_counter() - t0) * 1e3

    def dma(s, d, z):
        ops.swap_blocks_batch(s, d, z, is_src_access_order_any=True)

    def tri(s, d, z):
        # min_n=0 so the wrapper never falls back to DMA while we time it.
        swap_blocks_batch(s, d, z, bytes_per_chunk=bytes_per_chunk, min_n=0)

    ratios: list[list[float]] = [[] for _ in batch_sizes]
    try:
        # Real loads run on a dedicated stream (see transfer_async). On the
        # legacy default stream ops.swap_blocks_batch can't use
        # cuMemcpyBatchAsync and issues one cudaMemcpyAsync per copy, so time
        # both paths on one as well.
        with torch.cuda.stream(torch.cuda.Stream(device=device)):
            for round_idx in range(rounds):
                for i, n in enumerate(batch_sizes):
                    if round_idx == 0:
                        # Warm-up, which also compiles the Triton kernel.
                        run_ms(dma, n)
                        run_ms(tri, n)
                    for _ in range(reps):
                        # Back to back, so a slow spell hits both paths alike.
                        dma_ms = run_ms(dma, n)
                        ratios[i].append(run_ms(tri, n) / dma_ms)
    except Exception as measure_error:
        # A path that raised may have submitted some of its copies first.
        # They still write into scratch and read the descriptor tensors, so
        # wait for them before this frame releases those buffers.
        try:
            torch.accelerator.synchronize(device)
        except Exception as drain_error:
            raise LoadPathAutotuneError(
                "Could not wait for the copies submitted while autotuning the "
                "KV offload load path"
            ) from drain_error
        raise measure_error
    return [statistics.median(r) for r in ratios]
