# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the per-request activation-steering custom op.

The kernel fuses the gather, dtype cast, and add operations performed by
``apply_steering`` into a single launch and a single pass over
``hidden_states``. The eager Python implementation produces a fresh
output tensor (matching ``hidden_states.dtype``); this kernel preserves
that contract — output is written to a freshly allocated tensor, never
in place, to keep the ``torch.compile`` graph contract stable.

Layout assumptions:

- ``hidden_states`` is row-contiguous ``[N, H]`` in compute dtype.
- ``steering_table`` is row-contiguous ``[num_rows, H]`` in any dtype;
  values are cast to ``hidden_states.dtype`` inside the kernel.
- ``steering_index`` is ``int64`` and may be longer than ``N``; only
  the first ``N`` entries are read.

The kernel launches one program per token row (``grid = (N,)``) and
walks the hidden dimension in ``BLOCK_H`` chunks with masked loads
and stores so non-power-of-two hidden sizes are handled correctly.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _apply_steering_kernel(
    hidden_ptr,
    table_ptr,
    index_ptr,
    active_ptr,
    out_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    t_stride_r,
    t_stride_h,
    o_stride_n,
    o_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Compute ``out[i, j] = hidden[i, j] + cast(table[index[i], j])``.

    When the byte at ``active_ptr`` is zero, the kernel skips the gather
    and emits ``out[i, j] = hidden[i, j]`` so the inactive-hook short
    circuit keeps the same output-tensor contract as the active path.
    The active-flag is a tensor (not a Python branch) so the compiled
    graph topology stays stable across batches whose active-hook set
    differs.
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    active = tl.load(active_ptr)
    if active == 0:
        # Inactive: skip the table gather and the dtype cast entirely.
        # We still must produce ``out == hidden_states`` so the gather-
        # path callers see consistent value semantics; this branch
        # eliminates the table memory traffic and the cast, which is
        # the dominant cost when the table is bf16/fp16 and hidden_size
        # is large.  Combine with the in-place sibling branch
        # (``mutates_args=["hidden_states"]``) for a full skip with no
        # memcpy.
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
            tl.store(out_row_ptr + h_idx * o_stride_h, h_vals, mask=mask)
        return

    row = tl.load(index_ptr + pid_n)
    table_row_ptr = table_ptr + row * t_stride_r

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
        t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)
        # Cast table values to hidden dtype so dtype-mismatched tables
        # (fp32 table + bf16 hidden, common before PR 1 lands) work.
        result = h_vals + t_vals.to(h_vals.dtype)
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)


def _choose_block_h(hidden_size: int) -> int:
    """Pick a sensible ``BLOCK_H`` for the kernel given the hidden size.

    For small hidden sizes (< 2048) round up to the next power of two so
    a single iteration covers the row. For larger hidden sizes cap at
    2048 — the loop in the kernel handles multi-iteration walks.

    Uses a manual power-of-two computation rather than
    ``triton.next_power_of_2`` so the module remains importable on
    environments where Triton is disabled (e.g. CPU-only test runs);
    the kernel itself is only ever launched on CUDA.
    """
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


def apply_steering_triton(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Compute ``hidden_states + table[index[:N]].to(hidden_states.dtype)``.

    Returns a freshly allocated output tensor with the same shape and
    dtype as ``hidden_states``. Empty batches (``N == 0``) short-circuit
    without launching the kernel — Triton can fail on zero-sized grids.

    ``any_active`` is a single-element bool tensor; when ``False`` the
    kernel still launches but skips the table gather and emits
    ``hidden_states`` into the freshly-allocated output.
    """
    out = torch.empty_like(hidden_states)
    N = hidden_states.shape[0]
    if N == 0:
        return out

    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _apply_steering_kernel[(N,)](
        hidden_states,
        steering_table,
        steering_index,
        any_active,
        out,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        steering_table.stride(0),
        steering_table.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def _default_warmup_sizes() -> list[int]:
    """Fallback warmup batch sizes when no capture-size list is supplied.

    Mirrors the powers-of-two and small-batch shapes that vLLM commonly
    captures when ``cudagraph_capture_sizes`` is left to its default.
    Used only when the caller cannot pass an explicit list (e.g.
    standalone tests).
    """
    return [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def _dump_jit_cache_keys() -> None:
    """One-shot diagnostic — log the keys present in the kernel cache.

    Enabled when ``VLLM_STEERING_DUMP_JIT_CACHE=1``.  Walks
    ``_apply_steering_kernel.cache`` (a per-device dict from Triton's
    JITFunction) and emits each variant key at INFO so a benchmark run
    can reveal what specializations Triton actually built.
    """
    cache = getattr(_apply_steering_kernel, "cache", None)
    if cache is None:
        logger.info(
            "steering JIT cache dump requested but kernel has no cache "
            "attribute (Triton may be disabled)"
        )
        return
    total = 0
    for device_id, device_cache in cache.items():
        try:
            keys = list(device_cache.keys())
        except AttributeError:
            keys = []
        total += len(keys)
        logger.info(
            "steering JIT cache: device=%s variants=%d",
            device_id,
            len(keys),
        )
        for i, key in enumerate(keys):
            logger.info("  variant[%d]: %r", i, key)
    logger.info("steering JIT cache: total_variants=%d", total)


def _kernel_cache_size() -> int:
    """Return the total number of compiled variants across all devices.

    Returns 0 when the kernel has not yet been built (no ``cache``
    attribute, e.g. when Triton is disabled in the importing process).
    """
    cache = getattr(_apply_steering_kernel, "cache", None)
    if cache is None:
        return 0
    total = 0
    for device_cache in cache.values():
        try:
            total += len(device_cache)
        except TypeError:
            continue
    return total


def warmup_apply_steering_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile the kernel ahead of CUDA graph capture.

    The kernel is launched with dummy tensors at every shape vLLM will
    subsequently hit so Triton's first-call JIT cost — observed as
    ~10 ms ``cuLibraryLoadData`` events on a 3090 with gemma-3-4b-it,
    even in modes that never apply a non-zero steering vector — happens
    before any served request window.

    *capture_sizes* is the full list of batch dims that warmup should
    drive the kernel for.  Pass ``vllm_config.compilation_config.
    cudagraph_capture_sizes`` so the warmup stays in sync with the
    captured shapes.  When ``None``, falls back to a representative
    powers-of-two list.

    The ``apply_steering`` registered op is the path used at runtime; we
    route warmup through it (``torch.ops.vllm.apply_steering``) rather
    than calling the Triton wrapper directly, because going direct would
    compile a different stride-class specialization than what the
    dispatched runtime call ends up triggering.

    Both ``any_active`` states are exercised at every batch size — the
    inactive branch and the active branch share the same compiled
    artifact (the flag is a tensor, not a constexpr) but driving both
    flags is cheap insurance.

    Total compile count and cumulative warmup wall-clock are logged at
    INFO so the cost is visible.
    """
    if device.type != "cuda":
        return

    sizes = capture_sizes if capture_sizes else _default_warmup_sizes()
    # Defensive: deduplicate and sort to drive smaller shapes first
    # (smaller compiles tend to be slightly cheaper).
    sizes = sorted({int(s) for s in sizes if int(s) > 0})
    if not sizes:
        return

    cache_before = _kernel_cache_size()
    max_n = max(sizes)
    # One large allocation each; we only ever read/write the leading N
    # rows per launch, so reusing a single buffer per dtype avoids
    # ``max_n`` independent allocations.
    hidden_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    table_buf = torch.zeros(
        max(table_rows, 1), hidden_size, dtype=table_dtype, device=device
    )
    index_buf = torch.zeros(max_n, dtype=torch.long, device=device)
    active_flag = torch.zeros(1, dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    for n in sizes:
        hidden_view = hidden_buf[:n]
        index_view = index_buf[:n]
        # Inactive path first — exercises the short-circuit branch.
        active_flag.fill_(False)
        torch.ops.vllm.apply_steering(hidden_view, table_buf, index_view, active_flag)
        # Active path — exercises the gather + add.
        active_flag.fill_(True)
        torch.ops.vllm.apply_steering(hidden_view, table_buf, index_view, active_flag)
    # Block until every JIT compile (and cuLibraryLoadData) has retired so
    # the wall-clock measurement and cache-size readback reflect reality.
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    cache_after = _kernel_cache_size()
    new_variants = cache_after - cache_before
    logger.info(
        "steering kernel warmup: shapes=%d variants_compiled=%d "
        "cache_total=%d elapsed_ms=%.1f",
        len(sizes),
        new_variants,
        cache_after,
        elapsed_ms,
    )

    if os.environ.get("VLLM_STEERING_DUMP_JIT_CACHE", "0") == "1":
        _dump_jit_cache_keys()
