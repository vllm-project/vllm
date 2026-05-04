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

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _apply_steering_kernel(
    hidden_ptr,
    table_ptr,
    index_ptr,
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
    """Compute ``out[i, j] = hidden[i, j] + cast(table[index[i], j])``."""
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    row = tl.load(index_ptr + pid_n)

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    table_row_ptr = table_ptr + row * t_stride_r
    out_row_ptr = out_ptr + pid_n * o_stride_n

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
) -> torch.Tensor:
    """Compute ``hidden_states + table[index[:N]].to(hidden_states.dtype)``.

    Returns a freshly allocated output tensor with the same shape and
    dtype as ``hidden_states``. Empty batches (``N == 0``) short-circuit
    without launching the kernel — Triton can fail on zero-sized grids.
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


def warmup_apply_steering_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """JIT-compile the kernel ahead of CUDA graph capture.

    The kernel is launched with a tiny dummy batch so Triton's first-call
    JIT cost happens before any captured forward pass. The shapes used
    here only need to share ``BLOCK_H`` and dtypes with the real launch
    for the compiled artifact to be reused.
    """
    if device.type != "cuda":
        return
    dummy_hidden = torch.zeros(1, hidden_size, dtype=compute_dtype, device=device)
    dummy_table = torch.zeros(
        max(table_rows, 1), hidden_size, dtype=table_dtype, device=device
    )
    dummy_index = torch.zeros(1, dtype=torch.long, device=device)
    apply_steering_triton(dummy_hidden, dummy_table, dummy_index)
