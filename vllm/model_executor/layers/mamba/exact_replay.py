# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 exact-replay mode: one SSD step that reproduces single-shot prefill.

In exact-replay mode every SSD call for a sequence starts at the sequence's
last chunk boundary. The fp32 state at that boundary lives in the regular SSM
cache; the inputs of the partial chunk after it (x, raw dt, B) live in three
per-slot buffers appended to the mamba state. C is not buffered: it only
enters a token's own output row, and the re-fed tokens' rows are discarded.
Re-feeding those inputs in front of the current step's tokens makes the
chunked-scan kernels see exactly the chunk grid of a single-shot prefill, so
prefill, chunked prefill and decode produce identical bits.
"""

from typing import NamedTuple

import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_emit import (
    _bmm_chunk_workspace_range_fwd,
    _chunk_scan_workspace_range_fwd,
    _workspace_chunk_cumsum_fwd,
)


class ExactReplayBuffers(NamedTuple):
    """Per-slot partial-chunk input buffers, each ``(num_slots, chunk_size, ...)``.

    The tensors are views into the paged mamba cache, so the slot dimension is
    not contiguous with the position dimension: index them as ``buf[slot, pos]``
    and never flatten the two leading dimensions.
    """

    x: torch.Tensor
    """``(num_slots, chunk_size, nheads, head_dim)`` in the activation dtype."""
    dt: torch.Tensor
    """``(num_slots, chunk_size, nheads)``, raw dt before bias and softplus."""
    B: torch.Tensor
    """``(num_slots, chunk_size, ngroups, dstate)``."""


def exact_replay_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    slots: torch.Tensor,
    meta,
    chunk_size: int,
    buffers: ExactReplayBuffers,
) -> None:
    """Run one exact-replay SSD step for a batch of sequences.

    Works for prefill rows and decode rows alike. Every sequence is processed
    from its last chunk boundary: the buffered inputs of its partial chunk are
    re-fed first, followed by this step's tokens. Only this step's outputs are
    written to ``out``; the boundary state in ``ssm_state`` is advanced only
    for sequences that complete a chunk, and the inputs of the trailing partial
    chunk are stored back into ``buffers``.

    Args:
        x: ``(num_tokens, nheads, head_dim)`` inputs of this step's tokens.
        dt: ``(num_tokens, nheads)`` raw dt of this step's tokens.
        B: ``(num_tokens, ngroups, dstate)``.
        C: ``(num_tokens, ngroups, dstate)``.
        A: ``(nheads,)`` fp32 per-head decay parameter.
        D: ``(nheads,)`` skip-connection parameter.
        dt_bias: ``(nheads,)`` dt bias (softplus is applied by the kernel).
        out: ``(num_tokens, nheads, head_dim)`` preallocated output, written
            in place.
        ssm_state: the fp32 SSM cache, ``(num_slots, nheads, head_dim, dstate)``.
        slots: ``(num_seqs,)`` state slot of each sequence, in batch order.
            These are the calling layer's own state indices; the metadata
            only refers to batch rows.
        meta: an ``ExactReplayMetadata`` built for these sequences.
        chunk_size: the model's SSD chunk size.
        buffers: the partial-chunk input buffers.
    """
    nheads, head_dim = x.shape[1], x.shape[2]
    ngroups, dstate = B.shape[1], B.shape[2]
    n_aug = meta.num_aug_tokens
    slots64 = slots.to(torch.int64)
    augmented = meta.buffered_seq.numel() > 0
    if augmented:
        buffered_slot = slots64[meta.buffered_seq]
        x_aug = x.new_empty((n_aug, nheads, head_dim))
        dt_aug = dt.new_empty((n_aug, nheads))
        B_aug = B.new_empty((n_aug, ngroups, dstate))
        # C only affects a token's own output row and the re-fed rows are
        # discarded, so the buffered positions get zeros instead of history.
        C_aug = C.new_zeros((n_aug, ngroups, dstate))
        x_aug[meta.buffered_dst] = buffers.x[buffered_slot, meta.buffered_pos]
        dt_aug[meta.buffered_dst] = buffers.dt[buffered_slot, meta.buffered_pos]
        B_aug[meta.buffered_dst] = buffers.B[buffered_slot, meta.buffered_pos]
        x_aug[meta.step_dst] = x
        dt_aug[meta.step_dst] = dt
        B_aug[meta.step_dst] = B
        C_aug[meta.step_dst] = C
        out_aug = torch.empty_like(x_aug)
    else:
        x_aug, dt_aug, B_aug, C_aug, out_aug = x, dt, B, C, out

    initial_states = torch.where(
        meta.has_boundary_state[:, None, None, None], ssm_state[slots], 0
    )
    states = mamba_chunk_scan_combined_varlen(
        x_aug,
        dt_aug,
        A,
        B_aug,
        C_aug,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        seq_idx=meta.seq_idx,
        cu_seqlens=meta.cu_seqlens,
        cu_chunk_seqlens=meta.cu_chunk_seqlens,
        last_chunk_indices=meta.last_chunk_indices,
        initial_states=initial_states,
        return_intermediate_states=True,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        out=out_aug,
        state_dtype=ssm_state.dtype,
    )
    if augmented:
        out.copy_(out_aug[meta.step_dst])
    if meta.boundary_rows.numel() > 0:
        ssm_state[slots64[meta.boundary_rows]] = states[meta.boundary_chunk_idx]
    if meta.zero_state_rows.numel() > 0:
        # No chunk completed yet: a zero state lets a single-row decode read
        # the slot as its boundary state without a separate flag.
        ssm_state[slots64[meta.zero_state_rows]] = 0
    if meta.store_src.numel() > 0:
        store_slot = slots64[meta.store_seq]
        buffers.x[store_slot, meta.store_pos] = x_aug[meta.store_src]
        buffers.dt[store_slot, meta.store_pos] = dt_aug[meta.store_src]
        buffers.B[store_slot, meta.store_pos] = B_aug[meta.store_src]


def exact_replay_emit(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    slots: torch.Tensor,
    meta,
    chunk_size: int,
    buffers: ExactReplayBuffers,
) -> None:
    """One decode step, one token per sequence, through the row-gated kernels.

    Instead of re-running the chunked scan over each sequence's partial chunk,
    the step writes its token into the buffers and computes only that token's
    output row from the buffered inputs and the boundary state in
    ``ssm_state`` (zero while the sequence is still in its first chunk, see
    ``zero_state_rows``). Sequences that complete a chunk fold it into the
    boundary state with the single-shot kernels, so the bits match a
    single-shot prefill exactly.

    Args:
        x, dt, B, C: ``(num_seqs, ...)`` inputs of this step's tokens.
        A, D, dt_bias: per-head parameters as in :func:`exact_replay_ssd`.
        out: ``(num_seqs, nheads, head_dim)`` preallocated output.
        ssm_state: the fp32 SSM cache.
        slots: ``(num_seqs,)`` state slot of each sequence.
        meta: ``ExactReplayMetadata`` for these rows, all with one token.
        chunk_size: the model's SSD chunk size.
        buffers: the partial-chunk input buffers.
    """
    num_rows, nheads, head_dim = x.shape
    ngroups, dstate = B.shape[1], B.shape[2]
    device = x.device
    slots64 = slots.to(torch.int64)
    slots32 = slots.to(torch.int32)
    pos32 = meta.row_pos
    pos64 = pos32.to(torch.int64)

    buffers.x[slots64, pos64] = x
    buffers.dt[slots64, pos64] = dt
    buffers.B[slots64, pos64] = B

    rows = torch.arange(num_rows, dtype=torch.int32, device=device)
    dt_out = torch.empty(
        nheads, num_rows, chunk_size, dtype=torch.float32, device=device
    )
    dA_cumsum = torch.empty_like(dt_out)
    cb = torch.empty(num_rows, ngroups, chunk_size, dtype=torch.float32, device=device)
    _workspace_chunk_cumsum_fwd(
        buffers.dt,
        A,
        chunk_size,
        slots32,
        pos32,
        dt_bias,
        dt_out=dt_out,
        dA_cumsum=dA_cumsum,
    )
    _bmm_chunk_workspace_range_fwd(
        C, buffers.B, chunk_size, slots32, pos32, rows, pos32, out=cb
    )
    _chunk_scan_workspace_range_fwd(
        cb,
        buffers.x,
        dt_out,
        dA_cumsum,
        C,
        ssm_state,
        slots32,
        slots32,
        pos32,
        rows,
        pos32,
        rows,
        out,
        D=D,
    )

    if meta.boundary_rows.numel() == 0:
        return
    # Fold the completed chunks into their boundary states with the same
    # kernels a single-shot prefill uses; C does not enter the state.
    fold_slots = slots64[meta.boundary_rows]
    n_fold = fold_slots.numel()
    x_f = buffers.x[fold_slots].reshape(n_fold * chunk_size, nheads, head_dim)
    dt_f = buffers.dt[fold_slots].reshape(n_fold * chunk_size, nheads)
    B_f = buffers.B[fold_slots].reshape(n_fold * chunk_size, ngroups, dstate)
    cu = torch.arange(
        0, (n_fold + 1) * chunk_size, chunk_size, dtype=torch.int32, device=device
    )
    seq = torch.arange(n_fold, dtype=torch.int32, device=device)
    states = mamba_chunk_scan_combined_varlen(
        x_f,
        dt_f,
        A,
        B_f,
        torch.zeros_like(B_f),
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        seq_idx=seq,
        cu_seqlens=cu,
        cu_chunk_seqlens=cu,
        last_chunk_indices=seq,
        initial_states=ssm_state[fold_slots],
        return_intermediate_states=False,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        out=torch.empty_like(x_f),
        state_dtype=ssm_state.dtype,
    )
    ssm_state[fold_slots] = states
