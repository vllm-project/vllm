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

from dataclasses import dataclass
from typing import NamedTuple

import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d


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


@dataclass
class ExactReplayMetadata:
    """Per-step metadata for Mamba2 exact-replay mode.

    One instance describes the prefill rows of a step, another the decode
    rows. Every sequence is re-expanded to an *augmented* sequence that starts
    at its last chunk boundary: the buffered inputs of the partial chunk come
    first, then this step's tokens. Index tensors are int64 device tensors;
    the varlen/chunk metadata consumed by the SSD kernels is int32.

    The metadata refers to sequences by their row in the batch and never to
    state slots: the slots come from the layer's own state indices at call
    time. This keeps one instance valid for every KV cache group of a hybrid
    model, where the model runner builds the metadata once and only swaps the
    block table per group.

    Attributes:
        num_aug_tokens: total number of tokens in the augmented layout.
        buffered_seq: batch row of the sequence each buffered token belongs to.
        buffered_pos: position of each buffered token inside its slot's buffer.
        buffered_dst: destination of each buffered token in the augmented layout.
        step_dst: destination of each of this step's tokens in the augmented
            layout (in input order).
        cu_seqlens: ``(num_seqs + 1,)`` cumulative augmented sequence lengths.
        cu_chunk_seqlens: ``(num_chunks + 1,)`` chunk offsets in the augmented
            layout.
        last_chunk_indices: ``(num_seqs,)`` index of each sequence's last chunk.
        seq_idx: ``(num_chunks,)`` sequence index of each chunk.
        has_boundary_state: ``(num_seqs,)`` bool, whether the slot holds a valid
            boundary state (False while a sequence is still in its first chunk).
        boundary_rows: sequences that complete at least one chunk this step.
        boundary_chunk_idx: for each of those sequences, the chunk (indexed into
            the kernel's intermediate states) whose end state becomes the new
            boundary state.
        store_src: positions in the augmented layout of the trailing partial
            chunk's tokens that must be stored into the buffers.
        store_seq: batch row of the sequence each stored token belongs to.
        store_pos: destination position of each stored token.

    """

    num_aug_tokens: int
    buffered_seq: torch.Tensor
    buffered_pos: torch.Tensor
    buffered_dst: torch.Tensor
    step_dst: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_chunk_seqlens: torch.Tensor
    last_chunk_indices: torch.Tensor
    seq_idx: torch.Tensor
    has_boundary_state: torch.Tensor
    boundary_rows: torch.Tensor
    boundary_chunk_idx: torch.Tensor
    store_src: torch.Tensor
    store_seq: torch.Tensor
    store_pos: torch.Tensor


def build_exact_replay_metadata(
    num_computed: list[int],
    query_lens: list[int],
    chunk_size: int,
    device: torch.device,
) -> ExactReplayMetadata:
    """Build :class:`ExactReplayMetadata` on the host.

    Args:
        num_computed: per sequence, tokens processed before this step.
        query_lens: per sequence, tokens scheduled in this step.
        chunk_size: the model's SSD chunk size.
        device: device for the returned tensors.

    Returns:
        The metadata describing the augmented layout of these sequences.

    """
    buffered_seq: list[int] = []
    buffered_pos: list[int] = []
    buffered_dst: list[int] = []
    step_dst: list[int] = []
    cu_seqlens = [0]
    cu_chunk: list[int] = []
    seq_idx: list[int] = []
    last_chunk: list[int] = []
    has_boundary: list[bool] = []
    boundary_rows: list[int] = []
    boundary_chunk_idx: list[int] = []
    store_src: list[int] = []
    store_seq: list[int] = []
    store_pos: list[int] = []
    offset = 0
    for i, (nc, q) in enumerate(zip(num_computed, query_lens)):
        n_pre = nc % chunk_size
        aug_len = n_pre + q
        buffered_seq.extend([i] * n_pre)
        buffered_pos.extend(range(n_pre))
        buffered_dst.extend(range(offset, offset + n_pre))
        step_dst.extend(range(offset + n_pre, offset + aug_len))
        first_chunk = len(cu_chunk)
        n_chunks = cdiv(aug_len, chunk_size)
        cu_chunk.extend(offset + k * chunk_size for k in range(n_chunks))
        seq_idx.extend([i] * n_chunks)
        last_chunk.append(len(cu_chunk) - 1)
        has_boundary.append(nc - n_pre > 0)
        full = aug_len // chunk_size
        if full >= 1:
            boundary_rows.append(i)
            boundary_chunk_idx.append(first_chunk + full - 1)
        tail_len = aug_len % chunk_size
        if full == 0:
            # the buffer already holds positions [0, n_pre): append this step
            store_src.extend(range(offset + n_pre, offset + aug_len))
            store_seq.extend([i] * q)
            store_pos.extend(range(n_pre, aug_len))
        elif tail_len > 0:
            base = offset + aug_len - tail_len
            store_src.extend(range(base, base + tail_len))
            store_seq.extend([i] * tail_len)
            store_pos.extend(range(tail_len))
        offset += aug_len
        cu_seqlens.append(offset)
    cu_chunk.append(offset)

    def i64(v: list[int]) -> torch.Tensor:
        return async_tensor_h2d(v, dtype=torch.int64, device=device)

    def i32(v: list[int]) -> torch.Tensor:
        return async_tensor_h2d(v, dtype=torch.int32, device=device)

    return ExactReplayMetadata(
        num_aug_tokens=offset,
        buffered_seq=i64(buffered_seq),
        buffered_pos=i64(buffered_pos),
        buffered_dst=i64(buffered_dst),
        step_dst=i64(step_dst),
        cu_seqlens=i32(cu_seqlens),
        cu_chunk_seqlens=i32(cu_chunk),
        last_chunk_indices=i32(last_chunk),
        seq_idx=i32(seq_idx),
        has_boundary_state=async_tensor_h2d(
            has_boundary, dtype=torch.bool, device=device
        ),
        boundary_rows=i64(boundary_rows),
        boundary_chunk_idx=i64(boundary_chunk_idx),
        store_src=i64(store_src),
        store_seq=i64(store_seq),
        store_pos=i64(store_pos),
    )


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
    meta: ExactReplayMetadata,
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
    if meta.store_src.numel() > 0:
        store_slot = slots64[meta.store_seq]
        buffers.x[store_slot, meta.store_pos] = x_aug[meta.store_src]
        buffers.dt[store_slot, meta.store_pos] = dt_aug[meta.store_src]
        buffers.B[store_slot, meta.store_pos] = B_aug[meta.store_src]
