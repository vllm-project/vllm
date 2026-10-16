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
from vllm.model_executor.layers.mamba.ops.ssd_emit import (
    _bmm_chunk_workspace_range_fwd,
    _chunk_scan_workspace_range_fwd,
    _fold_chunk_fwd,
    _workspace_chunk_cumsum_fwd,
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
    zero_state_rows: torch.Tensor
    """Sequences that still have no completed chunk after this step; their
    SSM state slot is zeroed so a later single-row decode can read it as the
    boundary state."""


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
    zero_state_rows: list[int] = []
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
        if nc - n_pre == 0 and full == 0:
            zero_state_rows.append(i)
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
        zero_state_rows=i64(zero_state_rows),
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
    pos: torch.Tensor,
    chunk_size: int,
    buffers: ExactReplayBuffers,
) -> None:
    """One decode step, one token per row, through the row-gated kernels.

    Instead of re-running the chunked scan over each row's partial chunk, the
    kernels write the token into the buffers and compute only that token's
    output row from the buffered inputs and the boundary state in
    ``ssm_state`` (zero while the sequence is still in its first chunk, see
    ``ExactReplayMetadata.zero_state_rows``). Rows whose token completes a
    chunk fold it into the boundary state with the same arithmetic as the
    prefill kernels, so the bits match a single-shot prefill exactly.

    Every kernel runs over all rows with shapes that depend only on the batch
    size and skips rows with a negative slot or position, so the step can be
    captured in a CUDA graph and padding rows (position -1) touch no buffer
    or state; their output rows are zero.

    Args:
        x: ``(num_rows, nheads, head_dim)`` inputs of this step's tokens.
        dt: ``(num_rows, nheads)`` raw dt of this step's tokens.
        B: ``(num_rows, ngroups, dstate)`` of this step's tokens.
        C: ``(num_rows, ngroups, dstate)`` of this step's tokens.
        A: per-head parameter as in :func:`exact_replay_ssd`.
        D: per-head parameter as in :func:`exact_replay_ssd`.
        dt_bias: per-head parameter as in :func:`exact_replay_ssd`.
        out: ``(num_rows, nheads, head_dim)`` preallocated output.
        ssm_state: the fp32 SSM cache.
        slots: ``(num_rows,)`` int32 state slot of each row.
        pos: ``(num_rows,)`` int32 position of each row's token inside its
            partial chunk, i.e. its computed token count modulo ``chunk_size``;
            -1 for padding rows.
        chunk_size: the model's SSD chunk size.
        buffers: the partial-chunk input buffers.

    """
    num_rows, nheads, _ = x.shape
    ngroups = B.shape[1]
    device = x.device
    assert slots.dtype == torch.int32 and pos.dtype == torch.int32
    assert slots.shape == (num_rows,) and pos.shape == (num_rows,)

    dt_out = torch.empty(
        nheads, num_rows, chunk_size, dtype=torch.float32, device=device
    )
    dA_cumsum = torch.empty_like(dt_out)
    cb = torch.empty(num_rows, ngroups, chunk_size, dtype=torch.float32, device=device)
    _workspace_chunk_cumsum_fwd(
        buffers.dt,
        A,
        chunk_size,
        slots,
        pos,
        dt_bias,
        current_dt=dt,
        dt_out=dt_out,
        dA_cumsum=dA_cumsum,
    )
    _bmm_chunk_workspace_range_fwd(C, buffers.B, chunk_size, slots, pos, cb, B)
    _chunk_scan_workspace_range_fwd(
        cb,
        buffers.x,
        dt_out,
        dA_cumsum,
        C,
        ssm_state,
        slots,
        pos,
        out,
        D=D,
        current_x=x,
    )
    _fold_chunk_fwd(buffers.x, buffers.B, dt_out, dA_cumsum, ssm_state, slots, pos)


def exact_replay_spec_decode_layout(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_spec: int,
    chunk_size: int,
    pos_out: torch.Tensor | None = None,
    token_idx_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token emit positions of speculative decode rows.

    A speculative decode row holds the last sampled token followed by up to
    ``num_spec`` drafts. Its first token is always accepted, so when it
    completes a chunk the boundary state may be folded at once. A draft that
    completes a chunk would fold not-yet-verified inputs into the boundary
    state, which a rejection could not undo; the rejection sampler therefore
    rejects any draft at a chunk's last position (``draft_chunk_size`` of the
    model runner V2 ``RejectionSampler``), and this layout skips it and every
    later token of the row. Accepted tokens thus never cross an uncommitted
    chunk boundary and the state stays exactly that of non-speculative decode.

    Args:
        seq_lens: ``(num_rows,)`` sequence lengths including this step's
            tokens; 0 for CUDA graph padding rows.
        query_start_loc: ``(num_rows + 1,)`` token offsets of the rows.
        num_spec: the number of speculative tokens per step.
        chunk_size: the model's SSD chunk size.
        pos_out: optional ``(num_rows, 1 + num_spec)`` int32 output buffer.
        token_idx_out: optional ``(num_rows, 1 + num_spec)`` int64 output
            buffer.

    Returns:
        ``(pos, token_idx)``, each ``(num_rows, 1 + num_spec)``: the position
        of the row's j-th token inside its partial chunk and that token's
        index in the decode tokens, or -1 for both where the token is skipped.

    """
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_computed = seq_lens - query_lens
    j = torch.arange(1 + num_spec, device=seq_lens.device)
    # Steps from the first token to the next chunk-final position; a draft
    # landing there is rejected, so only the drafts before it are evaluated.
    to_chunk_end = chunk_size - 1 - torch.remainder(num_computed, chunk_size)
    draft_limit = torch.where(to_chunk_end == 0, chunk_size, to_chunk_end)
    valid = (
        (j < query_lens[:, None])
        & ((j == 0) | (j < draft_limit[:, None]))
        & (seq_lens[:, None] > 0)
    )
    pos = torch.where(valid, torch.remainder(num_computed[:, None] + j, chunk_size), -1)
    token_idx = torch.where(valid, query_start_loc[:-1, None] + j, -1)
    if pos_out is None:
        return pos.to(torch.int32), token_idx.to(torch.int64)
    assert token_idx_out is not None
    pos_out.copy_(pos)
    token_idx_out.copy_(token_idx)
    return pos_out, token_idx_out


def exact_replay_emit_spec(
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
    pos: torch.Tensor,
    token_idx: torch.Tensor,
    chunk_size: int,
    buffers: ExactReplayBuffers,
) -> None:
    """Speculative decode: :func:`exact_replay_emit` once per token slot.

    Call ``j`` runs the j-th token of every row, after calls ``0..j-1`` have
    written the row's earlier tokens into the buffers, so each token sees
    exactly the buffers and boundary state of non-speculative decode. Skipped
    tokens (``pos == -1``, see :func:`exact_replay_spec_decode_layout`) touch
    no buffer or state and get zero output.

    Args:
        x: ``(num_tokens, nheads, head_dim)`` inputs of the decode tokens.
        dt: ``(num_tokens, nheads)`` raw dt.
        B: ``(num_tokens, ngroups, dstate)``.
        C: ``(num_tokens, ngroups, dstate)``.
        A: per-head parameter as in :func:`exact_replay_ssd`.
        D: per-head parameter as in :func:`exact_replay_ssd`.
        dt_bias: per-head parameter as in :func:`exact_replay_ssd`.
        out: ``(num_tokens, nheads, head_dim)`` preallocated output.
        ssm_state: the fp32 SSM cache.
        slots: ``(num_rows,)`` int32 state slot of each row.
        pos: ``(num_rows, 1 + num_spec)`` int32 token positions.
        token_idx: ``(num_rows, 1 + num_spec)`` int64 token indices.
        chunk_size: the model's SSD chunk size.
        buffers: the partial-chunk input buffers.

    """
    num_tokens = x.shape[0]
    gather_idx = token_idx.clamp(min=0)
    # Skipped tokens scatter into a trailing scratch row.
    scatter_idx = torch.where(token_idx < 0, num_tokens, token_idx)
    out_ext = out.new_zeros((num_tokens + 1, *out.shape[1:]))
    slots = slots.contiguous()
    for j in range(pos.shape[1]):
        rows = gather_idx[:, j]
        out_j = out.new_empty((rows.shape[0], *out.shape[1:]))
        exact_replay_emit(
            x[rows],
            dt[rows],
            B[rows],
            C[rows],
            A=A,
            D=D,
            dt_bias=dt_bias,
            out=out_j,
            ssm_state=ssm_state,
            slots=slots,
            pos=pos[:, j].contiguous(),
            chunk_size=chunk_size,
            buffers=buffers,
        )
        out_ext[scatter_idx[:, j]] = out_j
    out.copy_(out_ext[:num_tokens])
