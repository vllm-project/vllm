# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-keyed ring cursors for the FlashInfer ReplaySSM speculative SSU.

FlashInfer owns none of this bookkeeping: ``checkpointing_ssu`` reads
``ring_start`` and ``prev_num_accepted_tokens`` and decides internally whether
to checkpoint, but never writes them back. The host must therefore mirror the
kernel's decision exactly so it knows how far to advance the origin next step.

Two things differ from the Triton commit in
``selective_state_update_replayssm_spec`` and neither is cosmetic:

* the ring is exactly ``B + T`` rows, so wraparound is a subtraction, not a
  power-of-two bitmask;
* the kernel's checkpoint predicate in varlen mode is
  ``pnat + seq_len > max_window`` using the **actual** per-row length
  (``mc_of`` in ``kernel_checkpointing_ssu_main.cuh``), while the Triton commit
  uses the maximum ``T``. Those diverge whenever a row is shorter than ``T``.

The Triton path keeps its own cursor kernels unchanged.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _commit_flashinfer_cursors_kernel(
    history_len_ptr,  # (num_blocks,) int32, = FlashInfer prev_num_accepted_tokens
    ring_start_ptr,  # (num_blocks,) int32, = FlashInfer ring_start
    is_flush_ptr,  # (num_blocks,) int8, this step's checkpoint decision
    num_accepted_ptr,  # (batch,) int32, previous step, INCLUDES the bonus token
    query_start_loc_ptr,  # (batch + 1,) int32, this step's packed offsets
    state_batch_indices_ptr,  # (batch,) int32
    null_block_id,
    batch,
    stride_state_indices_batch,
    MAX_WINDOW: tl.constexpr,  # B
    RING_LEN: tl.constexpr,  # R = B + T
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(
        state_batch_indices_ptr + offs * stride_state_indices_batch,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    valid = row_mask & (state_batch_idx != null_block_id)

    old_history = tl.load(history_len_ptr + state_batch_idx, mask=valid, other=0).to(
        tl.int32
    )
    old_origin = tl.load(ring_start_ptr + state_batch_idx, mask=valid, other=0).to(
        tl.int32
    )
    prev_flushed = tl.load(is_flush_ptr + state_batch_idx, mask=valid, other=0).to(
        tl.int32
    )
    accepted = tl.load(num_accepted_ptr + offs, mask=valid, other=0).to(tl.int32)
    accepted = tl.where(valid, accepted, 0)

    # Commit the previous step. A flush folded `old_history` tokens into the
    # checkpoint, so the ring restarts past them; otherwise history just grows.
    committed = accepted > 0
    advanced_origin = old_origin + old_history
    # old_origin < R and old_history <= B < R, so one conditional subtraction
    # is enough -- R is not a power of two, so no bitmask here.
    advanced_origin = tl.where(
        advanced_origin >= RING_LEN, advanced_origin - RING_LEN, advanced_origin
    )
    flush_now = committed & (prev_flushed != 0)
    new_origin = tl.where(flush_now, advanced_origin, old_origin).to(tl.int32)
    new_history = tl.where(
        committed,
        tl.where(prev_flushed != 0, accepted, old_history + accepted),
        old_history,
    ).to(tl.int32)

    # This step's decision, recomputed unconditionally: a row that accepted
    # nothing must not inherit the previous call's flag.
    cur_start = tl.load(query_start_loc_ptr + offs, mask=valid, other=0).to(tl.int32)
    cur_end = tl.load(query_start_loc_ptr + offs + 1, mask=valid, other=0).to(tl.int32)
    cur_len = cur_end - cur_start
    cur_is_flush = ((new_history + cur_len) > MAX_WINDOW).to(tl.int8)

    tl.store(history_len_ptr + state_batch_idx, new_history, mask=valid)
    tl.store(ring_start_ptr + state_batch_idx, new_origin, mask=valid)
    tl.store(is_flush_ptr + state_batch_idx, cur_is_flush, mask=valid)


@triton.jit
def _reset_flashinfer_cursors_kernel(
    history_len_ptr,
    ring_start_ptr,
    is_flush_ptr,
    needs_reset_ptr,  # (batch,) int8, once-per-admission flag
    query_start_loc_ptr,
    state_batch_indices_ptr,
    null_block_id,
    batch,
    stride_state_indices_batch,
    MAX_WINDOW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(
        state_batch_indices_ptr + offs * stride_state_indices_batch,
        mask=row_mask,
        other=null_block_id,
    ).to(tl.int64)
    needs_reset = tl.load(needs_reset_ptr + offs, mask=row_mask, other=0).to(tl.int32)
    do_reset = row_mask & (state_batch_idx != null_block_id) & (needs_reset != 0)

    zero = tl.zeros_like(state_batch_idx).to(tl.int32)
    tl.store(history_len_ptr + state_batch_idx, zero, mask=do_reset)
    tl.store(ring_start_ptr + state_batch_idx, zero, mask=do_reset)
    # With an empty ring the kernel checkpoints iff 0 + cur_len > B, which is
    # false because cur_len <= T <= B. Recomputed rather than hardcoded so the
    # flag stays correct if that invariant is ever relaxed.
    cur_start = tl.load(query_start_loc_ptr + offs, mask=do_reset, other=0).to(tl.int32)
    cur_end = tl.load(query_start_loc_ptr + offs + 1, mask=do_reset, other=0).to(
        tl.int32
    )
    init_is_flush = ((cur_end - cur_start) > MAX_WINDOW).to(tl.int8)
    tl.store(is_flush_ptr + state_batch_idx, init_is_flush, mask=do_reset)


def commit_replayssm_spec_flashinfer(
    history_len: torch.Tensor,
    ring_start: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_batch_indices: torch.Tensor,
    max_window: int,
    ring_len: int,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Advance the block-keyed cursors for this step.

    Folds the previous step's ``num_accepted_tokens`` (which includes the bonus
    token) into the ring, then records whether the FlashInfer call about to run
    will checkpoint. Fixed launch, so the captured graph is identical every step.
    """
    batch = state_batch_indices.shape[0]
    assert ring_len > max_window, (
        f"ring_len ({ring_len}) must exceed max_window ({max_window}) by the "
        "verify-window length"
    )
    assert query_start_loc.shape[0] >= batch + 1, (
        f"query_start_loc has {query_start_loc.shape[0]} entries, need "
        f"{batch + 1} for {batch} rows"
    )
    block = max(1, triton.next_power_of_2(batch))
    with torch.accelerator.device_index(history_len.device.index):
        _commit_flashinfer_cursors_kernel[(1,)](
            history_len,
            ring_start,
            is_flush,
            num_accepted_tokens,
            query_start_loc,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            MAX_WINDOW=max_window,
            RING_LEN=ring_len,
            BLOCK_SIZE=block,
            num_warps=1,
        )


def reset_replayssm_spec_flashinfer_cursors(
    history_len: torch.Tensor,
    ring_start: torch.Tensor,
    is_flush: torch.Tensor,
    needs_reset_mask: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_batch_indices: torch.Tensor,
    max_window: int,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Empty the ring for rows entering decode on a fresh admission.

    ``needs_reset_mask`` fires exactly once per admission, so unlike the Triton
    path there is no forced-flush case to neutralise a second fire. Page
    contents are left alone -- they are ignored while ``history_len == 0``.
    """
    batch = state_batch_indices.shape[0]
    block = max(1, triton.next_power_of_2(batch))
    with torch.accelerator.device_index(history_len.device.index):
        _reset_flashinfer_cursors_kernel[(1,)](
            history_len,
            ring_start,
            is_flush,
            needs_reset_mask,
            query_start_loc,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            MAX_WINDOW=max_window,
            BLOCK_SIZE=block,
            num_warps=1,
        )


__all__ = [
    "commit_replayssm_spec_flashinfer",
    "reset_replayssm_spec_flashinfer_cursors",
]
