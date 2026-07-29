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
def _update_flashinfer_cursors_kernel(
    history_len_ptr,  # (num_blocks,) int32, = FlashInfer prev_num_accepted_tokens
    ring_start_ptr,  # (num_blocks,) int32, = FlashInfer ring_start
    is_flush_ptr,  # (num_blocks,) int8, this step's checkpoint decision
    num_accepted_ptr,  # (batch,) int32, previous step, INCLUDES the bonus token
    query_start_loc_ptr,  # (batch + 1,) int32, this step's packed offsets
    state_batch_indices_ptr,  # (batch,) int32
    admission_ptr,  # classic: (batch,) int8; V2: (max_num_reqs,) int64
    request_state_indices_ptr,  # V2: batch row -> persistent request slot
    consumed_admission_ptr,  # V2: builder-owned (max_num_reqs,) int64
    null_block_id,
    batch,
    request_batch,
    stride_state_indices_batch,
    MAX_WINDOW: tl.constexpr,  # B
    RING_LEN: tl.constexpr,  # R = B + T
    USE_ADMISSION_EPOCH: tl.constexpr,
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

    if USE_ADMISSION_EPOCH:
        request_row_valid = valid & (offs < request_batch)
        request_state_idx = tl.load(
            request_state_indices_ptr + offs,
            mask=request_row_valid,
            other=-1,
        ).to(tl.int64)
        request_valid = request_row_valid & (request_state_idx >= 0)
        admission = tl.load(
            admission_ptr + request_state_idx,
            mask=request_valid,
            other=0,
        ).to(tl.int64)
        consumed_admission = tl.load(
            consumed_admission_ptr + request_state_idx,
            mask=request_valid,
            other=0,
        ).to(tl.int64)
        needs_reset = request_valid & (admission != consumed_admission)
    else:
        request_state_idx = tl.zeros_like(state_batch_idx)
        admission = tl.zeros_like(state_batch_idx)
        needs_reset = valid & (
            tl.load(admission_ptr + offs, mask=row_mask, other=0).to(tl.int32) != 0
        )

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

    # Reset wins over the commit: accepted may belong to the request that
    # previously occupied this cursor slot.
    new_origin = tl.where(needs_reset, 0, new_origin)
    new_history = tl.where(needs_reset, 0, new_history)

    # Recompute unconditionally: a zero-accept row must not inherit the old flag.
    cur_start = tl.load(query_start_loc_ptr + offs, mask=valid, other=0).to(tl.int32)
    cur_end = tl.load(query_start_loc_ptr + offs + 1, mask=valid, other=0).to(tl.int32)
    cur_len = cur_end - cur_start
    cur_is_flush = ((new_history + cur_len) > MAX_WINDOW).to(tl.int8)

    tl.store(history_len_ptr + state_batch_idx, new_history, mask=valid)
    tl.store(ring_start_ptr + state_batch_idx, new_origin, mask=valid)
    tl.store(is_flush_ptr + state_batch_idx, cur_is_flush, mask=valid)
    if USE_ADMISSION_EPOCH:
        tl.store(
            consumed_admission_ptr + request_state_idx,
            admission,
            mask=needs_reset,
        )


def update_replayssm_spec_flashinfer_cursors(
    history_len: torch.Tensor,
    ring_start: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_batch_indices: torch.Tensor,
    max_window: int,
    ring_len: int,
    *,
    needs_reset_mask: torch.Tensor | None = None,
    request_state_indices: torch.Tensor | None = None,
    admission_epoch: torch.Tensor | None = None,
    consumed_admission_epoch: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Commit the previous step and reset newly admitted rows in one launch.

    Folds the previous step's ``num_accepted_tokens`` (which includes the bonus
    token) into the ring, unless the row is entering decode on a new admission.
    The classic runner passes a batch-order reset mask; V2 passes persistent
    admission epochs and a batch-to-request-slot mapping.
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
    epoch_args = (
        request_state_indices,
        admission_epoch,
        consumed_admission_epoch,
    )
    has_admission_epoch_arg = any(arg is not None for arg in epoch_args)
    use_admission_epoch = all(arg is not None for arg in epoch_args)
    if has_admission_epoch_arg and not use_admission_epoch:
        raise ValueError("V2 admission epoch arguments must be provided together")
    if use_admission_epoch == (needs_reset_mask is not None):
        raise ValueError(
            "pass exactly one FlashInfer admission mode: needs_reset_mask or "
            "(request_state_indices, admission_epoch, consumed_admission_epoch)"
        )

    if use_admission_epoch:
        assert request_state_indices is not None
        assert admission_epoch is not None
        assert consumed_admission_epoch is not None
        admission_ptr = admission_epoch
        request_indices_ptr = request_state_indices
        consumed_ptr = consumed_admission_epoch
        request_batch = request_state_indices.shape[0]
    else:
        assert needs_reset_mask is not None
        admission_ptr = needs_reset_mask
        # Unused in the classic specialization; valid pointers keep the launch
        # signature uniform without allocating dummy tensors.
        request_indices_ptr = state_batch_indices
        consumed_ptr = needs_reset_mask
        request_batch = 0

    block = max(1, triton.next_power_of_2(batch))
    with torch.accelerator.device_index(history_len.device.index):
        _update_flashinfer_cursors_kernel[(1,)](
            history_len,
            ring_start,
            is_flush,
            num_accepted_tokens,
            query_start_loc,
            state_batch_indices,
            admission_ptr,
            request_indices_ptr,
            consumed_ptr,
            null_block_id,
            batch,
            request_batch,
            state_batch_indices.stride(0),
            MAX_WINDOW=max_window,
            RING_LEN=ring_len,
            USE_ADMISSION_EPOCH=use_admission_epoch,
            BLOCK_SIZE=block,
            num_warps=1,
        )


__all__ = ["update_replayssm_spec_flashinfer_cursors"]
