# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch GPU preparation for the shared-model Uno draft.

The regular Uno input path builds the same tensors with several eager Torch
operations.  This module keeps the request-state reads on the device and
performs the complete preparation in one Triton launch.  The launch owns the
full persistent buffers, including their CUDA-graph padding, so a replay never
observes values left by a differently sized request batch.

``step`` is a runtime scalar.  The kernel uses it in the same integer hash as
``vllm.v1.spec_decode.uno_noise.fill_uno_noise``.  Uno preparation runs before
the captured draft forward, so the live step must remain a launch argument.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.launch_key_debug import record_triton_launch


@triton.jit(
    do_not_specialize=["step", "target_query_len", "target_position_len"]
)
def _prepare_uno_inputs_kernel(
    # Request state and target batch inputs.
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    target_query_start_loc_ptr,
    target_positions_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    seeds_ptr,
    # Batch-local KV block table.
    block_table_ptr,
    block_table_stride,
    # Persistent Uno output buffers.
    out_input_ids_ptr,
    out_positions_ptr,
    out_slot_mapping_ptr,
    out_sample_idx_mapping_ptr,
    out_seq_lens_ptr,
    out_query_start_loc_ptr,
    # Runtime scalar bounds.  They describe the active target views without
    # making every view length a distinct Triton specialization.
    step,
    target_query_len,
    target_position_len,
    # Shape and arithmetic constants.
    NUM_REQS: tl.constexpr,
    K: tl.constexpr,
    COUNT: tl.constexpr,
    INPUT_CAP: tl.constexpr,
    POSITION_CAP: tl.constexpr,
    SLOT_CAP: tl.constexpr,
    SAMPLE_CAP: tl.constexpr,
    SEQ_CAP: tl.constexpr,
    QUERY_CAP: tl.constexpr,
    STATE_CAP: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
    NOISE_SEED: tl.constexpr,
    NOISE_LOW: tl.constexpr,
    NOISE_HIGH: tl.constexpr,
    HAS_REJECTED: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prepare all persistent output rows from one launch.

    Each fixed-size program owns a disjoint row range.  The wrapper launches
    enough programs to cover every persistent output row, so this remains one
    launch without an oversized vector or a second staging kernel.  Output
    capacities are independent constants because target input tensors may be
    slices while persistent output buffers are full graph-capacity tensors.
    """

    _NOISE_SEED_MULT: tl.constexpr = 0x1E3779B185EBCA8
    _NOISE_STEP_MULT: tl.constexpr = 0x11B54A32D192ED0
    _NOISE_MIX_1: tl.constexpr = 0x3F58476D1CE4E5B9
    _NOISE_MIX_2: tl.constexpr = 0x14D049BB133111EB
    _NOISE_MASK: tl.constexpr = (1 << 62) - 1

    pid = tl.program_id(0).to(tl.int64)
    row = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = row < COUNT
    req = row // K
    offset = row % K
    req_valid = req < NUM_REQS

    state_idx = tl.load(
        idx_mapping_ptr + req,
        mask=req_valid,
        other=0,
    ).to(tl.int64)
    state_valid = (state_idx >= 0) & (state_idx < STATE_CAP)

    rejected = tl.zeros((BLOCK,), dtype=tl.int64)
    if HAS_REJECTED:
        rejected = tl.load(
            num_rejected_ptr + req,
            mask=req_valid,
            other=0,
        ).to(tl.int64)

    target_query_end = tl.load(
        target_query_start_loc_ptr + req + 1,
        mask=req_valid & ((req + 1) < target_query_len),
        other=0,
    ).to(tl.int64)
    last_position_index = target_query_end - rejected - 1
    target_position_valid = (
        req_valid
        & (last_position_index >= 0)
        & (last_position_index < target_position_len)
    )
    last_position = tl.load(
        target_positions_ptr + last_position_index,
        mask=target_position_valid,
        other=0,
    ).to(tl.int64)
    first_position = last_position + 1
    position = first_position + offset

    # The reference path clamps positions for the model input but uses the
    # unclamped position for cache residency.  Keep those two semantics apart.
    clamped_position = tl.minimum(position, MAX_MODEL_LEN - 1)
    tl.store(
        out_positions_ptr + row,
        tl.where(active, clamped_position, 0).to(tl.int64),
        mask=row < POSITION_CAP,
    )

    sampled = tl.load(
        num_sampled_ptr + req,
        mask=req_valid,
        other=0,
    )
    last_sampled = tl.load(
        last_sampled_ptr + state_idx,
        mask=state_valid,
        other=0,
    ).to(tl.int64)
    prefill_token = tl.load(
        next_prefill_tokens_ptr + state_idx,
        mask=state_valid,
        other=0,
    ).to(tl.int64)
    bonus_token = tl.where(sampled > 0, last_sampled, prefill_token)

    # Match fill_uno_noise exactly: the request seed is expanded across K
    # rows, then the global flat output row is added before the 62-bit mix.
    request_seed = (
        tl.load(
            seeds_ptr + state_idx,
            mask=state_valid,
            other=0,
        ).to(tl.int64)
        + NOISE_SEED
    )
    step = tl.cast(step, tl.int64)
    step_term = (step * _NOISE_STEP_MULT) & _NOISE_MASK
    seed_term = (request_seed * _NOISE_SEED_MULT) & _NOISE_MASK
    mixed = (seed_term + step_term + row) & _NOISE_MASK
    mixed = ((mixed ^ (mixed >> 30)) * _NOISE_MIX_1) & _NOISE_MASK
    mixed = ((mixed ^ (mixed >> 27)) * _NOISE_MIX_2) & _NOISE_MASK
    mixed = mixed ^ (mixed >> 31)
    noise_token = mixed % (NOISE_HIGH - NOISE_LOW) + NOISE_LOW
    input_token = tl.where(offset == 0, bonus_token, noise_token)
    tl.store(
        out_input_ids_ptr + row,
        tl.where(active, input_token, 0).to(tl.int32),
        mask=row < INPUT_CAP,
    )

    # Convert batch-order rows back to persistent request-state slots for
    # sampling.  Padding must remain -1 so gumbel writes cannot use stale rows.
    tl.store(
        out_sample_idx_mapping_ptr + row,
        tl.where(active & state_valid, state_idx, -1).to(tl.int32),
        mask=row < SAMPLE_CAP,
    )

    block_number = position // BLOCK_SIZE
    block_valid = (
        active
        & (position >= 0)
        & (position < MAX_MODEL_LEN)
        & (block_number >= 0)
        & (block_number < BLOCK_COLS)
    )
    block_id = tl.load(
        block_table_ptr + req * block_table_stride + block_number,
        mask=block_valid,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block.  Treat negative/unallocated entries as
    # nonresident as well, so no invalid physical slot can be written.
    resident = block_valid & (block_id > 0)
    slot = block_id * BLOCK_SIZE + position % BLOCK_SIZE
    tl.store(
        out_slot_mapping_ptr + row,
        tl.where(resident, slot, PAD_ID).to(tl.int64),
        mask=row < SLOT_CAP,
    )

    # Per-request sequence lengths and query starts are padded through the
    # entire persistent buffers.  Every request has one lane (row == req) that
    # computes the same post-rejection first position without any host readback.
    request_row = row
    request_valid = request_row < NUM_REQS
    request_end = tl.load(
        target_query_start_loc_ptr + request_row + 1,
        mask=request_valid & ((request_row + 1) < target_query_len),
        other=0,
    ).to(tl.int64)
    request_rejected = tl.zeros((BLOCK,), dtype=tl.int64)
    if HAS_REJECTED:
        request_rejected = tl.load(
            num_rejected_ptr + request_row,
            mask=request_valid,
            other=0,
        ).to(tl.int64)
    request_last_index = request_end - request_rejected - 1
    request_position_valid = (
        request_valid
        & (request_last_index >= 0)
        & (request_last_index < target_position_len)
    )
    request_last_position = tl.load(
        target_positions_ptr + request_last_index,
        mask=request_position_valid,
        other=0,
    ).to(tl.int64)
    request_seq_len = tl.minimum(
        request_last_position + 1 + K,
        MAX_MODEL_LEN,
    )
    tl.store(
        out_seq_lens_ptr + request_row,
        tl.where(request_valid, request_seq_len, 0).to(tl.int32),
        mask=request_row < SEQ_CAP,
    )
    query_start = tl.where(request_row <= NUM_REQS, request_row * K, COUNT)
    tl.store(
        out_query_start_loc_ptr + request_row,
        query_start.to(tl.int32),
        mask=request_row < QUERY_CAP,
    )


def _target_input_lengths(input_batch: Any) -> tuple[int, int]:
    """Return logical target-view lengths as non-specialized kernel scalars."""
    return (
        int(input_batch.query_start_loc.numel()),
        int(input_batch.positions.numel()),
    )


def _prepare_uno_specialization_kwargs(
    buffers: Any,
    slot_mapping: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_reqs: int,
    k: int,
    state_capacity: int,
    block_size: int,
    max_model_len: int,
    noise_seed: int,
    noise_high: int,
    has_rejected: bool,
    block: int,
) -> dict[str, int | bool]:
    """Build the complete compile-time Uno preparation specialization.

    Target input views are deliberately absent.  Their logical lengths are
    passed as runtime scalars because both dummy warmup and real serving use
    the same persistent output buffers but legitimately have different
    target-view lengths.
    """
    return {
        "NUM_REQS": num_reqs,
        "K": k,
        "COUNT": num_reqs * k,
        "INPUT_CAP": buffers.input_ids.numel(),
        "POSITION_CAP": buffers.positions.numel(),
        "SLOT_CAP": slot_mapping.numel(),
        "SAMPLE_CAP": sample_idx_mapping.numel(),
        "SEQ_CAP": buffers.seq_lens.numel(),
        "QUERY_CAP": buffers.query_start_loc.numel(),
        "STATE_CAP": state_capacity,
        "BLOCK_COLS": block_table.shape[1],
        "BLOCK_SIZE": block_size,
        "MAX_MODEL_LEN": max_model_len,
        "NOISE_SEED": noise_seed,
        "NOISE_LOW": 1,
        "NOISE_HIGH": noise_high,
        "HAS_REJECTED": has_rejected,
        "PAD_ID": PAD_SLOT_ID,
        "BLOCK": block,
        "num_warps": 4,
    }


def prepare_uno_inputs_fused(
    buffers: Any,
    slot_mapping: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: Any,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor | None,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    k: int,
    max_model_len: int,
    noise_seed: int,
    noise_high: int,
    step: int,
) -> None:
    """Prepare Uno inputs with one Triton launch.

    The input batch fields are device tensors in MRV2.  ``input_batch.num_reqs``
    and the output buffer capacities are host-known shape metadata; no tensor
    value is copied to or inspected by the host.  ``block_table`` must already
    be the batch-local table returned by ``BlockTables.gather_block_tables``.

    The launch is parity-checked against the eager CPU
    ``prepare_uno_inputs_reference`` oracle in ``uno.py``;
    both paths must preserve the same request-state layouts and row ordering.

    ``step`` and the logical target-view lengths are runtime scalars rather
    than constexprs.  This preserves dynamic bounds while allowing dummy
    warmup and a real request to reuse the same specialization.
    """
    state_capacity = seeds.numel()
    if (
        last_sampled.ndim != 2
        or last_sampled.shape[1] != 1
        or last_sampled.shape[0] != state_capacity
        or not last_sampled.is_contiguous()
    ):
        raise ValueError("last_sampled must be contiguous with shape [max_num_reqs, 1]")
    if (
        next_prefill_tokens.ndim != 2
        or next_prefill_tokens.shape[0] < 1
        or next_prefill_tokens.shape[1] != state_capacity
        or not next_prefill_tokens.is_contiguous()
    ):
        raise ValueError(
            "next_prefill_tokens must be contiguous with shape "
            "[num_prefill_lookahead, max_num_reqs]"
        )
    if buffers.device.type != "cuda":
        raise ValueError("Fused Uno preparation requires CUDA buffers")
    if k < 1 or block_size < 1 or max_model_len < 1:
        raise ValueError(
            "Uno preparation requires positive K, block size, and max length"
        )
    if noise_high <= 1:
        raise ValueError("Uno noise range must include at least one token")

    num_reqs = int(input_batch.num_reqs)
    if num_reqs < 1:
        raise ValueError("Uno preparation requires at least one request")
    count = num_reqs * int(k)

    if input_batch.idx_mapping.numel() < num_reqs:
        raise ValueError("idx_mapping is shorter than input_batch.num_reqs")
    if num_sampled.numel() < num_reqs:
        raise ValueError("num_sampled is shorter than input_batch.num_reqs")
    if num_rejected is not None and num_rejected.numel() < num_reqs:
        raise ValueError("num_rejected is shorter than input_batch.num_reqs")
    if input_batch.query_start_loc.numel() < num_reqs + 1:
        raise ValueError("query_start_loc is shorter than input_batch.num_reqs + 1")
    if block_table.ndim != 2 or block_table.shape[0] < num_reqs:
        raise ValueError("block_table must have one batch-local row per request")
    if block_table.shape[1] < 1 or block_table.stride(1) != 1:
        raise ValueError("block_table must have contiguous rows")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative Python integer")
    if count > buffers.input_ids.numel():
        raise ValueError("input buffer is shorter than num_reqs * k")
    if count > buffers.positions.numel():
        raise ValueError("position buffer is shorter than num_reqs * k")
    if count > slot_mapping.numel() or count > sample_idx_mapping.numel():
        raise ValueError("slot/sample buffers are shorter than num_reqs * k")
    if buffers.seq_lens.numel() < num_reqs:
        raise ValueError("sequence-length buffer is shorter than num_reqs")
    if buffers.query_start_loc.numel() < num_reqs + 1:
        raise ValueError("query-start buffer is shorter than num_reqs + 1")

    output_capacity = max(
        buffers.input_ids.numel(),
        buffers.positions.numel(),
        slot_mapping.numel(),
        sample_idx_mapping.numel(),
        buffers.seq_lens.numel(),
        buffers.query_start_loc.numel(),
    )
    block = 256
    launch_grid = (triton.cdiv(output_capacity, block),)
    target_query_len, target_position_len = _target_input_lengths(input_batch)
    launch_kwargs = _prepare_uno_specialization_kwargs(
        buffers,
        slot_mapping,
        sample_idx_mapping,
        block_table,
        num_reqs=num_reqs,
        k=int(k),
        state_capacity=state_capacity,
        block_size=int(block_size),
        max_model_len=int(max_model_len),
        noise_seed=int(noise_seed),
        noise_high=int(noise_high),
        has_rejected=num_rejected is not None,
        block=block,
    )
    record_triton_launch(
        "_prepare_uno_inputs_kernel",
        _prepare_uno_inputs_kernel,
        launch_grid,
        input_batch.idx_mapping,
        num_sampled,
        num_sampled if num_rejected is None else num_rejected,
        input_batch.query_start_loc,
        input_batch.positions,
        last_sampled,
        next_prefill_tokens,
        seeds,
        block_table,
        block_table.stride(0),
        buffers.input_ids,
        buffers.positions,
        slot_mapping,
        sample_idx_mapping,
        buffers.seq_lens,
        buffers.query_start_loc,
        int(step),
        target_query_len,
        target_position_len,
        **launch_kwargs,
    )
    _prepare_uno_inputs_kernel[launch_grid](
        input_batch.idx_mapping,
        num_sampled,
        num_sampled if num_rejected is None else num_rejected,
        input_batch.query_start_loc,
        input_batch.positions,
        last_sampled,
        next_prefill_tokens,
        seeds,
        block_table,
        block_table.stride(0),
        buffers.input_ids,
        buffers.positions,
        slot_mapping,
        sample_idx_mapping,
        buffers.seq_lens,
        buffers.query_start_loc,
        int(step),
        target_query_len,
        target_position_len,
        **launch_kwargs,
    )


__all__ = ["prepare_uno_inputs_fused"]
