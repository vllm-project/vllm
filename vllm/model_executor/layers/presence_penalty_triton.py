# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place presence penalties for compact vocab-parallel sampling."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv


@triton.jit
def _apply_presence_penalty_from_counts_kernel(
    logits_ptr,
    logits_stride,
    local_token_ids_ptr,
    local_token_ids_stride,
    output_counts_ptr,
    output_counts_stride,
    request_indices_ptr,
    presence_penalties_ptr,
    num_cols,
    counts_vocab_size,
    org_vocab_start,
    num_org_elements,
    num_org_elements_padded,
    added_vocab_start,
    num_added_elements,
    BLOCK_SIZE: tl.constexpr,
    INDEXED: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < num_cols

    if INDEXED:
        local_ids = tl.load(
            local_token_ids_ptr + row * local_token_ids_stride + cols,
            mask=col_mask,
            other=0,
        )
    else:
        local_ids = cols

    is_org = local_ids < num_org_elements
    added_offsets = local_ids - num_org_elements_padded
    is_added = (added_offsets >= 0) & (added_offsets < num_added_elements)
    global_ids = tl.where(
        is_org,
        org_vocab_start + local_ids,
        added_vocab_start + added_offsets,
    )
    token_mask = col_mask & (is_org | is_added)
    token_mask &= (global_ids >= 0) & (global_ids < counts_vocab_size)

    request_idx = tl.load(request_indices_ptr + row)
    counts = tl.load(
        output_counts_ptr + request_idx * output_counts_stride + global_ids,
        mask=token_mask,
        other=0,
    )
    penalty = tl.load(presence_penalties_ptr + row)
    logits_row = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=col_mask,
        other=0.0,
    )
    logits_row -= tl.where(token_mask & (counts > 0), penalty, 0.0)
    tl.store(logits_ptr + row * logits_stride + cols, logits_row, mask=col_mask)


def apply_presence_penalty_from_counts(
    logits: torch.Tensor,
    output_token_counts: torch.Tensor,
    request_indices: torch.Tensor,
    presence_penalties: torch.Tensor,
    *,
    org_vocab_start: int,
    num_org_elements: int,
    num_org_elements_padded: int,
    added_vocab_start: int,
    num_added_elements: int,
    local_token_ids: torch.Tensor | None = None,
) -> None:
    """Subtract presence penalties without materializing a dense mask.

    ``local_token_ids`` is supplied for the compact refined candidate matrix;
    when omitted, columns are interpreted as the physical local shard layout.
    """
    if logits.numel() == 0:
        return
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("logits must be a contiguous rank-2 tensor")
    if output_token_counts.ndim != 2:
        raise ValueError("output_token_counts must be rank 2")
    if request_indices.shape != (logits.shape[0],):
        raise ValueError("request_indices must have one entry per logits row")
    if presence_penalties.shape != (logits.shape[0],):
        raise ValueError("presence_penalties must have one entry per logits row")
    if local_token_ids is not None and local_token_ids.shape != logits.shape:
        raise ValueError("local_token_ids must match logits")

    block_size = 256
    _apply_presence_penalty_from_counts_kernel[
        (logits.shape[0], cdiv(logits.shape[1], block_size))
    ](
        logits,
        logits.stride(0),
        local_token_ids,
        local_token_ids.stride(0) if local_token_ids is not None else 0,
        output_token_counts,
        output_token_counts.stride(0),
        request_indices,
        presence_penalties,
        logits.shape[1],
        output_token_counts.shape[1],
        org_vocab_start,
        num_org_elements,
        num_org_elements_padded,
        added_vocab_start,
        num_added_elements,
        BLOCK_SIZE=block_size,
        INDEXED=local_token_ids is not None,
    )


__all__ = ["apply_presence_penalty_from_counts"]
