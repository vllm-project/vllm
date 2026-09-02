# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense KSA attention for short correctness tests.

This module does not own persistent KV state. It must not be used for long
contexts or as a production attention backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KSAExpandedBatch:
    """Model-internal rows for a flattened variable-length batch."""

    expanded_input_ids: torch.Tensor
    expanded_positions: torch.Tensor
    text_row_indices: torch.Tensor
    summary_row_indices: torch.Tensor
    output_gather_indices: torch.Tensor
    row_to_request: torch.Tensor
    row_logical_positions: torch.Tensor
    is_summary: torch.Tensor
    logical_boundary_mask: torch.Tensor
    text_row_is_valid: torch.Tensor
    summary_row_is_active: torch.Tensor


@dataclass(frozen=True)
class KSAReferenceOutput:
    output: torch.Tensor
    lse: torch.Tensor
    visibility_mask: torch.Tensor | None = None


def _validate_query_start_loc(
    query_start_loc: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must be a one-dimensional boundary tensor")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_start_loc must use an integer dtype")
    if int(query_start_loc[0]) != 0 or int(query_start_loc[-1]) != token_count:
        raise ValueError("query_start_loc must start at 0 and end at token count")
    if torch.any(query_start_loc[1:] <= query_start_loc[:-1]):
        raise ValueError("empty requests are not supported")
    return query_start_loc


def infer_query_start_loc(positions: torch.Tensor) -> torch.Tensor:
    """Infer full-prompt request boundaries from logical position resets."""
    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if positions.numel() == 0:
        raise ValueError("at least one logical token is required")
    starts = torch.nonzero(positions == 0, as_tuple=False).flatten()
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError(
            "dense KSA forward requires each prompt to start at position 0"
        )
    return torch.cat(
        (
            starts.to(dtype=torch.int64),
            starts.new_tensor([positions.numel()], dtype=torch.int64),
        )
    )


def expand_ksa_batch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    query_start_loc: torch.Tensor,
    summary_chunk_size: int,
    summary_token_begin: int,
    summary_token_num: int = 1,
    num_computed_tokens: torch.Tensor | None = None,
    validate_position_steps: bool = True,
) -> KSAExpandedBatch:
    """Insert request-local virtual Summary rows into flattened logical rows."""
    if input_ids.ndim != 1 or positions.ndim != 1:
        raise ValueError("input_ids and positions must be one-dimensional")
    if input_ids.shape != positions.shape:
        raise ValueError("input_ids and positions must have the same shape")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must use an integer dtype")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must use an integer dtype")
    if summary_chunk_size <= 0:
        raise ValueError("summary_chunk_size must be positive")
    if summary_token_num <= 0:
        raise ValueError("summary_token_num must be positive")
    if summary_token_num != 1:
        raise NotImplementedError("the initial KSA implementation supports S=1 only")
    if summary_token_begin < 0:
        raise ValueError("summary_token_begin must be non-negative")

    query_start_loc = _validate_query_start_loc(
        query_start_loc.to(device=positions.device), input_ids.numel()
    )
    request_lengths = query_start_loc[1:] - query_start_loc[:-1]
    if num_computed_tokens is None:
        num_computed_tokens = torch.zeros_like(request_lengths)
    else:
        num_computed_tokens = num_computed_tokens.to(
            device=positions.device, dtype=torch.int64
        )
        if num_computed_tokens.shape != request_lengths.shape:
            raise ValueError("num_computed_tokens must contain one value per request")
        if torch.any(num_computed_tokens < 0):
            raise ValueError("num_computed_tokens must be non-negative")
    row_to_request = torch.repeat_interleave(
        torch.arange(
            request_lengths.numel(),
            device=input_ids.device,
            dtype=torch.int64,
        ),
        request_lengths,
    )

    if validate_position_steps and input_ids.numel() > 1:
        same_request = row_to_request[1:] == row_to_request[:-1]
        position_steps = positions[1:] - positions[:-1]
        if torch.any(same_request & (position_steps != 1)):
            raise ValueError("logical positions must be contiguous within each request")

    device = input_ids.device
    request_starts = torch.repeat_interleave(query_start_loc[:-1], request_lengths)
    request_query_rows = (
        torch.arange(input_ids.numel(), device=device, dtype=torch.int64)
        - request_starts
    )
    request_computed_rows = torch.repeat_interleave(
        num_computed_tokens, request_lengths
    )
    logical_positions = request_computed_rows + request_query_rows
    boundary = (logical_positions + 1).remainder(summary_chunk_size) == 0
    boundary_count = int(boundary.sum())
    expanded_count = input_ids.numel() + boundary_count * summary_token_num
    boundary_rows = boundary.to(dtype=torch.int64)
    prior_summary_rows = (
        boundary_rows.cumsum(dim=0) - boundary_rows
    ) * summary_token_num
    text_row_indices = (
        torch.arange(input_ids.numel(), device=device, dtype=torch.int64)
        + prior_summary_rows
    )

    expanded_input_ids = torch.empty(
        expanded_count, device=device, dtype=input_ids.dtype
    )
    expanded_positions = torch.empty(
        expanded_count, device=device, dtype=positions.dtype
    )
    expanded_row_to_request = torch.empty(
        expanded_count, device=device, dtype=torch.int64
    )
    row_logical_positions = torch.empty(
        expanded_count, device=device, dtype=positions.dtype
    )
    is_summary = torch.ones(expanded_count, device=device, dtype=torch.bool)

    expanded_input_ids[text_row_indices] = input_ids
    expanded_positions[text_row_indices] = positions
    expanded_row_to_request[text_row_indices] = row_to_request
    row_logical_positions[text_row_indices] = logical_positions.to(
        dtype=positions.dtype
    )
    is_summary[text_row_indices] = False

    if boundary_count:
        boundary_text_rows = text_row_indices[boundary]
        summary_offsets = torch.arange(
            1, summary_token_num + 1, device=device, dtype=torch.int64
        )
        summary_row_indices = (
            boundary_text_rows[:, None] + summary_offsets[None, :]
        ).reshape(-1)
        summary_token_offsets = torch.arange(
            summary_token_num, device=device, dtype=input_ids.dtype
        ).repeat(boundary_count)
        expanded_input_ids[summary_row_indices] = (
            summary_token_begin + summary_token_offsets
        )
        summary_positions = positions[boundary].repeat_interleave(summary_token_num)
        expanded_positions[summary_row_indices] = summary_positions
        row_logical_positions[summary_row_indices] = logical_positions[
            boundary
        ].repeat_interleave(summary_token_num)
        expanded_row_to_request[summary_row_indices] = row_to_request[
            boundary
        ].repeat_interleave(summary_token_num)
    else:
        summary_row_indices = torch.empty(0, device=device, dtype=torch.int64)

    return KSAExpandedBatch(
        expanded_input_ids=expanded_input_ids,
        expanded_positions=expanded_positions,
        text_row_indices=text_row_indices,
        summary_row_indices=summary_row_indices,
        output_gather_indices=text_row_indices,
        row_to_request=expanded_row_to_request,
        row_logical_positions=row_logical_positions,
        is_summary=is_summary,
        logical_boundary_mask=boundary,
        text_row_is_valid=torch.ones_like(boundary),
        summary_row_is_active=torch.ones(
            summary_row_indices.shape,
            device=device,
            dtype=torch.bool,
        ),
    )


def expand_ksa_cudagraph_decode(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    text_row_is_valid: torch.Tensor,
    summary_chunk_size: int,
    summary_token_begin: int,
    summary_token_num: int = 1,
) -> KSAExpandedBatch:
    """Build a fixed two-row layout for full-graph single-token decode.

    Every graph request owns one Text row and one Summary candidate row. The
    candidate is active only when the real Text row ends a Summary chunk. This
    keeps all captured shapes fixed while cache slot ``-1`` masks inactive rows.
    """
    if input_ids.ndim != 1 or positions.ndim != 1:
        raise ValueError("input_ids and positions must be one-dimensional")
    if input_ids.shape != positions.shape:
        raise ValueError("input_ids and positions must have the same shape")
    if text_row_is_valid.shape != input_ids.shape:
        raise ValueError("text_row_is_valid must contain one value per token")
    if summary_chunk_size <= 0:
        raise ValueError("summary_chunk_size must be positive")
    if summary_token_num != 1:
        raise NotImplementedError("CUDA Graph KSA currently requires S=1")

    token_count = input_ids.numel()
    device = input_ids.device
    request_indices = torch.arange(token_count, device=device, dtype=torch.int64)
    text_row_indices = request_indices * 2
    summary_row_indices = text_row_indices + 1
    summary_ids = torch.full_like(input_ids, summary_token_begin)
    expanded_input_ids = torch.stack((input_ids, summary_ids), dim=1).reshape(-1)
    expanded_positions = torch.stack((positions, positions), dim=1).reshape(-1)
    row_to_request = request_indices.repeat_interleave(2)
    row_logical_positions = expanded_positions
    is_summary = (
        torch.arange(
            token_count * 2,
            device=device,
            dtype=torch.int64,
        )
        .remainder(2)
        .eq(1)
    )
    text_row_is_valid = text_row_is_valid.to(device=device, dtype=torch.bool)
    boundary = (positions + 1).remainder(summary_chunk_size).eq(0)
    summary_row_is_active = boundary & text_row_is_valid

    return KSAExpandedBatch(
        expanded_input_ids=expanded_input_ids,
        expanded_positions=expanded_positions,
        text_row_indices=text_row_indices,
        summary_row_indices=summary_row_indices,
        output_gather_indices=text_row_indices,
        row_to_request=row_to_request,
        row_logical_positions=row_logical_positions,
        is_summary=is_summary,
        logical_boundary_mask=summary_row_is_active,
        text_row_is_valid=text_row_is_valid,
        summary_row_is_active=summary_row_is_active,
    )


def build_ksa_visibility_mask(
    expanded_batch: KSAExpandedBatch,
    *,
    summary_chunk_size: int,
    sliding_chunk_num: int,
) -> torch.Tensor:
    """Build the exact KSA visibility matrix for one decoder layer."""
    if summary_chunk_size <= 0:
        raise ValueError("summary_chunk_size must be positive")
    if sliding_chunk_num <= 0:
        raise ValueError("sliding_chunk_num must be positive")

    request = expanded_batch.row_to_request
    logical = expanded_batch.row_logical_positions.to(dtype=torch.int64)
    is_summary = expanded_batch.is_summary
    chunk = torch.div(logical, summary_chunk_size, rounding_mode="floor")

    same_request = request[:, None] == request[None, :]
    query_summary = is_summary[:, None]
    key_summary = is_summary[None, :]
    query_chunk = chunk[:, None]
    key_chunk = chunk[None, :]
    query_position = logical[:, None]
    key_position = logical[None, :]

    text_start = torch.clamp(
        (query_chunk - sliding_chunk_num) * summary_chunk_size,
        min=0,
    )
    visible_text = (
        (~key_summary) & (key_position >= text_start) & (key_position <= query_position)
    )
    visible_summary_count = torch.clamp(
        query_chunk - sliding_chunk_num,
        min=0,
    )
    visible_old_summary = key_summary & (key_chunk < visible_summary_count)
    text_query_mask = (
        (~query_summary) & same_request & (visible_text | visible_old_summary)
    )

    row_index = torch.arange(logical.numel(), device=logical.device, dtype=torch.int64)
    own_chunk_causal = (key_chunk == query_chunk) & (
        row_index[None, :] <= row_index[:, None]
    )
    summary_query_mask = query_summary & same_request & own_chunk_causal
    return text_query_mask | summary_query_mask


def dense_ksa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    expanded_batch: KSAExpandedBatch,
    *,
    summary_chunk_size: int,
    sliding_chunk_num: int,
    scale: float | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
    max_reference_len: int = 4096,
    return_debug_mask: bool = False,
) -> KSAReferenceOutput:
    """Compute dense KSA attention over flattened expanded rows."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must have shape [rows, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0]:
        raise ValueError("dense KSA attention requires aligned Q, K, and V rows")
    if query.shape[0] != expanded_batch.expanded_positions.numel():
        raise ValueError("QKV row count does not match expanded metadata")
    if query.shape[2] != key.shape[2]:
        raise ValueError("query and key head dimensions must match")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if max_reference_len <= 0:
        raise ValueError("max_reference_len must be positive")

    text_requests = expanded_batch.row_to_request[~expanded_batch.is_summary]
    logical_lengths = torch.bincount(text_requests)
    if logical_lengths.numel() and int(logical_lengths.max()) > max_reference_len:
        raise ValueError(
            f"dense KSA reference length exceeds max_reference_len={max_reference_len}"
        )

    kv_repeat = query.shape[1] // key.shape[1]
    if kv_repeat != 1:
        key = key.repeat_interleave(kv_repeat, dim=1)
        value = value.repeat_interleave(kv_repeat, dim=1)

    visibility_mask = build_ksa_visibility_mask(
        expanded_batch,
        summary_chunk_size=summary_chunk_size,
        sliding_chunk_num=sliding_chunk_num,
    )
    query_acc = query.to(dtype=accumulation_dtype)
    key_acc = key.to(dtype=accumulation_dtype)
    value_acc = value.to(dtype=accumulation_dtype)
    attention_scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("qhd,khd->hqk", query_acc, key_acc)
    scores.mul_(attention_scale)
    scores.masked_fill_(~visibility_mask.unsqueeze(0), -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value_acc)
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return KSAReferenceOutput(
        output=output.to(dtype=query.dtype),
        lse=lse,
        visibility_mask=visibility_mask if return_debug_mask else None,
    )


__all__ = [
    "KSAExpandedBatch",
    "KSAReferenceOutput",
    "build_ksa_visibility_mask",
    "dense_ksa_attention",
    "expand_ksa_batch",
    "infer_query_start_loc",
]
