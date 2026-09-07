# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, CUDA-graph-friendly kernels for compact top-k candidate rows."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tldevice


@triton.jit
def _pack_topk_pairs_kernel(
    local_values_ptr,
    local_ids_ptr,
    output_ptr,
    vocab_start: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < top_k
    input_offset = row_idx * top_k + offsets
    values = tl.load(local_values_ptr + input_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    ids = tl.load(local_ids_ptr + input_offset, mask=mask, other=0).to(tl.int64)
    output_offset = row_idx * top_k * 2 + offsets * 2
    tl.store(output_ptr + output_offset, values, mask=mask)
    tl.store(
        output_ptr + output_offset + 1,
        (ids + vocab_start).to(tl.float32),
        mask=mask,
    )


@triton.jit
def _select_compact_topk_pairs_kernel(
    gathered_pairs_ptr,
    output_values_ptr,
    output_ids_ptr,
    num_candidates: tl.constexpr,
    top_k: tl.constexpr,
    top_p: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    """Merge compact pairs, apply top-p, and preserve token-id tie order."""
    row_idx = tl.program_id(0)
    candidate_offsets = tl.arange(0, CANDIDATE_BLOCK)
    candidate_mask = candidate_offsets < num_candidates
    pair_offset = row_idx * num_candidates * 2 + candidate_offsets * 2
    values = tl.load(
        gathered_pairs_ptr + pair_offset,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        gathered_pairs_ptr + pair_offset + 1,
        mask=candidate_mask,
        other=float("inf"),
    ).to(tl.float32)
    values = tl.where(candidate_mask & (values == values), values, -float("inf"))

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_values = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), float("inf"), tl.float32)
    work_values = values
    work_ids = ids
    for rank in tl.static_range(0, top_k):
        max_value = tl.max(work_values, axis=0)
        equal_max = work_values == max_value
        # Candidate ids are exact FP32 values under the compact-path vocab gate.
        min_id = tl.min(tl.where(equal_max, work_ids, float("inf")), axis=0)
        winner = equal_max & (work_ids == min_id)
        top_values = tl.where(top_offsets == rank, max_value, top_values)
        top_ids = tl.where(top_offsets == rank, min_id, top_ids)
        work_values = tl.where(winner, -float("inf"), work_values)
        work_ids = tl.where(winner, float("inf"), work_ids)

    valid_top = top_offsets < top_k
    if top_p < 1.0:
        safe_max = tl.max(
            tl.where(valid_top & (top_values == top_values), top_values, -float("inf")),
            axis=0,
        )
        finite_top = valid_top & (top_values > -float("inf"))
        weights = tl.where(finite_top, tl.exp(top_values - safe_max), 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        previous_cumulative = tl.cumsum(probs, axis=0) - probs
        valid_top = valid_top & (previous_cumulative <= top_p)

    output_offset = row_idx * top_k + top_offsets
    output_mask = top_offsets < top_k
    tl.store(
        output_values_ptr + output_offset,
        tl.where(valid_top, top_values, -float("inf")),
        mask=output_mask,
    )
    tl.store(
        output_ids_ptr + output_offset,
        top_ids.to(tl.int64),
        mask=output_mask,
    )


def pack_topk_pairs(
    local_values: torch.Tensor,
    local_ids: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Pack local logits and ids into interleaved FP32 communication pairs."""
    if (
        not HAS_TRITON
        or not local_values.is_cuda
        or local_values.ndim != 2
        or local_ids.shape != local_values.shape
        or local_values.dtype != torch.float32
        or local_ids.dtype != torch.int64
    ):
        global_ids = local_ids + vocab_start
        return torch.stack(
            [local_values, global_ids.to(torch.float32)], dim=-1
        ).flatten(start_dim=-2)

    batch_size, top_k = local_values.shape
    output = torch.empty(
        (batch_size, top_k * 2), dtype=torch.float32, device=local_values.device
    )
    if batch_size == 0 or top_k == 0:
        return output
    if not local_values.is_contiguous():
        local_values = local_values.contiguous()
    if not local_ids.is_contiguous():
        local_ids = local_ids.contiguous()
    _pack_topk_pairs_kernel[(batch_size,)](
        local_values,
        local_ids,
        output,
        vocab_start=int(vocab_start),
        top_k=top_k,
        BLOCK_SIZE=triton.next_power_of_2(top_k),
    )
    return output


def select_compact_topk_pairs(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge compact pair rows and apply top-p in one CUDA kernel."""
    if (
        not HAS_TRITON
        or not gathered_pairs.is_cuda
        or gathered_pairs.ndim != 3
        or gathered_pairs.shape[-1] != 2
        or gathered_pairs.dtype != torch.float32
        or not 0 < top_k <= 64
        or not top_k <= gathered_pairs.shape[1] <= 2048
    ):
        values = gathered_pairs[..., 0]
        ids = gathered_pairs[..., 1].to(torch.int64)
        top_values, positions = torch.topk(values, top_k, dim=-1)
        top_ids = ids.gather(-1, positions)
        if top_p < 1.0:
            probs = top_values.softmax(dim=-1, dtype=torch.float32)
            remove = probs.cumsum(dim=-1) - probs > top_p
            top_values = top_values.masked_fill(remove, -float("inf"))
        return top_values, top_ids

    batch_size, num_candidates, _ = gathered_pairs.shape
    output_values = torch.empty(
        (batch_size, top_k), dtype=torch.float32, device=gathered_pairs.device
    )
    output_ids = torch.empty(
        (batch_size, top_k), dtype=torch.int64, device=gathered_pairs.device
    )
    if batch_size == 0:
        return output_values, output_ids
    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()
    _select_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        output_values,
        output_ids,
        num_candidates=num_candidates,
        top_k=top_k,
        top_p=float(top_p),
        CANDIDATE_BLOCK=triton.next_power_of_2(num_candidates),
        TOPK_BLOCK=triton.next_power_of_2(top_k),
    )
    return output_values, output_ids


@triton.jit(do_not_specialize=["num_candidates"])
def _sample_compact_topk_pairs_kernel(
    gathered_pairs_ptr,
    output_token_ids_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    num_candidates,
    top_k: tl.constexpr,
    top_p: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    """Merge compact TP pairs and sample with the keyed Gumbel stream."""
    row_idx = tl.program_id(0).to(tl.int64)
    candidate_offsets = tl.arange(0, CANDIDATE_BLOCK)
    candidate_mask = candidate_offsets < num_candidates
    pair_offset = row_idx * num_candidates * 2 + candidate_offsets * 2
    values = tl.load(
        gathered_pairs_ptr + pair_offset,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        gathered_pairs_ptr + pair_offset + 1,
        mask=candidate_mask,
        other=float("inf"),
    ).to(tl.float32)
    values = tl.where(
        candidate_mask & (values == values), values, -float("inf")
    )

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_values = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), float("inf"), tl.float32)
    work_values = values
    work_ids = ids
    for rank in tl.static_range(0, top_k):
        max_value = tl.max(work_values, axis=0)
        equal_max = work_values == max_value
        min_id = tl.min(
            tl.where(equal_max, work_ids, float("inf")), axis=0
        )
        winner = equal_max & (work_ids == min_id)
        top_values = tl.where(top_offsets == rank, max_value, top_values)
        top_ids = tl.where(top_offsets == rank, min_id, top_ids)
        work_values = tl.where(winner, -float("inf"), work_values)
        work_ids = tl.where(winner, float("inf"), work_ids)

    valid_top = top_offsets < top_k
    if top_p < 1.0:
        safe_max = tl.max(
            tl.where(
                valid_top & (top_values == top_values),
                top_values,
                -float("inf"),
            ),
            axis=0,
        )
        finite_top = valid_top & (top_values > -float("inf"))
        weights = tl.where(finite_top, tl.exp(top_values - safe_max), 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        previous_cumulative = tl.cumsum(probs, axis=0) - probs
        valid_top = valid_top & (previous_cumulative <= top_p)

    request_idx = tl.load(expanded_idx_mapping_ptr + row_idx).to(tl.int64)
    seed = tl.load(seeds_ptr + request_idx)
    pos = tl.load(pos_ptr + row_idx)
    gumbel_seed = tl.randint(seed, pos)
    u = tl_rand32(gumbel_seed, top_ids.to(tl.int64), includes_zero=False)
    noise = -tl.log(-tldevice.log1p(-u))
    scores = tl.where(valid_top, top_values + noise, -float("inf"))
    _, winner_idx = tl.max(scores, axis=0, return_indices=True)
    winner_id = tl.sum(
        tl.where(top_offsets == winner_idx, top_ids, 0.0), axis=0
    )
    has_valid = tl.sum(valid_top.to(tl.int32), axis=0) > 0
    tl.store(
        output_token_ids_ptr + row_idx,
        tl.where(has_valid, winner_id, 0.0).to(tl.int64),
    )


def sample_compact_topk_pairs(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
    expanded_idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse compact top-k/top-p merge and keyed sampling on CUDA.

    ``None`` requests the caller's reference path when the compact kernel's
    bounded candidate shape is not applicable.
    """
    if (
        not HAS_TRITON
        or not gathered_pairs.is_cuda
        or gathered_pairs.ndim != 3
        or gathered_pairs.shape[-1] != 2
        or gathered_pairs.dtype != torch.float32
        or not 0 < top_k <= 64
        or not top_k <= gathered_pairs.shape[1] <= 2048
        or expanded_idx_mapping.ndim != 1
        or expanded_idx_mapping.shape[0] != gathered_pairs.shape[0]
        or seeds.ndim != 1
        or pos.ndim != 1
    ):
        return None
    batch_size, num_candidates, _ = gathered_pairs.shape
    if batch_size == 0:
        return torch.empty(
            (0,), dtype=torch.int64, device=gathered_pairs.device
        )
    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    seeds = seeds.contiguous()
    pos = pos.contiguous()
    sampled = torch.empty(
        (batch_size,), dtype=torch.int64, device=gathered_pairs.device
    )
    _sample_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        sampled,
        expanded_idx_mapping,
        seeds,
        pos,
        num_candidates,
        top_k=top_k,
        top_p=float(top_p),
        CANDIDATE_BLOCK=triton.next_power_of_2(num_candidates),
        TOPK_BLOCK=triton.next_power_of_2(top_k),
    )
    return sampled


__all__ = [
    "pack_topk_pairs",
    "sample_compact_topk_pairs",
    "select_compact_topk_pairs",
]
