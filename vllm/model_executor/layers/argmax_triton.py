from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import next_power_of_2


if HAS_TRITON:

    @triton.jit
    def _global_pair_argmax_kernel(
        gathered_pairs,
        out_indices,
        pair_stride_0: tl.constexpr,
        pair_stride_1: tl.constexpr,
        tp_size: tl.constexpr,
        block_tp: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, block_tp)
        mask = lane < tp_size
        base = row * pair_stride_0 + lane * 2 * pair_stride_1
        values = tl.load(gathered_pairs + base, mask=mask, other=-float("inf"))
        indices = tl.load(
            gathered_pairs + base + pair_stride_1,
            mask=mask,
            other=0,
        )
        _, lane_idx = tl.max(values, axis=0, return_indices=True)
        token_id = tl.max(
            tl.where(lane == lane_idx, indices, 0.0),
            axis=0,
        ).to(tl.int32)
        tl.store(out_indices + row, token_id)

    @triton.jit
    def _indexed_argmax_kernel(
        values,
        token_ids,
        out_values,
        out_token_ids,
        value_stride_0: tl.constexpr,
        token_stride_0: tl.constexpr,
        num_candidates: tl.constexpr,
        block_candidates: tl.constexpr,
        index_offset: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, block_candidates)
        mask = lane < num_candidates
        values = tl.load(
            values + row * value_stride_0 + lane,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        token_ids = tl.load(
            token_ids + row * token_stride_0 + lane,
            mask=mask,
            other=0x7FFFFFFF,
        ).to(tl.int32)
        values = tl.where(values == values, values, -float("inf"))
        max_value = tl.max(values, axis=0)
        min_token_id = tl.min(
            tl.where(values == max_value, token_ids, 0x7FFFFFFF),
            axis=0,
        )
        tl.store(out_values + row, max_value)
        tl.store(out_token_ids + row, min_token_id + index_offset)


def indexed_argmax_triton(
    values: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    index_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for indexed_argmax_triton")
    assert values.ndim == 2
    assert values.is_cuda
    assert values.shape == token_ids.shape
    assert 0 < values.shape[1] <= 1024
    if not values.is_contiguous():
        values = values.contiguous()
    if not token_ids.is_contiguous():
        token_ids = token_ids.contiguous()

    batch_size, num_candidates = values.shape
    out_values = torch.empty(
        (batch_size,),
        device=values.device,
        dtype=torch.float32,
    )
    out_token_ids = torch.empty(
        (batch_size,),
        device=values.device,
        dtype=torch.int32,
    )
    _indexed_argmax_kernel[(batch_size,)](
        values,
        token_ids,
        out_values,
        out_token_ids,
        value_stride_0=values.stride(0),
        token_stride_0=token_ids.stride(0),
        num_candidates=num_candidates,
        block_candidates=next_power_of_2(num_candidates),
        index_offset=index_offset,
    )
    return out_values, out_token_ids


def reduce_global_argmax_triton(
    gathered_pairs: torch.Tensor,
    *,
    tp_size: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for reduce_global_argmax_triton")
    assert gathered_pairs.ndim == 2
    assert gathered_pairs.is_cuda
    assert gathered_pairs.shape[1] == tp_size * 2
    batch_size = gathered_pairs.shape[0]
    out_indices = torch.empty(
        (batch_size,),
        device=gathered_pairs.device,
        dtype=torch.int32,
    )
    _global_pair_argmax_kernel[(batch_size,)](
        gathered_pairs,
        out_indices,
        gathered_pairs.stride(0),
        gathered_pairs.stride(1),
        tp_size=tp_size,
        block_tp=next_power_of_2(tp_size),
    )
    return out_indices


__all__ = [
    "indexed_argmax_triton",
    "reduce_global_argmax_triton",
]
