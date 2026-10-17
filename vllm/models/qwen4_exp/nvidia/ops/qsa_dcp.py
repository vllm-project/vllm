# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ownership helpers for QSA selections."""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_localize_dcp_kernel(
    src_ptr,
    dst_ptr,
    stride_src_row,
    stride_dst_row,
    num_rows,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if row >= num_rows:
        return

    src = src_ptr + row * stride_src_row
    dst = dst_ptr + row * stride_dst_row
    valid_count = tl.load(src + SELECTION_WIDTH)

    columns = tl.arange(0, BLOCK_W)
    in_range = columns < SELECTION_WIDTH
    within_count = columns < valid_count
    g = tl.load(src + columns, mask=in_range, other=-1)

    owned = (g >= 0) & in_range & within_count & (((g // INTERLEAVE) % WORLD) == RANK)
    local = (g // (WORLD * INTERLEAVE)) * INTERLEAVE + (g % INTERLEAVE)

    dest = tl.cumsum(owned.to(tl.int32), axis=0) - 1
    kept = tl.sum(owned.to(tl.int32), axis=0)

    tl.store(dst + columns, -1, mask=in_range & (columns >= kept))
    tl.store(dst + dest, local, mask=owned)
    tl.store(dst + SELECTION_WIDTH, kept)


def qsa_localize_dcp_indices(
    packed_indices: torch.Tensor,
    out: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> torch.Tensor:
    """Copy the selection ids this rank owns into local slots."""
    assert packed_indices.ndim == 2
    assert out.shape == packed_indices.shape
    assert out.dtype == packed_indices.dtype
    rows, width_plus_count = packed_indices.shape
    assert width_plus_count >= 2
    assert 0 <= dcp_rank < dcp_world_size
    assert cp_kv_cache_interleave_size > 0
    if dcp_world_size <= 1:
        out.copy_(packed_indices)
        return out
    if rows == 0:
        return out

    selection_width = width_plus_count - 1
    _qsa_localize_dcp_kernel[(rows,)](
        packed_indices,
        out,
        packed_indices.stride(0),
        out.stride(0),
        rows,
        WORLD=dcp_world_size,
        RANK=dcp_rank,
        INTERLEAVE=cp_kv_cache_interleave_size,
        SELECTION_WIDTH=selection_width,
        BLOCK_W=triton.next_power_of_2(max(selection_width, 1)),
    )
    return out


def warmup_qsa_localize_dcp_indices(
    *,
    selection_width: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    """Warm the QSA DCP localization kernel."""
    if dcp_world_size <= 1:
        return
    _qsa_localize_dcp_kernel.warmup(
        TritonWarmupTensor(torch.int32, shape=(16, selection_width + 1)),
        TritonWarmupTensor(torch.int32, shape=(16, selection_width + 1)),
        selection_width + 1,
        selection_width + 1,
        16,
        WORLD=dcp_world_size,
        RANK=dcp_rank,
        INTERLEAVE=cp_kv_cache_interleave_size,
        SELECTION_WIDTH=selection_width,
        BLOCK_W=triton.next_power_of_2(max(selection_width, 1)),
        grid=(16,),
    )
