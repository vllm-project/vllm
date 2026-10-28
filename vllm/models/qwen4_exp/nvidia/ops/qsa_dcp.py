# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP helpers for QSA: selection ownership and the LSE combine."""

from collections.abc import Callable

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.attention.ops.dcp import CPTritonContext, cp_lse_ag_out_rs


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
    """Copy the selection ids this rank owns into local slots.

    ``out`` may be ``packed_indices`` (in place): one program handles one
    row and loads the whole row before it stores.
    """
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


@triton.jit(do_not_specialize=["num_rows", "dcp_rank"])
def _correct_qsa_dcp_heads_first_kernel(
    partial_ptr,
    lses_ptr,
    heads_first_ptr,
    num_rows,
    dcp_rank,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DCP_WORLD: tl.constexpr,
):
    """Scale one row/head of a partial output by its share of the DCP LSE.

    The arithmetic is that of upstream ``CorrectAttnCPOutKernel`` in base 2
    (same operations), so the values are bit-identical. The result goes to a
    head-major ``[heads, rows, D]`` buffer: the layout the dim-0
    reduce-scatter needs, without a copy.
    """
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim_offsets = tl.arange(0, HEAD_DIM)
    rank_offsets = tl.arange(0, DCP_WORLD)
    # lses is [world, rows, heads] contiguous.
    lse = tl.load(
        lses_ptr + rank_offsets * num_rows * NUM_HEADS + row * NUM_HEADS + head
    ).to(tl.float32)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log2(lse_acc)
    lse += lse_max
    lse_own = tl.load(
        lses_ptr + dcp_rank * num_rows * NUM_HEADS + row * NUM_HEADS + head
    ).to(tl.float32)
    lse_finally = lse_own - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp2(lse_finally)
    output = tl.load(partial_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)
    tl.store(heads_first_ptr + (head * num_rows + row) * HEAD_DIM + dim_offsets, output)


def qsa_dcp_lse_ag_out_rs(
    partial: torch.Tensor,
    lse: torch.Tensor,
    group,
    heads_first: torch.Tensor,
) -> torch.Tensor:
    """``cp_lse_ag_out_rs`` for base-2 LSE without its two layout copies.

    The LSE all-gather is the same dim-0 gather. The correction writes the
    head-major ``heads_first`` (``partial.numel()`` elements, dtype of
    ``partial``), so the reduce-scatter runs on dim 0 and returns
    ``[local_heads, rows, D]`` directly; upstream moves the head dimension
    first and back with two ``contiguous`` copies.
    """
    num_rows, num_heads, head_dim = partial.shape
    assert partial.is_contiguous() and lse.shape == (num_rows, num_heads)
    assert heads_first.dtype == partial.dtype
    lses = group.all_gather(lse.contiguous(), dim=0)
    heads_first = heads_first.view(num_heads, num_rows, head_dim)
    if num_rows:
        _correct_qsa_dcp_heads_first_kernel[(num_rows, num_heads)](
            partial,
            lses,
            heads_first,
            num_rows,
            group.rank_in_group,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DCP_WORLD=group.world_size,
            num_warps=4,
        )
    return group.reduce_scatter(heads_first, dim=0)


def qsa_dcp_combine(
    partial: torch.Tensor,
    lse: torch.Tensor,
    group,
    combine: Callable[..., torch.Tensor],
    heads_first: torch.Tensor,
    *,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e: bool = False,
) -> torch.Tensor:
    """Merge the ranks' partial outputs by their LSE.

    Returns this rank's heads as ``[rows, local_heads, D]`` (any row and head
    strides, unit last stride). ``cp_lse_ag_out_rs`` with a base-2 LSE runs as
    the copy-free ``qsa_dcp_lse_ag_out_rs`` with ``heads_first`` as its buffer;
    any other case calls ``combine`` unchanged, with ``ctx`` and
    ``is_lse_base_on_e``.
    """
    if combine is cp_lse_ag_out_rs and not is_lse_base_on_e:
        return qsa_dcp_lse_ag_out_rs(partial, lse, group, heads_first).transpose(0, 1)
    return combine(partial, lse, group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e)


@triton.jit(do_not_specialize=["stride_combined_row", "stride_combined_head"])
def _gate_qsa_dcp_output_kernel(
    combined_ptr,
    output_gate_ptr,
    output_ptr,
    stride_combined_row,
    stride_combined_head,
    stride_gate_row,
    stride_gate_head,
    stride_output_row,
    stride_output_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Write ``combined * sigmoid(gate)`` in fp32, rounded to the output dtype.

    The sigmoid is ``1 / (1 + exp(-x))`` with libdevice ``expf`` and an
    IEEE division, the formula of PyTorch's fp32 CUDA sigmoid, so the result
    matches ``out.copy_(combined.float() * torch.sigmoid(gate.float()))``.
    """
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim_offsets = tl.arange(0, BLOCK_D)
    mask = dim_offsets < HEAD_DIM
    combined = tl.load(
        combined_ptr
        + row * stride_combined_row
        + head * stride_combined_head
        + dim_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        output_gate_ptr + row * stride_gate_row + head * stride_gate_head + dim_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid = tl.math.div_rn(1.0, 1.0 + tldevice.exp(-gate))
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        (combined * sigmoid).to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def gate_qsa_dcp_output(
    combined: torch.Tensor, output_gate: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Write ``combined * sigmoid(output_gate)`` to ``out`` in one launch.

    ``combined`` is ``[rows, heads, D]`` with any row and head strides and a
    unit last stride; ``output_gate`` has the element count of ``out``.
    """
    num_rows, num_heads, head_dim = out.shape
    assert combined.shape == out.shape and combined.stride(2) == 1
    assert out.stride(2) == 1
    gate = output_gate.view_as(out)
    assert gate.stride(2) == 1
    if num_rows:
        _gate_qsa_dcp_output_kernel[(num_rows, num_heads)](
            combined,
            gate,
            out,
            combined.stride(0),
            combined.stride(1),
            gate.stride(0),
            gate.stride(1),
            out.stride(0),
            out.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
    return out
