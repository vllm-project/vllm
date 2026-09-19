# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localize a global QSA selection to the positions one DCP rank owns.

The selector runs on a replicated cache, so every rank produces the same global
selection. The main KV is sharded, so a rank can only attend over the part of
that selection it owns. This writes a localized copy of the packed selection
buffer: owned entries become compact-local ids in a dense prefix, and the
trailing count column holds how many survived.

The copy is not optional. The selection buffer belongs to the layer and MTP
draft steps reuse the rows frozen at step 0, so localizing in place would
localize an already-local id on the next step.

Compact-local addressing, with ``W`` ranks and interleave ``I``:

    owner(g)    = (g // I) % W
    local_id(g) = (g // (W * I)) * I + (g % I)

This has to agree with the slot mapping exactly, or a rank attends over keys it
does not hold. The authority is the DCP branch of the slot-mapping kernel in
``vllm/v1/worker/block_table.py``, which works inside a virtual block of
``P * W`` positions (``P`` = page size):

    vbo  = g % (P * W)
    owner_of(g) = (vbo // I) % W
    lbo  = (vbo // (W * I)) * I + (vbo % I)

The two forms agree when ``I`` divides ``P``, and ``vllm/config/vllm.py``
asserts exactly that whenever DCP is on. Under that invariant the flat form
above reduces to ``(g // (P * W)) * P + lbo``, which is what the block table
indexes. Without it the flat form drifts by a block per virtual block, so the
impl re-checks the invariant rather than trusting it.

After this runs the attention kernel needs no change. It already reads a token
id, divides by the page size and indexes the block table, which under DCP holds
this rank's own physical blocks.
"""

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
    # The trailing column is the count the attention kernel uses as its tile
    # bound. It is never a token id.
    valid_count = tl.load(src + SELECTION_WIDTH)

    columns = tl.arange(0, BLOCK_W)
    in_range = columns < SELECTION_WIDTH
    within_count = columns < valid_count
    g = tl.load(src + columns, mask=in_range, other=-1)

    owned = (g >= 0) & in_range & within_count & (((g // INTERLEAVE) % WORLD) == RANK)
    local = (g // (WORLD * INTERLEAVE)) * INTERLEAVE + (g % INTERLEAVE)

    # Dense prefix: an owned entry lands at the count of owned entries before
    # it. Shapes never change, so this is safe under a captured graph.
    dest = tl.cumsum(owned.to(tl.int32), axis=0) - 1
    kept = tl.sum(owned.to(tl.int32), axis=0)

    # Clear first, then scatter, so a stale id from a reused buffer cannot
    # survive past the new count.
    tl.store(dst + columns, -1, mask=in_range)
    tl.store(dst + dest, local, mask=owned)
    tl.store(dst + SELECTION_WIDTH, kept)


def qsa_localize_dcp_indices(
    packed_indices: torch.Tensor,
    out: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> torch.Tensor:
    """Write this rank's owned, compact-local ids into ``out``.

    ``packed_indices`` is ``[rows, selection_width + 1]``. The final column
    holds the valid-entry count. ``out`` takes the same shape and receives the
    kept ids in a dense prefix, with its own count column.

    ``packed_indices`` is left untouched, because it is the layer's selection
    buffer and later MTP steps read it again.
    """
    if packed_indices.ndim != 2:
        raise ValueError("QSA packed indices must be two-dimensional")
    if out.shape != packed_indices.shape:
        raise ValueError("QSA localized output must match the selection shape")
    if out.dtype != packed_indices.dtype:
        raise ValueError("QSA localized output must match the selection dtype")
    rows, width_plus_count = packed_indices.shape
    if width_plus_count < 2:
        raise ValueError("QSA packed indices need selection columns plus a count")
    if not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("QSA DCP rank is outside the world")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError("QSA DCP interleave must be positive")
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


def qsa_dcp_empty_owner_rows(packed_indices: torch.Tensor) -> torch.Tensor:
    """Rows where this rank owns none of the selected positions.

    Derive emptiness from the count column, never from scanning the ids for
    ``-1``. A reused buffer holds stale ids past its count, and the kernel
    honours the count while a scan does not. The difference only shows under a
    captured graph, so an eager run passes either way.

    This is not the same as an empty shard. A rank can hold plenty of KV for a
    sequence and still own none of what the selector chose, which happens
    routinely at short context.
    """
    if packed_indices.ndim != 2 or packed_indices.shape[1] < 2:
        raise ValueError("QSA packed indices need selection columns plus a count")
    return packed_indices[:, -1] <= 0


def qsa_neutralize_empty_owner_lse_(
    lse: torch.Tensor,
    empty_rows: torch.Tensor,
) -> None:
    """Give an empty owner the identity of the cross-rank merge: ``lse = -inf``.

    The attention kernel already emits ``-inf`` for a row it never entered, so
    this restates a contract rather than repairing one. It is kept because that
    contract lives in another file, and a row that silently carries a finite
    LSE would weight garbage into every rank's result.

    The output needs no matching fix. ``correct_attn_out`` multiplies each
    rank's output by its own ``exp2(lse - global_lse)`` and then forces the row
    to zero wherever that factor is zero, so a NaN payload cannot survive. That
    is worth relying on: zeroing ``[tokens, heads, head_dim]`` here would cost a
    full read-modify-write of the partial output on every layer.

    Masked in place with no host synchronization, so this is safe to capture.
    """
    if empty_rows.dtype != torch.bool:
        raise ValueError("empty_rows must be a boolean mask")
    if lse.shape[0] != empty_rows.shape[0]:
        raise ValueError("empty_rows must have one entry per row of lse")
    mask = empty_rows.to(device=lse.device)
    lse.masked_fill_(mask.view(-1, *([1] * (lse.ndim - 1))), float("-inf"))


def warmup_qsa_localize_dcp_indices(
    *,
    selection_width: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    """Compile the localization kernel before the first request needs it.

    Only reachable under DCP, so the rest of the warmup never sees it. The row
    count is do_not_specialize'd; the constexprs below are what it specializes
    on, and they are fixed for a deployment.
    """
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
