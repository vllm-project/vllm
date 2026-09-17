# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localize a global QSA selection to the positions one DCP rank owns.

The selector runs on a replicated cache, so every rank produces the same global
selection. The main KV is sharded, so a rank can only attend over the part of
that selection it owns. This rewrites the packed selection buffer in place:
owned entries become compact-local ids in a dense prefix, and the trailing count
column is set to how many survived.

Compact-local addressing, with ``W`` ranks and interleave ``I``:

    owner(g)    = (g // I) % W
    local_id(g) = (g // (W * I)) * I + (g % I)

Do not pick the global block first and compact the offset afterwards. That form
is only correct when the block size is at least the interleave, and it walks
onto a neighbouring block otherwise.

After this runs the attention kernel needs no change. It already reads a token
id, divides by the page size and indexes the block table, which under DCP holds
this rank's own physical blocks.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qsa_localize_dcp_kernel(
    indices_ptr,
    stride_indices_row,
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

    base = indices_ptr + row * stride_indices_row
    # The trailing column is the count the attention kernel uses as its tile
    # bound. It is never a token id.
    valid_count = tl.load(base + SELECTION_WIDTH)

    columns = tl.arange(0, BLOCK_W)
    in_range = columns < SELECTION_WIDTH
    within_count = columns < valid_count
    g = tl.load(base + columns, mask=in_range, other=-1)

    owned = (g >= 0) & in_range & within_count & (((g // INTERLEAVE) % WORLD) == RANK)
    local = (g // (WORLD * INTERLEAVE)) * INTERLEAVE + (g % INTERLEAVE)

    # Dense prefix: an owned entry lands at the count of owned entries before
    # it. Shapes never change, so this is safe under a captured graph.
    dest = tl.cumsum(owned.to(tl.int32), axis=0) - 1
    kept = tl.sum(owned.to(tl.int32), axis=0)

    # Clear first, then scatter, so a stale id from a reused buffer cannot
    # survive past the new count.
    tl.store(base + columns, -1, mask=in_range)
    tl.store(base + dest, local, mask=owned)
    tl.store(base + SELECTION_WIDTH, kept)


def qsa_localize_dcp_indices(
    packed_indices: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> torch.Tensor:
    """Rewrite a global selection to this rank's owned, compact-local ids.

    ``packed_indices`` is ``[rows, selection_width + 1]``. The final column
    holds the valid-entry count and is rewritten to the number kept.

    Returns the same tensor, modified in place.
    """
    if dcp_world_size <= 1:
        return packed_indices
    if packed_indices.ndim != 2:
        raise ValueError("QSA packed indices must be two-dimensional")
    rows, width_plus_count = packed_indices.shape
    if width_plus_count < 2:
        raise ValueError("QSA packed indices need selection columns plus a count")
    if not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("QSA DCP rank is outside the world")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError("QSA DCP interleave must be positive")
    if rows == 0:
        return packed_indices

    selection_width = width_plus_count - 1
    _qsa_localize_dcp_kernel[(rows,)](
        packed_indices,
        packed_indices.stride(0),
        rows,
        WORLD=dcp_world_size,
        RANK=dcp_rank,
        INTERLEAVE=cp_kv_cache_interleave_size,
        SELECTION_WIDTH=selection_width,
        BLOCK_W=triton.next_power_of_2(max(selection_width, 1)),
    )
    return packed_indices


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


def qsa_neutralize_empty_owner_(
    out: torch.Tensor,
    lse: torch.Tensor,
    empty_rows: torch.Tensor,
) -> None:
    """Give an empty owner the identity of the cross-rank merge.

    ``out = 0`` and ``lse = -inf``. The merge weights by ``exp(lse - max)``, so
    ``-inf`` contributes nothing, and zeroing the output keeps an undefined
    payload from reaching the reduction: a stale or NaN value multiplied by a
    zero weight is still NaN, and it would corrupt every rank.

    Masked in place with no host synchronization, so this is safe to capture.
    """
    if empty_rows.dtype != torch.bool:
        raise ValueError("empty_rows must be a boolean mask")
    if out.shape[0] != empty_rows.shape[0] or lse.shape[0] != empty_rows.shape[0]:
        raise ValueError("empty_rows must have one entry per row of out and lse")
    mask = empty_rows.to(device=out.device)
    out.masked_fill_(mask.view(-1, *([1] * (out.ndim - 1))), 0.0)
    lse.masked_fill_(mask.view(-1, *([1] * (lse.ndim - 1))), float("-inf"))
