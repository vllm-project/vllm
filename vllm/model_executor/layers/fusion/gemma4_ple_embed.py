# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 per-layer embedding lookup as one Triton kernel.

  out[r, :] = table[ids[rows[r]], col_start : col_start + ncols] * scale

replaces the gather over all 42 layers' columns, the scale multiply and the copy into the static
per-layer-embedding buffer (three passes over [T, 10752] in the eager path) with one pass over the
columns that are needed. Ids outside [0, vocab) map to row 0, as in ``get_per_layer_inputs``.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ple_embed_gather_kernel(
    ids_ptr,
    rows_ptr,
    table_ptr,
    scale_ptr,
    out_ptr,
    vocab,
    stride_table,
    stride_out,
    col_start,
    NCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ROWS: tl.constexpr,
):
    r = tl.program_id(0)
    if HAS_ROWS:
        src = tl.load(rows_ptr + r)
    else:
        src = r
    tok = tl.load(ids_ptr + src).to(tl.int64)
    tok = tl.where((tok >= 0) & (tok < vocab), tok, 0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < NCOLS
    v = tl.load(table_ptr + tok * stride_table + col_start + cols, mask=mask)
    s = tl.load(scale_ptr).to(tl.float32)
    v = (v.to(tl.float32) * s).to(v.dtype)
    tl.store(out_ptr + r.to(tl.int64) * stride_out + cols, v, mask=mask)


def gemma4_ple_embed_gather(
    ids: torch.Tensor,
    rows: torch.Tensor | None,
    table: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    col_start: int,
    ncols: int,
    num_rows: int,
) -> None:
    """Write ``num_rows`` rows of the scaled embedding columns into ``out`` ([>=num_rows, ncols]).

    ``rows`` (int tensor, optional) selects which entries of ``ids`` are looked up.
    """
    assert table.stride(1) == 1 and out.stride(1) == 1
    BLOCK = 512
    grid = (num_rows, triton.cdiv(ncols, BLOCK))
    _ple_embed_gather_kernel[grid](
        ids,
        rows if rows is not None else ids,
        table,
        scale,
        out,
        table.shape[0],
        table.stride(0),
        out.stride(0),
        col_start,
        NCOLS=ncols,
        BLOCK=BLOCK,
        HAS_ROWS=rows is not None,
        num_warps=2,
    )
