# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row gather for the MLA chunked-context loop under DCP."""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton

_BLOCK_ROWS = 32


# A chunk's row indices start at any offset into the batch's indices, so the
# kernel does not specialize on their alignment and one compile serves every
# chunk.
@triton.jit(do_not_specialize=["row_indices_ptr", "num_rows"])
def _gather_context_rows_kernel(
    gathered_ptr,
    row_indices_ptr,
    latent_ptr,
    k_pe_ptr,
    num_rows,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    LATENT_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    src_rows = tl.load(row_indices_ptr + rows, mask=row_mask, other=0)
    src = gathered_ptr + src_rows.to(tl.int64)[:, None] * (LATENT_DIM + ROPE_DIM)
    dst_rows = rows.to(tl.int64)[:, None]

    cols = tl.arange(0, LATENT_BLOCK)[None, :]
    mask = row_mask[:, None] & (cols < LATENT_DIM)
    latent = tl.load(src + cols, mask=mask)
    tl.store(latent_ptr + dst_rows * LATENT_DIM + cols, latent, mask=mask)

    cols = tl.arange(0, ROPE_BLOCK)[None, :]
    mask = row_mask[:, None] & (cols < ROPE_DIM)
    k_pe = tl.load(src + LATENT_DIM + cols, mask=mask)
    tl.store(k_pe_ptr + dst_rows * ROPE_DIM + cols, k_pe, mask=mask)


def _gather_context_rows_warmup_inputs(
    *, dtype: torch.dtype, kv_lora_rank: int, rope_dim: int
) -> dict[str, object]:
    return dict(
        gathered=TritonWarmupTensor(dtype, shape=(1, kv_lora_rank + rope_dim)),
        row_indices=TritonWarmupTensor(torch.int64),
        latent=TritonWarmupTensor(dtype, shape=(1, 1, kv_lora_rank)),
        k_pe=TritonWarmupTensor(dtype, shape=(1, 1, rope_dim)),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_gather_context_rows_kernel,
    warmup_inputs=_gather_context_rows_warmup_inputs,
)
def _gather_context_rows(
    gathered: torch.Tensor,
    row_indices: torch.Tensor,
    latent: torch.Tensor,
    k_pe: torch.Tensor,
) -> DispatchSpec:
    num_rows = row_indices.shape[0]
    kv_lora_rank = latent.shape[-1]
    rope_dim = k_pe.shape[-1]
    return (triton.cdiv(num_rows, _BLOCK_ROWS),), dict(
        num_rows=num_rows,
        LATENT_DIM=kv_lora_rank,
        ROPE_DIM=rope_dim,
        LATENT_BLOCK=triton.next_power_of_2(kv_lora_rank),
        # A NoPE layer has no rope part; the mask then skips the one-wide block.
        ROPE_BLOCK=triton.next_power_of_2(max(rope_dim, 1)),
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=4,
    )


def gather_dcp_context_rows(
    gathered: torch.Tensor, row_indices: torch.Tensor, kv_lora_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the context rows out of the all-gathered DCP workspace.

    Args:
        gathered: Contiguous all-gathered rows ``[N, kv_lora_rank + rope_dim]``.
        row_indices: Contiguous int64 ``[C]``, the row of ``gathered`` that
            each output row is copied from.
        kv_lora_rank: Width of the latent part of a row.

    Returns:
        The latent ``[C, 1, kv_lora_rank]`` and ``k_pe`` ``[C, 1, rope_dim]``,
        both contiguous and in the dtype of ``gathered``.

    """
    assert gathered.dim() == 2 and gathered.is_contiguous()
    assert row_indices.dim() == 1 and row_indices.is_contiguous()
    assert row_indices.dtype == torch.int64
    assert row_indices.device == gathered.device
    assert 0 < kv_lora_rank <= gathered.shape[1]
    num_rows = row_indices.shape[0]
    rope_dim = gathered.shape[1] - kv_lora_rank
    latent = gathered.new_empty((num_rows, 1, kv_lora_rank))
    k_pe = gathered.new_empty((num_rows, 1, rope_dim))
    if num_rows:
        _gather_context_rows(gathered, row_indices, latent, k_pe)
    return latent, k_pe
