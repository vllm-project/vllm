# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row gather for the MLA chunked-context loop under DCP."""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BLOCK_ROWS = 32


# A chunk's row indices start at any offset into the batch's indices, so the
# kernel does not specialize on their alignment and one compile serves every
# chunk. The k_scale pointer is read once and needs no alignment either.
@triton.jit(do_not_specialize=["row_indices_ptr", "k_scale_ptr", "num_rows"])
def _gather_context_rows_kernel(
    gathered_ptr,
    row_indices_ptr,
    k_scale_ptr,
    latent_ptr,
    k_pe_ptr,
    num_rows,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    LATENT_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    DEQUANT: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    src_rows = tl.load(row_indices_ptr + rows, mask=row_mask, other=0)
    src = gathered_ptr + src_rows.to(tl.int64)[:, None] * (LATENT_DIM + ROPE_DIM)
    dst_rows = rows.to(tl.int64)[:, None]
    if DEQUANT:
        k_scale = tl.load(k_scale_ptr)

    cols = tl.arange(0, LATENT_BLOCK)[None, :]
    mask = row_mask[:, None] & (cols < LATENT_DIM)
    latent = tl.load(src + cols, mask=mask)
    if DEQUANT:
        # The fp32 product rounded to nearest even, as the dequantizing cache
        # gather computes it.
        latent = (latent.to(tl.float32) * k_scale).to(
            latent_ptr.dtype.element_ty, fp_downcast_rounding="rtne"
        )
    tl.store(latent_ptr + dst_rows * LATENT_DIM + cols, latent, mask=mask)

    cols = tl.arange(0, ROPE_BLOCK)[None, :]
    mask = row_mask[:, None] & (cols < ROPE_DIM)
    k_pe = tl.load(src + LATENT_DIM + cols, mask=mask)
    if DEQUANT:
        k_pe = (k_pe.to(tl.float32) * k_scale).to(
            k_pe_ptr.dtype.element_ty, fp_downcast_rounding="rtne"
        )
    tl.store(k_pe_ptr + dst_rows * ROPE_DIM + cols, k_pe, mask=mask)


def _gather_context_rows_warmup_inputs(
    *,
    dtype: torch.dtype,
    kv_lora_rank: int,
    rope_dim: int,
    out_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    # With out_dtype, the rows are fp8 and are dequantized into out_dtype.
    inputs: dict[str, object] = dict(
        gathered=TritonWarmupTensor(dtype, shape=(1, kv_lora_rank + rope_dim)),
        row_indices=TritonWarmupTensor(torch.int64),
        latent=TritonWarmupTensor(out_dtype or dtype, shape=(1, 1, kv_lora_rank)),
        k_pe=TritonWarmupTensor(out_dtype or dtype, shape=(1, 1, rope_dim)),
    )
    if out_dtype is not None:
        inputs["k_scale"] = TritonWarmupTensor(torch.float32)
    return inputs


@triton_kernel_dispatcher_with_warmup(
    kernel=_gather_context_rows_kernel,
    warmup_inputs=_gather_context_rows_warmup_inputs,
)
def _gather_context_rows(
    gathered: torch.Tensor,
    row_indices: torch.Tensor,
    latent: torch.Tensor,
    k_pe: torch.Tensor,
    k_scale: torch.Tensor | None = None,
) -> DispatchSpec:
    num_rows = row_indices.shape[0]
    kv_lora_rank = latent.shape[-1]
    rope_dim = k_pe.shape[-1]
    return (triton.cdiv(num_rows, _BLOCK_ROWS),), dict(
        # Not read when copying.
        k_scale_ptr=gathered if k_scale is None else k_scale,
        num_rows=num_rows,
        LATENT_DIM=kv_lora_rank,
        ROPE_DIM=rope_dim,
        LATENT_BLOCK=triton.next_power_of_2(kv_lora_rank),
        # A NoPE layer has no rope part; the mask then skips the one-wide block.
        ROPE_BLOCK=triton.next_power_of_2(max(rope_dim, 1)),
        BLOCK_ROWS=_BLOCK_ROWS,
        DEQUANT=k_scale is not None,
        num_warps=4,
    )


def gather_dcp_context_rows(
    gathered: torch.Tensor,
    row_indices: torch.Tensor,
    kv_lora_rank: int,
    k_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the context rows out of the all-gathered DCP workspace.

    With ``k_scale``, the rows are fp8 as the KV cache stores them and are
    dequantized into ``out_dtype``: each value is widened to fp32, multiplied
    by ``k_scale`` and rounded to nearest even, which is the arithmetic of the
    dequantizing cache gather (``gather_and_maybe_dequant_cache``). Every DCP
    rank holds the same k_scale, so dequantizing the rows after the all-gather
    equals dequantizing each rank's rows before it.

    Args:
        gathered: Contiguous all-gathered rows ``[N, kv_lora_rank + rope_dim]``.
        row_indices: Contiguous ``[C]``, the row of ``gathered`` that each
            output row is copied from.
        kv_lora_rank: Width of the latent part of a row.
        k_scale: One-element fp32 tensor with the layer's k_scale, or None to
            copy the rows unchanged. It is read on the device, so an in-place
            update applies to the next call.
        out_dtype: bf16 or fp16, the dtype to dequantize into. Given exactly
            when ``k_scale`` is.

    Returns:
        The latent ``[C, 1, kv_lora_rank]`` and ``k_pe`` ``[C, 1, rope_dim]``,
        both contiguous, in ``out_dtype`` if given, else in the dtype of
        ``gathered``.

    """
    assert gathered.dim() == 2 and gathered.is_contiguous()
    assert row_indices.dim() == 1 and row_indices.is_contiguous()
    assert row_indices.dtype in (torch.int32, torch.int64)
    assert row_indices.device == gathered.device
    assert 0 < kv_lora_rank <= gathered.shape[1]
    assert (k_scale is None) == (out_dtype is None)
    if k_scale is not None:
        assert gathered.dtype == current_platform.fp8_dtype(), gathered.dtype
        assert k_scale.dtype == torch.float32 and k_scale.numel() == 1, k_scale
        assert k_scale.device == gathered.device, k_scale.device
        assert out_dtype in (torch.bfloat16, torch.float16), out_dtype
    dtype = gathered.dtype if out_dtype is None else out_dtype
    num_rows = row_indices.shape[0]
    rope_dim = gathered.shape[1] - kv_lora_rank
    latent = gathered.new_empty((num_rows, 1, kv_lora_rank), dtype=dtype)
    k_pe = gathered.new_empty((num_rows, 1, rope_dim), dtype=dtype)
    if num_rows:
        _gather_context_rows(gathered, row_indices, latent, k_pe, k_scale)
    return latent, k_pe
