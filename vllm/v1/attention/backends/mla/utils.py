# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_DEFAULT_BLOCK_SIZE = 1024


if HAS_TRITON:

    @triton.jit
    def _zero_out_decode_padding_kernel(
        out_ptr,
        seq_lens_ptr,
        row_stride,
        num_cols,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)

        if tl.load(seq_lens_ptr + row) != 0:
            return

        col_offsets = tl.arange(0, BLOCK_SIZE)
        out_ptrs = out_ptr + row * row_stride + col_offsets
        for c in tl.range(0, tl.cdiv(num_cols, BLOCK_SIZE)):
            mask = col_offsets + c * BLOCK_SIZE < num_cols
            tl.store(
                out_ptrs,
                tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
                mask=mask,
            )
            out_ptrs += BLOCK_SIZE


def _zero_out_decode_padding_triton(
    out: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Zero rows in `out` where `seq_lens == 0` using a Triton kernel."""
    if not out.is_cuda or not seq_lens.is_cuda:
        raise ValueError("out and seq_lens must be CUDA tensors.")
    if out.size(0) != seq_lens.numel():
        raise ValueError(
            f"out.size(0) {out.size()} must matchseq_lens.numel() ({seq_lens.numel()})."
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous.")

    BLOCK_SIZE = 1024

    out_2d = out.view(out.size(0), -1)
    grid = (out_2d.size(0),)
    _zero_out_decode_padding_kernel[grid](
        out_2d,
        seq_lens,
        out_2d.stride(0),
        out_2d.size(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def zero_out_decode_padding(out: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    if HAS_TRITON:
        _zero_out_decode_padding_triton(out, seq_lens)
    else:
        out[seq_lens == 0] = 0
    return out
