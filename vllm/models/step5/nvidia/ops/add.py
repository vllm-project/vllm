# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add low-precision tensors without materializing their FP32 conversions."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["N"])
def _add_fp32(X, Y, Output, N, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
    y = tl.load(Y + offset, offset < N, other=0).to(tl.float32)
    tl.store(Output + offset, x + y, offset < N)


def add_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return x.float() + y.float(), preserving the FP32 addition."""
    if (
        not x.is_cuda
        or x.device != y.device
        or x.shape != y.shape
        or not x.is_contiguous()
        or not y.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or y.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or (x.dtype == torch.float32 and y.dtype == torch.float32)
    ):
        return x.float() + y.float()
    output = torch.empty_like(x, dtype=torch.float32)
    if x.numel():
        _add_fp32[(triton.cdiv(x.numel(), 1024),)](
            x, y, output, x.numel(), 1024, enable_fp_fusion=False
        )
    return output
