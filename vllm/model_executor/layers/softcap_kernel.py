# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for logit softcapping."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _softcap_fwd_kernel(
    X_ptr,
    Out_ptr,
    cap: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask)
    # Upcast to fp32 for tanh precision
    x_fp32 = x.to(tl.float32)
    x_fp32 = x_fp32 / cap
    x_fp32 = tl.math.tanh(x_fp32)
    x_fp32 = x_fp32 * cap
    out = x_fp32.to(x.dtype)
    tl.store(Out_ptr + offsets, out, mask=mask)


def softcap_logits(
    logits: torch.Tensor,
    soft_cap: float,
    inplace: bool = True,
) -> torch.Tensor:
    logits = logits.contiguous()
    n_elements = logits.numel()
    out = logits if inplace else torch.empty_like(logits)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _softcap_fwd_kernel[grid](
        logits,
        out,
        cap=soft_cap,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
