# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused ``attn_out * sigmoid(gate)`` for MLA output gating.
"""

from functools import cache

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sigmoid_mul_kernel(
    attn_ptr,
    gate_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(gate).to(gate_ptr.dtype.element_ty).to(tl.float32)
    attn = tl.load(attn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(attn_ptr + offs, attn * s, mask=mask)


@cache
def _on_gfx942() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


def fused_sigmoid_mul_(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # Fall back to non fused if layouts dont qualify
    if (
        attn_out.shape != gate.shape
        or not attn_out.is_contiguous()
        or not gate.is_contiguous()
        or attn_out.dtype != gate.dtype
    ):
        return attn_out * gate.sigmoid()
    numel = attn_out.numel()
    if numel == 0:
        return attn_out
    BLOCK = 4096 if _on_gfx942() else 1024
    _sigmoid_mul_kernel[(triton.cdiv(numel, BLOCK),)](
        attn_out, gate, numel, BLOCK=BLOCK, num_warps=4
    )
    return attn_out
