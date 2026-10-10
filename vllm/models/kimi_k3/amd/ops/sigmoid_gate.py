# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA output gate, attn_out * sigmoid(gate), in one kernel."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sigmoid_gate_mul_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + offs, (x * tl.sigmoid(g)).to(out_ptr.dtype.element_ty), mask=mask
    )


def sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return x * sigmoid(gate). Falls back to torch for non-contiguous input."""
    if x.shape != gate.shape or not (x.is_contiguous() and gate.is_contiguous()):
        return x * gate.sigmoid()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    block = 1024
    _sigmoid_gate_mul_kernel[(triton.cdiv(n, block),)](x, gate, out, n, BLOCK=block)
    return out
