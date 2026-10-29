# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit(do_not_specialize=["size"])
def _silu_mul_kernel(X, Y, size, width: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    src = i // width * (2 * width) + i % width
    gate = tl.load(X + src, i < size, other=0).to(tl.float32)
    up = tl.load(X + src + width, i < size, other=0).to(tl.float32)
    act = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(Y + i, act * up, i < size)


def beam_silu_mul(x: torch.Tensor) -> torch.Tensor:
    """Compute SiLU and multiplication in FP32, rounding only the product."""
    width = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    if output.numel():
        _silu_mul_kernel[(triton.cdiv(output.numel(), 1024),)](
            x, output, output.numel(), width, 1024, enable_fp_fusion=False
        )
    return output


@triton.jit
def _softcap_logits_kernel(
    X,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + col, col < WIDTH, 0).to(tl.float32)
    y = libdevice.tanh(x * (1.0 / CAP)) * CAP
    tl.store(X + row * STRIDE + col, y, col < WIDTH)


def beam_softcap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Softcap in FP32 and round back into the original logits buffer."""
    assert logits.ndim == 2 and logits.stride(1) == 1
    if logits.numel():
        _softcap_logits_kernel[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
            logits,
            logits.shape[1],
            logits.stride(0),
            cap,
            1024,
            enable_fp_fusion=False,
        )
    return logits
