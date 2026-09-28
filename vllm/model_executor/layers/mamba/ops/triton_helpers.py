# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


def td_compatible(t: torch.Tensor | None) -> bool:
    """Whether ``t`` can back a 2D tensor descriptor: unit inner stride and a
    16-byte aligned base and outer strides. ``None`` (unused operand) is fine.
    """
    if t is None:
        return True
    align = 16 // t.element_size()
    return (
        t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % align == 0 for s in t.stride()[:-1])
    )


@triton.jit
def fast_exp(x):
    """Faster alternative to tl.exp() using the hardware exp2 instruction.

    tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
    while tl.exp goes through libdevice __nv_expf which adds function call
    overhead and extra range checking.
    """
    # exp(x) = exp2(x * log2(e)), where log2(e) = 1/ln(2) = 1.4426950408889634
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)
