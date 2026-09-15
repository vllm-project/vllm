# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.triton_utils import tl, triton


def batch_invariant_autotune_configs(
    configs: list[triton.Config], pinned: triton.Config
) -> list[triton.Config]:
    """Autotune candidates for a Mamba2 SSD kernel.

    Under ``VLLM_BATCH_INVARIANT`` the kernel runs one fixed configuration.
    Autotuning picks a tile by timing, so two processes that share these kernels
    (a trainer and an inference engine, or two engine restarts) can otherwise
    settle on different tiles, and tile sizes change the reduction order and
    therefore the bits.
    """
    if envs.VLLM_BATCH_INVARIANT:
        return [pinned]
    return configs


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
